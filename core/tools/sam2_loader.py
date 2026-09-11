"""Shared, cross-platform SAM2 model loading.

Two kinds of "paths" live here — do not mix them up:

* The checkpoint is a real file on disk: use pathlib.Path for it.
* Hydra config names are package-relative lookup keys resolved inside the
  installed `sam2` package. They must use '/' on every OS and must stay
  plain strings — wrapping them in Path() yields '\\' on Windows and
  breaks resolution.
"""
from pathlib import Path

import torch
from sam2.build_sam import build_sam2

from core.tools import sam_registry
from services.logger import get_logger

logger = get_logger(__name__)


def _is_config_lookup_error(exc) -> bool:
    """Did Hydra fail to *find/compose* the config (as opposed to the
    checkpoint not fitting the architecture it describes)?"""
    module = type(exc).__module__ or ""
    return (module.startswith(("hydra", "omegaconf"))
            or isinstance(exc, (ImportError, FileNotFoundError)))


def _build_sam2_without_hydra(config_name, checkpoint, device):
    """Build a SAM2 model from the package's YAML without Hydra's compose().

    Fallback for the packaged app: Hydra's config search path relies on
    package resources that PyInstaller does not always expose, but the YAML
    files themselves are bundled with the sam2 package, so read the file
    directly and instantiate it the same way build_sam2 does -- including
    the dynamic-multimask post-processing overrides it applies.
    """
    import sam2
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    yaml_path = Path(sam2.__file__).resolve().parent / config_name
    if not yaml_path.is_file():
        raise FileNotFoundError(f"SAM2 config {config_name} not found at {yaml_path}")

    cfg = OmegaConf.load(yaml_path)
    for key, value in (("dynamic_multimask_via_stability", True),
                       ("dynamic_multimask_stability_delta", 0.05),
                       ("dynamic_multimask_stability_thresh", 0.98)):
        OmegaConf.update(cfg, f"model.sam_mask_decoder_extra_args.{key}", value, force_add=True)
    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model, _recursive_=True)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state)
    if missing or unexpected:
        raise RuntimeError(
            f"{Path(checkpoint).name} does not fit {config_name} "
            f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
    return model.to(device).eval()


def _build_sam2(config_name, checkpoint, device):
    """build_sam2, falling back to a direct YAML load when Hydra cannot
    resolve the config name (the usual failure inside a frozen build)."""
    try:
        return build_sam2(config_name, str(checkpoint), device=device)
    except Exception as exc:
        if not _is_config_lookup_error(exc):
            raise
        logger.info("Hydra could not resolve %r (%s); loading the YAML directly", config_name, exc)
        return _build_sam2_without_hydra(config_name, checkpoint, device)


def load_sam2_model(device, variant=None):
    """Build a SAM2-family model on `device`, working on Linux and Windows.

    `variant` is a sam_registry.SamVariant (any sam2/sam2.1 entry); omitted,
    it falls back to the registry default (SAM2 Tiny).

    Raises FileNotFoundError if the checkpoint is missing, RuntimeError if
    no known config name resolves in the installed sam2 package.
    """
    if variant is None:
        variant = sam_registry.SAM_VARIANTS[sam_registry.DEFAULT_KEY]

    checkpoint = sam_registry.checkpoint_path(variant)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"{variant.label} checkpoint not found at {checkpoint}. "
            "Download it (or pick another model) in Settings -> Models."
        )

    last_error = None
    for config_name in variant.configs:
        try:
            model = _build_sam2(config_name, checkpoint, device)
        except Exception as exc:
            last_error = exc
            logger.debug("SAM2 config %r did not resolve: %s", config_name, exc)
            continue
        logger.info("Loaded %s model (config=%s, checkpoint=%s, device=%s)",
                    variant.label, config_name, checkpoint, device)
        return model

    raise RuntimeError(
        f"Could not load {variant.label} with any known config name "
        f"{variant.configs}. Check that the 'sam2' package is installed "
        "correctly (pip install git+https://github.com/facebookresearch/sam2.git)."
    ) from last_error


def _config_mismatch(config_name, checkpoint_name):
    """Sort key putting configs that match the checkpoint filename first,
    so e.g. 'my_finetuned_sam2.1_hiera_small.pt' tries the SAM2.1 Small
    config before the seven others."""
    stem = config_name.rsplit("/", 1)[-1].removesuffix(".yaml")  # sam2.1_hiera_b+
    size = stem.rsplit("_", 1)[-1]  # t / s / b+ / l
    size_word = {"t": "tiny", "s": "small", "b+": "base_plus", "l": "large"}[size]
    mismatch = 0
    if ("2.1" in checkpoint_name) != stem.startswith("sam2.1"):
        mismatch += 2
    if size_word not in checkpoint_name and f"_{size}." not in checkpoint_name:
        mismatch += 1
    return mismatch


def load_sam2_custom_model(device, checkpoint_path):
    """Build a SAM2-family model from a user-supplied checkpoint file.

    Custom checkpoints (e.g. fine-tuned models) don't say which Hydra
    config they were trained with, so every known SAM2/2.1 config is
    tried — best filename matches first — until the state dict loads
    cleanly.

    Raises FileNotFoundError if the file is missing, RuntimeError if no
    known SAM2/2.1 architecture fits the checkpoint.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found at {checkpoint}.")

    candidates = []
    for variant in sam_registry.SAM_VARIANTS.values():
        for config_name in variant.configs:
            if config_name not in candidates:
                candidates.append(config_name)
    candidates.sort(key=lambda c: _config_mismatch(c, checkpoint.name.lower()))

    last_error = None
    for config_name in candidates:
        try:
            model = _build_sam2(config_name, checkpoint, device)
        except Exception as exc:
            last_error = exc
            logger.debug("SAM2 config %r did not fit %s: %s",
                         config_name, checkpoint.name, exc)
            continue
        logger.info("Loaded custom SAM2 model (config=%s, checkpoint=%s, device=%s)",
                    config_name, checkpoint, device)
        return model

    raise RuntimeError(
        f"{checkpoint.name} does not match any known SAM2/2.1 architecture. "
        "Custom checkpoints must be SAM2 or SAM2.1 models (e.g. fine-tuned "
        "from an official checkpoint)."
    ) from last_error

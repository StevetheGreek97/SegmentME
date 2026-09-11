"""Shared, cross-platform SAM3 model loading.

SAM3 has no Hydra configs — the architecture is built in code by
`build_sam3_image_model`, so only the checkpoint lives on disk (in the
models folder, next to the SAM2 ones -- see services.model_store).

The full SAM3 image model is ~3.4 GB, so unlike SAM2 the built predictor
is cached at module level while the tool is in use (ToolManager may
re-create the tool object, e.g. on a variant hot-reload). When the SAM
button is toggled off, ToolManager calls unload_sam3_predictor() so the
memory is returned; the next toggle rebuilds the model.
"""
import hashlib
import sys
import types

import numpy as np
import torch
from PIL import Image

from core.tools import sam_registry
from services.logger import get_logger

logger = get_logger(__name__)

# NOTE: nothing from the `sam3` package is imported at module level. Its
# import chain reaches sam3.model.edt, which does `import triton` -- a package
# that only exists next to CUDA torch on Linux -- so an eager import would
# crash the whole app on CPU builds, Windows and macOS. See
# _ensure_sam3_importable() and the lazy imports in load_sam3_predictor().

_cached_predictor = None
_cached_device = None
_cpu_patch_done = False
_triton_missing = False


def _ensure_sam3_importable():
    """Make `import sam3` possible on machines without triton.

    sam3.model.edt (a Triton GPU kernel for the Euclidean distance
    transform, used only by the video tracker this app never runs) does
    `import triton` at module level. Where triton is missing, register a
    stand-in for that one module before sam3 is imported. The name
    `triton` itself is deliberately left alone: torch checks for it and
    must keep seeing it as absent.
    """
    global _triton_missing
    try:
        import triton  # noqa: F401
        return
    except ImportError:
        _triton_missing = True
    if "sam3.model.edt" in sys.modules:
        return

    def edt_triton(*_args, **_kwargs):
        raise RuntimeError(
            "SAM3's distance transform needs Triton GPU kernels, which are not "
            "available in this build (it is only used for video tracking).")

    stand_in = types.ModuleType("sam3.model.edt")
    stand_in.__doc__ = "SegmentME stand-in for sam3.model.edt (triton is not installed)."
    stand_in.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = stand_in
    logger.info("triton is not installed; sam3.model.edt replaced by a stand-in")


def _patch_nms_for_missing_triton():
    """Without triton, sam3's mask NMS cannot handle CUDA tensors (it falls
    back to a Triton kernel when the optional torch_generic_nms package is
    absent) -- e.g. CUDA torch on Windows, which ships no triton. Route
    those through the CPU implementation instead."""
    from sam3.perflib import nms as nms_module

    if getattr(nms_module, "_segmentme_patched", False):
        return
    original = nms_module.generic_nms

    def generic_nms_cpu_routed(ious, scores, iou_threshold, *args, **kwargs):
        if ious.is_cuda:
            result = original(ious.cpu(), scores.cpu(), iou_threshold, *args, **kwargs)
            return result.to(scores.device)
        return original(ious, scores, iou_threshold, *args, **kwargs)

    nms_module.generic_nms = generic_nms_cpu_routed
    nms_module._segmentme_patched = True


def _patch_sam3_for_cpu():
    """Make sam3 buildable on machines without CUDA.

    Two sam3 modules precompute caches on device="cuda" at __init__ time
    (an optimization for torch.compile); the caches are otherwise filled
    lazily on the right device at inference. On CPU-only machines those
    lines raise "No CUDA GPUs are available", so skip/redirect them.
    """
    global _cpu_patch_done
    if _cpu_patch_done:
        return

    import torch.nn.functional as F
    from sam3.model import vitdet
    from sam3.model.position_encoding import PositionEmbeddingSine
    from sam3.model.decoder import TransformerDecoder

    # vitdet's MLP always routes fc1 through a fused addmm that casts to
    # bfloat16; the following fc2 stays float32 and errors on CPU. Use a
    # plain float32 linear + activation instead.
    def addmm_act_fp32(activation, linear, mat1):
        out = linear(mat1)
        if activation in (F.relu, torch.nn.ReLU):
            return F.relu(out)
        if activation in (F.gelu, torch.nn.GELU):
            return F.gelu(out)
        raise ValueError(f"Unexpected activation {activation}")

    vitdet.addmm_act = addmm_act_fp32

    orig_pes_init = PositionEmbeddingSine.__init__

    def pes_init_no_precompute(self, *args, **kwargs):
        kwargs.pop("precompute_resolution", None)
        orig_pes_init(self, *args, **kwargs)

    PositionEmbeddingSine.__init__ = pes_init_no_precompute

    orig_get_coords = TransformerDecoder._get_coords

    @staticmethod
    def get_coords_cpu(H, W, device):
        if str(device).startswith("cuda"):
            device = "cpu"
        return orig_get_coords(H, W, device)

    TransformerDecoder._get_coords = get_coords_cpu

    _cpu_patch_done = True
    logger.debug("Patched sam3 init-time cuda precomputations for CPU")


class Sam3InteractivePredictor:
    """Adapter giving the SAM3 image model the SAM2ImagePredictor API.

    SAM3's inst_interactive_predictor has no backbone of its own — image
    features must come from the main model via Sam3Processor.set_image and
    model.predict_inst. This wrapper hides that, so the SAM3 tool can call
    set_image/predict exactly like the SAM2 tool does.
    """

    def __init__(self, model, device):
        from sam3.model.sam3_image_processor import Sam3Processor

        self.model = model
        self.processor = Sam3Processor(model, device=device)
        self.state = None
        self._image_key = None

    def set_image(self, image):
        # Encoding an image takes ~20s on CPU, so skip it when the same
        # image is set again (every E press re-sets the current image).
        if isinstance(image, np.ndarray):
            key = (image.shape, hashlib.md5(image.tobytes()).hexdigest())
            if key == self._image_key and self.state is not None:
                logger.debug("SAM3: image unchanged, reusing encoded features")
                return
            self._image_key = key
            # Sam3Processor reads numpy dimensions as CHW; go through PIL
            # so HWC images from the app keep their true width/height.
            image = Image.fromarray(image)
        else:
            self._image_key = None
        self.state = self.processor.set_image(image)

    @torch.inference_mode()
    def predict(self, point_coords=None, point_labels=None, box=None,
                multimask_output=False):
        if self.state is None:
            raise RuntimeError("Call set_image before predict.")
        return self.model.predict_inst(
            self.state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=multimask_output,
        )


def load_sam3_predictor(device):
    """Return the SAM3 interactive image predictor on `device`.

    The predictor has the same API as SAM2ImagePredictor (set_image /
    predict with point_coords, point_labels, box). Built once per process;
    subsequent calls return the cached instance.

    Raises FileNotFoundError if the checkpoint is missing.
    """
    global _cached_predictor, _cached_device

    if _cached_predictor is not None and _cached_device == str(device):
        return _cached_predictor

    variant = sam_registry.SAM_VARIANTS["sam3"]
    checkpoint = sam_registry.checkpoint_path(variant)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SAM3 checkpoint not found at {checkpoint}. Download sam3.pt from "
            f"{variant.download_url} (gated: request access first) and import "
            "it in Settings -> Models."
        )

    _ensure_sam3_importable()
    from sam3.model_builder import build_sam3_image_model

    if _triton_missing:
        _patch_nms_for_missing_triton()
    if not torch.cuda.is_available():
        _patch_sam3_for_cpu()

    logger.info("Building SAM3 model (checkpoint=%s, device=%s) — this can "
                "take a while on CPU...", checkpoint, device)
    model = build_sam3_image_model(
        device=device,
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=True,
    )

    _cached_predictor = Sam3InteractivePredictor(model, device)
    _cached_device = str(device)
    logger.info("Loaded SAM3 model (checkpoint=%s, device=%s)", checkpoint, device)
    return _cached_predictor


def unload_sam3_predictor():
    """Drop the cached SAM3 predictor so its ~3.4 GB can be reclaimed.

    Callers should follow up with gc.collect(). The next
    load_sam3_predictor call rebuilds the model from the checkpoint.
    """
    global _cached_predictor, _cached_device
    if _cached_predictor is None:
        return
    _cached_predictor = None
    _cached_device = None
    logger.info("Unloaded cached SAM3 model")

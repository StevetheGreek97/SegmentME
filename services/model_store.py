"""Where model checkpoints live, how they are found, and how they get there.

This module is deliberately free of PyQt6 imports: the inference and
training worker subprocesses use it too, and they must stay lightweight.

Layout
------
Checkpoints are plain files in ONE flat "models" folder that sits next to
the application:

* frozen (PyInstaller) build .... <folder containing the exe>/models
                                  (macOS: the folder containing the .app)
* running from source ........... <repo root>/models

The SEGMENTME_MODELS_DIR environment variable overrides that location.
If the folder next to the app cannot be written to (e.g. the app was
installed into Program Files / /Applications by an administrator),
downloads fall back to a per-user folder that is always writable, and
lookups check both places. Running from source, the legacy dev layout
(sam2_configs/ and models/dextr/) is searched as well so an existing
checkout keeps working unchanged.

Only the SAM2 Tiny checkpoint ships inside the packaged app (build.py
copies it into the models folder). Everything else is fetched on demand
through `download()` -- from Settings -> Models, or from the prompt shown
when a tool needs a model that is not installed yet -- or hand-copied in
with `import_checkpoint()` for gated models such as SAM3.
"""
import os
import platform
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ENV_MODELS_DIR = "SEGMENTME_MODELS_DIR"
_USER_DIR_NAME = "segmentme"  # same per-user folder family as logs/recent.json

_SAM2_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824"
_SAM21_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"


class DownloadCancelled(Exception):
    """Raised by download()/import_checkpoint() when should_cancel() says stop."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    family: str          # "sam2" (covers 2.1), "sam3", "dextr"
    checkpoint: str      # filename inside the models folder
    download_url: str = ""
    size_bytes: int = 0  # exact size of the file when known (0 = unknown)
    description: str = ""
    configs: tuple = ()  # Hydra config name candidates (sam2 family only)
    gated: bool = False  # cannot be fetched anonymously; user imports the file

    @property
    def size_label(self) -> str:
        return format_size(self.size_bytes) if self.size_bytes else "size unknown"


MODELS = {
    m.key: m for m in (
        ModelSpec(
            key="sam2_tiny", label="SAM2 Tiny", family="sam2",
            checkpoint="sam2_hiera_tiny.pt",
            configs=("configs/sam2/sam2_hiera_t.yaml", "sam2_hiera_t.yaml"),
            download_url=f"{_SAM2_BASE_URL}/sam2_hiera_tiny.pt",
            size_bytes=155_906_050,
            description="Fastest, lowest accuracy. Included with the app.",
        ),
        ModelSpec(
            key="sam2_small", label="SAM2 Small", family="sam2",
            checkpoint="sam2_hiera_small.pt",
            configs=("configs/sam2/sam2_hiera_s.yaml", "sam2_hiera_s.yaml"),
            download_url=f"{_SAM2_BASE_URL}/sam2_hiera_small.pt",
            size_bytes=184_309_650,
            description="Fast, a bit more accurate than Tiny.",
        ),
        ModelSpec(
            key="sam2_base_plus", label="SAM2 Base+", family="sam2",
            checkpoint="sam2_hiera_base_plus.pt",
            configs=("configs/sam2/sam2_hiera_b+.yaml", "sam2_hiera_b+.yaml"),
            download_url=f"{_SAM2_BASE_URL}/sam2_hiera_base_plus.pt",
            size_bytes=323_493_298,
            description="Balanced speed/accuracy.",
        ),
        ModelSpec(
            key="sam2_large", label="SAM2 Large", family="sam2",
            checkpoint="sam2_hiera_large.pt",
            configs=("configs/sam2/sam2_hiera_l.yaml", "sam2_hiera_l.yaml"),
            download_url=f"{_SAM2_BASE_URL}/sam2_hiera_large.pt",
            size_bytes=897_952_466,
            description="Most accurate SAM2, slower.",
        ),
        ModelSpec(
            key="sam2.1_tiny", label="SAM2.1 Tiny", family="sam2",
            checkpoint="sam2.1_hiera_tiny.pt",
            configs=("configs/sam2.1/sam2.1_hiera_t.yaml",),
            download_url=f"{_SAM21_BASE_URL}/sam2.1_hiera_tiny.pt",
            size_bytes=156_008_466,
            description="Improved SAM2 release, fastest.",
        ),
        ModelSpec(
            key="sam2.1_small", label="SAM2.1 Small", family="sam2",
            checkpoint="sam2.1_hiera_small.pt",
            configs=("configs/sam2.1/sam2.1_hiera_s.yaml",),
            download_url=f"{_SAM21_BASE_URL}/sam2.1_hiera_small.pt",
            size_bytes=184_416_285,
            description="Improved SAM2 release, fast.",
        ),
        ModelSpec(
            key="sam2.1_base_plus", label="SAM2.1 Base+", family="sam2",
            checkpoint="sam2.1_hiera_base_plus.pt",
            configs=("configs/sam2.1/sam2.1_hiera_b+.yaml",),
            download_url=f"{_SAM21_BASE_URL}/sam2.1_hiera_base_plus.pt",
            size_bytes=323_606_802,
            description="Improved SAM2 release, balanced.",
        ),
        ModelSpec(
            key="sam2.1_large", label="SAM2.1 Large", family="sam2",
            checkpoint="sam2.1_hiera_large.pt",
            configs=("configs/sam2.1/sam2.1_hiera_l.yaml",),
            download_url=f"{_SAM21_BASE_URL}/sam2.1_hiera_large.pt",
            size_bytes=898_083_611,
            description="Improved SAM2 release, most accurate.",
        ),
        ModelSpec(
            key="sam3", label="SAM3", family="sam3",
            checkpoint="sam3.pt",
            download_url="https://huggingface.co/facebook/sam3",
            size_bytes=3_450_062_241,
            description="Newest and most accurate; large and slow on CPU. "
                        "Gated on Hugging Face: request access, download "
                        "sam3.pt yourself, then import it here.",
            gated=True,
        ),
        ModelSpec(
            key="dextr", label="DEXTR", family="dextr",
            checkpoint="dextr_pascal-sbd.pth",
            download_url="https://data.vision.ee.ethz.ch/csergi/share/DEXTR/dextr_pascal-sbd.pth",
            size_bytes=195_710_751,
            description="Deep Extreme Cut: segment from 4 extreme points.",
        ),
    )
}


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------
def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """Repository root when running from source."""
    return Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    """The folder the application lives in.

    Frozen: the folder holding the executable, or -- for a macOS .app
    bundle -- the folder holding the bundle, so "next to the app" means
    what a user would expect. From source: the repo root.
    """
    if not is_frozen():
        return project_root()
    exe = Path(sys.executable).resolve()
    if platform.system() == "Darwin" and exe.parent.name == "MacOS" \
            and exe.parent.parent.name == "Contents":
        return exe.parents[2].parent  # <dir>/SegmentME.app/Contents/MacOS/exe
    return exe.parent


def user_data_dir() -> Path:
    """Per-user, always-writable app folder (same family as logs/recent.json)."""
    system = platform.system()
    if system == "Windows":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
        return base / _USER_DIR_NAME
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / _USER_DIR_NAME
    return Path.home() / ".config" / _USER_DIR_NAME


def models_dir() -> Path:
    """The primary models folder (next to the app, or the env override)."""
    override = os.environ.get(ENV_MODELS_DIR)
    if override:
        return Path(override).expanduser()
    return app_dir() / "models"


def user_models_dir() -> Path:
    return user_data_dir() / "models"


def search_dirs() -> list:
    """Every folder a checkpoint may be found in, most preferred first."""
    dirs = [models_dir(), user_models_dir()]
    if is_frozen():
        dirs.append(Path(getattr(sys, "_MEIPASS", app_dir())) / "models")
    else:
        root = project_root()
        # Legacy dev layout: SAM checkpoints in sam2_configs/, DEXTR in models/dextr/
        dirs += [root / "sam2_configs", root / "models" / "dextr"]
    seen, unique = set(), []
    for d in dirs:
        key = str(d)
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def writable_models_dir() -> Path:
    """Where downloads and imports go: the folder next to the app when it
    can be written to, otherwise the per-user folder. Created on demand."""
    errors = []
    for candidate in (models_dir(), user_models_dir()):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.touch()
            probe.unlink()
            return candidate
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")
    raise OSError("No writable models folder:\n" + "\n".join(errors))


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------
def find_checkpoint(filename: str):
    """Path of `filename` in the first search dir that has it, else None."""
    for directory in search_dirs():
        candidate = directory / filename
        if candidate.is_file():
            return candidate
    return None


def checkpoint_path(spec: ModelSpec) -> Path:
    """Where the checkpoint is, or where it would be put if downloaded."""
    return find_checkpoint(spec.checkpoint) or (models_dir() / spec.checkpoint)


def is_available(spec: ModelSpec) -> bool:
    return find_checkpoint(spec.checkpoint) is not None


def installed_yolo_models() -> list:
    """User-provided YOLO weights: every .pt in a `yolo/` subfolder of any
    models folder (plus the legacy repo models/yolo when running from source)."""
    dirs = [d / "yolo" for d in search_dirs()]
    if not is_frozen():
        dirs.append(project_root() / "models" / "yolo")
    found, seen = [], set()
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.pt")):
            if path.name not in seen:
                seen.add(path.name)
                found.append(path)
    return found


def trained_models(project_root_dir) -> list:
    """best.pt of every training run inside a project (newest first)."""
    if not project_root_dir:
        return []
    trainings = Path(project_root_dir) / "trainings"
    if not trainings.is_dir():
        return []
    weights = list(trainings.glob("*/*/weights/best.pt"))
    weights.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return weights


# --------------------------------------------------------------------------
# Transfers
# --------------------------------------------------------------------------
def format_size(num_bytes: int) -> str:
    """Decimal units, matching how the model publishers (and browsers) quote sizes."""
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"


def _content_range_total(response, fallback: int) -> int:
    header = response.headers.get("Content-Range", "")  # "bytes 100-999/1000"
    if "/" in header:
        try:
            return int(header.rsplit("/", 1)[1])
        except ValueError:
            pass
    return fallback


def download_url(url: str, target: Path, expected_size: int = 0,
                 progress=None, should_cancel=None, chunk_size: int = 1 << 20) -> Path:
    """Stream `url` into `target`, resuming a previous partial download.

    Data is written to `<target>.part` and renamed only once complete and
    (when `expected_size` is known) the right size. `progress(done, total)`
    is called after every chunk; `should_cancel()` returning True aborts
    with DownloadCancelled, keeping the partial file for a later resume.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    existing = part.stat().st_size if part.exists() else 0

    if expected_size and existing == expected_size:
        part.replace(target)
        return target

    headers = {"User-Agent": "SegmentME-model-downloader"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and existing:
            # Server says our partial file is already past the end: start over.
            part.unlink()
            return download_url(url, target, expected_size, progress, should_cancel, chunk_size)
        raise RuntimeError(f"Download failed: HTTP {exc.code} {exc.reason} ({url})") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Download failed: {exc.reason} ({url})") from exc

    with response:
        if existing and response.status == 206:
            mode = "ab"
            total = _content_range_total(response, expected_size)
        else:
            mode, existing = "wb", 0
            total = int(response.headers.get("Content-Length") or 0) or expected_size

        done = existing
        with open(part, mode) as fh:
            while True:
                if should_cancel is not None and should_cancel():
                    raise DownloadCancelled()
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)

    actual = part.stat().st_size
    if expected_size and actual != expected_size:
        part.unlink()
        raise RuntimeError(
            f"Downloaded file is {format_size(actual)} but {format_size(expected_size)} "
            f"was expected; the download was discarded. Please try again."
        )
    part.replace(target)
    return target


def download(spec: ModelSpec, progress=None, should_cancel=None) -> Path:
    """Fetch `spec`'s checkpoint into the writable models folder."""
    if spec.gated or not spec.download_url:
        raise ValueError(f"{spec.label} cannot be downloaded automatically; import the file instead.")
    target = writable_models_dir() / spec.checkpoint
    return download_url(spec.download_url, target, spec.size_bytes, progress, should_cancel)


def import_checkpoint(source, spec: ModelSpec = None, progress=None,
                      should_cancel=None, chunk_size: int = 8 << 20) -> Path:
    """Copy a checkpoint the user obtained by hand into the models folder.

    With `spec` given, the copy takes the filename the app expects for that
    model (so e.g. any download of SAM3 becomes sam3.pt). Returns the path
    of the installed file.
    """
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"{source} is not a file.")
    if spec is not None and spec.size_bytes and source.stat().st_size != spec.size_bytes:
        raise ValueError(
            f"{source.name} is {format_size(source.stat().st_size)}, but {spec.label} "
            f"should be {spec.size_label}. This does not look like the right file."
        )

    target = writable_models_dir() / (spec.checkpoint if spec else source.name)
    if source.resolve() == target.resolve():
        return target

    part = target.with_name(target.name + ".part")
    total = source.stat().st_size
    done = 0
    try:
        with open(source, "rb") as fin, open(part, "wb") as fout:
            while True:
                if should_cancel is not None and should_cancel():
                    raise DownloadCancelled()
                chunk = fin.read(chunk_size)
                if not chunk:
                    break
                fout.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    part.replace(target)
    return target

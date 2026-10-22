"""
How big a pixel is, per image.

Two sources, in this order:
  1. The resolution stored in the image file (DPI / pixels per cm), when
     "use_file_dpi" is on and the file has a real one (flatbed scans do:
     2400 DPI = 10.58 um per pixel). Values that are just software defaults
     (72 / 96 DPI) are ignored.
  2. The project scale, set by drawing along a ruler or typing a value; used
     for images without a stored resolution.

Saved in the project file (.SEproj) under "scale". Masks stay in pixels;
only what is shown or exported is converted, so changing the scale never
changes an annotation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

from services.logger import get_logger
from services.pil_limits import large_images_allowed

logger = get_logger(__name__)

UM_PER_INCH = 25400.0
UM_PER_CM = 10000.0
# DPI values that image software writes by default; they say nothing about the
# real size (a 72 DPI "scan" is a screenshot or a camera JPEG).
_DEFAULT_DPI = {1.0, 72.0, 96.0}
_MIN_REAL_DPI = 50.0

# Units offered for display / export: name -> micrometres per unit
UNITS = {"mm": 1000.0, "cm": 10000.0, "µm": 1.0}

# Sources of a scale (shown to the user)
FROM_FILE = "file"
FROM_PROJECT = "project"


def _dpi_from_header(im) -> Optional[float]:
    """Horizontal resolution in pixels per inch from an open PIL image, or None."""
    if im.format == "TIFF":
        tags = im.tag_v2
        res = tags.get(282)            # XResolution
        unit = tags.get(296, 2)        # ResolutionUnit: 1 none, 2 inch, 3 cm
        if res is None or unit == 1:
            return None
        try:
            value = float(res)
        except (TypeError, ValueError, ZeroDivisionError):
            return None
        return value * 2.54 if unit == 3 else value
    dpi = im.info.get("dpi")
    if not dpi:
        return None
    try:
        return float(dpi[0] if isinstance(dpi, (tuple, list)) else dpi)
    except (TypeError, ValueError):
        return None


def file_um_per_px(path) -> Optional[float]:
    """Micrometres per pixel from the resolution stored in the image file,
    or None when it has none (or only a software default such as 72 DPI).
    Reads the header only."""
    try:
        with large_images_allowed(), Image.open(path) as im:
            dpi = _dpi_from_header(im)
    except Exception as exc:
        logger.debug("No resolution read from %s: %s", path, exc)
        return None
    # PNG stores pixels per metre, so 96 DPI comes back as 96.012: compare loosely
    if dpi is None:
        return None
    if dpi < _MIN_REAL_DPI or any(abs(dpi - d) < 0.5 for d in _DEFAULT_DPI):
        logger.debug("Ignoring %.4g DPI in %s (a software default, not a real size)", dpi, path)
        return None
    return UM_PER_INCH / dpi


def dpi_from_um_per_px(um_per_px: float) -> float:
    return UM_PER_INCH / um_per_px


def describe(um_per_px: Optional[float]) -> str:
    """'10.58 µm per pixel (2400 DPI, 1 mm = 94.5 px)' -- for the user."""
    if not um_per_px:
        return "no scale (pixels)"
    return (f"{um_per_px:.4g} µm per pixel ({dpi_from_um_per_px(um_per_px):.0f} DPI, "
            f"1 mm = {1000.0 / um_per_px:.1f} px)")


@dataclass
class ProjectScale:
    """The project's scale settings plus a per-image cache of file resolutions."""
    um_per_px: Optional[float] = None    # for images without their own resolution
    source: str = ""                     # how um_per_px was set: "ruler", "typed" or ""
    use_file_dpi: bool = True
    unit: str = "mm"                     # display / export unit for lengths
    show_bar: bool = True
    _file_cache: Dict[str, Optional[float]] = field(default_factory=dict, repr=False)

    # ---------------------------------------------------------------- storage
    @classmethod
    def from_dict(cls, data) -> "ProjectScale":
        data = data if isinstance(data, dict) else {}
        value = data.get("um_per_px")
        try:
            value = float(value) if value else None
        except (TypeError, ValueError):
            logger.warning("Ignoring invalid project scale %r in the project file", value)
            value = None
        unit = data.get("unit", "mm")
        if unit not in UNITS:
            logger.warning("Unknown unit %r in the project file; using mm", unit)
        return cls(um_per_px=value if value and value > 0 else None,
                   source=str(data.get("source", "")),
                   use_file_dpi=bool(data.get("use_file_dpi", True)),
                   unit=unit if unit in UNITS else "mm",
                   show_bar=bool(data.get("show_bar", True)))

    def to_dict(self) -> dict:
        return {"um_per_px": self.um_per_px, "source": self.source,
                "use_file_dpi": self.use_file_dpi, "unit": self.unit,
                "show_bar": self.show_bar}

    @classmethod
    def load(cls, project_root) -> "ProjectScale":
        from services.seproj import read_seproj
        return cls.from_dict(read_seproj(project_root).get("scale"))

    def save(self, project_root):
        from services.seproj import read_seproj, write_seproj
        data = read_seproj(project_root)
        data["scale"] = self.to_dict()
        data.setdefault("name", Path(project_root).name)
        write_seproj(project_root, data)
        logger.info("Saved scale to the project file: %s; file DPI %s; unit %s",
                    describe(self.um_per_px), "used" if self.use_file_dpi else "ignored", self.unit)

    # ---------------------------------------------------------------- lookup
    def file_scale(self, image_path) -> Optional[float]:
        """The file's own resolution (cached; the header is read once)."""
        key = str(image_path)
        if key not in self._file_cache:
            value = file_um_per_px(image_path)
            self._file_cache[key] = value
            logger.debug("Resolution of %s: %s", Path(key).name,
                         describe(value) if value else "none stored")
        return self._file_cache[key]

    def for_image(self, image_path) -> Tuple[Optional[float], str]:
        """(micrometres per pixel, FROM_FILE / FROM_PROJECT / "") for an image."""
        if self.use_file_dpi and image_path:
            value = self.file_scale(image_path)
            if value:
                return value, FROM_FILE
        if self.um_per_px:
            return self.um_per_px, FROM_PROJECT
        return None, ""

    @property
    def unit_um(self) -> float:
        return UNITS[self.unit]

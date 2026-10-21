"""
Import images together with their YOLO polygon annotations into a project.

A dataset is a folder of images plus one .txt per image (same file name),
either side by side or in an images/ + labels/ pair (the layout this app's own
YOLO export writes, optionally split into train/val/test subfolders), and a
data.yaml whose `names` give the class of each class number.

Each .txt line is one polygon: `class x1 y1 x2 y2 ... xn yn`, coordinates as
fractions (0-1) of the image size. The rules:

* A trailing confidence value (prediction files such as HiReS output, which
  leave an odd number of values) is ignored and the polygon imported.
* A .txt with any broken line -- text instead of numbers, a box (`class x y w
  h`) instead of a polygon, a class number missing from data.yaml, too few
  values, coordinates outside the image -- is skipped as a whole; its image is
  imported without masks and the reason reported.
* Polygons without area are dropped (reported); the rest of the file is kept.
* An image without a .txt is imported without masks (reported); an empty .txt
  simply means "no objects".
* Images whose name is already in the project are skipped (reported), as are
  later images with the same name as an earlier one in the dataset: masks are
  stored per image name, so two images may never share one.

Nothing here touches the GUI; ui/dialogs/import_dialog.py drives it.
"""
import math
import re
import shutil
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from services.logger import get_logger

logger = get_logger(__name__)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
# Coordinates may overshoot the image edge by this much (rounding) and are
# clipped; anything further out makes the line broken.
COORD_TOLERANCE = 0.01
# Valid class names in this app (see the sidebar's Add class).
CLASS_NAME_RE = re.compile(r"[A-Za-z0-9_]+")


class ImportCancelled(Exception):
    """The user cancelled a scan or an import."""


class ImportFailed(Exception):
    """The import could not be done (message is shown to the user)."""


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
@dataclass
class LabelResult:
    """What a .txt file contributes: polygons, or why it is skipped."""
    polygons: List[Tuple[int, np.ndarray]] = field(default_factory=list)  # (class, Nx2 in 0-1)
    error: Optional[str] = None     # set: the whole file is skipped
    dropped_empty: int = 0          # polygons without area, left out
    confidence_ignored: int = 0     # lines whose trailing confidence was ignored


@dataclass
class ImageEntry:
    path: Path
    stem: str
    status: str                      # "import", "exists", "duplicate" or "unreadable"
    note: str = ""                   # why it is skipped (status != "import")
    size: Optional[Tuple[int, int]] = None   # (width, height) as the app shows it
    label_path: Optional[Path] = None
    label: Optional[LabelResult] = None

    @property
    def masks_ok(self) -> bool:
        return self.label is not None and self.label.error is None


@dataclass
class ScanResult:
    images_dir: Path
    labels_dir: Path
    class_names: Dict[int, str]
    entries: List[ImageEntry]
    orphan_labels: List[Path]        # .txt files with no image of that name

    def with_status(self, status) -> List[ImageEntry]:
        return [e for e in self.entries if e.status == status]

    @property
    def to_import(self) -> List[ImageEntry]:
        return self.with_status("import")

    @property
    def without_label(self) -> List[ImageEntry]:
        return [e for e in self.to_import if e.label is None]

    @property
    def broken_labels(self) -> List[ImageEntry]:
        return [e for e in self.to_import if e.label is not None and e.label.error]

    @property
    def class_counts(self) -> Counter:
        """Masks per class number among the masks that will be imported."""
        counts = Counter()
        for e in self.to_import:
            if e.masks_ok:
                counts.update(cls for cls, _ in e.label.polygons)
        return counts

    @property
    def mask_count(self) -> int:
        return sum(self.class_counts.values())

    @property
    def import_bytes(self) -> int:
        return sum(_file_size(e.path) for e in self.to_import)


@dataclass
class ImportResult:
    images: List[Path]               # copied into the project
    masks: int
    classes_added: List[str]
    classes_merged: List[Tuple[str, str]]   # (name in data.yaml, project class)


# --------------------------------------------------------------------------
# Dataset layout and class names
# --------------------------------------------------------------------------
def detect_layout(folder) -> Tuple[Path, Path, Optional[Path]]:
    """
    (images folder, labels folder, data.yaml or None) for a chosen folder:
    a dataset root with images/ (+ labels/), an images/ folder with a sibling
    labels/, or a folder holding images and .txt files side by side.
    """
    folder = Path(folder)
    if (folder / "images").is_dir():
        images_dir = folder / "images"
        labels_dir = folder / "labels" if (folder / "labels").is_dir() else images_dir
    elif folder.name.lower() == "images" and (folder.parent / "labels").is_dir():
        images_dir, labels_dir = folder, folder.parent / "labels"
    else:
        images_dir = labels_dir = folder

    # data.yaml sits in the dataset root: the chosen folder, or its parent when
    # the images/ folder itself was chosen (never further up: an unrelated
    # data.yaml there would give wrong class names).
    yaml_path = None
    places = [folder] + ([folder.parent] if folder.name.lower() == "images" else [])
    for directory in places:
        for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
            if (directory / name).is_file():
                yaml_path = directory / name
                break
        if yaml_path:
            break
    return images_dir, labels_dir, yaml_path


def read_class_names(yaml_path) -> Dict[int, str]:
    """Class number -> name from a data.yaml (`names:` as a list or a mapping)."""
    try:
        with open(yaml_path, "r", encoding="utf-8-sig") as fh:
            data = yaml.safe_load(fh)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ImportFailed(f"{Path(yaml_path).name} could not be read: {exc}") from exc

    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        result = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        try:
            result = {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError) as exc:
            raise ImportFailed(f"{Path(yaml_path).name}: class numbers in 'names' must be "
                               f"whole numbers") from exc
    else:
        raise ImportFailed(f"{Path(yaml_path).name} has no class names ('names:')")
    if not result:
        raise ImportFailed(f"{Path(yaml_path).name} lists no classes")
    return result


def clean_class_name(name: str, index: int) -> str:
    """A name valid in this app (letters, digits, _): 'D. galeata' -> 'D_galeata'."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or f"class_{index}"


# --------------------------------------------------------------------------
# One .txt file
# --------------------------------------------------------------------------
def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def parse_label_file(path, class_names: Dict[int, str]) -> LabelResult:
    """Read one YOLO polygon .txt (see the module docstring for the rules)."""
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return LabelResult(error=f"can't be read ({exc.strerror or exc})")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return LabelResult(error="is not a text file")
    if "\x00" in text:
        return LabelResult(error="is not a text file")

    result = LabelResult()
    known = f"0-{max(class_names)}" if class_names else "none"
    for n, line in enumerate(text.splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        bad = next((t for t in tokens if not _is_number(t)), None)
        if bad is not None:
            return LabelResult(error=f"line {n}: '{bad[:20]}' is not a number")
        values = [float(t) for t in tokens]
        if not all(math.isfinite(v) for v in values):
            return LabelResult(error=f"line {n}: contains an invalid number (nan/inf)")

        cls_value, coords = values[0], values[1:]
        if cls_value < 0 or cls_value != int(cls_value):
            return LabelResult(error=f"line {n}: class '{tokens[0]}' is not a class number")
        cls = int(cls_value)
        if cls not in class_names:
            return LabelResult(error=f"line {n}: class {cls} is not in data.yaml (it has {known})")
        if len(coords) in (4, 5):
            return LabelResult(error=f"line {n}: a box (class x y w h), not a polygon")
        if len(coords) < 6:
            return LabelResult(error=f"line {n}: too few values for a polygon ({len(coords)})")
        if len(coords) % 2:
            coords = coords[:-1]  # trailing confidence (prediction files): ignored
            result.confidence_ignored += 1

        points = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
        low, high = float(points.min()), float(points.max())
        if low < -COORD_TOLERANCE or high > 1 + COORD_TOLERANCE:
            return LabelResult(error=f"line {n}: coordinates outside the image "
                                     f"({low:.3g} to {high:.3g}; expected 0 to 1)")
        points = np.clip(points, 0.0, 1.0)
        if len(points) > 3 and np.array_equal(points[0], points[-1]):
            points = points[:-1]  # closed ring (first point repeated): one copy is enough
        if len(np.unique(points, axis=0)) < 3 or _polygon_area(points) < 1e-12:
            result.dropped_empty += 1
            continue
        result.polygons.append((cls, points.astype(np.float32)))
    return result


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def image_size(path) -> Tuple[int, int]:
    """
    (width, height) as the app shows the image, read from the file header only
    (big scans are not decoded). Photos rotated by EXIF are shown rotated, so
    their size is taken from how OpenCV loads them.
    """
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # large scans are expected, not a "decompression bomb"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Image.open(path) as im:
                width, height = im.size
                orientation = 1
                # Camera rotation tags live in JPEG/TIFF headers; for PNG,
                # Pillow would decode the whole image just to look for one.
                if im.format in ("JPEG", "MPO", "TIFF"):
                    try:
                        orientation = im.getexif().get(0x0112, 1)
                    except Exception:
                        orientation = 1
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    if orientation not in (1, None):
        loaded = cv2.imread(str(path))  # rare: let OpenCV decide how it shows it
        if loaded is not None:
            height, width = loaded.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image has no size")
    return width, height


def _list_images(images_dir: Path, include_subfolders: bool) -> List[Path]:
    pattern = "**/*" if include_subfolders else "*"
    found = []
    for p in images_dir.glob(pattern):
        rel = p.relative_to(images_dir)
        if any(part.startswith(".") for part in rel.parts):
            continue  # hidden files/folders, macOS "._" copies
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
            found.append(p)
    return sorted(found, key=lambda p: str(p.relative_to(images_dir)).lower())


def _is_label_candidate(p: Path) -> bool:
    name = p.name.lower()
    return name != "classes.txt" and not name.startswith("autosplit_")


def _find_label(image: Path, images_dir: Path, labels_dir: Path) -> Optional[Path]:
    """The image's .txt: mirrored under labels/ (train/val splits), flat in
    labels/, or next to the image."""
    rel_parent = image.parent.relative_to(images_dir)
    for candidate in (labels_dir / rel_parent / f"{image.stem}.txt",
                      labels_dir / f"{image.stem}.txt",
                      image.with_suffix(".txt")):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
def scan_dataset(images_dir, labels_dir, class_names: Dict[int, str], project_image_names,
                 include_subfolders: bool = True,
                 progress: Optional[Callable[[int, int], bool]] = None) -> ScanResult:
    """
    Look at every image of the dataset and its .txt; nothing is changed.

    project_image_names: file names of the images already in the project.
    progress(done, total) is called per image; returning False cancels
    (ImportCancelled).
    """
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    taken = {Path(n).stem.lower() for n in project_image_names}
    images = _list_images(images_dir, include_subfolders)

    entries: List[ImageEntry] = []
    first_with_stem: Dict[str, ImageEntry] = {}
    used_labels = set()
    for i, path in enumerate(images):
        if progress is not None and progress(i, len(images)) is False:
            raise ImportCancelled()
        stem = path.stem
        key = stem.lower()
        label_path = _find_label(path, images_dir, labels_dir)
        if label_path is not None:
            used_labels.add(label_path.resolve())

        if key in taken:
            entries.append(ImageEntry(path, stem, "exists", "already in the project"))
            continue
        if key in first_with_stem:
            other = first_with_stem[key].path.relative_to(images_dir)
            entries.append(ImageEntry(path, stem, "duplicate", f"same name as {other}"))
            continue
        try:
            size = image_size(path)
        except Exception as exc:  # unreadable or not really an image
            entries.append(ImageEntry(path, stem, "unreadable", f"can't be read as an image ({exc})"))
            continue

        entry = ImageEntry(path, stem, "import", size=size, label_path=label_path)
        if label_path is not None:
            entry.label = parse_label_file(label_path, class_names)
        first_with_stem[key] = entry
        entries.append(entry)
    if progress is not None:
        progress(len(images), len(images))

    pattern = "**/*.txt" if include_subfolders else "*.txt"
    orphans = sorted(p for p in labels_dir.glob(pattern)
                     if p.is_file() and _is_label_candidate(p)
                     and not any(part.startswith(".") for part in p.relative_to(labels_dir).parts)
                     and p.resolve() not in used_labels)
    return ScanResult(images_dir, labels_dir, class_names, entries, orphans)


# --------------------------------------------------------------------------
# Import
# --------------------------------------------------------------------------
def pick_new_colors(count: int, existing_hex) -> List[str]:
    """Distinct colours for new classes (golden-angle hues, away from existing ones)."""
    existing = []
    for h in existing_hex:
        try:
            existing.append(cv2.cvtColor(np.uint8([[[int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)]]]),
                                         cv2.COLOR_BGR2HSV)[0, 0, 0] / 180.0)
        except (ValueError, IndexError, TypeError):
            continue
    colors, hue, tries = [], 0.07, 0
    while len(colors) < count:
        hue = (hue + 0.618033988749895) % 1.0
        tries += 1
        if tries < 200 and any(min(abs(hue - e), 1 - abs(hue - e)) < 0.04 for e in existing):
            continue
        existing.append(hue)
        r, g, b = cv2.cvtColor(np.uint8([[[int(hue * 180) % 180, 190, 235]]]), cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


def run_import(scan: ScanResult, class_targets: Dict[int, str], existing_classes: Dict[str, str],
               db, dest_images_dir,
               progress: Optional[Callable[[int, int, str], bool]] = None) -> ImportResult:
    """
    Copy the scan's images into the project and add their masks and any new
    classes -- all or nothing: on cancel (ImportCancelled) or error the copied
    files are removed again and the database is left as it was.

    class_targets: class number -> project class name for every class used.
    existing_classes: project class name -> colour (hex).
    progress(done, total, file name) is called per image; returning False cancels.
    """
    dest_images_dir = Path(dest_images_dir)
    entries = scan.to_import
    used = sorted(scan.class_counts)
    missing = [c for c in used if not class_targets.get(c)]
    if missing:
        raise ImportFailed(f"No project class chosen for class number(s) {missing}")
    invalid = [class_targets[c] for c in used if not CLASS_NAME_RE.fullmatch(class_targets[c])]
    if invalid:
        raise ImportFailed(f"Invalid class name(s) {invalid}: use only letters, numbers and _")

    needed = scan.import_bytes
    dest_images_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest_images_dir).free
    if needed + 50 * 1024 ** 2 > free:
        raise ImportFailed(f"Not enough disk space: the images need {needed / 1024 ** 2:.0f} MB, "
                           f"{free / 1024 ** 2:.0f} MB are free.")

    # Classes: same name as a project class (ignoring case) -> that class; else new.
    by_lower = {name.lower(): name for name in existing_classes}
    new_names, merged = [], []
    for cls in used:
        target = class_targets[cls]
        if target.lower() in by_lower:
            merged.append((scan.class_names[cls], by_lower[target.lower()]))
        elif target.lower() not in {n.lower() for n in new_names}:
            new_names.append(target)
    resolve = {}
    for cls in used:
        target = class_targets[cls]
        resolve[cls] = by_lower.get(target.lower()) or next(
            n for n in new_names if n.lower() == target.lower())
    colors = pick_new_colors(len(new_names), existing_classes.values())

    copied: List[Path] = []
    try:
        for i, entry in enumerate(entries):
            if progress is not None and progress(i, len(entries), entry.path.name) is False:
                raise ImportCancelled()
            dest = dest_images_dir / entry.path.name
            if dest.exists():  # appeared since the scan
                raise ImportFailed(f"{dest.name} already exists in the project's images folder")
            shutil.copy(entry.path, dest)
            copied.append(dest)
        if progress is not None:
            progress(len(entries), len(entries), "")

        rows = []
        for entry in entries:
            if not entry.masks_ok:
                continue
            w, h = entry.size
            scale = np.array([w, h], dtype=np.float32)
            for cls, points in entry.label.polygons:
                # Rounded to 0.1 px: 6-decimal fractions give e.g. 79.99996 for
                # pixel 80, which the app (truncating to whole pixels) would
                # draw one pixel off.
                pixels = np.round(points.astype(np.float64) * scale, 1).astype(np.float32)
                area = cv2.contourArea(pixels.astype(np.int32))  # as MaskDatabaseManager.save_mask
                rows.append((entry.stem, pixels.tobytes(), resolve[cls], area))

        with db.transaction():
            db.executemany("INSERT INTO classes (class_name, color) VALUES (?, ?)",
                           list(zip(new_names, colors)))
            db.executemany("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                           "VALUES (?, ?, ?, ?)", rows)
    except BaseException:
        for path in copied:
            try:
                path.unlink()
            except OSError:
                logger.warning("Could not remove %s after a failed import", path)
        raise

    logger.info("Imported %d image(s) and %d mask(s) from %s; new classes %s",
                len(copied), len(rows), scan.images_dir, new_names)
    return ImportResult(copied, len(rows), new_names, merged)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def describe(scan: ScanResult, result: Optional[ImportResult] = None, limit: int = 0) -> str:
    """Human-readable list of everything notable (limit > 0 caps each list)."""
    def listing(items):
        items = list(items)
        shown = items if not limit else items[:limit]
        lines = [f"    {s}" for s in shown]
        if len(items) > len(shown):
            lines.append(f"    ... and {len(items) - len(shown)} more")
        return lines

    rel = lambda p, base: str(Path(p).relative_to(base)) if str(p).startswith(str(base)) else str(p)
    out = []
    if result is not None:
        out.append(f"Imported {len(result.images)} image(s) with {result.masks} mask(s).")
        if result.classes_added:
            out.append("New classes: " + ", ".join(result.classes_added))
        if result.classes_merged:
            out.append("Merged into existing classes: " + ", ".join(
                f"{a} -> {b}" if a != b else a for a, b in result.classes_merged))
        out.append("")

    groups = [
        ("Skipped - already in the project", scan.with_status("exists"), "name"),
        ("Skipped - same name as another image in this dataset", scan.with_status("duplicate"), "note"),
        ("Skipped - not a readable image", scan.with_status("unreadable"), "note"),
        ("Imported without masks - .txt skipped", scan.broken_labels, "label"),
        ("Imported without masks - no .txt file", scan.without_label, "name"),
    ]
    for title, entries, detail in groups:
        if not entries:
            continue
        out.append(f"{title} ({len(entries)}):")
        if detail == "name":
            out += listing(rel(e.path, scan.images_dir) for e in entries)
        elif detail == "note":
            out += listing(f"{rel(e.path, scan.images_dir)}: {e.note}" for e in entries)
        else:
            out += listing(f"{rel(e.label_path, scan.labels_dir)}: {e.label.error}" for e in entries)
        out.append("")

    dropped = [e for e in scan.to_import if e.masks_ok and e.label.dropped_empty]
    if dropped:
        out.append(f"Polygons without area left out ({sum(e.label.dropped_empty for e in dropped)} "
                   f"in {len(dropped)} file(s)):")
        out += listing(f"{rel(e.label_path, scan.labels_dir)}: {e.label.dropped_empty}" for e in dropped)
        out.append("")
    conf = [e for e in scan.to_import if e.masks_ok and e.label.confidence_ignored]
    if conf:
        out.append(f"Confidence values ignored in {len(conf)} file(s) "
                   f"({sum(e.label.confidence_ignored for e in conf)} line(s)).")
        out.append("")
    if scan.orphan_labels:
        out.append(f".txt files without an image of the same name ({len(scan.orphan_labels)}):")
        out += listing(rel(p, scan.labels_dir) for p in scan.orphan_labels)
        out.append("")
    return "\n".join(out).rstrip() + "\n"

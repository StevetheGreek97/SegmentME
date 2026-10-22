"""
Import YOLO polygon annotations for images that are already in the project.

The user picks a folder; every .txt in it and in its subfolders is read (a
YOLO export's labels/ folder, with or without train/val/test splits). A .txt
belongs to the project image with exactly the same file name (stem, case
sensitive): masks are stored per image name. The dataset's data.yaml gives
the class of each class number. No image files are copied or changed.

Each .txt line is one polygon: `class x1 y1 x2 y2 ... xn yn`, coordinates as
fractions (0-1) of the image size. The rules:

* A trailing confidence value (prediction files such as HiReS output, which
  leave an odd number of values) is ignored and the polygon imported.
* A .txt with any broken line -- text instead of numbers, a box (`class x y w
  h`) instead of a polygon, a class number missing from data.yaml, too few
  values, coordinates outside the image -- is skipped as a whole (reported).
* Polygons without area are dropped (reported); the rest of the file is kept.
* An empty .txt simply means "no objects".
* A .txt without a project image of that name is skipped (reported), and so
  are .txt files whose name occurs more than once in the folder (e.g.
  train/x.txt and val/x.txt): it is unclear which one belongs to the image.
* Masks are added next to masks the image already has.

Nothing here touches the GUI; ui/dialogs/import_dialog.py drives it.
"""
import re
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

from services.logger import get_logger

logger = get_logger(__name__)

# Coordinates may overshoot the image edge by this much (rounding) and are
# clipped; anything further out makes the line broken.
COORD_TOLERANCE = 0.01
# Valid class names in this app (see the sidebar's Add class).
CLASS_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
YAML_NAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
# Image headers read at once while scanning (see _read_matched).
_READ_WORKERS = 16


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


# Statuses of a .txt file
IMPORT = "import"            # matched to a project image; label may still be broken
NO_IMAGE = "no_image"        # no project image with this name
DUPLICATE = "duplicate"      # another .txt in the folder has the same name
AMBIGUOUS = "ambiguous"      # more than one project image has this name (a.png + a.jpg)
UNREADABLE = "unreadable"    # the project image can't be read (e.g. linked original gone)


@dataclass
class LabelEntry:
    label_path: Path
    stem: str
    status: str
    note: str = ""                           # why it is skipped (status != IMPORT)
    image_path: Optional[Path] = None        # the matching project image
    size: Optional[Tuple[int, int]] = None   # (width, height) as the app shows it
    label: Optional[LabelResult] = None
    existing_masks: int = 0                  # masks the image has already

    @property
    def masks_ok(self) -> bool:
        return self.status == IMPORT and self.label is not None and self.label.error is None


@dataclass
class ScanResult:
    folder: Path
    class_names: Dict[int, str]
    entries: List[LabelEntry]
    images_without_label: int = 0            # project images no .txt was found for

    def with_status(self, status) -> List[LabelEntry]:
        return [e for e in self.entries if e.status == status]

    @property
    def to_import(self) -> List[LabelEntry]:
        """Matched .txt files that will give masks (an empty file gives none)."""
        return [e for e in self.entries if e.masks_ok]

    @property
    def broken_labels(self) -> List[LabelEntry]:
        return [e for e in self.with_status(IMPORT) if e.label is not None and e.label.error]

    @property
    def with_existing_masks(self) -> List[LabelEntry]:
        return [e for e in self.to_import if e.existing_masks and e.label.polygons]

    @property
    def class_counts(self) -> Counter:
        """Masks per class number among the masks that will be imported."""
        counts = Counter()
        for e in self.to_import:
            counts.update(cls for cls, _ in e.label.polygons)
        return counts

    @property
    def mask_count(self) -> int:
        return sum(self.class_counts.values())

    @property
    def images_with_masks(self) -> int:
        return sum(1 for e in self.to_import if e.label.polygons)


@dataclass
class ImportResult:
    images: int                       # project images that received masks
    masks: int
    classes_added: List[str]
    classes_merged: List[Tuple[str, str]]   # (name in data.yaml, project class)


# --------------------------------------------------------------------------
# data.yaml and class names
# --------------------------------------------------------------------------
def find_yaml(folder) -> Optional[Path]:
    """The dataset's data.yaml: in the chosen folder, or next to it when a
    labels/ folder was chosen (never further up: an unrelated data.yaml there
    would give wrong class names)."""
    folder = Path(folder)
    places = [folder] + ([folder.parent] if folder.name.lower() == "labels" else [])
    for directory in places:
        for name in YAML_NAMES:
            if (directory / name).is_file():
                return directory / name
    return None


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
    cross = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]) + x[-1] * y[0] - y[-1] * x[0]
    return 0.5 * abs(float(cross))


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
        try:
            values = np.array(tokens, dtype=np.float64)   # all at once: files can hold 100k+ values
        except ValueError:
            # numpy is stricter than float() (e.g. "1_000"); decide as float() does
            bad = next((t for t in tokens if not _is_number(t)), None)
            if bad is not None:
                return LabelResult(error=f"line {n}: '{bad[:20]}' is not a number")
            values = np.array([float(t) for t in tokens], dtype=np.float64)
        if not np.isfinite(values).all():
            return LabelResult(error=f"line {n}: contains an invalid number (nan/inf)")

        cls_value, coords = float(values[0]), values[1:]
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

        points = coords.reshape(-1, 2)
        low, high = float(points.min()), float(points.max())
        if low < -COORD_TOLERANCE or high > 1 + COORD_TOLERANCE:
            return LabelResult(error=f"line {n}: coordinates outside the image "
                                     f"({low:.3g} to {high:.3g}; expected 0 to 1)")
        points = np.clip(points, 0.0, 1.0)
        if len(points) > 3 and np.array_equal(points[0], points[-1]):
            points = points[:-1]  # closed ring (first point repeated): one copy is enough
        # (Fewer than 3 distinct points has no area either.)
        if _polygon_area(points) < 1e-12:
            result.dropped_empty += 1
            continue
        result.polygons.append((cls, points.astype(np.float32)))
    return result


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
class _LargeImagesAllowed:
    """Lifts PIL's "decompression bomb" size check (large scans are expected)
    and its warnings while active. Both are process-wide settings, so set
    them once around a batch rather than per image in each thread."""

    def __enter__(self):
        self._limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None
        self._warnings = warnings.catch_warnings()
        self._warnings.__enter__()
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        return self

    def __exit__(self, *exc):
        self._warnings.__exit__(*exc)
        Image.MAX_IMAGE_PIXELS = self._limit
        return False


def image_size(path) -> Tuple[int, int]:
    """
    (width, height) as the app shows the image, read from the file header only
    (big scans are not decoded). Photos rotated by EXIF are shown rotated, so
    their size is taken from how OpenCV loads them.
    """
    with _LargeImagesAllowed():
        return _image_size(path)


def _image_size(path) -> Tuple[int, int]:
    """image_size() without touching process-wide settings (safe in threads,
    inside _LargeImagesAllowed)."""
    with Image.open(path) as im:
        width, height = im.size
        orientation = 1
        # Camera rotation tags live in JPEG/TIFF headers; for PNG, Pillow
        # would decode the whole image just to look for one.
        try:
            if im.format == "TIFF":
                orientation = im.tag_v2.get(0x0112, 1)   # already read with the header
            elif im.format in ("JPEG", "MPO"):
                orientation = im.getexif().get(0x0112, 1)
        except Exception:
            orientation = 1
    if orientation not in (1, None):
        loaded = cv2.imread(str(path))  # rare: let OpenCV decide how it shows it
        if loaded is not None:
            height, width = loaded.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError("image has no size")
    return width, height



# --------------------------------------------------------------------------
# Scan
# --------------------------------------------------------------------------
def _is_label_candidate(p: Path) -> bool:
    name = p.name.lower()
    return name != "classes.txt" and not name.startswith("autosplit_")


def list_label_files(folder) -> List[Path]:
    """Every .txt in `folder` and its subfolders (not hidden ones, not
    classes.txt or autosplit_*.txt), sorted."""
    folder = Path(folder)
    found = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts):
            continue  # hidden files/folders, macOS "._" copies
        if p.suffix.lower() == ".txt" and _is_label_candidate(p) and p.is_file():
            found.append(p)
    return sorted(found, key=lambda p: str(p.relative_to(folder)).lower())


def scan_annotations(folder, class_names: Dict[int, str], project_images: Iterable,
                     existing_masks: Optional[Dict[str, int]] = None,
                     progress: Optional[Callable[[int, int], bool]] = None) -> ScanResult:
    """
    Read every .txt under `folder` and match it to a project image; nothing
    is changed.

    project_images: paths of the project's images.
    existing_masks: image name (stem) -> number of masks it has already.
    progress(done, total) is called per file; returning False cancels
    (ImportCancelled).
    """
    folder = Path(folder)
    existing_masks = existing_masks or {}
    by_stem = defaultdict(list)
    for p in project_images:
        by_stem[Path(p).stem].append(Path(p))

    labels = list_label_files(folder)
    count = Counter(p.stem for p in labels)

    entries: List[LabelEntry] = []
    for path in labels:
        stem = path.stem
        if count[stem] > 1:
            others = [str(q.relative_to(folder)) for q in labels if q.stem == stem and q != path]
            entries.append(LabelEntry(path, stem, DUPLICATE, "same name as " + ", ".join(others)))
            continue
        images = by_stem.get(stem, [])
        if not images:
            entries.append(LabelEntry(path, stem, NO_IMAGE, "no project image with this name"))
            continue
        if len(images) > 1:
            entries.append(LabelEntry(path, stem, AMBIGUOUS, "the project has "
                                      + " and ".join(sorted(q.name for q in images))))
            continue
        entries.append(LabelEntry(path, stem, IMPORT, image_path=images[0],
                                  existing_masks=int(existing_masks.get(stem, 0))))

    _read_matched(entries, class_names, progress)

    matched = {e.stem for e in entries if e.status in (IMPORT, UNREADABLE, DUPLICATE, AMBIGUOUS)}
    without = sum(len(v) for k, v in by_stem.items() if k not in matched)
    return ScanResult(folder, class_names, entries, without)


def _read_matched(entries: List[LabelEntry], class_names, progress) -> None:
    """Image size and parsed .txt for every matched entry, several at a time:
    reading image headers waits on the disk -- or the network, when project
    images are links into a cloud folder -- so they are read in parallel."""
    todo = [e for e in entries if e.status == IMPORT]

    def read(entry):
        try:
            entry.size = _image_size(entry.image_path)
        except Exception as exc:  # e.g. the original of a linked image is gone
            entry.status = UNREADABLE
            entry.note = f"{entry.image_path.name} can't be read ({exc})"
            return
        entry.label = parse_label_file(entry.label_path, class_names)

    if progress is not None and progress(0, len(todo)) is False:
        raise ImportCancelled()
    with _LargeImagesAllowed(), ThreadPoolExecutor(max_workers=_READ_WORKERS) as pool:
        futures = [pool.submit(read, e) for e in todo]
        try:
            for done, future in enumerate(as_completed(futures), 1):
                future.result()
                if progress is not None and progress(done, len(todo)) is False:
                    raise ImportCancelled()
        except BaseException:
            for f in futures:
                f.cancel()
            raise


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
               db, progress: Optional[Callable[[int, int, str], bool]] = None) -> ImportResult:
    """
    Add the scan's masks (next to any masks the images have) and any new
    classes -- all or nothing: on cancel (ImportCancelled) or error the
    database is left as it was. No files are touched.

    class_targets: class number -> project class name for every class used.
    existing_classes: project class name -> colour (hex).
    progress(done, total, file name) is called per file; returning False cancels.
    """
    entries = scan.to_import
    used = sorted(scan.class_counts)
    missing = [c for c in used if not class_targets.get(c)]
    if missing:
        raise ImportFailed(f"No project class chosen for class number(s) {missing}")
    invalid = [class_targets[c] for c in used if not CLASS_NAME_RE.fullmatch(class_targets[c])]
    if invalid:
        raise ImportFailed(f"Invalid class name(s) {invalid}: use only letters, numbers and _")

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

    rows, images = [], 0
    for i, entry in enumerate(entries):
        if progress is not None and progress(i, len(entries), entry.label_path.name) is False:
            raise ImportCancelled()
        w, h = entry.size
        scale = np.array([w, h], dtype=np.float32)
        images += bool(entry.label.polygons)
        for cls, points in entry.label.polygons:
            # Rounded to 0.1 px: 6-decimal fractions give e.g. 79.99996 for
            # pixel 80, which the app (truncating to whole pixels) would
            # draw one pixel off.
            pixels = np.round(points.astype(np.float64) * scale, 1).astype(np.float32)
            area = cv2.contourArea(pixels.astype(np.int32))  # as MaskDatabaseManager.save_mask
            rows.append((entry.stem, pixels.tobytes(), resolve[cls], area))
    if progress is not None:
        progress(len(entries), len(entries), "")

    with db.transaction():
        db.executemany("INSERT INTO classes (class_name, color) VALUES (?, ?)",
                       list(zip(new_names, colors)))
        db.executemany("INSERT INTO masks (image_name, mask_data, class_name, surface_area) "
                       "VALUES (?, ?, ?, ?)", rows)

    logger.info("Imported %d mask(s) for %d image(s) from %s; new classes %s",
                len(rows), images, scan.folder, new_names)
    return ImportResult(images, len(rows), new_names, merged)


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

    rel = lambda p: str(Path(p).relative_to(scan.folder)) if str(p).startswith(str(scan.folder)) else str(p)
    out = []
    if result is not None:
        out.append(f"Imported {result.masks} mask(s) for {result.images} image(s).")
        if result.classes_added:
            out.append("New classes: " + ", ".join(result.classes_added))
        if result.classes_merged:
            out.append("Merged into existing classes: " + ", ".join(
                f"{a} -> {b}" if a != b else a for a, b in result.classes_merged))
        out.append("")

    groups = [
        ("Skipped - no project image with this name", scan.with_status(NO_IMAGE), False),
        ("Skipped - same name as another .txt in the folder", scan.with_status(DUPLICATE), True),
        ("Skipped - more than one project image with this name", scan.with_status(AMBIGUOUS), True),
        ("Skipped - the project image can't be read", scan.with_status(UNREADABLE), True),
    ]
    for title, entries, with_note in groups:
        if not entries:
            continue
        out.append(f"{title} ({len(entries)}):")
        out += listing(f"{rel(e.label_path)}: {e.note}" if with_note else rel(e.label_path)
                       for e in entries)
        out.append("")
    broken = scan.broken_labels
    if broken:
        out.append(f"Skipped - broken .txt ({len(broken)}):")
        out += listing(f"{rel(e.label_path)}: {e.label.error}" for e in broken)
        out.append("")

    dropped = [e for e in scan.to_import if e.label.dropped_empty]
    if dropped:
        out.append(f"Polygons without area left out ({sum(e.label.dropped_empty for e in dropped)} "
                   f"in {len(dropped)} file(s)):")
        out += listing(f"{rel(e.label_path)}: {e.label.dropped_empty}" for e in dropped)
        out.append("")
    conf = [e for e in scan.to_import if e.label.confidence_ignored]
    if conf:
        out.append(f"Confidence values ignored in {len(conf)} file(s) "
                   f"({sum(e.label.confidence_ignored for e in conf)} line(s)).")
        out.append("")
    existing = scan.with_existing_masks
    if existing:
        out.append(f"Added next to masks the image already had ({len(existing)} image(s)):")
        out += listing(f"{e.image_path.name}: {e.existing_masks} existing" for e in existing)
        out.append("")
    if scan.images_without_label:
        out.append(f"{scan.images_without_label} project image(s) had no .txt in the folder.")
        out.append("")
    return "\n".join(out).rstrip() + "\n"

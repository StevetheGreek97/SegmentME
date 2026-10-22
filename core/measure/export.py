"""
File > Save Results: one CSV row per mask with its measurements, in real
units where the image has a scale and in pixels always.

Runs fine in a worker thread: it opens its own read-only connection to the
project database and only reads image headers (in parallel: project images
may be links into a slow network folder).
"""
import csv
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

from core.database.connection import connect_read_only
from core.measure import morphometrics
from core.measure.scale import FROM_FILE, ProjectScale
from services.logger import get_logger

logger = get_logger(__name__)

_SCALE_WORKERS = 16


class ExportCancelled(Exception):
    pass


@dataclass
class ExportResult:
    files: List[Path] = field(default_factory=list)
    rows: int = 0
    without_scale: int = 0          # images exported in pixels only
    skipped: int = 0                # masks without area


def columns(unit: str) -> List[str]:
    u, u2 = unit, f"{unit}2"
    return ["image_name", "mask_id", "class_name",
            f"area_{u2}", f"length_{u}", f"width_{u}", f"perimeter_{u}",
            "orientation_deg", "aspect_ratio", "solidity", "circularity",
            f"centroid_x_{u}", f"centroid_y_{u}",
            "area_px", "length_px", "width_px", "um_per_px", "scale_source"]


def _fmt(value, digits=6):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{value:.{digits}g}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "class"


def export_measurements(db_path, out_path, image_paths: Dict[str, str], scale: ProjectScale,
                        unit: str = "mm", image_names: Optional[Iterable[str]] = None,
                        classes: Optional[Iterable[str]] = None, per_class: bool = False,
                        progress: Optional[Callable[[int, int, str], None]] = None,
                        cancelled: Optional[Callable[[], bool]] = None) -> ExportResult:
    """
    Write the measurement CSV(s).

    image_paths: image name (stem) -> image file (for its scale).
    image_names / classes: limit the export (None = all).
    per_class: one file per class, named <out stem>_<class>.csv, else one file.
    progress(done, total, stage) is called now and then; cancelled() -> True stops
    (ExportCancelled; files already written are removed).
    """
    from core.measure.scale import UNITS
    unit_um = UNITS[unit]
    out_path = Path(out_path)
    started = time.perf_counter()
    conn = connect_read_only(db_path)
    result = ExportResult()
    writers, handles = {}, {}
    try:
        where, params = [], []
        if image_names is not None:
            names = sorted(set(image_names))
            where.append(f"image_name IN ({','.join('?' * len(names))})")
            params += names
        if classes is not None:
            wanted = sorted(set(classes))
            where.append(f"class_name IN ({','.join('?' * len(wanted))})")
            params += wanted
        sql_where = (" WHERE " + " AND ".join(where)) if where else ""
        total = conn.execute(f"SELECT COUNT(*) FROM masks{sql_where}", params).fetchone()[0]
        stems = [r[0] for r in conn.execute(
            f"SELECT DISTINCT image_name FROM masks{sql_where}", params)]
        logger.info("Measuring %d mask(s) on %d image(s) -> %s (unit %s, classes %s, %s)",
                    total, len(stems), out_path, unit,
                    "all" if classes is None else sorted(set(classes)),
                    "one file per class" if per_class else "one file")

        # Scale per image, header reads in parallel
        if progress:
            progress(0, total, "Reading image resolutions…")
        with ThreadPoolExecutor(max_workers=_SCALE_WORKERS) as pool:
            scales = dict(zip(stems, pool.map(
                lambda stem: scale.for_image(image_paths.get(stem)), stems)))
        result.without_scale = sum(1 for v, _ in scales.values() if not v)
        no_scale = sorted(s for s, (v, _) in scales.items() if not v)
        logger.info("Image scales read in %.1f s: %d from file DPI, %d project scale, %d none",
                    time.perf_counter() - started,
                    sum(1 for _, src in scales.values() if src == FROM_FILE),
                    sum(1 for v, src in scales.values() if v and src != FROM_FILE), len(no_scale))
        if no_scale:
            logger.warning("No scale for %d image(s), sizes in pixels only: %s%s", len(no_scale),
                           ", ".join(no_scale[:10]), " …" if len(no_scale) > 10 else "")
        missing = [s for s in stems if s not in image_paths]
        if missing:
            logger.warning("%d image name(s) with masks are not in the project's image list "
                           "(no file to read a resolution from): %s", len(missing),
                           ", ".join(missing[:10]))

        def writer_for(class_name):
            key = class_name if per_class else ""
            if key not in writers:
                path = (out_path.with_name(f"{out_path.stem}_{_safe(class_name)}{out_path.suffix}")
                        if per_class else out_path)
                fh = open(path, "w", newline="", encoding="utf-8")
                handles[key] = fh
                writers[key] = csv.writer(fh)
                writers[key].writerow(columns(unit))
                result.files.append(path)
            return writers[key]

        cur = conn.execute(f"SELECT id, image_name, class_name, mask_data FROM masks{sql_where} "
                           f"ORDER BY image_name, id", params)
        done = 0
        while True:
            batch = cur.fetchmany(2000)
            if not batch:
                break
            if cancelled is not None and cancelled():
                raise ExportCancelled()
            for mask_id, image_name, class_name, blob in batch:
                done += 1
                px = morphometrics.measure(np.frombuffer(blob, np.float32).reshape(-1, 2))
                if px is None:
                    result.skipped += 1
                    logger.debug("Mask %s on %s has no area; left out", mask_id, image_name)
                    continue
                um_per_px, source = scales.get(image_name, (None, ""))
                real = morphometrics.to_units(px, um_per_px, unit_um)
                writer_for(class_name).writerow([
                    image_name, mask_id, class_name,
                    _fmt(real["area"]), _fmt(real["length"]), _fmt(real["width"]),
                    _fmt(real["perimeter"]), _fmt(real["orientation_deg"], 5),
                    _fmt(real["aspect_ratio"]), _fmt(real["solidity"]), _fmt(real["circularity"]),
                    _fmt(real["centroid_x"]), _fmt(real["centroid_y"]),
                    _fmt(px["area"]), _fmt(px["length"]), _fmt(px["width"]),
                    _fmt(um_per_px), ("file DPI" if source == FROM_FILE else source),
                ])
                result.rows += 1
            if progress:
                progress(done, total, "Measuring masks…")
    except BaseException as exc:
        if isinstance(exc, ExportCancelled):
            logger.info("Measurement export cancelled; partial file(s) removed")
        else:
            logger.exception("Measurement export to %s failed", out_path)
        for fh in handles.values():
            fh.close()
        for path in result.files:
            try:
                path.unlink()
            except OSError:
                pass
        raise
    finally:
        conn.close()
    for fh in handles.values():
        fh.close()
    logger.info("Measured %d mask(s) in %.1f s -> %s (%d without area left out)",
                result.rows, time.perf_counter() - started,
                ", ".join(str(p) for p in result.files) or "no file", result.skipped)
    return result

"""Subprocess entry point for batch inference (YOLO, HiReS tiled YOLO, SAM2/2.1).

Runs as a plain OS process spawned by core.inference_manager.InferenceManager
via QProcess -- deliberately has no PyQt6 import so it stays cheap to start
and can never pull a GUI event loop into the child. Talks to the parent over
stdio with NDJSON: reads one JSON request line from stdin, writes one JSON
result line per image to stdout, then a final {"done": true} line.
Human-readable progress goes to stderr; the GUI shows it as the run's log
(the progress dialog's "Show log" button).

Dispatched from main.py before any GUI imports happen, so the exact same
mechanism works whether the app runs as `python main.py` or as a frozen
PyInstaller executable (both re-invoke themselves with --inference-worker).
"""
import json
import os
import sys
import time
import traceback
from collections import Counter

# The NDJSON protocol to the GUI travels on stdout. _setup_streams() moves it
# to a private copy of the stdout file descriptor and points fd 1 at stderr,
# so everything else that writes to "stdout" -- print(), ultralytics' logger
# (bound to sys.stdout when ultralytics is imported), C libraries -- ends up
# in the log instead of in the middle of the protocol. Until then, or if the
# swap is not possible, the protocol simply uses sys.stdout.
_PROTOCOL_OUT = sys.stdout


def _setup_streams():
    global _PROTOCOL_OUT
    for stream in (sys.stdout, sys.stderr):
        try:
            # Pipes to the GUI: UTF-8 everywhere (a Windows code page would
            # choke on non-ASCII class names) and line by line, as it happens.
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass
    try:
        sys.stdout.flush()
        protocol_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return  # protocol stays on stdout; the GUI logs any stray lines
    _PROTOCOL_OUT = open(protocol_fd, "w", encoding="utf-8", errors="replace", buffering=1)


def _emit(obj):
    """One protocol message to the GUI."""
    _PROTOCOL_OUT.write(json.dumps(obj) + "\n")
    _PROTOCOL_OUT.flush()


def _log(message=""):
    """One line of the run's log (stderr), timestamped like the hires CLI."""
    stamp = f"[{time.strftime('%H:%M:%S')}] " if message else ""
    sys.stderr.write(f"{stamp}{message}\n")
    sys.stderr.flush()


def _log_failure(exc):
    """Why an image failed; with the traceback unless the reason is a plain
    file or settings problem that the message already explains."""
    _log(f"  Failed: {exc}")
    if not isinstance(exc, (ValueError, OSError)):
        _log(traceback.format_exc().rstrip())


def _duration(start):
    seconds = time.perf_counter() - start
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    return f"{minutes} min {seconds} s"


def _class_name(names, class_id):
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)


def _counts_text(class_names):
    """'3 daphnid, 1 egg' -- most frequent first."""
    return ", ".join(f"{n} {name}" for name, n in Counter(class_names).most_common())


def _result_line(n_masks, class_names, start):
    detail = f" ({_counts_text(class_names)})" if class_names else ""
    return f"  Result: {n_masks} mask{'' if n_masks == 1 else 's'}{detail} in {_duration(start)}"


def _run_yolo(request):
    from PIL import Image
    from ultralytics import YOLO

    paths = request["image_paths"]
    _log(f"YOLO inference: {len(paths)} image(s) | model: {os.path.basename(request['model_path'])}"
         f" | conf={float(request['conf']):g}, image size={tuple(request['dims'])}")
    model = YOLO(request["model_path"])
    for index, image_path in enumerate(paths, start=1):
        start = time.perf_counter()
        _log(f"[{index}/{len(paths)}] {os.path.basename(image_path)}")
        try:
            _emit({"status": "processing", "image": image_path})
            Image.open(image_path).convert("RGB")  # fail fast on unreadable files
            results = model.predict(
                image_path, save_txt=False, save_crop=False, imgsz=request["dims"],
                verbose=False, conf=request["conf"], iou=0.2, agnostic_nms=True,
            )
            masks_xy = results[0].masks.xy if results[0].masks else []
            class_ids = results[0].boxes.cls.tolist() if results[0].boxes is not None else []
            class_names = [model.names[int(i)] for i in class_ids]
            _log(_result_line(len(masks_xy), class_names, start))
            _emit({
                "image": image_path,
                "masks": [m.tolist() for m in masks_xy],
                "classes": class_names,
            })
        except Exception as exc:
            _log_failure(exc)
            _emit({"image": image_path, "error": str(exc)})


# --------------------------------------------------------------------------
# HiReS: tiled YOLO inference for large images (pip package `hireseg`)
# --------------------------------------------------------------------------
_HIRES_TMP_PREFIX = "segmentme-hires-"


def _cleanup_stale_hires_dirs():
    """Remove tile folders left behind by HiReS workers that were killed
    (Cancel kills the worker outright, so its TemporaryDirectory never gets
    to clean up). Folders are named segmentme-hires-<pid>-*; only those
    whose process is gone are removed."""
    import shutil
    import tempfile

    import psutil

    root = tempfile.gettempdir()
    try:
        entries = os.listdir(root)
    except OSError:
        return
    for name in entries:
        if not name.startswith(_HIRES_TMP_PREFIX):
            continue
        try:
            pid = int(name[len(_HIRES_TMP_PREFIX):].split("-", 1)[0])
        except ValueError:
            continue
        if pid != os.getpid() and not psutil.pid_exists(pid):
            shutil.rmtree(os.path.join(root, name), ignore_errors=True)


def _hires_settings(request):
    """Validated HiReS parameters from the request (defaults = HiReS CLI)."""
    params = request.get("hires") or {}
    tile_w, tile_h = (int(v) for v in params.get("chunk_size", (1024, 1024)))
    overlap = int(params.get("overlap", 150))
    if min(tile_w, tile_h) < 32:
        raise ValueError("HiReS tile size must be at least 32 px.")
    if not 0 <= overlap < min(tile_w, tile_h):
        raise ValueError(
            f"HiReS overlap ({overlap} px) must be smaller than the tile size "
            f"({tile_w}x{tile_h} px).")
    edge = float(params.get("edge_threshold", 0.01))
    if not 0.0 <= edge < 0.5:
        raise ValueError("HiReS edge filter must be between 0 and 0.5.")
    iou = float(params.get("iou_thresh", 0.7))
    if not 0.0 <= iou <= 1.0:
        raise ValueError("HiReS NMS IoU must be between 0 and 1.")
    return {
        "chunk_size": (tile_w, tile_h),
        "overlap": overlap,
        "imgsz": int(params.get("imgsz", 1024)),
        "edge_threshold": edge,
        "iou_thresh": iou,
    }


def _read_image_unchanged(image_path):
    """The image as stored (first page, alpha kept, EXIF rotation ignored),
    the same on every OS -- what cv2.imread(path, IMREAD_UNCHANGED) gives on
    Linux/macOS.

    Not cv2.imread itself: on Windows, importing ultralytics replaces it with
    a version that copes with non-English paths but stacks all pages of a
    TIFF, so a scan with a thumbnail page fails there. Reading the bytes with
    numpy works with any path on every OS."""
    import cv2
    import numpy as np

    data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not read this image: {image_path}")
    return image


def _hires_image(image_path, predictor, settings, conf, device, workdir, tiles):
    """One image through the HiReS steps (as in hires' SegmentationPipeline):
    tile -> YOLO per tile -> drop edge-touching polygons -> merge into
    full-image coordinates -> drop image-border polygons -> polygon NMS.

    `tiles` is the progress state the per-tile callback reads (see
    _run_hires). Returns (polygons in pixel coordinates, class ids).
    """
    import tempfile
    from pathlib import Path

    from PIL import Image

    from hires.models.collection import Collection
    from hires.operations.ops import unify_collections
    from hires.processing.chunker import ImageChunker

    with Image.open(image_path) as img:  # header only; hires lifts PIL's size cap
        width, height = img.size
    _log(f"  Image: {width} x {height} px")

    tile_w, tile_h = settings["chunk_size"]
    edge = settings["edge_threshold"]

    def inside(ann, t=edge):
        return ann.is_inside_unit_box(threshold=t)

    # ignore_cleanup_errors: on Windows a freshly written tile can still be
    # locked (e.g. by a virus scanner) when the folder is removed; that must
    # not fail an image whose prediction worked. Leftovers are removed later
    # (_cleanup_stale_hires_dirs, and the GUI after the worker exits).
    with tempfile.TemporaryDirectory(prefix=f"{_HIRES_TMP_PREFIX}{os.getpid()}-",
                                     dir=workdir, ignore_cleanup_errors=True) as tmp:
        tiles_dir = Path(tmp) / "chunks"
        pred_dir = Path(tmp) / "pred"
        tiles_dir.mkdir()
        pred_dir.mkdir()

        if height <= tile_h:
            # hires' ImageChunker makes no tiles when the image is not
            # taller than one tile; say so instead of returning nothing.
            raise ValueError(
                f"HiReS made no tiles for this {width}x{height} image: it must be "
                f"taller than the tile height ({tile_h} px). "
                "Use a smaller tile size, or plain YOLO inference.")

        start = time.perf_counter()
        image = _read_image_unchanged(image_path)
        # Tiles from our own read (ImageChunker.slice would read the file
        # itself; see _read_image_unchanged for why that differs on Windows).
        ImageChunker(str(image_path))._chunk_and_save(
            image, str(tiles_dir), Path(image_path).stem,
            settings["chunk_size"], settings["overlap"])
        del image
        n_tiles = sum(1 for _ in tiles_dir.glob("*.png"))
        _log(f"  Chunking: {n_tiles} tiles of {tile_w} x {tile_h} px, "
             f"overlap {settings['overlap']} px ({_duration(start)})")
        if n_tiles == 0:
            raise ValueError(f"HiReS could not write its tiles to {tiles_dir}.")

        _emit({"status": "processing", "image": image_path,
               "detail": f"Predicting {n_tiles} tiles...", "fraction": 0.0})
        tiles.update(image=image_path, done=0, total=n_tiles)
        start = time.perf_counter()
        predictor.output_dir = str(pred_dir)
        predictor.predict(image_dir=str(tiles_dir), conf=conf,
                          imgsz=settings["imgsz"], device=device)

        tile_collections, raw = {}, 0
        for txt in sorted(pred_dir.glob("*.txt")):
            coll = Collection.read_txt(txt, collection_name=txt.stem)
            raw += len(coll)
            tile_collections[txt.name] = coll.filter(predicate=inside)
        _log(f"  Prediction: {raw} detections in {n_tiles} tiles ({_duration(start)})")

    in_tiles = sum(len(coll) for coll in tile_collections.values())
    _log(f"  Edge filter (within {edge:g} of a tile border): {raw} -> {in_tiles}")
    merged = unify_collections(
        chunk_collections=tile_collections,
        chunk_size=settings["chunk_size"],
        full_img_path=str(image_path),
    )
    in_image = merged.filter(predicate=inside)
    _log(f"  Merged into image coordinates; image-border filter: {len(merged)} -> {len(in_image)}")
    kept = in_image.nms(iou_threshold=settings["iou_thresh"], class_aware=False)
    _log(f"  NMS (duplicates from overlapping tiles, IoU > {settings['iou_thresh']:g}): "
         f"{len(in_image)} -> {len(kept)}")

    polygons, class_ids = [], []
    for ann in kept.annotations:
        coords = list(ann.polygon.exterior.coords)[:-1]  # drop closing point
        if len(coords) < 3:
            continue
        polygons.append([[x * width, y * height] for x, y in coords])
        class_ids.append(int(ann.class_id))
    return polygons, class_ids


def _run_hires(request):
    import tempfile

    import torch

    try:
        from hires.processing.predictor import YOLOSegPredictor
    except ImportError:
        raise RuntimeError(
            "HiReS inference needs the 'hireseg' package, which is not installed. "
            "Install it into the SegmentME environment with:  "
            "pip install --no-deps hireseg==1.2.0")

    paths = request["image_paths"]
    settings = _hires_settings(request)
    device = 0 if torch.cuda.is_available() else "cpu"
    conf = float(request["conf"])
    workdir = tempfile.gettempdir()
    _cleanup_stale_hires_dirs()

    model_name = os.path.basename(request["model_path"])
    tile_w, tile_h = settings["chunk_size"]
    _log(f"HiReS tiled inference: {len(paths)} image(s) | model: {model_name} | "
         f"device: {'cuda:0' if device == 0 else 'cpu'}")
    _log(f"Settings: conf={conf:g}, tile={tile_w} x {tile_h} px, overlap={settings['overlap']} px, "
         f"inference size={settings['imgsz']} px, edge filter={settings['edge_threshold']:g}, "
         f"NMS IoU={settings['iou_thresh']:g}")
    _log(f"Temporary tiles: {workdir}")

    _emit({"status": "loading_model", "variant": model_name})
    start = time.perf_counter()
    predictor = YOLOSegPredictor(request["model_path"], output_dir=workdir)
    names = predictor.model.names
    labels = names.items() if isinstance(names, dict) else enumerate(names)
    _log(f"Model loaded in {_duration(start)}; classes: "
         + ", ".join(f"{i}={name}" for i, name in labels))

    # One log line + one progress update per predicted tile, from ultralytics'
    # own per-batch callback (hires' predictor consumes the result stream).
    tiles = {"image": None, "done": 0, "total": 0, "broken": False}

    def on_tile_predicted(yolo):
        try:
            for path, result in zip(yolo.batch[0], yolo.results):
                tiles["done"] += 1
                done, total = tiles["done"], tiles["total"]
                x, y = os.path.splitext(os.path.basename(path))[0].rsplit("_", 2)[1:]
                found = ([_class_name(result.names, int(c)) for c in result.boxes.cls.tolist()]
                         if result.boxes is not None else [])
                what = f"{len(found)} detection{'' if len(found) == 1 else 's'}"
                if found:
                    what += f" ({_counts_text(found)})"
                ms = result.speed.get("inference", 0.0) if result.speed else 0.0
                _log(f"  tile {done}/{total} at x={x}, y={y}: {what}, {ms:.0f} ms")
                _emit({"status": "processing", "image": tiles["image"],
                       "detail": f"Predicting tile {done}/{total}",
                       "fraction": done / total if total else 0.0})
        except Exception as exc:  # progress reporting must never break inference
            if not tiles["broken"]:
                tiles["broken"] = True
                _log(f"  (per-tile progress unavailable: {exc})")

    predictor.model.add_callback("on_predict_batch_end", on_tile_predicted)

    run_start, total_masks, failed = time.perf_counter(), 0, 0
    for index, image_path in enumerate(paths, start=1):
        start = time.perf_counter()
        _log()
        _log(f"[{index}/{len(paths)}] {os.path.basename(image_path)}")
        try:
            _emit({"status": "processing", "image": image_path})
            polygons, class_ids = _hires_image(
                image_path, predictor, settings, conf, device, workdir, tiles)
            classes = [_class_name(names, i) for i in class_ids]
            _log(_result_line(len(polygons), classes, start))
            total_masks += len(polygons)
            _emit({"image": image_path, "masks": polygons, "classes": classes})
        except Exception as exc:
            failed += 1
            _log_failure(exc)
            _emit({"image": image_path, "error": str(exc)})

    _log()
    _log(f"Finished: {len(paths) - failed} of {len(paths)} image(s), "
         f"{total_masks} masks in total ({_duration(run_start)})")


def _mask_to_polygon(seg):
    """Largest external contour of a binary uint8 mask as [[x, y], ...],
    or None if the mask has no usable contour."""
    import cv2

    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    return largest.squeeze(axis=1).tolist()


def _read_image_rgb(image_path):
    import cv2

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _detect_custom_family(model_path):
    """Classify a user-browsed checkpoint by the tensor names inside it.

    Returns a key of _CUSTOM_RUNNERS, or None if unrecognized:
    * "sam2" -- SAM2/2.1 save format (a dict wrapping a "model" state dict),
      covering official and fine-tuned checkpoints alike.
    * "cellpose_sam" -- Cellpose 4 "cpsam" models: a raw state dict with a
      SAM ViT encoder ("encoder.*") plus Cellpose's readout head.
    """
    import torch

    try:
        ckpt = torch.load(model_path, map_location="cpu",
                          weights_only=True, mmap=True)
    except Exception:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        return None
    if isinstance(ckpt.get("model"), dict):
        return "sam2"
    if any(k.startswith("encoder.") for k in ckpt) and "out.weight" in ckpt:
        return "cellpose_sam"
    return None


def _run_sam2_generator(request, model):
    """Shared 'segment everything' loop for any SAM2-family model."""
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    generator = SAM2AutomaticMaskGenerator(model)

    paths = request["image_paths"]
    for index, image_path in enumerate(paths, start=1):
        start = time.perf_counter()
        _log(f"[{index}/{len(paths)}] {os.path.basename(image_path)}")
        try:
            # Automatic mask generation runs ~1000 point-prompt forward passes
            # per image, so a single image can take a minute or more on CPU --
            # report the start too, not just the result, so a caller polling
            # stdout can tell "still working" from "hung".
            _emit({"status": "processing", "image": image_path})
            image = _read_image_rgb(image_path)

            masks = generator.generate(image)
            polygons = []
            for mask_data in masks:
                polygon = _mask_to_polygon(mask_data["segmentation"].astype("uint8"))
                if polygon:
                    polygons.append(polygon)

            _log(_result_line(len(polygons), [], start))
            _emit({
                "image": image_path,
                "masks": polygons,
                "classes": ["object"] * len(polygons),
            })
        except Exception as exc:
            _log_failure(exc)
            _emit({"image": image_path, "error": str(exc)})


def _run_sam2_custom(request, model_path):
    import torch

    from core.tools.sam2_loader import load_sam2_custom_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"SAM automatic segmentation: {len(request['image_paths'])} image(s) | "
         f"custom model: {os.path.basename(model_path)} | device: {device}")
    _emit({"status": "loading_model", "variant": os.path.basename(model_path)})
    _run_sam2_generator(request, load_sam2_custom_model(device, model_path))


def _run_cellpose_sam(request, model_path):
    import numpy as np
    import torch

    try:
        from cellpose import models as cellpose_models
    except ImportError:
        raise RuntimeError(
            f"{os.path.basename(model_path)} is a Cellpose-SAM model, which "
            "needs the 'cellpose' package (not installed). Install it into "
            "the SegmentME environment with:  pip install cellpose"
        )

    paths = request["image_paths"]
    _log(f"Cellpose-SAM segmentation: {len(paths)} image(s) | model: "
         f"{os.path.basename(model_path)} | GPU: {torch.cuda.is_available()}")
    _emit({"status": "loading_model", "variant": os.path.basename(model_path)})
    model = cellpose_models.CellposeModel(
        gpu=torch.cuda.is_available(), pretrained_model=model_path
    )

    for index, image_path in enumerate(paths, start=1):
        start = time.perf_counter()
        _log(f"[{index}/{len(paths)}] {os.path.basename(image_path)}")
        try:
            _emit({"status": "processing", "image": image_path})
            image = _read_image_rgb(image_path)

            # eval returns (labels, flows, styles); labels is an HxW int
            # array where 0 = background and 1..N = one cell each.
            labels = model.eval(image)[0]
            polygons = []
            for label in range(1, int(labels.max()) + 1):
                polygon = _mask_to_polygon((labels == label).astype(np.uint8))
                if polygon:
                    polygons.append(polygon)

            _log(_result_line(len(polygons), [], start))
            _emit({
                "image": image_path,
                "masks": polygons,
                "classes": ["object"] * len(polygons),
            })
        except Exception as exc:
            _log_failure(exc)
            _emit({"image": image_path, "error": str(exc)})


# User-browsed checkpoints, keyed by _detect_custom_family result.
# Supporting a new model family = one detect clause + one runner here.
_CUSTOM_RUNNERS = {
    "sam2": _run_sam2_custom,
    "cellpose_sam": _run_cellpose_sam,
}


def _run_sam(request):
    import torch

    from core.tools import sam_registry
    from core.tools.sam2_loader import load_sam2_model

    model_path = request.get("model_path")
    if model_path:
        runner = _CUSTOM_RUNNERS.get(_detect_custom_family(model_path))
        if runner is None:
            raise RuntimeError(
                f"{os.path.basename(model_path)} is not a recognized model type. "
                "Supported custom models: SAM2/SAM2.1 (official or fine-tuned) "
                "and Cellpose-SAM (e.g. cpsam.pt)."
            )
        return runner(request, model_path)

    variant_key = request.get("sam_variant_key")
    variant = sam_registry.SAM_VARIANTS.get(variant_key)
    if variant is None or variant.family != "sam2":
        raise RuntimeError(
            f"Batch SAM inference only supports SAM2/2.1 models; "
            f"{variant.label if variant else variant_key!r} is interactive-only. "
            "Pick a SAM2 variant in Settings -> SAM Model, or use the sidebar "
            "SAM tool for interactive SAM3 segmentation."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"SAM automatic segmentation: {len(request['image_paths'])} image(s) | "
         f"model: {variant.label} | device: {device}")
    _emit({"status": "loading_model", "variant": variant.label})
    _run_sam2_generator(request, load_sam2_model(device, variant))


def main():
    _setup_streams()
    line = sys.stdin.readline()
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        _log(f"Could not parse worker request: {exc}")
        _emit({"fatal": f"Could not parse worker request: {exc}"})
        return 1

    mode = request.get("mode")
    try:
        if mode == "yolo":
            _run_yolo(request)
        elif mode == "hires":
            _run_hires(request)
        elif mode == "sam":
            _run_sam(request)
        else:
            _log(f"Unsupported inference mode: {mode!r}")
            _emit({"fatal": f"Unsupported inference mode: {mode!r}"})
            return 1
    except Exception as exc:
        _log(f"Inference stopped: {exc}")
        _log(traceback.format_exc().rstrip())
        _emit({"fatal": str(exc)})
        return 1

    _emit({"done": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())

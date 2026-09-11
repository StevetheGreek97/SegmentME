"""Subprocess entry point for batch inference (YOLO + SAM2/2.1).

Runs as a plain OS process spawned by core.inference_manager.InferenceManager
via QProcess -- deliberately has no PyQt6 import so it stays cheap to start
and can never pull a GUI event loop into the child. Talks to the parent over
stdio with NDJSON: reads one JSON request line from stdin, writes one JSON
result line per image to stdout, then a final {"done": true} line.

Dispatched from main.py before any GUI imports happen, so the exact same
mechanism works whether the app runs as `python main.py` or as a frozen
PyInstaller executable (both re-invoke themselves with --inference-worker).
"""
import json
import os
import sys


def _emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _run_yolo(request):
    from PIL import Image
    from ultralytics import YOLO

    model = YOLO(request["model_path"])
    for image_path in request["image_paths"]:
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
            _emit({
                "image": image_path,
                "masks": [m.tolist() for m in masks_xy],
                "classes": class_names,
            })
        except Exception as exc:
            _emit({"image": image_path, "error": str(exc)})


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

    for image_path in request["image_paths"]:
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

            _emit({
                "image": image_path,
                "masks": polygons,
                "classes": ["object"] * len(polygons),
            })
        except Exception as exc:
            _emit({"image": image_path, "error": str(exc)})


def _run_sam2_custom(request, model_path):
    import torch

    from core.tools.sam2_loader import load_sam2_custom_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    _emit({"status": "loading_model", "variant": os.path.basename(model_path)})
    model = cellpose_models.CellposeModel(
        gpu=torch.cuda.is_available(), pretrained_model=model_path
    )

    for image_path in request["image_paths"]:
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

            _emit({
                "image": image_path,
                "masks": polygons,
                "classes": ["object"] * len(polygons),
            })
        except Exception as exc:
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
    _emit({"status": "loading_model", "variant": variant.label})
    _run_sam2_generator(request, load_sam2_model(device, variant))


def main():
    line = sys.stdin.readline()
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        _emit({"fatal": f"Could not parse worker request: {exc}"})
        return 1

    mode = request.get("mode")
    try:
        if mode == "yolo":
            _run_yolo(request)
        elif mode == "sam":
            _run_sam(request)
        else:
            _emit({"fatal": f"Unsupported inference mode: {mode!r}"})
            return 1
    except Exception as exc:
        _emit({"fatal": str(exc)})
        return 1

    _emit({"done": True})
    return 0


if __name__ == "__main__":
    sys.exit(main())

import os
import re
import numpy as np
import sys
import yaml
from pathlib import Path

from services.logger import get_logger

logger = get_logger(__name__)
def natural_key(name):
    """Sort key that orders names as people do: case-insensitive, and the
    numbers in them by value ("img2" before "img10")."""
    return [(0, int(part), "") if part.isdigit() else (1, 0, part.casefold())
            for part in re.split(r"(\d+)", os.path.basename(name))]


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")


def is_image_file(path) -> bool:
    return str(path).lower().endswith(IMAGE_EXTENSIONS)   # IMG_01.JPG, scan.TIF too


def find_images(folder):
    """Every image in `folder` and all its subfolders, in natural name order
    (by file name, then folder). Hidden folders (".segmentme", ".Trash", ...)
    are skipped; folder links are not followed (no loops)."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        found.extend(os.path.join(root, f) for f in files if is_image_file(f))
    return sorted(found, key=lambda p: (natural_key(p), natural_key(os.path.dirname(p))))


def loader(folder_path):
    """The images in a folder, sorted by name (natural order). os.listdir's
    order depends on the file system -- it differs between Linux, Windows and
    two copies of the same folder -- so the slider, Next / Previous and the
    first image shown would otherwise too."""
    if folder_path:
        image_paths = []
        for f in os.listdir(folder_path):
            joined = os.path.join(folder_path, f)
            if is_image_file(joined):
                image_paths.append(joined)
        return sorted(image_paths, key=natural_key)
    else:
        return None




def normalize_coordinates(mask, img_width, img_height):
    """
    Normalize mask coordinates to the image dimensions using matrix operations.

    Args:
        mask (np.ndarray): Array of shape (N, 2) containing (x, y) coordinates.
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        np.ndarray: Array of normalized (x, y) coordinates.
    """
    normalization_matrix = np.array([img_width, img_height], dtype=np.float32)
    return mask / normalization_matrix  # Element-wise division


def format_masks_to_yolo(class_ids, masks, img_width, img_height):
    """
    Format all masks to YOLO segmentation format using matrix operations.

    Args:
        masks (list of np.ndarray): List of masks, each as an array of shape (N, 2).
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        list of str: YOLO formatted strings for each mask.
    """
    yolo_annotations = []
    for class_id, mask in zip(class_ids, masks):
        normalized_mask = normalize_coordinates(mask, img_width, img_height)
        flattened_coords = normalized_mask.flatten()
        annotation = f"{class_id} " + " ".join(f"{coord:.6f}" for coord in flattened_coords)
        yolo_annotations.append(annotation)
    return yolo_annotations


def write_annotations_to_file(image_name, yolo_annotations, export_dir):
    """
    Write YOLO annotations to a file.

    Args:
        image_name (str): Name of the image without extension.
        yolo_annotations (list of str): List of YOLO formatted annotation strings.
        export_dir (str): Directory to save the .txt file.
    """
    txt_filename = os.path.join(export_dir, f"{image_name}.txt")
    with open(txt_filename, 'w') as file:
        file.write("\n".join(yolo_annotations) + '\n')
    logger.debug("Wrote %d annotation(s) to %s", len(yolo_annotations), txt_filename)



def get_resource_path(rel_path):
    """Get the absolute path to a resource, adjusting for executable and normal script execution.

    Accepts rel_path with '/' separators on any OS (str or Path) and returns
    a str so callers can pass it straight to cv2/torch/open.
    """
    if getattr(sys, 'frozen', False):
        # If the application is running as an executable (PyInstaller)
        base_path = Path(sys._MEIPASS)
    else:
        # Normal script execution: move one directory up to get out of 'services'
        base_path = Path(__file__).resolve().parent.parent

    return str(base_path / rel_path)

def get_tooltip(tool):
    with open(get_resource_path('resources/docs/tooltips.yml'), 'r') as file:
        TOOLTIPS = yaml.safe_load(file)
        return TOOLTIPS[tool]

from pathlib import Path
import numpy as np
import yaml
from shapely.geometry import Polygon
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from PyQt6.QtWidgets import QApplication, QProgressDialog
from PyQt6.QtCore import Qt

from services.logger import get_logger

logger = get_logger(__name__)

class BaseExporter:
    """Writes the project's masks as a training dataset.

    state: the app's StateManager (images, masks, classes, project folder);
    parent: the window, used only as the parent of dialogs."""

    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.export_dir = self.set_export_dir()

    def set_export_dir(self):
        base_dir = Path(self.state.project_root) / "labels"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def generate_data_yaml(self):
        class_manager = self.state.class_manager
        class_names = class_manager.get_all_class_names()
        num_classes = len(class_names)

        data_yaml = {
            "path": str(self.state.project_root),
            "test": "autosplit_test.txt",
            "train": "autosplit_train.txt",
            "val": "autosplit_val.txt",
            "nc": num_classes,
            "names": {
                class_manager.get_idx_by_name(name) - 1: name for name in class_names
            }
        }

        yaml_path = Path(self.state.project_root) / "data.yaml"
        with yaml_path.open("w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)
        logger.info("Wrote dataset config (%d classes) to %s", num_classes, yaml_path)

    def _show_progress_dialog(self, total):
        progress_dialog = QProgressDialog("Exporting annotations...", "Cancel", 0, total, self.parent)
        progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        progress_dialog.setValue(0)
        return progress_dialog

    def _read_image_sizes(self, image_paths, progress_dialog, workers=16):
        """(width, height) of each image from its file header, several at a
        time, keeping the window responsive. Reading whole images just for
        their size took ~4.5 s per 270 MB scan from a cloud folder.
        Returns {path: (w, h) or the exception}; None if cancelled."""
        from core.importers.yolo_importer import image_size   # EXIF-aware, as the app shows it
        paths = list(dict.fromkeys(str(p) for p in image_paths))
        sizes = {}
        if not paths:
            return sizes
        started = time.perf_counter()
        progress_dialog.setLabelText("Reading image sizes…")
        progress_dialog.setMaximum(len(paths))
        progress_dialog.setValue(0)

        def read(path):
            try:
                return path, image_size(path)
            except Exception as exc:  # reported per image by the caller
                return path, exc

        pool = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [pool.submit(read, p) for p in paths]
            for done, future in enumerate(as_completed(futures), 1):
                path, size = future.result()
                sizes[path] = size
                progress_dialog.setValue(done)
                QApplication.processEvents()
                if progress_dialog.wasCanceled():
                    for f in futures:
                        f.cancel()
                    logger.info("Export cancelled while reading image sizes")
                    return None
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        logger.info("Read %d image size(s) in %.1f s", len(paths), time.perf_counter() - started)
        progress_dialog.setLabelText("Exporting annotations...")
        return sizes

    @staticmethod
    def _size_of(sizes, image_path):
        size = sizes.get(str(image_path))
        if isinstance(size, Exception):
            raise ValueError(f"Failed to read image {image_path}: {size}")
        if size is None:
            raise ValueError(f"No size read for {image_path}")
        return size

    @staticmethod
    def simplify_polygon(mask, tolerance=0.01):
        polygon = Polygon(mask)
        return np.array(polygon.simplify(tolerance, preserve_topology=True).exterior.coords)

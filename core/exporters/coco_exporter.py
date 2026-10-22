import json
import random
import shutil
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox
from core.exporters.base_exporter import BaseExporter
from services.logger import get_logger

logger = get_logger(__name__)


class COCOExporter(BaseExporter):
    def set_export_dir(self):
        base_dir = Path(self.state.project_root) / "labels_coco"
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    def export_all_annotations(self, train_pct, val_pct, test_pct):
        image_paths = self.state.image_paths
        if not image_paths:
            logger.warning("Export requested with no images loaded; ignoring")
            return

        if any(self.export_dir.iterdir()):
            reply = QMessageBox.warning(
                self.parent,
                "Overwrite Existing Labels",
                f"The COCO annotations folder already exists at:\n\n{self.export_dir}\n\n"
                "Do you want to overwrite it?\nAll existing annotation files will be deleted.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                logger.info("Export cancelled by user (labels folder exists)")
                return

            shutil.rmtree(self.export_dir)
            self.export_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Deleted existing COCO labels folder %s", self.export_dir)

        splits = self._assign_splits(image_paths, train_pct, val_pct, test_pct)
        categories = self._build_categories()
        docs = {
            name: {"images": [], "annotations": [], "categories": categories}
            for name in ("train", "val", "test")
        }

        progress_dialog = self._show_progress_dialog(len(image_paths))
        logger.info("Exporting COCO annotations for %d image(s) to %s (split %d/%d/%d)",
                    len(image_paths), self.export_dir, train_pct, val_pct, test_pct)
        self._sizes = self._read_image_sizes(image_paths, progress_dialog)
        if self._sizes is None:            # cancelled
            progress_dialog.close()
            return
        progress_dialog.setMaximum(len(image_paths))

        image_id = 1
        annotation_id = 1
        for index, image_path in enumerate(image_paths):
            if progress_dialog.wasCanceled():
                logger.info("Export cancelled by user after %d image(s)", index)
                break

            try:
                doc = docs[splits[image_path]]
                image_id, annotation_id = self._process_image(
                    Path(image_path), doc, image_id, annotation_id
                )
            except Exception:
                logger.exception("Failed to export annotations for %s; continuing", image_path)

            progress_dialog.setValue(index + 1)

        for split_name, doc in docs.items():
            if not doc["images"]:
                continue
            out_path = self.export_dir / f"instances_{split_name}.json"
            with out_path.open("w") as f:
                json.dump(doc, f)
            logger.info("Wrote %d image(s) / %d annotation(s) to %s",
                        len(doc["images"]), len(doc["annotations"]), out_path)

        progress_dialog.close()
        logger.info("COCO export finished")

    def _assign_splits(self, image_paths, train_pct, val_pct, test_pct):
        shuffled = list(image_paths)
        random.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_pct / 100)
        n_val = round(n * val_pct / 100)

        assignments = {}
        for i, path in enumerate(shuffled):
            if i < n_train:
                assignments[path] = "train"
            elif i < n_train + n_val:
                assignments[path] = "val"
            else:
                assignments[path] = "test"
        return assignments

    def _build_categories(self):
        class_manager = self.state.class_manager
        return [
            {"id": class_manager.get_idx_by_name(name), "name": name, "supercategory": "object"}
            for name in class_manager.get_all_class_names()
        ]

    def _process_image(self, image_path: Path, doc, image_id: int, annotation_id: int):
        w, h = self._size_of(self._sizes, image_path)   # header only, read in advance

        doc["images"].append({
            "id": image_id,
            "file_name": image_path.name,
            "width": w,
            "height": h,
        })

        class_manager = self.state.class_manager
        masks = self.state.mask_manager.load_masks(image_path.stem)

        for mask_id, mask_array, class_name, surface_area in masks:
            if mask_array is None or len(mask_array) < 3:
                continue

            category_id = class_manager.get_idx_by_name(class_name)
            if category_id is None:
                logger.warning("Class %r not found for mask %s; mask skipped in export",
                               class_name, mask_id)
                continue

            polygon = self.simplify_polygon(mask_array)
            x_min, y_min = polygon.min(axis=0)
            x_max, y_max = polygon.max(axis=0)
            area = float(surface_area) if surface_area else float((x_max - x_min) * (y_max - y_min))

            doc["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [polygon.flatten().tolist()],
                "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
                "area": area,
                "iscrowd": 0,
            })
            annotation_id += 1

        return image_id + 1, annotation_id

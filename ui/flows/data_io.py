"""
File menu jobs that move data in and out of the project: Import Images,
Import Annotations and Export (YOLO / COCO). Each is a function of the main
window; the window's menu methods call them.
"""
import os
from pathlib import Path

from PyQt6.QtWidgets import QDialog, QFileDialog, QMessageBox

from core.exporters.coco_exporter import COCOExporter
from core.exporters.yolo_exporter import YOLOExporter
from core.importers.yolo_importer import ImportCancelled, ImportFailed, describe, run_import
from services import image_files
from services.file_handlers import loader
from services.logger import get_logger
from services.seproj import update_classes as seproj_update_classes
from services.seproj import update_images as seproj_update_images
from ui.dialogs.background_task import run_in_background
from ui.dialogs.export_dialog import ExportDialog
from ui.dialogs.image_mode import ImageModeDialog
from ui.dialogs.import_dialog import ImportAnnotationsDialog, ImportReportDialog
from ui.dialogs.place_progress import enough_space, run_placement

logger = get_logger(__name__)


def import_images(window):
    files, _ = QFileDialog.getOpenFileNames(
        window, "Select Images to Import",
        filter="Images (*.png *.jpg *.jpeg *.bmp *.tif)"
    )
    if not files:
        return
    mode_dialog = ImageModeDialog(window, len(files))
    if mode_dialog.exec() != QDialog.DialogCode.Accepted:
        return
    mode = mode_dialog.mode()

    # window.config: the open project's settings (set by initialize_project)
    images_dir = window.config.get_images_dir()
    os.makedirs(images_dir, exist_ok=True)

    # Names already in the project are skipped (place_images checks too).
    jobs = [(f, os.path.join(images_dir, os.path.basename(f))) for f in files]
    new_sources = [src for src, dest in jobs if not os.path.lexists(dest)]
    if not enough_space(window, new_sources, images_dir, mode, "Import Images"):
        return
    # Copying runs on a worker thread with a progress window and Cancel.
    logger.info("Import Images: %d file(s) chosen, %s, into %s", len(files), mode, images_dir)
    run = run_placement(window, jobs, mode, title="Import Images")
    if run.cancelled:
        QMessageBox.information(window, "Import cancelled", "Nothing was added.")
        return
    if run.error is not None:
        QMessageBox.critical(window, "Import failed",
                             f"{run.error}\n\nNothing was added.")
        return
    result = run.result
    imported_count, placements = len(result.placed), result.placements
    failed = [f"{name}: {why}" for name, why in result.failed]

    image_paths = loader(images_dir)
    window.state_manager.set_image_paths(image_paths)
    window.slider.set_image_count(len(image_paths))
    current = window.state_manager.current_image_path
    if current != window.image_display.shown_path:   # the image on screen stays: no re-read
        window.image_display.show_image(current)

    # Keep .SEproj in sync with the folder
    project_root = getattr(window, "project_root", None)
    if project_root:
        seproj_update_images(project_root, [os.path.basename(p) for p in image_paths])

    text = f"✅ Imported {imported_count} images into your project."
    placed = image_files.summarize(placements)
    if placed and mode == image_files.LINK:
        text += "\n" + placed
    if failed:
        text += f"\n\n{len(failed)} could not be added:\n" + "\n".join(failed[:10])
    QMessageBox.information(window, "Import Complete", text)


def import_annotations(window):
    """
    File > Import Annotations: YOLO polygon .txt files plus the dataset's
    data.yaml, for images already in the project (each .txt goes to the
    image with exactly the same name; see core/importers/yolo_importer.py).
    A review window shows what will happen first; the import itself is
    all or nothing. Images are not copied or changed.
    """
    if not window.state_manager.image_paths:
        QMessageBox.information(window, "Import Annotations",
                                "This project has no images yet. Add images first "
                                "(File > Import Images), then import their annotations.")
        return
    project_root = Path(window.state_manager.project_root)
    if getattr(window, "project_root", None) is None:
        window.project_root = str(project_root)

    existing_classes = {row[1]: row[2] for row in window.state_manager.class_manager.list_classes()}
    existing_masks = window.state_manager.mask_manager.counts_per_image()
    # The folder is chosen inside the dialog (its Browse… button).
    dialog = ImportAnnotationsDialog(window, None, window.state_manager.image_paths,
                                     existing_classes, existing_masks)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        logger.debug("Import Annotations closed without importing")
        return
    folder = dialog.scan_result.folder
    scan, targets = dialog.scan_result, dialog.class_targets()
    logger.info("Import Annotations from %s: %d mask(s) for %d image(s); classes %s",
                folder, scan.mask_count, scan.images_with_masks,
                {scan.class_names[c]: t for c, t in targets.items()})

    db_path = window.state_manager.db.db_path

    def work(progress, cancelled):
        # Own connection: the worker thread writes while the window's
        # connection keeps reading (WAL). All or nothing, as before.
        from core.database.connection import DatabaseConnection
        db = DatabaseConnection(db_path)
        try:
            def report(done, total, name):
                progress(done, total, f"Reading {name}" if name else "Saving the masks…")
                return not cancelled()
            return run_import(scan, targets, existing_classes, db, report)
        finally:
            db.conn.close()

    run = run_in_background(window, "Import Annotations", work, cancel_exception=ImportCancelled,
                            start_text="Importing annotations…")
    if run.cancelled:
        QMessageBox.information(window, "Import cancelled", "Nothing was imported.")
        return
    if isinstance(run.error, ImportFailed):
        QMessageBox.warning(window, "Import failed", f"{run.error}\n\nNothing was imported.")
        return
    if run.error is not None:
        QMessageBox.critical(window, "Import failed", f"{run.error}\n\nNothing was imported.")
        return
    result = run.result
    logger.info("Import Annotations wrote %d mask(s) in %.1f s (in the background)",
                result.masks, run.seconds)

    refresh_after_annotation_import(window)
    text = describe(scan, result)
    logger.info("Import report for %s:\n%s", folder, text)
    ImportReportDialog(window, text, save_dir=window.project_root).exec()


def refresh_after_annotation_import(window):
    """Show the new classes and masks (the current image included)."""
    rows = window.state_manager.class_manager.list_classes()
    seproj_update_classes(window.project_root, [{"name": r[1], "color": r[2]} for r in rows])
    # the import wrote through its own connection: tell everyone
    window.state_manager.classes_changed.emit()
    window.state_manager.masks_updated.emit()


def export_annotations_dataset(window):
    dialog = ExportDialog(window)
    if dialog.exec():
        settings = dialog.get_settings()
        format_ = settings["format"]
        train = settings["train"]
        val = settings["val"]
        test = settings["test"]

        if format_ == "yolo":
            exporter = YOLOExporter(window.state_manager, window)
            exporter.export_all_annotations(train, val, test)
        elif format_ == "coco":
            exporter = COCOExporter(window.state_manager, window)
            exporter.export_all_annotations(train, val, test)
        else:
            QMessageBox.warning(window, "Unsupported Format", f"Export format '{format_}' not supported.")

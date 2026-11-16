"""
The training settings window (what it sends to YOLO, what it remembers,
the dataset summary) and the training monitor (progress, time left, best
score, how a run ends).
"""
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtGui import QColor

from tests.conftest import make_image


@pytest.fixture(autouse=True)
def forget_training_settings():
    """Each test starts without remembered training settings (the test
    settings file is shared by the whole run)."""
    from PyQt6.QtCore import QSettings
    from ui.dialogs.training_dialog import SETTINGS_GROUP, SETTINGS_ORG
    settings = QSettings(SETTINGS_ORG, SETTINGS_ORG)
    settings.remove(SETTINGS_GROUP)
    settings.sync()
    yield
    settings.remove(SETTINGS_GROUP)
    settings.sync()


def _project(tmp_path, names=("a.png", "b.png", "c.png"), name="proj"):
    root = tmp_path / name
    (root / ".segmentme").mkdir(parents=True)
    for n in names:
        make_image(root / "images" / n)
    (root / f"{name}.SEproj").write_text(json.dumps({"name": name, "images": list(names)}))
    return root


def _square(x=5):
    return np.array([[x, x], [x + 10, x], [x + 10, x + 10], [x, x + 10]], dtype=np.float32)


@pytest.fixture
def window(qapp, tmp_path, no_popups):
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    root = _project(tmp_path)
    db = str(root / ".segmentme" / "masks.db")
    win = MainApp(db_path=db)
    initialize_project(win, db)
    qapp.processEvents()
    yield win
    win.state_manager.db.conn.close()


@pytest.fixture
def dialog(window):
    from ui.dialogs.training_dialog import TrainingDialog
    d = TrainingDialog(window)
    yield d
    d.deleteLater()


def _export(root):
    for f in ("autosplit_train.txt", "autosplit_val.txt", "autosplit_test.txt", "data.yaml"):
        (root / f).write_text("x")
    (root / "labels").mkdir(exist_ok=True)


# ---------------------------------------------------------------- settings
def test_small_numbers_reach_yolo_unrounded(dialog):
    """The old window rounded every number to 2 decimals: weight decay 0.0005
    was sent as 0, hue 0.015 as 0.02."""
    s = dialog.get_settings()
    assert s.weight_decay == pytest.approx(0.0005)
    assert s.hsv_h == pytest.approx(0.015)
    assert s.lr0 == pytest.approx(0.01) and s.momentum == pytest.approx(0.937)
    dialog.perspective.setValue(0.0003)
    assert dialog.get_settings().perspective == pytest.approx(0.0003)


def test_defaults_match_yolo(dialog):
    from core.trainer.trainer_settings import TrainingSettings
    s = dialog.get_settings()
    for field in ("hsv_s", "hsv_v", "translate", "scale", "fliplr", "mosaic", "erasing", "patience"):
        assert getattr(s, field) == pytest.approx(getattr(TrainingSettings, field)), field
    assert s.epochs == 100 and s.imgsz == 640 and s.time is None
    assert s.model.endswith("yolo11n-seg.pt")


def test_auto_batch_is_sent_as_minus_one(dialog):
    assert dialog.batch_box.value() == 0
    assert dialog.batch_box.text().startswith("Auto")
    assert dialog.get_settings().batch == -1
    dialog.batch_box.setValue(8)
    assert dialog.get_settings().batch == 8


def test_image_size_snaps_to_a_multiple_of_32(dialog):
    dialog.imgsz_box.setValue(1000)
    dialog._snap_image_size()
    assert dialog.imgsz_box.value() == 992
    assert dialog.get_settings().imgsz % 32 == 0


def test_output_folder_never_overwrites_an_old_run(dialog):
    dialog.project_name.setText("daphnia")
    dialog.run_name.setText("run")
    first = dialog.get_settings().output_dir
    os.makedirs(first)
    second = dialog.get_settings()
    assert second.output_dir.endswith(os.path.join("daphnia", "run1"))
    assert second.name == "run1" and second.project == "daphnia"
    assert "run1" in dialog.output_dir_input.text()


def test_empty_names_fall_back_to_defaults(dialog):
    dialog.project_name.setText("")
    dialog.run_name.setText("  ")
    s = dialog.get_settings()
    assert s.project == "my-model" and s.name == "run"


def test_names_cannot_hold_folder_separators(dialog):
    from PyQt6.QtGui import QValidator
    validator = dialog.project_name.validator()
    assert validator.validate("my model-2", 0)[0] == QValidator.State.Acceptable
    assert validator.validate("../evil", 0)[0] != QValidator.State.Acceptable


def test_continue_from_an_earlier_training(window, tmp_path):
    from ui.dialogs.training_dialog import TrainingDialog
    weights = tmp_path / "proj" / "trainings" / "daphnia" / "run" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"x")
    d = TrainingDialog(window)
    index = d.start_from.findData(str(weights))
    assert index == 1                                       # right after "A new model"
    assert "daphnia/run" in d.start_from.itemText(1)
    d.start_from.setCurrentIndex(1)
    assert not d.model_selector.isEnabled()                 # its size is fixed
    assert d.get_settings().model == str(weights)
    d.start_from.setCurrentIndex(0)
    assert d.model_selector.isEnabled()
    assert d.get_settings().model.endswith(".pt") and "trainings" not in d.get_settings().model


def _checkpoint(path, task):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_args": {"task": task}}, str(path))
    return path


def test_start_from_does_not_list_the_models_folder(window, tmp_path, monkeypatch):
    """Only a new model, this project's trainings and Browse (the models
    folder also holds non-YOLO files such as FastSAM)."""
    from services import model_store
    from ui.dialogs.training_dialog import BROWSE, TrainingDialog
    mine = _checkpoint(tmp_path / "models" / "yolo" / "FastSAM-s.pt", "segment")
    monkeypatch.setattr(model_store, "installed_yolo_models", lambda: [mine])
    d = TrainingDialog(window)
    assert d.start_from.findData(str(mine)) < 0
    texts = [d.start_from.itemText(i) for i in range(d.start_from.count())]
    assert not any("Your model" in t for t in texts)
    assert d.start_from.itemData(d.start_from.count() - 1) == BROWSE


def test_browse_for_a_custom_model(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from ui.dialogs.training_dialog import BROWSE, TrainingDialog
    custom = _checkpoint(tmp_path / "elsewhere" / "DaphnAI.pt", "segment")
    d = TrainingDialog(window)
    browse = d.start_from.findData(BROWSE)
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (str(custom), ""))
    d.start_from.setCurrentIndex(browse)
    d._start_from_activated(browse)
    assert d.start_from.currentData() == str(custom)
    assert d.start_from.currentText() == "Custom: DaphnAI.pt"
    assert not d.model_selector.isEnabled()
    assert d.get_settings().model == str(custom)
    assert d.start_from.findData(BROWSE) == d.start_from.count() - 1   # browse stays last
    d.use_custom_model(str(custom))                                   # chosen again: not added twice
    assert sum(d.start_from.itemData(i) == str(custom) for i in range(d.start_from.count())) == 1


def test_browse_cancelled_keeps_the_previous_choice(dialog, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog
    from ui.dialogs.training_dialog import BROWSE
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    browse = dialog.start_from.findData(BROWSE)
    dialog.start_from.setCurrentIndex(browse)
    dialog._start_from_activated(browse)
    assert dialog.start_from.currentIndex() == 0
    assert dialog.get_settings().model.endswith("yolo11n-seg.pt")


def test_a_detection_model_is_refused(window, tmp_path, monkeypatch, no_popups):
    from ui.dialogs.training_dialog import TrainingDialog
    detect = _checkpoint(tmp_path / "yolo11n.pt", "detect")
    d = TrainingDialog(window)
    assert d.use_custom_model(str(detect)) is False
    assert no_popups and no_popups[-1][1] == "Not a Segmentation Model"
    assert d.start_from.findData(str(detect)) < 0
    assert d.start_from.currentIndex() == 0


def test_custom_model_is_remembered(window, tmp_path, monkeypatch):
    from ui.dialogs.training_dialog import TrainingDialog
    custom = _checkpoint(tmp_path / "elsewhere" / "DaphnAI.pt", "segment")
    d = TrainingDialog(window)
    d.use_custom_model(str(custom))
    d.remember()
    again = TrainingDialog(window)
    assert again.start_from.currentData() == str(custom)
    custom.unlink()                                          # moved away since
    assert TrainingDialog(window).start_from.currentIndex() == 0


def test_settings_are_remembered(window, monkeypatch):
    from ui.dialogs.training_dialog import TrainingDialog
    d = TrainingDialog(window)
    _export(d.project_root)
    monkeypatch.setattr(d, "export_is_outdated", lambda: False)
    d.epoch_box.setValue(250)
    d.imgsz_box.setValue(1024)
    d._set("model", "yolo11s-seg.pt")
    d.project_name.setText("daphnia")
    d.weight_decay.setValue(0.0007)
    d.degrees.setValue(180)
    d.flipud.setValue(0.5)
    d.validate_and_accept()
    assert d.result() == 1

    again = TrainingDialog(window)
    assert again.epoch_box.value() == 250 and again.imgsz_box.value() == 1024
    assert again.model_selector.currentData() == "yolo11s-seg.pt"
    assert again.project_name.text() == "daphnia"
    assert again.weight_decay.value() == pytest.approx(0.0007)
    assert again.degrees.value() == 180 and again.flipud.value() == pytest.approx(0.5)

    again.reset_to_defaults()
    assert again.epoch_box.value() == 100 and again.degrees.value() == 0
    assert again.flipud.value() == 0


def test_cancelled_settings_are_not_remembered(window):
    from ui.dialogs.training_dialog import TrainingDialog
    d = TrainingDialog(window)
    d.epoch_box.setValue(333)
    d.reject()
    assert TrainingDialog(window).epoch_box.value() == 100


def test_augmentation_has_no_preset_menu(dialog):
    from PyQt6.QtWidgets import QComboBox
    combos = dialog.augment_tab.findChildren(QComboBox)
    assert set(combos) == {dialog.copy_paste_mode, dialog.auto_augment}
    dialog.mosaic.setValue(0.3)
    assert dialog.get_settings().mosaic == pytest.approx(0.3)


def test_summary_counts_images_and_masks_per_class(window):
    from ui.dialogs.training_dialog import TrainingDialog
    state = window.state_manager
    state.class_manager.add_class("pulex", QColor("#2a78d6"))
    state.class_manager.add_class("galeata", QColor("#eb6834"))
    state.class_manager.add_class("empty", QColor("#1baf7a"))
    for image, cls in (("a", "pulex"), ("a", "pulex"), ("b", "galeata")):
        state.mask_manager.save_mask(_square(), image, cls)
    d = TrainingDialog(window)
    annotated, total, classes = d.dataset_summary()
    assert annotated == 2 and total == 3
    counts = {name: n for name, _colour, n in classes}
    assert counts["pulex"] == 2 and counts["galeata"] == 1 and counts["empty"] == 0
    assert "2 of 3 images annotated" in d.summary_headline.text()
    assert "3 masks" in d.summary_headline.text()
    assert "empty" in d.summary_classes.text()             # listed as not learned
    assert "Not exported yet" in d.summary_export.text()


def test_summary_says_when_the_export_is_current(window):
    from ui.dialogs.training_dialog import TrainingDialog
    root = window.state_manager.project_root
    _export(root)
    future = time.time() + 60
    os.utime(root / "data.yaml", (future, future))
    assert "up to date" in TrainingDialog(window).summary_export.text()


def test_device_is_shown_without_a_popup(window, no_popups):
    from ui.dialogs.training_dialog import TrainingDialog
    d = TrainingDialog(window)
    assert no_popups == []                                 # the old window always popped one up
    assert d.device_info.label in d.device_label.text()
    assert d.get_settings().device == d.device_info.device


@pytest.mark.parametrize("power", [0.3, 0.5, 1.0])
def test_class_weights_match_ultralytics(power):
    """The preview uses Ultralytics' own formula (DetectionTrainer, >= 8.4)."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ui.dialogs.training_dialog import TrainingDialog
    counts = np.array([8545, 192, 0, 152], np.float32)          # one class without masks
    theirs = DetectionTrainer.compute_class_weights(
        SimpleNamespace(args=SimpleNamespace(cls_pw=power)), counts)
    theirs = theirs / theirs.mean()
    assert TrainingDialog.class_weights(counts, power) == pytest.approx(list(theirs), rel=1e-5)


def test_class_balance_reaches_yolo_and_shows_the_weights(window):
    from ui.dialogs.training_dialog import TrainingDialog
    state = window.state_manager
    state.class_manager.add_class("object", QColor("#2a78d6"))
    state.class_manager.add_class("short", QColor("#eb6834"))
    for i in range(9):
        state.mask_manager.save_mask(_square(), "a", "object")
    state.mask_manager.save_mask(_square(), "b", "short")
    d = TrainingDialog(window)
    assert d.cls_pw.text() == "Off" and d.get_settings().cls_pw == 0
    assert "every class counts the same" in d.class_weights_label.text()
    assert d.get_settings().to_train_overrides()["cls_pw"] == 0
    d.cls_pw.setValue(0.5)
    text = d.class_weights_label.text()
    assert "object" in text and "short" in text and "×0.50" in text and "×1.50" in text
    s = d.get_settings()
    assert s.cls_pw == pytest.approx(0.5)
    assert s.to_train_overrides()["cls_pw"] == pytest.approx(0.5)
    d.remember()
    assert TrainingDialog(window).cls_pw.value() == pytest.approx(0.5)


def test_every_setting_has_the_ultralytics_tooltip(dialog):
    from PyQt6.QtWidgets import QFormLayout
    from ui.dialogs.training_dialog import DOCS
    for attr, (argument, _default, description) in DOCS.items():
        tip = getattr(dialog, attr).toolTip()
        assert f"<b>{argument}</b>" in tip and description[:40] in tip, attr
    labelled = 0
    for form in dialog._forms:                             # the labels carry it too
        for row in range(form.rowCount()):
            field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            if field and label and label.widget() and field.widget() in [getattr(dialog, a) for a in DOCS]:
                assert label.widget().toolTip() == field.widget().toolTip()
                labelled += 1
    assert labelled == len(DOCS)


def test_cards_use_two_columns_when_wide(dialog, qapp):
    from PyQt6.QtWidgets import QAbstractSpinBox
    grid = dialog.training_tab.widget()
    dialog.resize(1100, 700)
    dialog.show()
    qapp.processEvents()
    assert grid._columns == 2
    dialog.resize(600, 700)
    qapp.processEvents()
    assert grid._columns == 1
    widths = {box.width() for box in dialog.findChildren(QAbstractSpinBox)}
    assert len(widths) == 1                                  # every number box the same width


def test_cards_go_to_one_column_with_large_fonts(window, qapp):
    """With wide fonts (Windows) two cards side by side are wider than the
    two-column threshold; the grid must still drop to one column instead of
    staying wide behind a horizontal scrollbar."""
    from ui.dialogs.training_dialog import TrainingDialog
    old = qapp.font()
    big = qapp.font()
    big.setPointSize(16)
    qapp.setFont(big)
    try:
        d = TrainingDialog(window)
        grid = d.training_tab.widget()
        d.resize(1400, 700)
        d.show()
        qapp.processEvents()
        assert grid._columns == 2
        d.resize(600, 700)
        qapp.processEvents()
        assert grid._columns == 1
        d.close()
        d.deleteLater()
    finally:
        qapp.setFont(old)


def test_input_boxes_hold_only_the_value(dialog):
    from PyQt6.QtWidgets import QDoubleSpinBox, QSpinBox
    for box in dialog.findChildren((QSpinBox, QDoubleSpinBox)):
        assert box.suffix() == "" and box.prefix() == "", box
    assert dialog.epoch_box.text() == "100"
    assert dialog.patience_box.text() == "100"
    dialog.patience_box.setValue(0)
    assert dialog.patience_box.text() == "Off"
    assert dialog.batch_box.text() == "Auto"


def test_describe_device_on_a_cpu(monkeypatch):
    import torch
    from services import hardware
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    info = hardware.describe_device()
    assert info.device == "cpu" and not info.fast


# ----------------------------------------------------------------- monitor
def _results(path, rows, epochs_total=10):
    lines = ["epoch,time,train/box_loss,train/seg_loss,train/cls_loss,train/dfl_loss,"
             "metrics/precision(M),metrics/recall(M),metrics/mAP50(M),metrics/mAP50-95(M),val/seg_loss"]
    for e in range(1, rows + 1):
        score = [0.2, 0.5, 0.4][min(e - 1, 2)]
        lines.append(f"{e},{60 * e},{1.0 / e},{2.0 / e},{0.5 / e},{0.3 / e},"
                     f"0.6,0.5,0.7,{score},{2.5 / e}")
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def monitor(qapp, tmp_path):
    from ui.dialogs.training_monitor import TrainingMonitor
    out = tmp_path / "trainings" / "daphnia" / "run"
    out.mkdir(parents=True)
    m = TrainingMonitor()
    m.set_run(SimpleNamespace(output_dir=str(out), epochs=10))
    yield m, out
    m.timer.stop()
    m.deleteLater()


def test_monitor_shows_progress_time_left_and_best_score(monitor):
    m, out = monitor
    assert m.chip.text() == "Starting…"
    assert "daphnia/run" in m.title.text()
    _results(out / "results.csv", 3)
    m._read_results()
    assert m.chip.text() == "Training"
    assert m.progress.value() == 30 and m.progress.format() == "Epoch 3 of 10"
    assert m.best_tile.value.text() == "0.500" and m.best_tile.detail.text() == "epoch 2"
    assert m.left_tile.value.text() == "about 7:00"         # 60 s per epoch, 7 to go
    assert m.loss_tile.detail.text() == "↓ falling"
    xs, ys = m._curves["score:mAP50-95"].getData()
    assert list(xs) == [1, 2, 3] and list(ys) == pytest.approx([0.2, 0.5, 0.4])
    assert len(m._curves["val/seg_loss"].getData()[0]) == 3


def test_monitor_rereads_only_a_changed_file(monitor, monkeypatch):
    import pandas as pd
    m, out = monitor
    _results(out / "results.csv", 2)
    reads = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: reads.append(1) or real(*a, **k))
    m._read_results()
    m._read_results()
    m._read_results()
    assert reads == [1]


def test_monitor_curves_are_updated_not_duplicated(monitor):
    m, out = monitor
    for rows in (1, 2, 3):
        _results(out / "results.csv", rows)
        m._read_results(force=True)
    assert len(m.trainval_plot.listDataItems()) == 4
    assert len(m.metrics_plot.listDataItems()) == 5


def test_monitor_falls_back_to_box_scores(monitor):
    import pandas as pd
    m, _out = monitor
    m.show_results(pd.DataFrame({"epoch": [1, 2], "metrics/mAP50-95(B)": [0.1, 0.3]}))
    assert m.best_tile.value.text() == "0.300"


def test_monitor_finished_offers_close_and_the_folder(monitor, monkeypatch):
    from PyQt6.QtGui import QDesktopServices
    m, out = monitor
    _results(out / "results.csv", 10)
    m.set_finished("finished")
    assert m.chip.text() == "Finished" and m.progress.value() == 100
    assert m.stop_btn.text() == "Close" and m.stop_btn.isEnabled()
    assert not m.open_btn.isHidden()
    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()))
    m.open_btn.click()
    assert [Path(p) for p in opened] == [out]      # QUrl gives "/" even on Windows
    assert not m.timer.isActive()


def test_monitor_says_when_training_ended_early(monitor):
    m, out = monitor
    _results(out / "results.csv", 4)                       # 4 of 10: patience ended it
    m.set_finished("finished")
    assert "Finished early" in m.progress.format()
    assert m.left_tile.value.text() == "Done"


def test_monitor_failure_opens_the_log(monitor):
    m, _out = monitor
    m.append_log("Traceback: CUDA out of memory")
    m.set_finished("failed")
    assert m.chip.text() == "Failed"
    assert m.log_toggle.isChecked() and not m.log_view.isHidden()
    assert "out of memory" in m.log_view.toPlainText()


def test_monitor_stopped_by_the_user(monitor):
    m, _out = monitor
    stopped = []
    m.set_stop_callback(lambda: stopped.append(1))
    m._on_stop()
    assert stopped == [1] and m.stop_btn.text() == "Stopping…" and not m.stop_btn.isEnabled()
    m.set_finished("finished")                 # the worker ends normally after the stop
    assert m.chip.text() == "Stopped" and m.stop_btn.text() == "Close"


def test_monitor_log_shows_the_last_line_while_folded(monitor):
    m, _out = monitor
    assert m.log_view.isHidden()
    m.append_log("Epoch 3/10  seg_loss 1.2")
    assert m.last_line.text() == "Epoch 3/10  seg_loss 1.2"
    m.log_toggle.setChecked(True)
    assert not m.log_view.isHidden()


def test_worker_reports_how_training_ended(monkeypatch):
    from core.trainer import worker as worker_module
    from core.trainer.worker import TrainingWorker

    class FakeProcess:
        def __init__(self, code):
            self.code = code
            self.stdin = SimpleNamespace(write=lambda b: None, close=lambda: None)

        def wait(self):
            return self.code

    settings = SimpleNamespace(to_train_overrides=lambda: {"model": "m.pt"})
    for code, stopping, expected in ((0, False, "finished"), (1, False, "failed"), (-15, True, "stopped")):
        w = TrainingWorker(settings)
        monkeypatch.setattr(worker_module.subprocess, "Popen", lambda *a, **k: FakeProcess(code))
        monkeypatch.setattr(w, "_pump_output", lambda: None)
        w._stopping = stopping
        w.run()
        assert w.outcome == expected

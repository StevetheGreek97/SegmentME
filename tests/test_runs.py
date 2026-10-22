"""Run Inference and Train Custom Model, up to the point where the long job
would start (the job itself is replaced): dialogs answered, the right things
started and stopped."""
import json

import pytest

from tests.conftest import make_image

pytest.importorskip("sam2", reason="the main window needs SAM2 installed")


@pytest.fixture
def win(qapp, tmp_path, no_popups):
    from ui.main_window import MainApp
    from services.recent_projects import initialize_project
    from tests.conftest import wait_for_image
    root = tmp_path / "proj"
    (root / ".segmentme").mkdir(parents=True)
    make_image(root / "images" / "a.png")
    (root / "proj.SEproj").write_text(json.dumps({"name": "proj", "images": ["a.png"]}))
    db = str(root / ".segmentme" / "masks.db")
    w = MainApp(db_path=db)
    initialize_project(w, db)
    wait_for_image(w)
    yield w
    w.close()
    qapp.processEvents()
    w.state_manager.db.conn.close()
    w.deleteLater()
    qapp.processEvents()


def _inference_dialog(monkeypatch, accept, mode="yolo", model="/models/best.pt"):
    from ui.dialogs.inference_dialog import InferenceDialog
    monkeypatch.setattr(InferenceDialog, "exec", lambda self: accept)
    monkeypatch.setattr(InferenceDialog, "get_mode", lambda self: mode)
    monkeypatch.setattr(InferenceDialog, "get_selected_model", lambda self: model)
    monkeypatch.setattr(InferenceDialog, "get_threshold", lambda self: 0.4)
    monkeypatch.setattr(InferenceDialog, "get_image_dimensions", lambda self: 640)


def _fake_runs(monkeypatch):
    from core.inference_manager import InferenceManager
    runs, stops = [], []
    monkeypatch.setattr(InferenceManager, "run_inference",
                        lambda self, model, paths, *a, **k: runs.append((model, list(paths), k)))
    monkeypatch.setattr(InferenceManager, "stop_inference", lambda self: stops.append(1))
    return runs, stops


def test_inference_cancelled_starts_nothing(win, monkeypatch):
    _inference_dialog(monkeypatch, accept=False)
    runs, _ = _fake_runs(monkeypatch)
    win.popup_inference_dialog("Run Inference")
    assert runs == []


def test_inference_without_a_model_says_so(win, monkeypatch, no_popups):
    _inference_dialog(monkeypatch, accept=True, model=None)
    runs, _ = _fake_runs(monkeypatch)
    win.popup_inference_dialog("Run Inference")
    assert runs == [] and any("No YOLO model" in p[2] for p in no_popups)


def test_inference_runs_on_the_project_images_and_stops_on_close(win, monkeypatch):
    _inference_dialog(monkeypatch, accept=True)
    runs, stops = _fake_runs(monkeypatch)
    win.popup_inference_dialog("Run Inference")
    [(model, paths, kwargs)] = runs
    assert model == "/models/best.pt" and paths == win.state_manager.image_paths
    win.popup_inference_dialog("Run Inference")           # a second run stops the first
    assert len(runs) == 2 and stops == [1]
    win.close()                                            # closing the window stops it too
    assert stops == [1, 1]


def test_training_cancelled_starts_nothing(win, monkeypatch):
    from ui.dialogs.training_dialog import TrainingDialog
    from core.trainer.worker import TrainingWorker
    monkeypatch.setattr(TrainingDialog, "exec", lambda self: False)
    started = []
    monkeypatch.setattr(TrainingWorker, "run", lambda self: started.append(1))
    win.popup_training_dialog()
    assert started == []


def test_training_thread_ends_when_training_ends(win, monkeypatch, qapp, tmp_path):
    """Close (after training) must leave no thread running: a running QThread
    at exit aborts the app."""
    import time
    from types import SimpleNamespace
    from ui.dialogs.training_dialog import TrainingDialog
    from ui.dialogs.training_monitor import TrainingMonitor
    from core.trainer.worker import TrainingWorker
    monkeypatch.setattr(TrainingDialog, "exec", lambda self: True)
    monkeypatch.setattr(TrainingDialog, "get_settings", lambda self: SimpleNamespace(output_dir=str(tmp_path)))
    monkeypatch.setattr(TrainingWorker, "run", lambda self: (self.log_signal.emit("epoch 1"),
                                                             self.finished_signal.emit()))
    threads = []

    def monitor_exec(self):
        threads.append(self.thread)
        end = time.time() + 5
        while time.time() < end and "Close" not in self.stop_btn.text():
            qapp.processEvents()
            time.sleep(0.01)
        self.stop_btn.click()                       # the user presses "Close"
        return 1
    monkeypatch.setattr(TrainingMonitor, "exec", monitor_exec)
    win.popup_training_dialog()
    [thread] = threads
    end = time.time() + 5
    while time.time() < end:
        qapp.processEvents()
        try:
            if not thread.isRunning():
                break
        except RuntimeError:                        # already deleted: finished
            break
        time.sleep(0.01)
    try:
        running = thread.isRunning()
    except RuntimeError:
        running = False
    assert not running

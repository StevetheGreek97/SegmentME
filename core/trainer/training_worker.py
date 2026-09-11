"""Subprocess entry point for YOLO training.

Spawned by core.trainer.worker.TrainingWorker as `<app> --training-worker`
(see services.app_process). Like the inference worker it has no PyQt6
import and is dispatched from the very top of main.py, so it behaves the
same whether the app runs from source or as a frozen executable -- which
is exactly what the old approach of shelling out to the `yolo` console
script could not do (that script only exists inside a Python environment
with ultralytics installed, never inside a PyInstaller bundle).

Protocol: one JSON line on stdin holding the ultralytics train() keyword
arguments (plus "model", the weights to start from). Everything
ultralytics prints goes to stdout as-is for the GUI's Training Monitor to
show. Exit code 0 on success, 1 on failure, 130 when interrupted.
"""
import json
import sys
import traceback


def _configure_stdout():
    # stdout is a pipe to the GUI: line-buffer it so log lines arrive as
    # they happen, and never let a stray emoji from ultralytics raise
    # UnicodeEncodeError on a Windows code page.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError):
            pass


def main() -> int:
    _configure_stdout()

    line = sys.stdin.readline()
    try:
        overrides = json.loads(line)
        model_path = overrides.pop("model")
    except (json.JSONDecodeError, KeyError, AttributeError) as exc:
        print(f"Could not parse the training request: {exc}")
        return 1

    frozen = bool(getattr(sys, "frozen", False))
    print(f"SegmentME training worker (python {sys.version.split()[0]}, "
          f"{'packaged app' if frozen else 'source checkout'})")

    try:
        import torch
        from ultralytics import YOLO

        print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
        print(f"Base model: {model_path}")

        model = YOLO(model_path)
        model.train(**overrides)
    except KeyboardInterrupt:
        print("Training interrupted.")
        return 130
    except Exception as exc:
        print(f"Training failed: {exc}")
        traceback.print_exc(file=sys.stdout)
        return 1

    save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir:
        print(f"Training complete. Results saved to {save_dir}")
    else:
        print("Training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

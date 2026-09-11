#!/usr/bin/env python3
"""Package SegmentME with PyInstaller for the OS this script runs on.

    python build.py                  # CPU-only build (default)
    python build.py --flavor cuda    # CUDA build (Windows / Linux, NVIDIA GPUs)
    python build.py --skip-env       # reuse the interpreter running this
                                     # script (must already have every runtime
                                     # dependency + requirements-build.txt)
    python build.py --no-archive     # leave the folder in dist/, don't zip it

Steps
  1. Create build_env/<flavor>/ (a venv) with PyTorch of the chosen flavour,
     requirements.txt, requirements-build.txt and the GitHub-only packages
     (SAM2, SAM3, DEXTR) -- the same set install.sh installs.
  2. Run PyInstaller with SegmentME.spec.
  3. Copy SAM2 Tiny -- the only model that ships -- into <output>/models/
     next to the executable, with a README listing the other filenames.
  4. Archive the result as dist/SegmentME-<version>-<os>-<flavor>.<zip|tar.gz>.

PyInstaller does not cross-compile: run this once per OS you want to ship.
Needs Python 3.10-3.12 and an internet connection; git is not required.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from version import __version__  # noqa: E402
from services import model_store  # noqa: E402

SYSTEM = platform.system()  # "Windows", "Linux" or "Darwin"
OS_TAG = {"Windows": "windows", "Linux": "linux", "Darwin": "macos"}[SYSTEM]

TORCH_INDEX = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cuda": "https://download.pytorch.org/whl/cu126",
}
SAM2_ARCHIVE = "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip"
SAM3_ARCHIVE = "https://github.com/facebookresearch/sam3/archive/refs/heads/main.zip"
DEXTR_ARCHIVE = "https://github.com/StevetheGreek97/DEXTR-SMe/archive/refs/heads/master.zip"


def run(cmd, **kwargs):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kwargs)


def app_name(flavor):
    return "SegmentME" + ("-cuda" if flavor == "cuda" else "")


# ----------------------------------------------------------------- step 1
def venv_python(env_dir):
    return env_dir / ("Scripts/python.exe" if SYSTEM == "Windows" else "bin/python")


def create_env(flavor):
    env_dir = ROOT / "build_env" / flavor
    python = venv_python(env_dir)
    if not python.exists():
        run([sys.executable, "-m", "venv", env_dir])
    pip = [python, "-m", "pip", "install", "--upgrade"]

    run(pip + ["pip"])
    torch_cmd = pip + ["torch", "torchvision"]
    if SYSTEM != "Darwin":  # macOS has a single wheel (CPU + Metal)
        torch_cmd += ["--index-url", TORCH_INDEX[flavor]]
    run(torch_cmd)
    run(pip + ["-r", ROOT / "requirements.txt", "-r", ROOT / "requirements-build.txt"])

    # SAM2: never try to compile its optional CUDA extension (needs nvcc).
    run(pip + [SAM2_ARCHIVE], env=dict(os.environ, SAM2_BUILD_CUDA="0"))
    run(pip + [SAM3_ARCHIVE, "einops", "pycocotools"])
    install_dextr(pip)
    return python


def install_dextr(pip):
    """DEXTR-SMe ships without a package __init__.py; add it before installing."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "dextr.zip"
        urllib.request.urlretrieve(DEXTR_ARCHIVE, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        source = next(p for p in Path(tmp).iterdir() if p.is_dir())
        (source / "src" / "DEXTR" / "__init__.py").touch()
        run(pip + [source])


# ----------------------------------------------------------------- step 2
def run_pyinstaller(python, flavor):
    name = app_name(flavor)
    for stale in (ROOT / "dist" / name, ROOT / "dist" / f"{name}.app",
                  ROOT / "dist" / f"{name}-macos", ROOT / "build" / "SegmentME"):
        shutil.rmtree(stale, ignore_errors=True)
    env = dict(os.environ, SEGMENTME_FLAVOR=flavor)
    run([python, "-m", "PyInstaller", "--noconfirm", "--clean", ROOT / "SegmentME.spec"],
        cwd=ROOT, env=env)


def output_dir(flavor):
    """The folder that gets shipped: exe + _internal/ + models/ on Windows and
    Linux; on macOS a folder holding the .app bundle and models/."""
    name = app_name(flavor)
    dist = ROOT / "dist"
    if SYSTEM == "Darwin":
        out = dist / f"{name}-macos"
        out.mkdir()
        shutil.move(str(dist / f"{name}.app"), str(out / f"{name}.app"))
        shutil.rmtree(dist / name, ignore_errors=True)  # bare onedir; the .app is the deliverable
        return out
    return dist / name


# ----------------------------------------------------------------- step 3
def _print_progress(done, total):
    if total:
        print(f"\r  {done * 100 // total:3d}%  {model_store.format_size(done)}", end="", flush=True)


def add_bundled_models(out):
    spec = model_store.MODELS["sam2_tiny"]
    models = out / "models"
    models.mkdir(exist_ok=True)

    source = model_store.find_checkpoint(spec.checkpoint)  # a dev copy, if any
    if source is None or source.stat().st_size != spec.size_bytes:
        source = ROOT / "build_cache" / spec.checkpoint
        if not (source.exists() and source.stat().st_size == spec.size_bytes):
            print(f"Downloading {spec.label} ({spec.size_label})...")
            model_store.download_url(spec.download_url, source, spec.size_bytes,
                                     progress=_print_progress)
            print()
    print(f"Bundling {spec.label} from {source}")
    shutil.copy2(source, models / spec.checkpoint)
    (models / "README.txt").write_text(models_readme(), encoding="utf-8")


def models_readme():
    lines = [
        "SegmentME models folder",
        "========================",
        "",
        "Model checkpoints go in this folder. The easiest way to add one is",
        "Settings -> Models inside the app, which downloads it for you; you can",
        "also copy the files here by hand. The app looks for these exact names:",
        "",
    ]
    for spec in model_store.MODELS.values():
        note = "included" if spec.key == "sam2_tiny" else spec.size_label
        if spec.gated:
            note += f", manual download: {spec.download_url}"
        lines.append(f"  {spec.checkpoint:<28} {spec.label} ({note})")
    lines += [
        "",
        "YOLO weights for Actions -> Run Inference go in a 'yolo' subfolder.",
        "Models trained inside a project are picked up automatically from",
        "<project>/trainings/.",
        "",
        f"To keep models somewhere else, set the {model_store.ENV_MODELS_DIR}",
        "environment variable to that folder.",
        "",
    ]
    return "\n".join(lines)


def add_linux_desktop_script(out):
    """Ship install-desktop.sh next to the executable: it registers the
    applications-menu entry and the .SEproj file type for the current user."""
    target = out / "install-desktop.sh"
    shutil.copy2(ROOT / "install-desktop.sh", target)
    target.chmod(target.stat().st_mode | 0o111)


# ----------------------------------------------------------------- step 4
def archive(out, flavor):
    base = ROOT / "dist" / f"SegmentME-{__version__}-{OS_TAG}-{flavor}"
    if SYSTEM == "Darwin":
        # ditto keeps the .app's symlinks and permissions intact.
        target = base.with_suffix(".zip")
        target.unlink(missing_ok=True)
        run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", out, target])
        return target
    fmt = "zip" if SYSTEM == "Windows" else "gztar"  # tar keeps the exec bit on Linux
    return Path(shutil.make_archive(str(base), fmt, root_dir=out.parent, base_dir=out.name))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flavor", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-env", action="store_true",
                        help="use this interpreter instead of creating build_env/<flavor>")
    parser.add_argument("--no-archive", action="store_true", help="don't zip/tar the result")
    args = parser.parse_args()

    if SYSTEM == "Darwin" and args.flavor == "cuda":
        sys.exit("CUDA builds are not possible on macOS; use --flavor cpu.")

    python = Path(sys.executable) if args.skip_env else create_env(args.flavor)
    run_pyinstaller(python, args.flavor)
    out = output_dir(args.flavor)
    add_bundled_models(out)
    if SYSTEM == "Linux":
        add_linux_desktop_script(out)

    if args.no_archive:
        print(f"\nDone: {out}")
    else:
        print(f"\nDone: {archive(out, args.flavor)}")


if __name__ == "__main__":
    main()

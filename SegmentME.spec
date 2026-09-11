# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for SegmentME (SegmentME).

Build with build.py (recommended, sets up the environment too) or directly
from an environment that has every runtime dependency plus
requirements-build.txt installed:

    pyinstaller --noconfirm --clean SegmentME.spec

SEGMENTME_FLAVOR=cuda names the output SegmentME-cuda; the flavour itself
is decided by which torch wheel is installed in the build environment.

No model checkpoints are bundled here on purpose: build.py copies SAM2 Tiny
into <output>/models/ after the build, and users fetch the rest from
Settings -> Models (see services/model_store.py for the folder layout).
"""
import importlib.util
import os
import sys

from PyInstaller.utils.hooks import (
    collect_data_files, collect_dynamic_libs, collect_submodules,
)

ROOT = os.path.abspath(SPECPATH)
sys.path.insert(0, ROOT)
from version import __version__  # noqa: E402

FLAVOR = os.environ.get("SEGMENTME_FLAVOR", "cpu").lower()
APP_NAME = "SegmentME" + ("-cuda" if FLAVOR == "cuda" else "")


def _installed(package):
    return importlib.util.find_spec(package) is not None


def _modules_on_disk(package, skip=()):
    """Names of every module in `package`, found by walking its folder.

    For packages that cannot be *imported* at build time -- sam3's import
    chain needs triton, absent from CPU builds -- so collect_submodules()
    is unusable. PyInstaller analyzes the listed modules statically without
    executing them; `skip` drops subpackages (training/eval code) the app
    never needs.
    """
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return []
    root = spec.submodule_search_locations[0]
    names = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        prefix = package if rel == "." else package + "." + rel.replace(os.sep, ".")
        if prefix.startswith(skip) or "__pycache__" in prefix:
            dirnames[:] = []
            continue
        for filename in filenames:
            if filename.endswith(".py"):
                names.append(prefix if filename == "__init__.py" else f"{prefix}.{filename[:-3]}")
    return names


datas = [(os.path.join(ROOT, "resources"), "resources")]
binaries = []
hiddenimports = []

# Packages that read their own non-Python files at runtime: ultralytics
# (cfg/default.yaml at import time), sam2 (Hydra YAML configs), sam3 (BPE
# vocab), hydra (its own conf/), qtawesome (icon fonts).
for package in ("ultralytics", "sam2", "sam3", "hydra", "omegaconf", "qtawesome"):
    if _installed(package):
        datas += collect_data_files(package)
# DEXTR's repo keeps a 196 MB checkpoint and demo images inside the package
# tree; an editable install would expose them, and they must never ship
# (the checkpoint is fetched from Settings -> Models instead).
if _installed("DEXTR"):
    datas += collect_data_files(
        "DEXTR", excludes=["**/*.pth", "**/*.jpg", "**/*.png", "**/*.gif", "**/*.sh", "**/*.md"])

# Packages whose modules are imported by name from config strings (Hydra
# `_target_` entries, ultralytics model YAMLs, Hydra core plugins) rather
# than by import statements PyInstaller can see.
for package in ("sam2", "hydra", "DEXTR"):
    if _installed(package):
        hiddenimports += collect_submodules(package)
# sam3 cannot be imported where triton is missing (see core/tools/sam3_loader.py),
# so list its modules from disk instead of collect_submodules().
hiddenimports += _modules_on_disk("sam3", skip=("sam3.train", "sam3.eval", "sam3.agent"))

# ultralytics: skip the subpackages the app never uses whose import-time
# check_requirements() would pip-install extras (trackers -> lap,
# SAM/FastSAM/YOLOE -> clip, solutions/hub -> streamlit etc.) -- collecting
# them would run those installs during analysis and bloat the bundle.
_ULTRALYTICS_SKIP = (
    "ultralytics.trackers", "ultralytics.models.sam", "ultralytics.models.fastsam",
    "ultralytics.models.yoloe", "ultralytics.models.nas", "ultralytics.models.rtdetr",
    "ultralytics.solutions", "ultralytics.hub",
)
if _installed("ultralytics"):
    hiddenimports += collect_submodules(
        "ultralytics", filter=lambda name: not name.startswith(_ULTRALYTICS_SKIP))

# torchvision's compiled ops (_C.so / image.so, loaded with
# torch.ops.load_library rather than imported) are not picked up by the
# standard hooks; without them `import torchvision` fails with
# "operator torchvision::nms does not exist".
if _installed("torchvision"):
    binaries += collect_dynamic_libs(
        "torchvision", search_patterns=["*.so", "*.so.*", "*.pyd", "*.dll", "*.dylib"])
    # ...and the codec libraries image_stable.so links against, which live in
    # a sibling torchvision.libs/ folder (Linux wheels) that no hook covers.
    _tv_libs = os.path.join(
        os.path.dirname(importlib.util.find_spec("torchvision").submodule_search_locations[0]),
        "torchvision.libs")
    if os.path.isdir(_tv_libs):
        binaries += [(os.path.join(_tv_libs, f), "torchvision.libs") for f in os.listdir(_tv_libs)]

# The app's own worker entry points, dispatched from main.py.
hiddenimports += ["core.inference_worker", "core.trainer.training_worker"]

# CUDA torch on Linux loads the CUDA runtime from separate nvidia-* wheels
# (on Windows the DLLs sit inside torch/lib and the torch hook collects them).
if FLAVOR == "cuda" and sys.platform.startswith("linux"):
    for package in ("nvidia", "cusparselt"):
        if _installed(package):
            binaries += collect_dynamic_libs(package, search_patterns=["lib*.so*"])

excludes = [
    "triton",       # torch.compile backend; ~700 MB, never used by the app
    "tkinter", "IPython", "jupyter", "notebook", "pytest",
    "pip",
    # NOT setuptools or wheel: pkg_resources (pulled in by sam3) needs
    # setuptools' vendored packages (jaraco, more_itertools, wheel, ...),
    # else the app dies at startup with "No module named 'jaraco'" in
    # pyi_rth_pkgres, and excluding wheel makes the setuptools hook itself
    # fail ("already imported as ExcludedModule").
    "PyQt5", "PySide2", "PySide6",
]

a = Analysis(
    [os.path.join(ROOT, "main.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if sys.platform == "win32":
    icon = os.path.join(ROOT, "resources", "icons", "icon.ico")
else:
    # PyInstaller converts PNG to the platform format via Pillow.
    icon = os.path.join(ROOT, "resources", "icons", "desktop.png")

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX-compressed torch/Qt DLLs crash on Windows
    console=False,      # windowed app; logs go to the per-user log file
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=icon,
        bundle_identifier="com.segmentme.app",
        info_plist={
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
        },
    )

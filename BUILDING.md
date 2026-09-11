# Building the SegmentME desktop app

The app is packaged with [PyInstaller](https://pyinstaller.org) into a
self-contained folder (Windows and Linux) or an `.app` bundle (macOS). Python
is bundled; users install nothing. PyInstaller does **not** cross-compile, so
each platform is built on that platform.

## Prerequisites

| Platform | Needs |
|----------|-------|
| All      | Python 3.10–3.12 on PATH, internet access (dependencies and the SAM2 Tiny checkpoint are downloaded) |
| Windows  | Nothing else. Build on the oldest Windows you want to support. |
| macOS    | Xcode Command Line Tools (`xcode-select --install`). Build on the oldest macOS you want to support; an Apple-silicon build does not run on Intel Macs and vice versa. |
| Linux    | Build on the oldest distribution you want to support (the bundle links against that glibc). |

`git` is not needed: the GitHub-only packages (SAM2, SAM3, DEXTR) are
installed from archive downloads.

## Build

```bash
python build.py                  # CPU-only build (default)
python build.py --flavor cuda    # CUDA build, Windows/Linux only
```

`build.py` creates `build_env/<flavor>/` (a venv with the right PyTorch
wheel, `requirements.txt`, `requirements-build.txt`, SAM2, SAM3 and DEXTR),
runs PyInstaller with `SegmentME.spec`, copies SAM2 Tiny into the output's
`models/` folder and archives the result:

| Platform | Output |
|----------|--------|
| Windows  | `dist/SegmentME-<version>-windows-<flavor>.zip` → `SegmentME/SegmentME.exe` |
| Linux    | `dist/SegmentME-<version>-linux-<flavor>.tar.gz` → `SegmentME/SegmentME` |
| macOS    | `dist/SegmentME-<version>-macos-cpu.zip` → `SegmentME-macos/SegmentME.app` |

Each archive contains a `models/` folder beside the executable (beside the
`.app` on macOS) holding `sam2_hiera_tiny.pt` and a README. The version
number comes from `version.py`.

The Linux archive also contains `install-desktop.sh`. Running it once adds
SegmentME to the applications menu and registers the `.SEproj` file type
with SegmentME as its handler, so double-clicking a project file opens it
(per user, no root; `--uninstall` reverses it). The same script works in a
source checkout, where `install.sh` runs it automatically.

Useful flags: `--no-archive` leaves the folder in `dist/` without zipping it;
`--skip-env` uses the interpreter running the script instead of creating
`build_env/` (it must already have every dependency plus
`requirements-build.txt` installed).

Measured on Linux: the CPU build is 1.2 GB unpacked without models
(1.4 GB with SAM2 Tiny). Expect 3–4 GB more for a CUDA build.

## Flavours

* **cpu** — installs torch from the CPU wheel index. Works everywhere; on
  macOS this is the only flavour (the default macOS wheel also supports
  Apple Metal).
* **cuda** — installs torch from the CUDA 12.6 wheel index and bundles the
  CUDA runtime, so users with an NVIDIA GPU can train and run SAM on it
  without installing CUDA themselves. The output is named `SegmentME-cuda`.

## How the packaged app differs from `python main.py`

* **Helper processes.** Inference and training run in subprocesses that are
  the app itself started with `--inference-worker` / `--training-worker`
  (`services/app_process.py`). `main.py` dispatches those flags before
  importing anything GUI-related. The old approach of shelling out to the
  `yolo` command cannot work in a bundle, where no console scripts exist.
* **`multiprocessing.freeze_support()`** is the first statement in
  `main.py`; without it the DataLoader workers ultralytics spawns during
  training would start new copies of the GUI on Windows.
* **Models** are looked up through `services/model_store.py`, never through
  `sys._MEIPASS`, so users can add checkpoints to the `models/` folder next
  to the app (or download them from Settings → Models). If that folder is
  not writable, downloads go to the per-user folder (`~/.config/segmentme/models`,
  `~/Library/Application Support/segmentme/models`,
  `%APPDATA%\segmentme\models`). `SEGMENTME_MODELS_DIR` overrides the
  location.
* **Package data** for ultralytics, sam2, sam3, hydra and qtawesome is
  collected explicitly in the spec, and the SAM2 loader falls back to reading
  the YAML config directly if Hydra cannot resolve it inside the bundle.
* **SAM3 without triton.** The `sam3` package imports `triton` (a GPU kernel
  compiler that only exists next to CUDA torch on Linux) at import time,
  through `sam3.model.edt`, which only the video tracker uses. The SAM3
  loader (`core/tools/sam3_loader.py`) imports sam3 lazily and, when triton
  is missing, registers a stand-in for that one module and routes mask NMS
  through the CPU implementation. SAM3 inference on CPU has been verified in
  a triton-less build environment; the CUDA-on-Windows path (no triton
  there either) uses the same NMS fallback but has not been exercised.
  `triton` is excluded from the bundle in every flavour (~700 MB, never
  used).
* **Spec pitfalls already handled:** `setuptools` and `wheel` must not be
  excluded (pkg_resources, pulled in by sam3, needs setuptools' vendored
  packages, and excluding wheel breaks PyInstaller's setuptools hook);
  `collect_submodules("ultralytics")` is filtered because importing the
  trackers / SAM / FastSAM subpackages at analysis time makes ultralytics
  pip-install `lap` and `clip` into whatever Python is first on PATH; sam3's
  modules are listed from disk because the package cannot be imported
  without triton; torchvision's compiled ops (`_C_stable.so`,
  `image_stable.so` since torchvision 0.29) and its `torchvision.libs/`
  codecs are collected explicitly because the stock hook looks for the old
  `_C` name and `import torchvision` fails without them.
* **Worker imports must be literal.** `main.py` imports the worker modules
  with plain `from ... import` statements. PyInstaller only bundles what it
  can see imported, so an `importlib.import_module(name)` dispatch leaves
  the workers out and the packaged app fails with
  `No module named 'core.inference_worker'`.
* The executable is windowed (no console). Logs are written to the per-user
  log folder shown in the user guide's Troubleshooting section.

## Smoke test after building

1. Start the app, open a project, enable **SAM** – it should load SAM2 Tiny
   from `models/`.
2. **Settings → Models → Download** DEXTR (196 MB), then enable **DEXTR**.
3. **Actions → Train Custom Model** on an exported project for 1 epoch – the
   Training Monitor must show ultralytics' log lines, not an error about
   `yolo` not being found.
4. **Actions → Run Inference** with the trained `best.pt`.
5. Linux: run `./install-desktop.sh`, then double-click a `.SEproj` file in
   the file manager – the project must open directly, with the SegmentME
   icon shown on the file.

## Code signing (not automated)

Windows SmartScreen and macOS Gatekeeper warn about unsigned apps. Signing
and notarising need developer certificates and are outside `build.py`; on
macOS an unsigned `.app` can be opened once with right-click → Open.

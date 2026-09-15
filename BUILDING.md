# Building the SegmentME desktop app

The app is packaged with [PyInstaller](https://pyinstaller.org) into a
self-contained folder (Windows and Linux) or an `.app` bundle (macOS). Python
is bundled; users install nothing. PyInstaller does **not** cross-compile, so
each platform is built on that platform.

**Don't have a Mac or a Windows machine?** `.github/workflows/build.yml`
builds all three platforms (and both Mac architectures) on GitHub-hosted
runners. Push a version tag (`v2.0.1`, matching `version.py`) to build
everything and publish it as a GitHub Release, or run it manually from the
Actions tab for a CPU-only test build without publishing anything. Needs
"Read and write permissions" under Settings -> Actions -> General ->
Workflow permissions, set once, for the release step to be able to publish.

## Prerequisites

| Platform | Needs |
|----------|-------|
| All      | Python 3.10–3.12 on PATH, internet access (dependencies and the SAM2 Tiny checkpoint are downloaded) |
| Windows  | Nothing else. Build on the oldest Windows you want to support. |
| macOS    | Xcode Command Line Tools (`xcode-select --install`). Build on the oldest macOS you want to support; an Apple-silicon build does not run on Intel Macs and vice versa. |
| Linux    | Build on the oldest distribution you want to support (the bundle links against that glibc), or use `docker/build.sh`, which builds inside Ubuntu 22.04 – see *Building for older distributions*. |

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

## Debian package (Ubuntu, Debian, Mint)

```bash
python build.py --deb                              # build, tarball and .deb
python build.py --skip-build --deb --no-archive    # wrap an existing dist/ folder
```

Output: `dist/segmentme_<version>_<arch>.deb` (`segmentme-cuda_…` for the CUDA
flavour; the two packages conflict because both install to the same paths).
End users install it with a double-click (App Center, GDebi) or

```bash
sudo apt install ./segmentme_1.0.0_amd64.deb
```

and remove it with `sudo apt remove segmentme`. The package installs:

* `/opt/segmentme/` – the PyInstaller folder, `models/` with SAM2 Tiny
  included. It is root-owned, so models downloaded from Settings → Models go
  to `~/.config/segmentme/models` instead (`services/model_store.py` falls
  back to the per-user folder automatically).
* `/usr/bin/segmentme` – a launcher script.
* `/usr/share/applications/segmentme.desktop`, `/usr/share/mime/packages/segmentme.xml`
  and icons under `/usr/share/icons/hicolor/` – the menu entry, the `.SEproj`
  file type and its icon for every user. dpkg's file triggers refresh the
  desktop, MIME and icon caches, so the package has no maintainer scripts.

Notes:

* `Depends` is computed while packaging: `ldd` over every binary in the
  bundle finds the libraries loaded from the system (about a dozen: glibc,
  libxcb, OpenGL/EGL, Wayland – PyInstaller ships the rest) and `dpkg -S`
  maps them to package names. The `libc6 (>= …)` floor is the highest glibc
  symbol version any bundled binary references (`objdump -T`), which is the
  build host's generation because PyInstaller copies libpython, glib,
  libstdc++ and friends from the host. Built on Ubuntu 24.04 that is 2.38, so
  apt refuses the package on 22.04 – correctly, the tarball would crash there
  with `GLIBC_2.38 not found`. See *Building for older distributions*.
* Set `DEB_MAINTAINER` and `DEB_HOMEPAGE` in `build.py` before publishing.
* Needs `dpkg-deb` (part of dpkg) and Pillow in the build venv (already
  there, torchvision depends on it). The archive is gzip-compressed, so it
  is about the size of the tarball and takes under two minutes.
* Not covered: signing, an apt repository, and lintian cleanliness (no
  changelog, `copyright` is a plain copy of `LICENSE`).

## Building for older distributions

A Linux bundle runs only on distributions at least as new as the machine it
was built on. To support Ubuntu 22.04 (glibc 2.35) and everything newer,
build inside the Ubuntu 22.04 container instead of on the host:

```bash
docker/build.sh --deb          # same arguments as build.py
```

`docker/Dockerfile` installs Python 3.10 plus the runtime libraries
PyInstaller bundles from the host (Qt's xcb dependencies, GTK, fontconfig,
D-Bus, …; if one is missing from the image it is missing from the bundle).
`docker/build.sh` builds that image, then runs `build.py` in it with the
repository mounted, as your own user id, with its venv in
`build_env/ubuntu22.04-<flavor>/` so the second run is quick. Output goes to
`dist/` exactly as for a host build, and `build.py` prints the resulting
glibc floor. The package `Depends` come out with 22.04 names, which also
match Debian 12 and later.

Needs Docker with your user in the `docker` group. The first run downloads
about 1 GB (base image, PyTorch, the other wheels).

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

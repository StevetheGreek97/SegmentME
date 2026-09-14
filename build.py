#!/usr/bin/env python3
"""Package SegmentME with PyInstaller for the OS this script runs on.

    python build.py                  # CPU-only build (default)
    python build.py --flavor cuda    # CUDA build (Windows / Linux, NVIDIA GPUs)
    python build.py --skip-env       # reuse the interpreter running this
                                     # script (must already have every runtime
                                     # dependency + requirements-build.txt)
    python build.py --no-archive     # leave the folder in dist/, don't zip it
    python build.py --deb            # also build a Debian package (Linux)
    python build.py --installer      # also build a Windows setup .exe
    python build.py --skip-build --deb --no-archive
                                     # package the dist/ folder of an earlier
                                     # run as a .deb without rebuilding
    python build.py --skip-build --installer --no-archive
                                     # same, as a Windows setup .exe
    docker/build.sh --deb            # the same, inside an Ubuntu 22.04 container,
                                     # for a bundle that runs on older distros

Steps
  1. Create build_env/<flavor>/ (a venv) with PyTorch of the chosen flavour,
     requirements.txt, requirements-build.txt and the GitHub-only packages
     (SAM2, SAM3, DEXTR) -- the same set install.sh installs.
  2. Run PyInstaller with SegmentME.spec.
  3. Copy SAM2 Tiny -- the only model that ships -- into <output>/models/
     next to the executable, with a README listing the other filenames.
  4. Archive the result as dist/SegmentME-<version>-<os>-<flavor>.<zip|tar.gz>.
  5. With --deb (Linux): wrap the same folder as dist/segmentme_<version>_<arch>.deb
     -- the app in /opt/segmentme, a launcher in /usr/bin, and the menu
     entry, .SEproj file type and icons installed system-wide.

PyInstaller does not cross-compile: run this once per OS you want to ship.
Needs Python 3.10-3.12 and an internet connection; git is not required.
"""
import argparse
import os
import platform
import re
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


def create_env(flavor, env_dir):
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


# ----------------------------------------------------------------- step 5
DEB_MAINTAINER = "StevetheGreek97 <mavrianosstelios@icloud.com>"
DEB_HOMEPAGE = "https://github.com/StevetheGreek97/AquaVision"
# Needed for dpkg's file triggers, which rebuild the desktop, MIME and icon
# caches after installation (so the package needs no maintainer scripts).
DEB_EXTRA_DEPENDS = ("desktop-file-utils", "shared-mime-info", "hicolor-icon-theme")


def existing_output(flavor):
    """The dist/ folder a previous run left behind (for --skip-build)."""
    name = app_name(flavor)
    out = ROOT / "dist" / (f"{name}-macos" if SYSTEM == "Darwin" else name)
    if not out.exists():
        sys.exit(f"{out} not found: run build.py without --skip-build first.")
    return out


def glibc_floor(out, exe):
    """Lowest glibc the bundle runs on: the highest GLIBC_x.y symbol version
    any bundled binary references. PyInstaller copies system libraries such
    as libpython, libglib and libstdc++ from the build host, so this is the
    host distribution's generation, not torch's manylinux baseline -- build
    on the oldest release you want to support. None if objdump is missing."""
    if shutil.which("objdump") is None:
        return None
    pattern = re.compile(r"GLIBC_(\d+)\.(\d+)")
    newest = (0, 0)
    for elf in [out / exe, *(p for p in out.rglob("*.so*") if not p.is_symlink())]:
        dump = subprocess.run(["objdump", "-T", str(elf)], capture_output=True, text=True)
        for major, minor in pattern.findall(dump.stdout):
            newest = max(newest, (int(major), int(minor)))
    return "%d.%d" % newest if newest > (0, 0) else None


def system_depends(out, exe):
    """Debian packages owning the shared libraries the bundle loads from the
    system: whatever ldd resolves to a path outside the bundle (PyInstaller
    ships everything else). Computed on the build host, so the package names
    match the distribution the bundle was linked against."""
    bundled = {p.name for p in out.rglob("*.so*")}
    elfs = [out / exe] + [p for p in out.rglob("*.so*") if not p.is_symlink()]
    paths = set()
    for elf in elfs:
        ldd = subprocess.run(["ldd", str(elf)], capture_output=True, text=True)
        for line in ldd.stdout.splitlines():
            name, arrow, rest = line.strip().partition(" => ")
            path = rest.split(" (")[0].strip()
            if arrow and path.startswith("/") and not path.startswith(str(out)) \
                    and name not in bundled:
                # dpkg registers the file under /lib on releases before the
                # /usr merge (Ubuntu 22.04) and under /usr/lib after it, so
                # ask about both spellings; the miss only prints to stderr.
                paths.update({path, os.path.realpath(path)})
    query = subprocess.run(["dpkg", "-S", *sorted(paths)], capture_output=True, text=True)
    packages = {line.split(":")[0] for line in query.stdout.splitlines() if ": " in line}
    floor = glibc_floor(out, exe)
    if floor is None:  # no objdump: fall back to the build host's glibc
        host = subprocess.run(["dpkg-query", "-W", "-f=${Version}", "libc6"],
                              capture_output=True, text=True).stdout.strip()
        floor = host.split(":")[-1].split("-")[0] or None
    packages.discard("libc6")  # always needed, always versioned
    if floor:
        packages.add(f"libc6 (>= {floor})")
        print(f"glibc floor: {floor} -- the package refuses to install on older "
              "distributions; build on an older one to lower it (see BUILDING.md)")
    else:
        packages.add("libc6")
        print("warning: could not determine the glibc floor; libc6 left unversioned")
    return sorted(packages | set(DEB_EXTRA_DEPENDS))


def write_icons(python, source, hicolor):
    """desktop.png (1024 px) scaled to the usual hicolor sizes, as both the
    application icon and the .SEproj file icon. Runs in the build venv,
    which has Pillow (a torchvision dependency)."""
    script = """
import sys
from pathlib import Path
from PIL import Image
img = Image.open(sys.argv[1]).convert("RGBA")
for size in (48, 64, 128, 256, 512):
    scaled = img.resize((size, size), Image.Resampling.LANCZOS)
    for context, name in (("apps", "segmentme"),
                          ("mimetypes", "application-x-segmentme-project")):
        target = Path(sys.argv[2]) / f"{size}x{size}" / context / f"{name}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        scaled.save(target)
"""
    subprocess.run([str(python), "-c", script, str(source), str(hicolor)], check=True)


def make_deb(out, flavor, python):
    """Debian package around the PyInstaller folder: /opt/segmentme holds the
    app (models/ included), /usr/bin/segmentme launches it, and the desktop
    entry, MIME type and icons go to /usr/share so every user gets the menu
    entry and .SEproj association without running install-desktop.sh."""
    if SYSTEM != "Linux" or shutil.which("dpkg-deb") is None:
        sys.exit("--deb needs dpkg-deb, i.e. a Debian-based Linux build host.")
    package = "segmentme" + ("-cuda" if flavor == "cuda" else "")
    other = "segmentme-cuda" if flavor == "cpu" else "segmentme"  # same paths: never both
    exe = app_name(flavor)
    arch = subprocess.check_output(["dpkg", "--print-architecture"], text=True).strip()

    root = ROOT / "build" / f"deb-{flavor}"
    shutil.rmtree(root, ignore_errors=True)
    opt = root / "opt" / "segmentme"
    print(f"Staging {out} as /opt/segmentme ...", flush=True)
    shutil.copytree(out, opt, symlinks=True, ignore=shutil.ignore_patterns("install-desktop.sh"))

    usr = root / "usr"
    launcher = usr / "bin" / "segmentme"
    launcher.parent.mkdir(parents=True)
    launcher.write_text(f'#!/bin/sh\nexec /opt/segmentme/{exe} "$@"\n', encoding="utf-8")
    launcher.chmod(0o755)

    linux_res = ROOT / "resources" / "linux"
    apps = usr / "share" / "applications"
    apps.mkdir(parents=True)
    template = (linux_res / "segmentme.desktop").read_text(encoding="utf-8")
    (apps / "segmentme.desktop").write_text(
        template.replace("@EXEC@", f"/opt/segmentme/{exe}").replace("@ICON@", "segmentme"),
        encoding="utf-8")
    mime = usr / "share" / "mime" / "packages"
    mime.mkdir(parents=True)
    shutil.copy2(linux_res / "segmentme.xml", mime / "segmentme.xml")
    write_icons(python, ROOT / "resources" / "icons" / "desktop.png", usr / "share" / "icons" / "hicolor")
    doc = usr / "share" / "doc" / package
    doc.mkdir(parents=True)
    shutil.copy2(ROOT / "LICENSE", doc / "copyright")

    # Debian convention (and dpkg's expectation): 0755 directories and
    # executables, 0644 everything else -- not whatever the umask left.
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)

    installed_kb = sum(p.stat().st_size for p in root.rglob("*")
                       if p.is_file() and not p.is_symlink()) // 1024
    depends = system_depends(out, exe)
    flavor_note = ("CPU-only build." if flavor == "cpu"
                   else "CUDA build: bundles the CUDA runtime for NVIDIA GPUs.")
    control = "\n".join([
        f"Package: {package}",
        f"Version: {__version__}",
        "Section: graphics",
        "Priority: optional",
        f"Architecture: {arch}",
        f"Maintainer: {DEB_MAINTAINER}",
        f"Installed-Size: {installed_kb}",
        f"Depends: {', '.join(depends)}",
        f"Conflicts: {other}",
        f"Replaces: {other}",
        f"Homepage: {DEB_HOMEPAGE}",
        "Description: Image segmentation annotation tool",
        " SegmentME annotates images with segmentation masks -- by hand or with",
        " AI assistance (SAM2, SAM3, DEXTR) -- exports YOLO datasets and trains",
        " Ultralytics YOLO segmentation models. Opens .SEproj project files.",
        f" {flavor_note}",
        "",
    ])
    (root / "DEBIAN").mkdir()
    (root / "DEBIAN" / "control").write_text(control, encoding="utf-8")
    print("Depends:", ", ".join(depends))

    deb = ROOT / "dist" / f"{package}_{__version__}_{arch}.deb"
    deb.unlink(missing_ok=True)
    # gzip: every dpkg understands it and it is far quicker than xz on 1.4 GB.
    run(["dpkg-deb", "--build", "--root-owner-group", "-Zgzip", root, deb])
    shutil.rmtree(root, ignore_errors=True)
    return deb


def find_iscc():
    """Locate the Inno Setup 6 command-line compiler (ISCC.exe).

    winget installs it per-user under %LOCALAPPDATA%; other installers put
    it under Program Files (x86) or Program Files.
    """
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Inno Setup 6" / "ISCC.exe",
        Path("C:/Program Files (x86)/Inno Setup 6/ISCC.exe"),
        Path("C:/Program Files/Inno Setup 6/ISCC.exe"),
    ]
    for c in candidates:
        if c.exists():
            return c
    found = shutil.which("iscc") or shutil.which("ISCC")
    if found:
        return Path(found)
    sys.exit(
        "--installer needs the Inno Setup 6 command-line compiler (ISCC.exe), not found.\n"
        "Install it with: winget install JRSoftware.InnoSetup\n"
        "or download it from https://jrsoftware.org/isinfo.php")


def make_installer(out, flavor):
    """A single Windows setup .exe (Inno Setup) wrapping the PyInstaller
    folder: installs to Program Files (or per-user, no admin needed), Start
    Menu + optional Desktop shortcut, .SEproj file association, uninstaller
    entry -- see installer/windows.iss for the details."""
    if SYSTEM != "Windows":
        sys.exit("--installer only applies to Windows builds.")
    iscc = find_iscc()
    run([
        iscc,
        f"/DAppVersion={__version__}",
        f"/DFlavor={flavor}",
        f"/DSourceDir={out}",
        ROOT / "installer" / "windows.iss",
    ])
    return ROOT / "dist" / f"SegmentME-Setup-{__version__}-windows-{flavor}.exe"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--flavor", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--skip-env", action="store_true",
                        help="use this interpreter instead of creating build_env/<flavor>")
    parser.add_argument("--no-archive", action="store_true", help="don't zip/tar the result")
    parser.add_argument("--deb", action="store_true",
                        help="also build a Debian package (Linux hosts with dpkg-deb)")
    parser.add_argument("--installer", action="store_true",
                        help="also build a Windows setup .exe (needs Inno Setup 6's ISCC.exe; "
                             "winget install JRSoftware.InnoSetup)")
    parser.add_argument("--skip-build", action="store_true",
                        help="reuse dist/ from a previous run instead of rebuilding "
                             "(e.g. --skip-build --deb --no-archive to package it)")
    parser.add_argument("--env-dir", type=Path,
                        help="where the build venv lives (default: build_env/<flavor>); "
                             "container builds use their own so they don't clobber the host's")
    args = parser.parse_args()
    env_dir = (args.env_dir or ROOT / "build_env" / args.flavor).resolve()

    if SYSTEM == "Darwin" and args.flavor == "cuda":
        sys.exit("CUDA builds are not possible on macOS; use --flavor cpu.")

    if args.skip_build:
        out = existing_output(args.flavor)
        env_python = venv_python(env_dir)
        python = env_python if env_python.exists() and not args.skip_env else Path(sys.executable)
    else:
        python = Path(sys.executable) if args.skip_env else create_env(args.flavor, env_dir)
        run_pyinstaller(python, args.flavor)
        out = output_dir(args.flavor)
        add_bundled_models(out)
        if SYSTEM == "Linux":
            add_linux_desktop_script(out)

    results = []
    if args.deb:
        results.append(make_deb(out, args.flavor, python))
    if args.installer:
        results.append(make_installer(out, args.flavor))
    results.append(out if args.no_archive else archive(out, args.flavor))
    print("\nDone:" + "".join(f"\n  {r}" for r in results))


if __name__ == "__main__":
    main()

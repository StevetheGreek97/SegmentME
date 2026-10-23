"""
Is there a newer SegmentME, and which file of it fits this computer?

New versions are GitHub Releases of the SegmentME repository (the Build
workflow publishes one per "vX.Y.Z" tag). This module asks GitHub for the
latest one, compares its version with ours, picks the installer or archive
built for this OS, flavour (CPU/CUDA), CPU architecture and install type,
and downloads it: files over 2 GiB are published in ".part-NN" pieces,
which are downloaded one by one and joined, and the result is checked
against the release's SHA256SUMS.

No PyQt6 here (the dialog lives in ui/dialogs/update_dialog.py), so it is
easy to test.
"""
import hashlib
import http.client
import json
import platform
import re
import shutil
import socket
import ssl
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from services import model_store
from services.logger import get_logger
from version import __version__

logger = get_logger(__name__)

REPOSITORY = "StevetheGreek97/SegmentME"
LATEST_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPOSITORY}/releases/latest"
USER_AGENT = f"SegmentME/{__version__} (update check)"

# Install types, i.e. how this copy of the app was installed.
WINDOWS_INSTALLER = "windows-installer"
MACOS_DMG = "macos-dmg"
LINUX_DEB = "linux-deb"
LINUX_TARBALL = "linux-tarball"
SOURCE = "source"           # running `python main.py` from a checkout

DEB_PREFIX = Path("/opt/segmentme")   # where the .deb installs the app (build.py)


class UpdateError(Exception):
    """The check or the download failed; the message is for the user."""


# --------------------------------------------------------------------------
# Versions
# --------------------------------------------------------------------------
def parse_version(text):
    """"v2.3.0" / "2.3.0" -> (2, 3, 0); None when it is not a version."""
    match = re.fullmatch(r"\s*v?(\d+)(?:\.(\d+))?(?:\.(\d+))?\s*", str(text or ""))
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def is_newer(candidate, current=__version__) -> bool:
    """True when version `candidate` comes after `current`."""
    new, old = parse_version(candidate), parse_version(current)
    return new is not None and old is not None and new > old


# --------------------------------------------------------------------------
# This copy of the app
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Installation:
    system: str      # "windows", "macos", "linux"
    arch: str        # "x64", "arm64"
    flavor: str      # "cpu", "cuda"
    kind: str        # one of the install types above


def _arch(machine=None):
    machine = machine or platform.machine()
    return {"AMD64": "x64", "x86_64": "x64", "aarch64": "arm64", "ARM64": "arm64"}.get(
        machine, machine.lower())


def current_installation() -> Installation:
    """Which build this is. The CUDA build's executable is "SegmentME-cuda"
    (see build.py's app_name), the .deb installs into /opt/segmentme."""
    system = {"Windows": "windows", "Darwin": "macos"}.get(platform.system(), "linux")
    exe = Path(sys.executable).resolve()
    flavor = "cuda" if model_store.is_frozen() and "cuda" in exe.stem.lower() else "cpu"
    if not model_store.is_frozen():
        kind = SOURCE
    elif system == "windows":
        kind = WINDOWS_INSTALLER
    elif system == "macos":
        kind = MACOS_DMG
    elif DEB_PREFIX in exe.parents:
        kind = LINUX_DEB
    else:
        kind = LINUX_TARBALL
    return Installation(system, _arch(), flavor, kind)


# --------------------------------------------------------------------------
# The latest release
# --------------------------------------------------------------------------
@dataclass
class Asset:
    name: str
    size: int
    url: str


@dataclass
class Release:
    version: str                 # "2.4.0"
    tag: str                     # "v2.4.0"
    notes: str                   # Markdown
    page: str                    # the release's web page
    assets: list = field(default_factory=list)

    @classmethod
    def from_github(cls, data):
        tag = data.get("tag_name") or ""
        version = parse_version(tag)
        if version is None:
            raise UpdateError(f"The latest release has an unexpected name ({tag!r}).")
        return cls(
            version=".".join(map(str, version)),
            tag=tag,
            notes=data.get("body") or "",
            page=data.get("html_url") or RELEASES_PAGE,
            assets=[Asset(a.get("name", ""), int(a.get("size") or 0),
                          a.get("browser_download_url", ""))
                    for a in data.get("assets", [])],
        )


def ssl_context():
    """HTTPS that also works in the packaged app, whose Python may not find
    the system's certificates (certifi ships with requests/ultralytics)."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def fetch_latest(timeout=15) -> Release:
    """Ask GitHub for the latest published release."""
    request = urllib.request.Request(LATEST_URL, headers={
        "User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
            data = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise UpdateError("No release has been published yet.") from exc
        if exc.code in (403, 429):
            # The API allows 60 questions an hour per network address, which
            # a shared university network can use up. The web pages have no
            # such limit.
            try:
                return fetch_latest_from_web(timeout)
            except Exception as fallback_exc:
                logger.info("Update check without the API failed too: %s", fallback_exc)
            raise UpdateError("GitHub is limiting requests right now; "
                              "please try again in an hour.") from exc
        raise UpdateError(f"GitHub answered HTTP {exc.code} ({exc.reason}).") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise UpdateError(f"Could not reach GitHub ({reason}). "
                          "Check the internet connection.") from exc
    except ValueError as exc:
        raise UpdateError("GitHub sent an answer that could not be read.") from exc
    return Release.from_github(data)


def fetch_latest_from_web(timeout=15) -> Release:
    """The latest release without the API: /releases/latest redirects to the
    release's page (giving the tag), and its file list is a small HTML page.
    File sizes are unknown this way (shown as they download)."""
    context = ssl_context()
    request = urllib.request.Request(RELEASES_PAGE, headers={"User-Agent": USER_AGENT},
                                     method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
        page = response.geturl()
    match = re.search(r"/releases/tag/([^/?#]+)$", page)
    if not match:
        raise UpdateError("No release has been published yet.")
    tag = urllib.parse.unquote(match.group(1))
    html = _read_text(f"https://github.com/{REPOSITORY}/releases/expanded_assets/{tag}",
                      context, timeout)
    prefix = f"/{REPOSITORY}/releases/download/{tag}/"
    names = []
    for href in re.findall(r'href="([^"]+)"', html):
        if href.startswith(prefix):
            name = urllib.parse.unquote(href[len(prefix):])
            if name not in names:
                names.append(name)
    return Release.from_github({
        "tag_name": tag,
        "body": f"[Read what's new on GitHub]({page})",
        "html_url": page,
        "assets": [{"name": n, "size": 0, "browser_download_url": f"https://github.com{prefix}{n}"}
                   for n in names],
    })


def notes_for_app(notes, page=""):
    """The release notes as the update window shows them: without the
    "join the .part-NN files by hand" section every release starts with
    (the updater joins them itself)."""
    kept, skipping = [], False
    for line in (notes or "").splitlines():
        if re.match(r"#+\s*Linux CUDA build", line):
            skipping = True
            continue
        if skipping and (line.startswith("#") or line.startswith("**Full Changelog")):
            skipping = False
        if not skipping:
            kept.append(line)
    text = "\n".join(kept).strip()
    if page and page not in text:
        text += f"\n\n[Release page on GitHub]({page})"
    return text.strip()


# --------------------------------------------------------------------------
# Which file fits this computer
# --------------------------------------------------------------------------
_VERSION = r"\d+(?:\.\d+)*"       # the version in a file name is not trusted (2.3.0 shipped as 2.2.1)


def _pattern(inst: Installation):
    """Regular expression for this build's file name (see build.py)."""
    cuda = inst.flavor == "cuda"
    if inst.kind == WINDOWS_INSTALLER:
        return rf"SegmentME-Setup-{_VERSION}-windows-{inst.flavor}-{inst.arch}\.exe"
    if inst.kind == MACOS_DMG:
        return rf"SegmentME-{_VERSION}-macos-cpu-{inst.arch}\.dmg"
    if inst.kind == LINUX_DEB:
        deb_arch = {"x64": "amd64"}.get(inst.arch, inst.arch)
        return rf"segmentme{'-cuda' if cuda else ''}_{_VERSION}_{deb_arch}\.deb"
    if inst.kind == LINUX_TARBALL:
        return rf"SegmentME-{_VERSION}-linux-{inst.flavor}-{inst.arch}\.tar\.gz"
    return None


@dataclass
class Download:
    """The file to fetch: one asset, or the .part-NN pieces of one file."""
    name: str                   # the joined file's name
    parts: list                 # [Asset, ...] in order
    checksums: Asset = None     # the release's SHA256SUMS, when there is one

    @property
    def size(self) -> int:
        return sum(p.size for p in self.parts)


def pick_download(release: Release, inst: Installation):
    """The Download for this computer, or None (running from source, or no
    file was built for it, e.g. macOS on Intel)."""
    pattern = _pattern(inst)
    if pattern is None:
        return None
    checksums = next((a for a in release.assets if a.name == "SHA256SUMS"), None)
    whole = [a for a in release.assets if re.fullmatch(pattern, a.name)]
    if whole:
        return Download(whole[0].name, [whole[0]], checksums)
    pieces = {}
    for a in release.assets:
        m = re.fullmatch(rf"({pattern})\.part-(\d+)", a.name)
        if m:
            pieces.setdefault(m.group(1), []).append((int(m.group(2)), a))
    if not pieces:
        return None
    name, numbered = sorted(pieces.items())[0]
    return Download(name, [a for _n, a in sorted(numbered)], checksums)


# --------------------------------------------------------------------------
# Downloading
# --------------------------------------------------------------------------
def _expected_sha256(sums_text, name):
    for line in sums_text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            return parts[0].lower()
    return None


def _sha256(path, should_cancel=None, chunk=8 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            if should_cancel is not None and should_cancel():
                raise model_store.DownloadCancelled()
            digest.update(block)
    return digest.hexdigest()


def free_space(folder) -> int:
    """Free bytes on the drive holding `folder` (which may not exist yet)."""
    folder = Path(folder)
    while not folder.exists() and folder != folder.parent:
        folder = folder.parent
    return shutil.disk_usage(folder).free


def space_needed(item: Download, folder) -> int:
    """Bytes the download still needs in `folder`: what is left to fetch,
    plus room for the joined file when it comes in parts. 0 when the
    sizes are unknown (found without the API)."""
    folder = Path(folder)
    if (folder / item.name).exists() or not all(p.size for p in item.parts):
        return 0

    def have(path):
        return path.stat().st_size if path.exists() else 0
    if len(item.parts) == 1:
        return max(0, item.size - have(folder / (item.name + ".part")))
    left = 0
    for part in item.parts:
        dest = folder / part.name
        got = part.size if have(dest) == part.size else have(dest.with_name(dest.name + ".part"))
        left += part.size - got
    return left + item.size


def friendly_error(exc, folder=None) -> str:
    """What went wrong, in words a user can act on."""
    errno = getattr(exc, "errno", None)
    if errno == 28 or getattr(exc, "winerror", None) in (39, 112):
        return ("The disk is full. Free some space and try again; the download "
                "continues where it stopped.")
    if isinstance(exc, PermissionError):
        return f"SegmentME is not allowed to save files in {folder or 'the download folder'}."
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return ("The internet connection stopped responding. Try again later; the "
                "download continues where it stopped.")
    if isinstance(exc, (ConnectionError, http.client.HTTPException, urllib.error.URLError)):
        return ("The internet connection was interrupted. Try again; the download "
                "continues where it stopped.")
    return str(exc) or type(exc).__name__


def download(item: Download, folder, progress=None, should_cancel=None) -> Path:
    """Fetch `item` into `folder` and return the finished file.

    progress(done, total, text) is called while it runs; should_cancel()
    returning True stops it with model_store.DownloadCancelled (partial
    files are kept, so a second try resumes). A file whose SHA256 does not
    match SHA256SUMS is deleted; if it was left over from an earlier try it
    is downloaded once more, otherwise UpdateError is raised. Network and
    disk problems also become UpdateError, with a message for the user.
    """
    folder = Path(folder)
    try:
        folder.mkdir(parents=True, exist_ok=True)
        leftover = (folder / item.name).exists()
        try:
            return _download_once(item, folder, progress, should_cancel)
        except _Damaged:
            if not leftover:
                raise UpdateError(f"{item.name} was damaged while downloading and has been "
                                  "deleted. Please try again.")
            logger.info("Old copy of %s did not match; downloading it again", item.name)
            try:
                return _download_once(item, folder, progress, should_cancel)
            except _Damaged:
                raise UpdateError(f"{item.name} was damaged while downloading and has been "
                                  "deleted. Please try again.")
    except (UpdateError, model_store.DownloadCancelled):
        raise
    except (OSError, http.client.HTTPException) as exc:
        logger.warning("Update download failed: %r", exc)
        raise UpdateError(friendly_error(exc, folder)) from exc


class _Damaged(Exception):
    pass


def _download_once(item, folder, progress, should_cancel):
    target = folder / item.name
    total = item.size
    context = ssl_context()

    if not target.exists():
        finished = 0
        piece_paths = []
        for i, part in enumerate(item.parts):
            label = (f"Downloading {item.name}" if len(item.parts) == 1
                     else f"Downloading {item.name} (part {i + 1} of {len(item.parts)})")
            dest = folder / part.name if len(item.parts) > 1 else target

            def on_progress(done, part_total, base=finished, label=label):
                if progress is not None:   # sizes can be unknown (see fetch_latest_from_web)
                    progress(base + done, total or (base + part_total), label)
            # A piece finished on an earlier try (the app closed while joining)
            done_before = len(item.parts) > 1 and dest.exists() and \
                (not part.size or dest.stat().st_size == part.size)
            if not done_before:
                try:
                    model_store.download_url(part.url, dest, part.size, on_progress,
                                             should_cancel, context=context)
                except RuntimeError as exc:
                    raise UpdateError(str(exc)) from exc
            finished += part.size or dest.stat().st_size
            piece_paths.append(dest)

        total = total or finished
        if len(item.parts) > 1:
            if progress is not None:
                progress(total, total, f"Joining the {len(item.parts)} parts…")
            joined = target.with_name(target.name + ".part")
            with open(joined, "wb") as out:
                for piece in piece_paths:
                    with open(piece, "rb") as fh:
                        while block := fh.read(8 << 20):
                            if should_cancel is not None and should_cancel():
                                raise model_store.DownloadCancelled()
                            out.write(block)
            joined.replace(target)
            for piece in piece_paths:
                piece.unlink(missing_ok=True)

    total = total or target.stat().st_size
    if item.checksums is not None:
        if progress is not None:
            progress(total, total, "Checking the download…")
        expected = _expected_sha256(_read_text(item.checksums.url, context), item.name)
        if expected and _sha256(target, should_cancel) != expected:
            target.unlink(missing_ok=True)
            raise _Damaged()
    return target


def _read_text(url, context, timeout=30):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
            return response.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError) as exc:
        raise UpdateError(f"Could not download the checksum list ({exc}).") from exc

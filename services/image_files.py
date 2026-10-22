"""
Putting image files into a project: a copy, or a link to the original file.

Every feature finds a project's images by listing its images folder (the image
list, inference, export, training, the missing-image check on open). A link
sits in that folder like a normal file, so linked images work everywhere,
while the original stays where it is and no disk space is used.

How a link is made:
  * macOS / Linux: a symbolic link.
  * Windows: a hard link (any user may make one, but only on the same drive),
    else a symbolic link (needs Developer Mode or admin rights).
If no link can be made (other drive on Windows, a FAT/exFAT USB stick,
permissions), the file is copied and the reason is reported.

Removing a link (deleting the project, undoing an import) never touches the
original file.
"""
import errno
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from services.logger import get_logger

logger = get_logger(__name__)

# What the user asked for
COPY = "copy"
LINK = "link"

# How a file ended up in the project
COPIED = "copied"
SYMLINK = "symlink"
HARDLINK = "hardlink"

_COPY_CHUNK = 8 * 1024 * 1024  # copy in pieces so a big scan can be cancelled midway
_WIN_NOT_SAME_DEVICE = 17      # ERROR_NOT_SAME_DEVICE
_WIN_NO_PRIVILEGE = 1314       # ERROR_PRIVILEGE_NOT_HELD (symbolic link without Developer Mode)


class PlacementCancelled(Exception):
    """The user cancelled while images were being added."""


def _why(exc: OSError) -> str:
    """A short, user-facing reason a link could not be made."""
    winerror = getattr(exc, "winerror", None)
    if exc.errno == errno.EXDEV or winerror == _WIN_NOT_SAME_DEVICE:
        return "the original is on another drive"
    if winerror == _WIN_NO_PRIVILEGE or exc.errno in (errno.EPERM, errno.EACCES):
        return "no permission to make links here"
    return exc.strerror or str(exc)


def _link_attempts():
    if sys.platform == "win32":
        return ((HARDLINK, os.link), (SYMLINK, os.symlink))
    return ((SYMLINK, os.symlink), (HARDLINK, os.link))


def _copy(src, dest, cancelled=None, on_bytes=None):
    """Copy src to a new file dest in pieces (cancel checked between pieces);
    a half-written dest is removed again."""
    try:
        with open(src, "rb") as fin, open(dest, "xb") as fout:   # "x": never overwrite
            while True:
                if cancelled is not None and cancelled():
                    raise PlacementCancelled()
                chunk = fin.read(_COPY_CHUNK)
                if not chunk:
                    break
                fout.write(chunk)
                if on_bytes is not None:
                    on_bytes(len(chunk))
        shutil.copymode(src, dest)
    except BaseException:
        if os.path.lexists(dest):
            try:
                os.unlink(dest)
            except OSError:
                pass
        raise


def place_image(src, dest, mode: str = COPY, cancelled: Optional[Callable[[], bool]] = None,
                on_bytes: Optional[Callable[[int], None]] = None) -> Tuple[str, Optional[str]]:
    """Put the image `src` into the project as `dest` (a full file path).

    Returns (how, reason): how is COPIED, SYMLINK or HARDLINK; reason says why
    a requested link became a copy (None otherwise). Raises FileExistsError
    if `dest` exists (also a broken link), OSError if copying fails, and
    PlacementCancelled when `cancelled()` turns true during a copy.
    on_bytes(n) is called as copied bytes are written.
    """
    if mode not in (COPY, LINK):
        raise ValueError(f"Unknown mode {mode!r}")
    dest = os.fspath(dest)
    if os.path.lexists(dest):
        raise FileExistsError(errno.EEXIST, "already exists", dest)

    reason = None
    if mode == LINK:
        # Point at the real file: links to links break when the middle one moves.
        target = os.path.realpath(os.fspath(src))
        reasons = []
        for how, make in _link_attempts():
            try:
                make(target, dest)
                return how, None
            except (OSError, NotImplementedError) as exc:
                reasons.append(_why(exc) if isinstance(exc, OSError) else str(exc))
        # Prefer the first (preferred) method's reason: it is the one that matters.
        reason = reasons[0] if reasons else None
        logger.info("Could not link %s (%s); copying instead", src, "; ".join(reasons))

    _copy(src, dest, cancelled, on_bytes)
    return COPIED, reason


def is_broken_link(path) -> bool:
    """True if `path` is a symbolic link whose original file is gone
    (moved, deleted, or on a drive that is not connected)."""
    path = os.fspath(path)
    return os.path.islink(path) and not os.path.exists(path)


def link_target(path) -> Optional[str]:
    """Where a symbolic link points, or None for a normal file."""
    try:
        return os.readlink(os.fspath(path)) if os.path.islink(os.fspath(path)) else None
    except OSError:
        return None


def missing_image_message(path) -> str:
    """User-facing text for an image that cannot be opened."""
    target = link_target(path)
    if target is not None and is_broken_link(path):
        return (f"The original of this linked image is missing:\n{target}\n\n"
                "It was moved, deleted, or is on a drive that is not connected. "
                "Its annotations are kept.")
    if not os.path.exists(os.fspath(path)):
        return f"This image file is missing:\n{os.fspath(path)}"
    return f"This image could not be read:\n{os.fspath(path)}"


def summarize(placements) -> str:
    """One line about how images were placed, e.g. for an import report.
    `placements` is a list of (how, reason)."""
    linked = sum(1 for how, _ in placements if how in (SYMLINK, HARDLINK))
    fallback = [reason for how, reason in placements if how == COPIED and reason]
    copied = sum(1 for how, _ in placements if how == COPIED)
    parts = []
    if linked:
        parts.append(f"{linked} linked to the original file(s)")
    if copied:
        text = f"{copied} copied into the project"
        if fallback:
            common = max(set(fallback), key=fallback.count)
            text += f" ({len(fallback)} could not be linked: {common})"
        parts.append(text)
    return "Images: " + ", ".join(parts) + "." if parts else ""


# --------------------------------------------------------------------------
# Several images at once (New Project, Import Images)
# --------------------------------------------------------------------------
@dataclass
class BatchResult:
    placed: List[str] = field(default_factory=list)              # dest paths added
    placements: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)             # name already in the project
    failed: List[Tuple[str, str]] = field(default_factory=list)  # (file name, why)


def bytes_to_copy(sources: Sequence, mode: str) -> int:
    """Disk space a batch needs: the files' size when copying, none for links
    (a link that can't be made becomes a copy, which the copy itself then
    reports if the disk fills up)."""
    if mode != COPY:
        return 0
    total = 0
    for src in sources:
        try:
            total += os.path.getsize(src)
        except OSError:
            pass
    return total


def place_images(jobs: Sequence[Tuple[str, str]], mode: str = COPY,
                 cancelled: Optional[Callable[[], bool]] = None,
                 progress: Optional[Callable[[int, int, str], None]] = None,
                 on_bytes: Optional[Callable[[int], None]] = None,
                 stop_on_error: bool = False) -> BatchResult:
    """Place each (src, dest). Names already taken are skipped. A file that
    fails is reported and the rest go on -- unless stop_on_error, or the disk
    is full: then everything added so far is removed and the error raised.
    On cancel, everything added so far is removed and PlacementCancelled
    raised. Originals are never touched. Safe to run in a worker thread.
    """
    result = BatchResult()

    def undo():
        for dest in reversed(result.placed):
            try:
                os.unlink(dest)
            except OSError:
                logger.warning("Could not remove %s again", dest)
        result.placed.clear()

    try:
        for i, (src, dest) in enumerate(jobs):
            name = os.path.basename(dest)
            if progress is not None:
                progress(i, len(jobs), name)
            if cancelled is not None and cancelled():
                raise PlacementCancelled()
            if os.path.lexists(dest):
                result.skipped.append(name)
                continue
            try:
                result.placements.append(place_image(src, dest, mode, cancelled, on_bytes))
                result.placed.append(dest)
            except PlacementCancelled:
                raise
            except OSError as exc:
                if stop_on_error or exc.errno == errno.ENOSPC:
                    raise
                logger.exception("Could not add %s to the project", src)
                result.failed.append((name, exc.strerror or str(exc)))
        if progress is not None:
            progress(len(jobs), len(jobs), "")
    except BaseException as exc:
        logger.info("Adding images stopped (%s); removing the %d already added",
                    "cancelled" if isinstance(exc, PlacementCancelled) else exc, len(result.placed))
        undo()
        raise
    kinds = {}
    for how, _ in result.placements:
        kinds[how] = kinds.get(how, 0) + 1
    logger.info("Added %d image(s) (%s), %d skipped (name taken), %d failed",
                len(result.placed), ", ".join(f"{n} {k}" for k, n in kinds.items()) or "none",
                len(result.skipped), len(result.failed))
    if result.skipped:
        logger.debug("Skipped, name already in the project: %s", ", ".join(result.skipped))
    return result

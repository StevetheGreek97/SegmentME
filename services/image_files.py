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
from typing import Optional, Tuple

from services.logger import get_logger

logger = get_logger(__name__)

# What the user asked for
COPY = "copy"
LINK = "link"

# How a file ended up in the project
COPIED = "copied"
SYMLINK = "symlink"
HARDLINK = "hardlink"

_WIN_NOT_SAME_DEVICE = 17      # ERROR_NOT_SAME_DEVICE
_WIN_NO_PRIVILEGE = 1314       # ERROR_PRIVILEGE_NOT_HELD (symbolic link without Developer Mode)


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


def place_image(src, dest, mode: str = COPY) -> Tuple[str, Optional[str]]:
    """Put the image `src` into the project as `dest` (a full file path).

    Returns (how, reason): how is COPIED, SYMLINK or HARDLINK; reason says why
    a requested link became a copy (None otherwise). Raises FileExistsError
    if `dest` exists (also a broken link), and OSError if copying fails.
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

    shutil.copy(src, dest)
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

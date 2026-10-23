"""
Automatic copies of a project's database (masks and classes), so a wrong
click -- deleting a class with all its masks, a bad import, masks deleted
by mistake -- can be undone with File > Restore Backup….

Copies go to <project>/.segmentme/backups/ as masks-<date>-<time>.db,
made with SQLite's own backup (safe while the app is writing). A copy is
made when the project opens, every 20 minutes while something changed,
and just before risky actions. Only the newest KEEP copies are kept, and
none is made when the disk is nearly full.

No PyQt6 here.
"""
import sqlite3
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.database.connection import connect_read_only
from services.logger import get_logger

logger = get_logger(__name__)

KEEP = 10
FOLDER = "backups"
PREFIX = "masks-"
MIN_FREE_FACTOR = 3        # leave at least 3x the database size free


@dataclass
class Backup:
    path: Path
    made: datetime
    reason: str
    masks: int
    classes: int
    images: int

    @property
    def label(self):
        return f"{self.made:%Y-%m-%d %H:%M}"


def backup_folder(db_path) -> Path:
    return Path(db_path).parent / FOLDER


def _counts(conn):
    def one(sql):
        try:
            return conn.execute(sql).fetchone()[0]
        except sqlite3.Error:
            return 0
    return (one("SELECT COUNT(*) FROM masks"), one("SELECT COUNT(*) FROM classes"),
            one("SELECT COUNT(DISTINCT image_name) FROM masks"))


_one_at_a_time = threading.Lock()     # the timer's copy and a "before …" copy can meet


def make_backup(db_path, reason="automatic"):
    """Copy the database now; returns the Backup, or None when there is
    nothing to copy, the disk is nearly full, or copying failed (a backup
    must never stop the user's work). Safe to call from any thread."""
    with _one_at_a_time:
        return _make_backup(Path(db_path), reason)


def _make_backup(db_path, reason):
    partial = None
    try:
        if not db_path.is_file():
            return None
        size = db_path.stat().st_size
        folder = backup_folder(db_path)
        folder.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(folder).free < MIN_FREE_FACTOR * size + 50_000_000:
            logger.warning("Not backing up %s: the disk is nearly full", db_path)
            return None
        stamp = datetime.now()
        target = folder / f"{PREFIX}{stamp:%Y%m%d-%H%M%S}.db"
        n = 1
        while target.exists():                   # two copies in the same second
            target = folder / f"{PREFIX}{stamp:%Y%m%d-%H%M%S}-{n}.db"
            n += 1
        partial = target.with_name(target.name + f".{uuid.uuid4().hex[:8]}.part")
        started = time.perf_counter()
        source = connect_read_only(db_path)
        try:
            dest = sqlite3.connect(partial)
            try:
                source.backup(dest)
                dest.execute("CREATE TABLE IF NOT EXISTS segmentme_backup (reason TEXT, made TEXT)")
                dest.execute("DELETE FROM segmentme_backup")
                dest.execute("INSERT INTO segmentme_backup VALUES (?, ?)", (reason, stamp.isoformat()))
                dest.commit()
                masks, classes, images = _counts(dest)
            finally:
                dest.close()
        finally:
            source.close()
        partial.replace(target)
        logger.info("Backed up %s (%s, %d masks) in %.2f s", db_path.name, reason, masks,
                    time.perf_counter() - started)
        prune(db_path)
        return Backup(target, stamp, reason, masks, classes, images)
    except Exception:
        logger.exception("Backup of %s failed", db_path)
        try:
            if partial is not None:
                partial.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def list_backups(db_path):
    """The project's backups, newest first."""
    found = []
    for path in sorted(backup_folder(db_path).glob(f"{PREFIX}*.db"), reverse=True):
        try:
            conn = connect_read_only(path)
            try:
                masks, classes, images = _counts(conn)
                try:
                    reason, made = conn.execute(
                        "SELECT reason, made FROM segmentme_backup").fetchone()
                    made = datetime.fromisoformat(made)
                except (sqlite3.Error, TypeError, ValueError):
                    reason, made = "", datetime.fromtimestamp(path.stat().st_mtime)
            finally:
                conn.close()
        except sqlite3.Error:
            logger.warning("Skipping unreadable backup %s", path)
            continue
        found.append(Backup(path, made, reason, masks, classes, images))
    found.sort(key=lambda b: b.made, reverse=True)
    return found


def prune(db_path, keep=KEEP):
    # Half-written copies left by a crash or power cut
    for part in backup_folder(db_path).glob("*.part"):
        try:
            if time.time() - part.stat().st_mtime > 3600:
                part.unlink()
        except OSError:
            pass
    for old in list_backups(db_path)[keep:]:
        try:
            old.path.unlink()
        except OSError:
            logger.warning("Could not remove old backup %s", old.path)


def restore(backup: Backup, live_connection, db_path):
    """Put a backup's masks and classes back into the open database. The
    current state is backed up first ("before restore"), so a restore can
    itself be undone. Runs on the thread that owns `live_connection`."""
    make_backup(db_path, reason="before restore")
    source = connect_read_only(backup.path)
    try:
        source.backup(live_connection)
    finally:
        source.close()
    live_connection.execute("DROP TABLE IF EXISTS segmentme_backup")
    logger.info("Restored %s from %s", Path(db_path).name, backup.path.name)

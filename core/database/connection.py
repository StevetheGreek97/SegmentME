# core/database/connection.py
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from services.logger import get_logger

logger = get_logger(__name__)

SLOW_QUERY_SECONDS = 0.5


def _short_sql(query):
    text = " ".join(str(query).split())
    return text if len(text) <= 300 else text[:300] + " …"


def _short_params(params):
    """Parameters as they can be logged: blobs (mask outlines) by size only."""
    try:
        items = list(params) if not isinstance(params, dict) else list(params.items())
    except TypeError:
        return repr(params)[:200]
    shown = []
    for p in items[:12]:
        if isinstance(p, (bytes, bytearray, memoryview)):
            shown.append(f"<{len(p)} bytes>")
        else:
            shown.append(repr(p)[:80])
    more = f", … {len(items) - 12} more" if len(items) > 12 else ""
    return "(" + ", ".join(shown) + more + ")"


def connect_read_only(db_path):
    """A read-only connection of its own (e.g. for a worker thread). The path
    goes in as a proper file: URI, so spaces, '#', '?' or '%' in folder
    names and Windows drive letters work."""
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)

class DatabaseConnection:
    """
    Handles database connection and query execution.
    Keeps ONE persistent connection so explicit transactions work.
    """

    def __init__(self, db_path="masks.db"):
        self.db_path = db_path
        logger.info("Opening database %s (SQLite %s)", db_path, sqlite3.sqlite_version)
        # isolation_level=None puts sqlite3 in autocommit mode; we control BEGIN/COMMIT/ROLLBACK.
        # A connection belongs to the thread that made it (sqlite3's default
        # check): work on another thread opens its own connection
        # (connect_read_only, or a DatabaseConnection of its own), so it can
        # never interleave with the window's transactions.
        try:
            self.conn = sqlite3.connect(self.db_path, isolation_level=None)
            self.conn.execute("PRAGMA journal_mode=WAL;")
        except sqlite3.Error:
            logger.exception("Could not open database %s", db_path)
            raise
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        try:
            self._initialize_database()
        except sqlite3.Error:
            logger.exception("Could not prepare the tables of %s (damaged file?)", db_path)
            raise

    def _initialize_database(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS masks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_name TEXT NOT NULL,
                mask_data BLOB NOT NULL,
                class_name TEXT NOT NULL,
                surface_area REAL
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS classes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT UNIQUE NOT NULL,
                color TEXT NOT NULL
            )
        """)
        # Helpful indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_masks_image ON masks(image_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_masks_class  ON masks(class_name)")
        # AUTOCOMMIT is active (isolation_level=None), so explicit commit is not needed here.

    # --- Transaction control ---
    def begin(self, immediate: bool = True):
        self._run("execute", "BEGIN IMMEDIATE" if immediate else "BEGIN", ())

    def commit(self):
        self._run("execute", "COMMIT", ())

    def rollback(self):
        # If no transaction is active, SQLite will raise OperationalError; guard it.
        try:
            self.conn.execute("ROLLBACK")
            logger.warning("Database transaction rolled back (%s)", self.db_path)
        except sqlite3.OperationalError:
            pass

    # --- Queries ---
    def _run(self, how, query, params):
        """Run one statement; a failure is logged with its SQL, parameters
        and database, then raised as before; slow statements are noted."""
        started = time.perf_counter()
        cur = self.conn.cursor()
        try:
            if how == "executemany":
                params = list(params)
                cur.executemany(query, params)
                shown = f"[{len(params)} rows]"
            else:
                cur.execute(query, params)
                shown = None
        except sqlite3.Error as exc:
            logger.error("Database error in %s: %s\n  SQL: %s\n  parameters: %s",
                         Path(self.db_path).name, exc, _short_sql(query),
                         f"[{len(params)} rows]" if how == "executemany" else _short_params(params))
            raise
        seconds = time.perf_counter() - started
        if seconds > SLOW_QUERY_SECONDS:
            logger.warning("Slow database statement (%.2f s): %s %s", seconds, _short_sql(query),
                           shown or _short_params(params))
        return cur

    def execute_query(self, query: str, params=()):
        # In autocommit mode, non-transactional writes commit automatically.
        return self._run("execute", query, params).rowcount  # may be -1 for some statements; OK.

    def executemany(self, query: str, seq_of_params):
        return self._run("executemany", query, seq_of_params).rowcount

    def fetch_all(self, query: str, params=()):
        return self._run("execute", query, params).fetchall()

    def fetch_one(self, query: str, params=()):
        return self._run("execute", query, params).fetchone()

    # Optional: context manager for scoped transactions
    @contextmanager
    def transaction(self, immediate: bool = True):
        self.begin(immediate=immediate)
        try:
            yield
        except Exception as exc:
            logger.warning("Transaction on %s failed (%r); rolling back", Path(self.db_path).name, exc)
            self.rollback()
            raise
        else:
            self.commit()

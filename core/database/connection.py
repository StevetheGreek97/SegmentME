# core/database/connection.py
import sqlite3
from contextlib import contextmanager
from pathlib import Path


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
        # isolation_level=None puts sqlite3 in autocommit mode; we control BEGIN/COMMIT/ROLLBACK.
        # A connection belongs to the thread that made it (sqlite3's default
        # check): work on another thread opens its own connection
        # (connect_read_only, or a DatabaseConnection of its own), so it can
        # never interleave with the window's transactions.
        self.conn = sqlite3.connect(self.db_path, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA temp_store=MEMORY;")
        self._initialize_database()

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
        self.conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")

    def commit(self):
        self.conn.execute("COMMIT")

    def rollback(self):
        # If no transaction is active, SQLite will raise OperationalError; guard it.
        try:
            self.conn.execute("ROLLBACK")
        except sqlite3.OperationalError:
            pass

    # --- Queries ---
    def execute_query(self, query: str, params=()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        # In autocommit mode, non-transactional writes commit automatically.
        return cur.rowcount  # may be -1 for some statements; OK.

    def executemany(self, query: str, seq_of_params):
        cur = self.conn.cursor()
        cur.executemany(query, seq_of_params)
        return cur.rowcount

    def fetch_all(self, query: str, params=()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchall()

    def fetch_one(self, query: str, params=()):
        cur = self.conn.cursor()
        cur.execute(query, params)
        return cur.fetchone()

    # Optional: context manager for scoped transactions
    @contextmanager
    def transaction(self, immediate: bool = True):
        self.begin(immediate=immediate)
        try:
            yield
        except Exception:
            self.rollback()
            raise
        else:
            self.commit()

import sqlite3
from pathlib import Path


_MEDIA_FILES_DDL = """
CREATE TABLE IF NOT EXISTS media_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    extension TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    modified_ns INTEGER NOT NULL,
    indexed_at TEXT,
    model_name TEXT,
    index_status TEXT NOT NULL DEFAULT 'pending'
);
"""

_APP_SETTINGS_DDL = """
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class Database:
    """Manages SQLite connection and schema initialization."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    def initialize(self) -> None:
        """Create parent directory and tables if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(_MEDIA_FILES_DDL)
            connection.execute(_APP_SETTINGS_DDL)
            connection.commit()
        finally:
            connection.close()

    def connect(self) -> sqlite3.Connection:
        """Create a new connection with row_factory set."""
        connection = sqlite3.connect(str(self._db_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
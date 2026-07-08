import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from semantic_media_search.domain.models import MediaFile, MediaType
from semantic_media_search.storage.database import Database


class FileRepository:
    """CRUD operations for media_files table."""

    def __init__(self, database: Database) -> None:
        self._database = database

    def save_all(self, files: list[MediaFile]) -> list[MediaFile]:
        """Insert files in a single transaction. Returns files with assigned IDs."""
        connection = self._database.connect()
        try:
            with connection:
                rows = [
                    (
                        str(f.path),
                        f.name,
                        f.extension,
                        f.media_type.value,
                        f.size_bytes,
                        f.modified_ns,
                        None,
                        None,
                        "pending",
                    )
                    for f in files
                ]
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO media_files
                        (path, name, extension, media_type,
                         size_bytes, modified_ns, indexed_at,
                         model_name, index_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

            # Retrieve IDs by paths
            return self._fetch_by_paths(
                connection, [f.path for f in files]
            )
        finally:
            connection.close()

    def find_by_ids(self, file_ids: list[int]) -> list[MediaFile]:
        """Fetch files by their IDs. Order is NOT guaranteed to match input."""
        if not file_ids:
            return []

        connection = self._database.connect()
        try:
            placeholders = ",".join("?" for _ in file_ids)
            rows = connection.execute(
                f"SELECT * FROM media_files WHERE id IN ({placeholders})",
                file_ids,
            ).fetchall()
            return [self._row_to_media_file(r) for r in rows]
        finally:
            connection.close()

    def find_all(self) -> list[MediaFile]:
        """Fetch all indexed files."""
        connection = self._database.connect()
        try:
            rows = connection.execute(
                "SELECT * FROM media_files"
            ).fetchall()
            return [self._row_to_media_file(r) for r in rows]
        finally:
            connection.close()

    def delete_all(self) -> None:
        """Remove all media file records."""
        connection = self._database.connect()
        try:
            with connection:
                connection.execute("DELETE FROM media_files")
        finally:
            connection.close()

    def update_index_status(
        self,
        file_ids: list[int],
        status: str,
        model_name: str | None = None,
    ) -> None:
        """Update index_status and optionally model_name for given IDs."""
        if not file_ids:
            return

        connection = self._database.connect()
        now = datetime.now(timezone.utc).isoformat()
        try:
            placeholders = ",".join("?" for _ in file_ids)
            if model_name:
                connection.execute(
                    f"""
                    UPDATE media_files
                    SET index_status = ?,
                        indexed_at = ?,
                        model_name = ?
                    WHERE id IN ({placeholders})
                    """,
                    [status, now, model_name] + file_ids,
                )
            else:
                connection.execute(
                    f"""
                    UPDATE media_files
                    SET index_status = ?,
                        indexed_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [status, now] + file_ids,
                )
            connection.commit()
        finally:
            connection.close()

    def _fetch_by_paths(
        self,
        connection: sqlite3.Connection,
        paths: list[Path],
    ) -> list[MediaFile]:
        """Fetch files by their paths within an existing connection."""
        path_strs = [str(p) for p in paths]
        placeholders = ",".join("?" for _ in path_strs)
        rows = connection.execute(
            f"SELECT * FROM media_files WHERE path IN ({placeholders})",
            path_strs,
        ).fetchall()
        return [self._row_to_media_file(r) for r in rows]

    @staticmethod
    def _row_to_media_file(row: sqlite3.Row) -> MediaFile:
        return MediaFile(
            id=row["id"],
            path=Path(row["path"]),
            name=row["name"],
            extension=row["extension"],
            media_type=MediaType(row["media_type"]),
            size_bytes=row["size_bytes"],
            modified_ns=row["modified_ns"],
        )
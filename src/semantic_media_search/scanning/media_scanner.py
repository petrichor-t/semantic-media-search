from pathlib import Path

from semantic_media_search.domain.models import MediaFile, MediaType


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


class MediaScanner:

    def scan_images(self, directory: Path) -> list[MediaFile]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {directory}"
            )

        media_files: list[MediaFile] = []

        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                stat = path.stat()
                media_files.append(
                    MediaFile(
                        id=None,
                        path=path.resolve(),
                        name=path.name,
                        extension=path.suffix.lower(),
                        media_type=MediaType.IMAGE,
                        size_bytes=stat.st_size,
                        modified_ns=stat.st_mtime_ns,
                    )
                )
            except (PermissionError, OSError):
                continue

        return media_files
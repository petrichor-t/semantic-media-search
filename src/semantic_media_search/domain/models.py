from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"

@dataclass(frozen=True, slots=True)
class SearchHit:
    path: Path
    score: float


@dataclass(slots=True)
class MediaFile:
    id: int | None
    path: Path
    name: str
    extension: str
    media_type: MediaType
    size_bytes: int
    modified_ns: int


@dataclass(slots=True)
class SearchResult:
    file_id: int
    path: Path
    name: str
    media_type: MediaType
    score: float


@dataclass(slots=True)
class IndexingProgress:
    processed: int
    total: int
    current_file: str
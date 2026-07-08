from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Return the project root directory (parent of src/)."""
    return Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True, slots=True)
class Settings:
    _root: Path = field(default_factory=_project_root, repr=False)
    
    image_model_name: str = (
        "sentence-transformers/clip-ViT-B-32"
    )
    text_model_name: str = (
        "sentence-transformers/"
        "clip-ViT-B-32-multilingual-v1"
    )
    image_batch_size: int = 16
    default_top_k: int = 20
    
    @property
    def database_path(self) -> Path:
        return self._root / "data" / "database" / "semantic_media.db"
    
    @property
    def image_index_path(self) -> Path:
        return self._root / "data" / "indexes" / "images.faiss"

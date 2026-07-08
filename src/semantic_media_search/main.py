import logging
import sys

from PySide6.QtWidgets import QApplication

from semantic_media_search.config.settings import Settings
from semantic_media_search.indexing.vector_index import VectorIndex
from semantic_media_search.ml.image_encoder import ImageEncoder
from semantic_media_search.ml.text_encoder import TextEncoder
from semantic_media_search.scanning.media_scanner import MediaScanner
from semantic_media_search.services.indexing_service import IndexingService
from semantic_media_search.services.search_service import SearchService
from semantic_media_search.storage.database import Database
from semantic_media_search.storage.file_repository import FileRepository
from semantic_media_search.ui.main_window import MainWindow


def build_application() -> MainWindow:
    """Composition root — assemble all dependencies and return the main window."""
    settings = Settings()

    # ---- Storage layer ----
    database = Database(settings.database_path)
    database.initialize()

    file_repository = FileRepository(database)
    file_scanner = MediaScanner()

    # ---- ML models ----
    from sentence_transformers import SentenceTransformer

    image_model = SentenceTransformer(settings.image_model_name)
    text_model = SentenceTransformer(settings.text_model_name)

    image_encoder = ImageEncoder(image_model)
    text_encoder = TextEncoder(text_model)

    # ---- Vector index ----
    indexing_service = IndexingService(
        scanner=file_scanner,
        repository=file_repository,
        image_encoder=image_encoder,
        index_path=settings.image_index_path,
        model_name=settings.image_model_name,
        batch_size=settings.image_batch_size,
    )

    image_index = indexing_service.load_or_create_index()

    # ---- Search service ----
    search_service = SearchService(
        text_encoder=text_encoder,
        image_index=image_index,
        file_repository=file_repository,
    )

    # ---- GUI ----
    return MainWindow(
        indexing_service=indexing_service,
        search_service=search_service,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Semantic Media Search")

    window = build_application()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
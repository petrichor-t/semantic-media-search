from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from semantic_media_search.services.indexing_service import IndexingService


class IndexingWorker(QObject):
    """Background worker for image indexing without blocking the GUI thread."""

    progress = Signal(int, int, str)  # processed, total, current_file
    finished = Signal(int)  # indexed_count
    failed = Signal(str)  # error_message

    def __init__(
        self,
        indexing_service: IndexingService,
        folder_path: Path,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._indexing_service = indexing_service
        self._folder_path = folder_path

    @Slot()
    def run(self) -> None:
        try:
            count = self._indexing_service.rebuild_image_index(
                self._folder_path,
                progress_callback=self._on_progress,
            )
            self.finished.emit(count)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _on_progress(self, progress) -> None:
        self.progress.emit(
            progress.processed,
            progress.total,
            progress.current_file,
        )
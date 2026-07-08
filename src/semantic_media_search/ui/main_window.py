from pathlib import Path

from PySide6.QtCore import QThread, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from semantic_media_search.domain.models import SearchResult
from semantic_media_search.services.indexing_service import IndexingService
from semantic_media_search.services.search_service import SearchService
from semantic_media_search.ui.indexing_worker import IndexingWorker


class MainWindow(QMainWindow):
    """Main application window for Semantic Media Search."""

    def __init__(
        self,
        indexing_service: IndexingService,
        search_service: SearchService,
    ) -> None:
        super().__init__()
        self._indexing_service = indexing_service
        self._search_service = search_service
        self._selected_folder: Path | None = None
        self._indexing_thread: QThread | None = None

        self.setWindowTitle("Semantic Media Search")
        self.setMinimumSize(900, 600)

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(10)

        # ---- Folder selection row ----
        folder_layout = QHBoxLayout()

        self._folder_button = QPushButton("Выбрать папку")
        self._folder_button.clicked.connect(self._on_select_folder)

        self._folder_label = QLabel("Папка не выбрана")
        self._folder_label.setStyleSheet("color: gray;")

        folder_layout.addWidget(self._folder_button)
        folder_layout.addWidget(self._folder_label, 1)
        root_layout.addLayout(folder_layout)

        # ---- Indexing row ----
        index_layout = QHBoxLayout()

        self._index_button = QPushButton("Индексировать")
        self._index_button.setEnabled(False)
        self._index_button.clicked.connect(self._on_index)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)

        self._progress_label = QLabel("")

        index_layout.addWidget(self._index_button)
        index_layout.addWidget(self._progress_bar, 1)
        index_layout.addWidget(self._progress_label)
        root_layout.addLayout(index_layout)

        # ---- Search row ----
        search_layout = QHBoxLayout()

        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Введите запрос, например: фото с морем")
        self._search_input.returnPressed.connect(self._on_search)

        self._search_button = QPushButton("Найти")
        self._search_button.clicked.connect(self._on_search)

        self._status_label = QLabel("")

        search_layout.addWidget(self._search_input, 1)
        search_layout.addWidget(self._search_button)
        root_layout.addLayout(search_layout)

        # ---- Status label ----
        root_layout.addWidget(self._status_label)

        # ---- Results table ----
        self._result_table = QTableWidget(0, 4)
        self._result_table.setHorizontalHeaderLabels(
            ["Имя", "Тип", "Путь", "Score"]
        )
        self._result_table.horizontalHeader().setStretchLastSection(False)
        self._result_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Interactive
        )
        self._result_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self._result_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        self._result_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.ResizeToContents
        )
        self._result_table.setColumnWidth(0, 200)
        self._result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._result_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._result_table.doubleClicked.connect(self._on_open_file)

        root_layout.addWidget(self._result_table, 1)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с изображениями"
        )
        if folder:
            self._selected_folder = Path(folder)
            self._folder_label.setText(str(self._selected_folder))
            self._folder_label.setStyleSheet("color: black;")
            self._index_button.setEnabled(True)
            self._status_label.setText("")

    def _on_index(self) -> None:
        if not self._selected_folder:
            return

        self._index_button.setEnabled(False)
        self._search_button.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)  # indeterminate while scanning
        self._progress_label.setText("Сканирование...")
        self._status_label.setText("Идёт индексация...")

        # Keep references alive as attributes to prevent premature GC
        self._indexing_thread = QThread()
        self._worker = IndexingWorker(
            self._indexing_service, self._selected_folder
        )
        self._worker.moveToThread(self._indexing_thread)

        self._indexing_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_indexing_progress)
        self._worker.finished.connect(self._on_indexing_finished)
        self._worker.failed.connect(self._on_indexing_failed)
        self._worker.finished.connect(self._indexing_thread.quit)
        self._worker.failed.connect(self._indexing_thread.quit)

        # Cleanup only after thread has fully stopped
        self._indexing_thread.finished.connect(self._worker.deleteLater)
        self._indexing_thread.finished.connect(
            self._indexing_thread.deleteLater
        )

        self._indexing_thread.start()

    def _on_indexing_progress(
        self, processed: int, total: int, current_file: str
    ) -> None:
        if self._progress_bar.maximum() != total:
            self._progress_bar.setRange(0, total)
        self._progress_bar.setValue(processed)
        self._progress_label.setText(
            f"{processed}/{total} — {current_file}"
        )

    def _on_indexing_finished(self, count: int) -> None:
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        self._index_button.setEnabled(True)
        self._search_button.setEnabled(True)

        if count == 0:
            self._status_label.setText(
                "В выбранной папке не найдено изображений (.jpg, .png, .webp, .bmp)"
            )
            return

        # Reload fresh index into search service
        new_index = self._indexing_service.load_or_create_index()
        self._search_service.update_index(new_index)
        self._status_label.setText(
            f"Индексация завершена. Проиндексировано файлов: {count}"
        )

    def _on_indexing_failed(self, error_msg: str) -> None:
        self._progress_bar.setVisible(False)
        self._progress_label.setText("")
        self._index_button.setEnabled(True)
        self._search_button.setEnabled(True)
        self._status_label.setText("")
        QMessageBox.critical(self, "Ошибка индексации", error_msg)

    def _on_search(self) -> None:
        query = self._search_input.text().strip()
        if not query:
            return

        try:
            results = self._search_service.search(query)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка поиска", str(exc))
            return

        self._display_results(results)
        self._status_label.setText(
            f"По запросу «{query}» найдено: {len(results)}"
        )

    def _display_results(self, results: list[SearchResult]) -> None:
        self._result_table.setRowCount(0)
        self._result_table.setRowCount(len(results))

        for row, result in enumerate(results):
            self._result_table.setItem(
                row, 0, QTableWidgetItem(result.name)
            )
            self._result_table.setItem(
                row, 1, QTableWidgetItem(result.media_type.value)
            )
            self._result_table.setItem(
                row, 2, QTableWidgetItem(str(result.path))
            )
            score_item = QTableWidgetItem(f"{result.score:.4f}")
            self._result_table.setItem(row, 3, score_item)

    def _on_open_file(self, index) -> None:
        row = index.row()
        path_item = self._result_table.item(row, 2)
        if path_item:
            file_path = Path(path_item.text())
            if file_path.exists():
                QDesktopServices.openUrl(
                    QUrl.fromLocalFile(str(file_path))
                )
            else:
                QMessageBox.warning(
                    self,
                    "Файл не найден",
                    f"Файл не существует:\n{file_path}",
                )
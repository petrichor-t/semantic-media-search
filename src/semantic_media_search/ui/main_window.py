from pathlib import Path

from PySide6.QtCore import QSize, QThread, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QIcon
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
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from semantic_media_search.domain.models import SearchResult
from semantic_media_search.services.indexing_service import IndexingService
from semantic_media_search.services.search_service import SearchService
from semantic_media_search.ui.indexing_worker import IndexingWorker

DARK_STYLE = """
QMainWindow {
    background-color: #1a1a2e;
}
QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QLabel {
    color: #c0c0d0;
    background: transparent;
    border: none;
}
QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #16213e;
    border-radius: 6px;
    padding: 8px 20px;
    font-weight: bold;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #16578a;
}
QPushButton:pressed {
    background-color: #0a2647;
}
QPushButton:disabled {
    background-color: #2a2a3e;
    color: #606070;
}
QLineEdit {
    background-color: #16213e;
    border: 2px solid #0f3460;
    border-radius: 6px;
    padding: 8px 12px;
    color: #e0e0e0;
    font-size: 14px;
}
QLineEdit:focus {
    border-color: #e94560;
}
QProgressBar {
    background-color: #16213e;
    border: 1px solid #0f3460;
    border-radius: 4px;
    text-align: center;
    color: #e0e0e0;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #e94560;
    border-radius: 3px;
}
QTableWidget {
    background-color: #16213e;
    alternate-background-color: #1a1a3e;
    border: 1px solid #0f3460;
    border-radius: 6px;
    gridline-color: #2a2a4e;
    selection-background-color: #e94560;
    selection-color: white;
}
QTableWidget::item {
    padding: 6px;
}
QHeaderView::section {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #1a1a2e;
    padding: 8px;
    font-weight: bold;
}
QTableWidget QTableCornerButton::section {
    background-color: #0f3460;
    border: 1px solid #1a1a2e;
}
QFileDialog {
    background-color: #1a1a2e;
}
QMessageBox {
    background-color: #1a1a2e;
}
QStatusBar {
    background-color: #0f3460;
    color: #808090;
    border-top: 1px solid #16213e;
}
"""


class MainWindow(QMainWindow):
    """Beautiful dark-themed semantic search application."""

    def __init__(
        self,
        indexing_service: IndexingService,
        search_service: SearchService,
        device_info: str = "CPU",
    ) -> None:
        super().__init__()
        self._indexing_service = indexing_service
        self._search_service = search_service
        self._device_info = device_info
        self._selected_folder: Path | None = None
        self._indexing_thread: QThread | None = None
        self._worker: IndexingWorker | None = None

        self.setWindowTitle("Semantic Media Search")
        self.setMinimumSize(1000, 680)
        self.setStyleSheet(DARK_STYLE)

        self._build_ui()
        self._status_bar.showMessage(f"  Device: {device_info}  |  No folder selected")

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        # ---- Title ----
        title = QLabel("Semantic Media Search")
        title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #e94560; padding: 4px 0;"
        )
        root.addWidget(title)

        # ---- Folder row ----
        folder_row = QHBoxLayout()
        self._folder_btn = QPushButton("  Select Folder")
        self._folder_btn.clicked.connect(self._on_select_folder)
        self._folder_lbl = QLabel("No folder selected")
        self._folder_lbl.setStyleSheet("color: #707080; font-style: italic;")
        folder_row.addWidget(self._folder_btn)
        folder_row.addWidget(self._folder_lbl, 1)
        root.addLayout(folder_row)

        # ---- Indexing row ----
        idx_row = QHBoxLayout()
        self._index_btn = QPushButton("  Index")
        self._index_btn.setEnabled(False)
        self._index_btn.clicked.connect(self._on_index)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress_lbl = QLabel("")
        idx_row.addWidget(self._index_btn)
        idx_row.addWidget(self._progress, 1)
        idx_row.addWidget(self._progress_lbl)
        root.addLayout(idx_row)

        # ---- Search row ----
        search_row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search... e.g. beach sunset, mountain landscape")
        self._search_input.returnPressed.connect(self._on_search)
        self._search_btn = QPushButton("  Search")
        self._search_btn.clicked.connect(self._on_search)
        search_row.addWidget(self._search_input, 1)
        search_row.addWidget(self._search_btn)
        root.addLayout(search_row)

        # ---- Status label ----
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #e94560; font-size: 12px; padding: 2px 0;")
        root.addWidget(self._status_lbl)

        # ---- Results table ----
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Name", "Type", "Path", "Score"])
        self._table.setAlternatingRowColors(True)
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.setColumnWidth(0, 200)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.doubleClicked.connect(self._on_open_file)
        self._table.setIconSize(QSize(48, 48))
        root.addWidget(self._table, 1)

        # ---- Status bar ----
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

    # ---- Slots ----

    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self._selected_folder = Path(folder)
            self._folder_lbl.setText(str(self._selected_folder))
            self._folder_lbl.setStyleSheet("color: #e0e0e0; font-style: normal;")
            self._index_btn.setEnabled(True)
            self._status_lbl.setText("")

    def _on_index(self) -> None:
        if not self._selected_folder:
            return
        self._index_btn.setEnabled(False)
        self._search_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._progress_lbl.setText("Scanning...")
        self._status_lbl.setText("Indexing...")

        self._indexing_thread = QThread()
        self._worker = IndexingWorker(self._indexing_service, self._selected_folder)
        self._worker.moveToThread(self._indexing_thread)

        self._indexing_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._indexing_thread.quit)
        self._worker.failed.connect(self._indexing_thread.quit)
        self._indexing_thread.finished.connect(self._worker.deleteLater)
        self._indexing_thread.finished.connect(self._indexing_thread.deleteLater)
        self._indexing_thread.start()

    def _on_progress(self, processed: int, total: int, current_file: str) -> None:
        if self._progress.maximum() != total:
            self._progress.setRange(0, total)
        self._progress.setValue(processed)
        self._progress_lbl.setText(f"{processed}/{total}  {current_file}")

    def _on_finished(self, count: int) -> None:
        self._progress.setVisible(False)
        self._progress_lbl.setText("")
        self._index_btn.setEnabled(True)
        self._search_btn.setEnabled(True)
        if count == 0:
            self._status_lbl.setText("No images found (.jpg, .png, .webp, .bmp)")
            return
        new_index = self._indexing_service.load_or_create_index()
        self._search_service.update_index(new_index)
        self._status_lbl.setText(f"Done — {count} images indexed")

    def _on_failed(self, error_msg: str) -> None:
        self._progress.setVisible(False)
        self._progress_lbl.setText("")
        self._index_btn.setEnabled(True)
        self._search_btn.setEnabled(True)
        self._status_lbl.setText("")
        QMessageBox.critical(self, "Indexing Error", error_msg)

    def _on_search(self) -> None:
        query = self._search_input.text().strip()
        if not query:
            return
        try:
            results = self._search_service.search(query)
        except Exception as exc:
            QMessageBox.critical(self, "Search Error", str(exc))
            return
        self._display_results(results)
        self._status_lbl.setText(f"'{query}' — {len(results)} results")

    def _display_results(self, results: list[SearchResult]) -> None:
        self._table.setRowCount(0)
        self._table.setRowCount(len(results))
        for row, r in enumerate(results):
            name = QTableWidgetItem(r.name)
            name.setIcon(QIcon(str(r.path)))
            self._table.setItem(row, 0, name)
            self._table.setItem(row, 1, QTableWidgetItem(r.media_type.value))
            self._table.setItem(row, 2, QTableWidgetItem(str(r.path)))
            score = QTableWidgetItem(f"{r.score:.4f}")
            if r.score >= 0.7:
                score.setForeground(Qt.green)
            elif r.score >= 0.5:
                score.setForeground(Qt.yellow)
            self._table.setItem(row, 3, score)

    def _on_open_file(self, index) -> None:
        row = index.row()
        item = self._table.item(row, 2)
        if item:
            path = Path(item.text())
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            else:
                QMessageBox.warning(self, "File not found", f"File not found:\n{path}")
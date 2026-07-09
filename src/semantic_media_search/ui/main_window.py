from pathlib import Path

from PySide6.QtCore import QThread, QUrl, Qt
from PySide6.QtGui import QDesktopServices, QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from semantic_media_search.domain.models import SearchResult
from semantic_media_search.services.indexing_service import IndexingService
from semantic_media_search.services.search_service import SearchService
from semantic_media_search.storage.database import Database
from semantic_media_search.ui.indexing_worker import IndexingWorker

STYLE = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    font-size: 13px;
}
QMainWindow {
    background-color: #000000;
}
QWidget {
    background-color: #000000;
    color: #ffffff;
}
QLabel {
    background: transparent;
    border: none;
}
QPushButton {
    background: transparent;
    color: #ffffff;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 7px 18px;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.3px;
}
QPushButton:hover {
    border-color: #ffffff;
}
QPushButton:pressed {
    background: #1a1a1a;
}
QPushButton:disabled {
    color: #444444;
    border-color: #222222;
}
QLineEdit {
    background-color: #0a0a0a;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 8px 14px;
    color: #ffffff;
    font-size: 13px;
    letter-spacing: 0.2px;
}
QLineEdit:focus {
    border-color: #ffffff;
}
QComboBox {
    background-color: #0a0a0a;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 5px 12px;
    color: #ffffff;
    font-size: 12px;
    min-width: 80px;
}
QComboBox:hover {
    border-color: #ffffff;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox::down-arrow {
    image: none;
    border: none;
}
QComboBox QAbstractItemView {
    background-color: #0a0a0a;
    border: 1px solid #333333;
    color: #ffffff;
    selection-background-color: #ffffff;
    selection-color: #000000;
}
QProgressBar {
    background-color: #0a0a0a;
    border: 1px solid #333333;
    border-radius: 3px;
    text-align: center;
    color: #888888;
    height: 16px;
    font-size: 10px;
}
QProgressBar::chunk {
    background-color: #ffffff;
    border-radius: 2px;
}
QSlider::groove:horizontal {
    background: #0a0a0a;
    border: 1px solid #333333;
    border-radius: 3px;
    height: 6px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSpinBox {
    background-color: #0a0a0a;
    border: 1px solid #333333;
    border-radius: 4px;
    padding: 4px 8px;
    color: #ffffff;
    font-size: 12px;
}
QSpinBox:hover {
    border-color: #ffffff;
}
QTableWidget {
    background-color: #000000;
    border: 1px solid #222222;
    border-radius: 4px;
    gridline-color: #111111;
    selection-background-color: #ffffff;
    selection-color: #000000;
}
QTableWidget::item {
    padding: 8px 10px;
}
QHeaderView::section {
    background-color: #000000;
    color: #ffffff;
    border: none;
    border-bottom: 1px solid #333333;
    padding: 10px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
QHeaderView {
    background-color: #000000;
}
QTableWidget QTableCornerButton::section {
    background-color: #000000;
    border-bottom: 1px solid #333333;
}
QScrollBar:vertical {
    background: #000000;
    width: 8px;
    border: none;
}
QScrollBar::handle:vertical {
    background: #333333;
    border-radius: 4px;
    min-height: 40px;
}
QScrollBar::handle:vertical:hover {
    background: #555555;
}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: #000000;
    height: 8px;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #333333;
    border-radius: 4px;
    min-width: 40px;
}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {
    width: 0;
}
QFileDialog {
    background-color: #000000;
}
QMessageBox {
    background-color: #000000;
    color: #ffffff;
}
"""


class MainWindow(QMainWindow):
    """Minimalist black‑theme semantic search with iterative rethink."""

    def __init__(
        self,
        indexing_service: IndexingService,
        search_service: SearchService,
        database: Database,
        device_info: str = "GPU",
    ) -> None:
        super().__init__()
        self._indexing_service = indexing_service
        self._search_service = search_service
        self._database = database
        self._device_info = device_info
        self._selected_folder: Path | None = None
        self._indexing_thread: QThread | None = None
        self._worker: IndexingWorker | None = None
        self._last_query: str = ""

        self.setWindowTitle("Semantic Media Search")
        self.setMinimumSize(960, 680)
        self.setStyleSheet(STYLE)

        self._build_ui()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(20, 20, 20, 20)

        # ---- Top bar ----
        top = QHBoxLayout()
        title = QLabel("Semantic Media Search")
        title.setStyleSheet("font-size: 18px; font-weight: 600; letter-spacing: -0.3px;")
        top.addWidget(title)
        top.addStretch()

        device_lbl = QLabel("Device")
        device_lbl.setStyleSheet("font-size: 11px; color: #888888; text-transform: uppercase;")
        top.addWidget(device_lbl)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["GPU", "CPU"])
        self._device_combo.setCurrentText(self._device_info)
        self._device_combo.currentTextChanged.connect(self._on_device_changed)
        top.addWidget(self._device_combo)
        root.addLayout(top)

        # ---- Folder ----
        folder_row = QHBoxLayout()
        self._folder_btn = QPushButton("Select Folder")
        self._folder_btn.clicked.connect(self._on_select_folder)
        self._folder_lbl = QLabel("No folder selected")
        self._folder_lbl.setStyleSheet("color: #666666; font-size: 12px;")
        folder_row.addWidget(self._folder_btn)
        folder_row.addWidget(self._folder_lbl, 1)
        root.addLayout(folder_row)

        # ---- Indexing ----
        idx_row = QHBoxLayout()
        self._index_btn = QPushButton("Index")
        self._index_btn.setEnabled(False)
        self._index_btn.clicked.connect(self._on_index)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        idx_row.addWidget(self._index_btn)
        idx_row.addWidget(self._progress, 1)
        idx_row.addWidget(self._progress_lbl)
        root.addLayout(idx_row)

        # ---- Search ----
        search_row = QHBoxLayout()
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Search images by description...")
        self._search_input.returnPressed.connect(self._on_search)
        self._search_btn = QPushButton("Search")
        self._search_btn.clicked.connect(self._on_search)
        search_row.addWidget(self._search_input, 1)
        search_row.addWidget(self._search_btn)
        root.addLayout(search_row)

        # ---- Rethink ----
        rethink_row = QHBoxLayout()
        self._rethink_btn = QPushButton("Rethink")
        self._rethink_btn.setEnabled(False)
        self._rethink_btn.clicked.connect(self._on_rethink)

        depth_lbl = QLabel("Depth")
        depth_lbl.setStyleSheet("color: #888888; font-size: 11px; text-transform: uppercase;")
        self._depth_slider = QSlider(Qt.Horizontal)
        self._depth_slider.setRange(20, 500)
        self._depth_slider.setValue(50)
        self._depth_slider.setFixedWidth(120)
        self._depth_val = QLabel("50")
        self._depth_val.setStyleSheet("color: #cccccc; font-size: 11px; min-width: 30px;")
        self._depth_slider.valueChanged.connect(lambda v: self._depth_val.setText(str(v)))

        timeout_lbl = QLabel("Timeout")
        timeout_lbl.setStyleSheet("color: #888888; font-size: 11px; text-transform: uppercase; margin-left: 12px;")
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1, 30)
        self._timeout_spin.setValue(5)
        self._timeout_spin.setSuffix("s")
        self._timeout_spin.setFixedWidth(70)

        rethink_row.addWidget(self._rethink_btn)
        rethink_row.addWidget(depth_lbl)
        rethink_row.addWidget(self._depth_slider)
        rethink_row.addWidget(self._depth_val)
        rethink_row.addWidget(timeout_lbl)
        rethink_row.addWidget(self._timeout_spin)
        rethink_row.addStretch()
        root.addLayout(rethink_row)

        # ---- Status ----
        self._status_lbl = QLabel("")
        self._status_lbl.setStyleSheet("color: #666666; font-size: 11px;")
        root.addWidget(self._status_lbl)

        # ---- Table ----
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Name", "Type", "Path", "Score"])
        hdr = self._table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        self._table.setColumnWidth(0, 220)
        self._table.setColumnWidth(1, 60)
        self._table.setColumnWidth(3, 70)
        self._table.verticalHeader().setVisible(False)
        self._table.setShowGrid(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.doubleClicked.connect(self._on_open_file)
        root.addWidget(self._table, 1)

    # ---- Slots ----

    def _on_device_changed(self, device: str) -> None:
        if device == self._device_info:
            return
        self._device_info = device
        conn = self._database.connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO app_settings (key, value) VALUES ('device', ?)",
                (device.lower(),),
            )
            conn.commit()
        finally:
            conn.close()
        QMessageBox.information(self, "Device Changed",
                                f"Device set to {device}.\nRestart to apply.")

    def _on_select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select folder with images")
        if folder:
            self._selected_folder = Path(folder)
            self._folder_lbl.setText(str(self._selected_folder))
            self._folder_lbl.setStyleSheet("color: #cccccc; font-size: 12px;")
            self._index_btn.setEnabled(True)
            self._status_lbl.setText("")

    def _on_index(self) -> None:
        if not self._selected_folder:
            return
        self._index_btn.setEnabled(False)
        self._search_btn.setEnabled(False)
        self._rethink_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 0)
        self._progress_lbl.setText("Scanning...")
        self._status_lbl.setText("Indexing…")

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
        self._last_query = self._search_input.text().strip()
        if not self._last_query:
            return
        try:
            results = self._search_service.search(self._last_query)
        except Exception as exc:
            QMessageBox.critical(self, "Search Error", str(exc))
            return
        self._display_results(results)
        self._rethink_btn.setEnabled(
            len(results) > 0 and self._search_service.has_reranker
        )
        self._status_lbl.setText(f"{len(results)} results")

    def _on_rethink(self) -> None:
        if not self._last_query:
            return
        depth = self._depth_slider.value()
        timeout = self._timeout_spin.value()
        self._rethink_btn.setEnabled(False)
        self._status_lbl.setText(f"Rethinking with depth {depth}, timeout {timeout}s…")
        try:
            results = self._search_service.rerank(
                self._last_query, depth=depth, timeout_seconds=timeout
            )
        except Exception as exc:
            QMessageBox.critical(self, "Rethink Error", str(exc))
            self._rethink_btn.setEnabled(True)
            return
        self._display_results(results)
        self._rethink_btn.setEnabled(True)
        self._status_lbl.setText(
            f"{len(results)} results — depth {depth}, timeout {timeout}s"
        )

    def _display_results(self, results: list[SearchResult]) -> None:
        self._table.setRowCount(0)
        self._table.setRowCount(len(results))
        for row, r in enumerate(results):
            name_item = QTableWidgetItem(r.name)
            name_item.setIcon(QIcon(str(r.path)))
            self._table.setItem(row, 0, name_item)
            self._table.setItem(row, 1, QTableWidgetItem(r.media_type.value))
            self._table.setItem(row, 2, QTableWidgetItem(str(r.path)))
            score_item = QTableWidgetItem(f"{r.score:.4f}")
            score_item.setTextAlignment(0x0084)
            self._table.setItem(row, 3, score_item)

    def _on_open_file(self, index) -> None:
        row = index.row()
        item = self._table.item(row, 2)
        if item:
            path = Path(item.text())
            if path.exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
            else:
                QMessageBox.warning(self, "File not found", f"File not found:\n{path}")
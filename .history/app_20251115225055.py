import json
import sys
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import db
from gesture_analyzer import process_video


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GestureAnalyzer")
        self.resize(900, 600)

        db.init_db()

        self.selected_video: Optional[Path] = None

        # Widgets
        self.path_label = QLabel("No video selected")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_select = QPushButton("Select Video…")
        self.btn_analyze = QPushButton("Analyze Selected")
        self.btn_catalog = QPushButton("View Catalog")

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.table = QTableWidget()
        self.table.setVisible(False)

        # Layout
        top_row = QHBoxLayout()
        top_row.addWidget(self.btn_select)
        top_row.addWidget(self.btn_analyze)
        top_row.addWidget(self.btn_catalog)
        top_row.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addWidget(self.path_label)
        layout.addWidget(self.output, stretch=2)
        layout.addWidget(self.table, stretch=2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connections
        self.btn_select.clicked.connect(self.select_video)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_catalog.clicked.connect(self.show_catalog)

    def select_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            str(Path.cwd()),
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if file_path:
            self.selected_video = Path(file_path)
            self.path_label.setText(str(self.selected_video))
            self.output.clear()

    def run_analysis(self) -> None:
        if not self.selected_video or not self.selected_video.exists():
            QMessageBox.warning(
                self,
                "No Video",
                "Please select a valid video first.",
            )
            return
        try:
            self.output.setPlainText("Processing… this may take a moment…")
            QApplication.processEvents()

            result = process_video(str(self.selected_video))
            row_id = db.save_analysis(
                str(self.selected_video),
                result.dataframe,
                result.summary,
            )

            pretty = json.dumps(result.summary, indent=2)
            self.output.setPlainText(
                f"Analysis saved (id={row_id}):\n\n{pretty}"
            )
            self.table.setVisible(False)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to analyze video:\n{e}",
            )

    def show_catalog(self) -> None:
        try:
            catalog = db.query_catalog()
            self.table.clear()
            self.table.setRowCount(len(catalog))
            self.table.setColumnCount(len(catalog.columns))
            self.table.setHorizontalHeaderLabels(list(catalog.columns))
            for r in range(len(catalog)):
                for c, col in enumerate(catalog.columns):
                    self.table.setItem(
                        r,
                        c,
                        QTableWidgetItem(str(catalog.iloc[r][col])),
                    )
            self.table.resizeColumnsToContents()
            self.table.setVisible(True)
            self.output.clear()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load catalog:\n{e}",
            )


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

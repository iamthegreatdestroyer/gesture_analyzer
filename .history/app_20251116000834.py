import json
import sys
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
    QDialogButtonBox,
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
        self._cancel = False
        self.catalog_df: Optional[pd.DataFrame] = None
        self._filtered_df: Optional[pd.DataFrame] = None

        # Widgets
        self.path_label = QLabel("No video selected")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        self.btn_select = QPushButton("Select Video…")
        self.btn_analyze = QPushButton("Analyze Selected")
        self.btn_catalog = QPushButton("View Catalog")

        self.progress = QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setValue(0)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)

        self.output = QTextEdit()
        self.output.setReadOnly(True)

        self.table = QTableWidget()
        self.table.setVisible(False)

        # Catalog controls
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by name/date/text…")
        self.btn_export_csv = QPushButton("Export CSV…")
        self.btn_export_json = QPushButton("Export JSON…")
        self.btn_edit_mapping = QPushButton("Edit Mapping…")

        # Layout
        top_row = QHBoxLayout()
        top_row.addWidget(self.btn_select)
        top_row.addWidget(self.btn_analyze)
        top_row.addWidget(self.btn_catalog)
        top_row.addStretch()

        prog_row = QHBoxLayout()
        prog_row.addWidget(self.progress)
        prog_row.addWidget(self.btn_cancel)

        catalog_row = QHBoxLayout()
        catalog_row.addWidget(self.filter_edit)
        catalog_row.addWidget(self.btn_export_csv)
        catalog_row.addWidget(self.btn_export_json)
        catalog_row.addWidget(self.btn_edit_mapping)
        catalog_row.addStretch()

        layout = QVBoxLayout()
        layout.addLayout(top_row)
        layout.addLayout(prog_row)
        layout.addWidget(self.path_label)
        layout.addWidget(self.output, stretch=2)
        layout.addLayout(catalog_row)
        layout.addWidget(self.table, stretch=2)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # Connections
        self.btn_select.clicked.connect(self.select_video)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_catalog.clicked.connect(self.show_catalog)
        self.btn_cancel.clicked.connect(self.cancel_analysis)
        self.filter_edit.textChanged.connect(self.apply_filter)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_export_json.clicked.connect(self.export_json)
        self.btn_edit_mapping.clicked.connect(self.edit_mapping)

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
            self._cancel = False
            self.btn_cancel.setEnabled(True)

            def on_prog(done: int, total: int) -> None:
                self.progress.setMaximum(max(total, 1))
                self.progress.setValue(min(done, total))
                QApplication.processEvents()

            def should_cancel() -> bool:
                return self._cancel

            result = process_video(
                str(self.selected_video),
                on_progress=on_prog,
                should_cancel=should_cancel,
            )
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
        finally:
            self.btn_cancel.setEnabled(False)
            self._cancel = False
            self.progress.setValue(0)

    def show_catalog(self) -> None:
        try:
            self.catalog_df = db.query_catalog()
            self.apply_filter()
            self.table.setVisible(True)
            self.output.clear()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load catalog:\n{e}",
            )

    def render_table(self, df: pd.DataFrame) -> None:
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(list(df.columns))
        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                self.table.setItem(
                    r,
                    c,
                    QTableWidgetItem(str(df.iloc[r][col])),
                )
        self.table.resizeColumnsToContents()

    def apply_filter(self) -> None:
        if self.catalog_df is None:
            return
        text = self.filter_edit.text().strip()
        df = self.catalog_df
        if text:
            name_mask = df["video_name"].astype(str).str.contains(
                text, case=False, na=False
            )
            date_mask = df["analysis_date"].astype(str).str.contains(
                text, case=False, na=False
            )
            patt_mask = df["patterns_json"].astype(str).str.contains(
                text, case=False, na=False
            )
            mask = name_mask | date_mask | patt_mask
            filtered = df[mask].reset_index(drop=True)
        else:
            filtered = df.reset_index(drop=True)
        self._filtered_df = filtered
        self.render_table(filtered)

    def export_csv(self) -> None:
        if self._filtered_df is None or self._filtered_df.empty:
            QMessageBox.information(self, "Export", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save CSV",
            str(Path.cwd() / "catalog.csv"),
            "CSV Files (*.csv)",
        )
        if path:
            self._filtered_df.to_csv(path, index=False)
            QMessageBox.information(self, "Export", f"Saved CSV to\n{path}")

    def export_json(self) -> None:
        if self._filtered_df is None or self._filtered_df.empty:
            QMessageBox.information(self, "Export", "No data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save JSON",
            str(Path.cwd() / "catalog.json"),
            "JSON Files (*.json)",
        )
        if path:
            self._filtered_df.to_json(path, orient="records", indent=2)
            QMessageBox.information(self, "Export", f"Saved JSON to\n{path}")

    def cancel_analysis(self) -> None:
        self._cancel = True

    def edit_mapping(self) -> None:
        try:
            # Lazy import to avoid circular import at module load
            import gesture_analyzer as ga  # noqa: WPS433
            ga._MAPPING_CACHE = None  # reset cache
            mapping = ga.load_gesture_mapping().copy()

            dialog = QDialog(self)
            dialog.setWindowTitle("Edit Gesture Mapping")
            layout = QVBoxLayout(dialog)
            table = QTableWidget(dialog)
            table.setColumnCount(2)
            table.setHorizontalHeaderLabels(["Gesture", "Word"])
            table.setRowCount(len(mapping))
            for row, (gesture, word) in enumerate(mapping.items()):
                table.setItem(row, 0, QTableWidgetItem(gesture))
                item = QTableWidgetItem(word)
                table.setItem(row, 1, item)
            table.resizeColumnsToContents()
            layout.addWidget(table)

            buttons = QDialogButtonBox(
                QDialogButtonBox.Save | QDialogButtonBox.Cancel,
                parent=dialog,
            )
            layout.addWidget(buttons)

            def on_save() -> None:
                new_map = {}
                for r in range(table.rowCount()):
                    g_item = table.item(r, 0)
                    w_item = table.item(r, 1)
                    if g_item and w_item:
                        g = g_item.text().strip()
                        w = w_item.text().strip()
                        if g:
                            new_map[g] = w or g
                path = Path(
                    os.environ.get(
                        "GESTURE_MAPPING_PATH", "gesture_mapping.json"
                    )
                )
                with path.open("w", encoding="utf-8") as f:
                    json.dump(new_map, f, indent=2, ensure_ascii=False)
                ga._MAPPING_CACHE = None
                QMessageBox.information(
                    self,
                    "Mapping Saved",
                    f"Saved mapping to\n{path}",
                )
                dialog.accept()

            def on_cancel() -> None:
                dialog.reject()

            buttons.accepted.connect(on_save)
            buttons.rejected.connect(on_cancel)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to edit mapping:\n{e}",
            )


def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

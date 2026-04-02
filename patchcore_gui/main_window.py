"""
Главное окно приложения контроля качества: тёмная тема, четыре зоны UI, конвейер по таймеру.
"""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from patchcore_gui.utils import (
    anomaly_map_to_bgr_heatmap,
    blend_rgb_with_heat_bgr,
    load_display_rgb_224,
    list_image_paths,
    numpy_rgb_to_qpixmap,
    scaled_pixmap,
)
from patchcore_gui.workers import InferenceWorker, normalize_score, select_device


class ViewMode(IntEnum):
    ORIGINAL = 0
    HEATMAP = 1
    OVERLAY = 2


class ScaledImageLabel(QLabel):
    """QLabel с масштабированием pixmap с сохранением пропорций при изменении размера."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 320)
        self._source: QPixmap = QPixmap()

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        self._apply_scale()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._apply_scale()

    def _apply_scale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        scaled = scaled_pixmap(self._source, max(1, self.width()), max(1, self.height()))
        super().setPixmap(scaled)


class MainWindow(QMainWindow):
    """Основное окно: управление, визуализация, вердикт и журнал."""

    def __init__(self, device_preference: str = "auto") -> None:
        super().__init__()
        self.setWindowTitle("PatchCore — визуальный контроль качества")
        self.resize(1280, 800)

        self._device_pref = device_preference
        self._model_path: str = ""
        self._image_dir: str = ""
        self._image_paths: list[str] = []
        self._conveyor_index: int = 0
        self._processed_count: int = 0
        self._running: bool = False

        self._worker: Optional[InferenceWorker] = None
        self._score_min: float = 0.0
        self._score_max: float = 1.0

        self._last_path: str = ""
        self._last_raw_score: float = 0.0
        self._last_norm_score: float = 0.0
        self._last_elapsed_ms: float = 0.0
        self._last_rgb: Optional[np.ndarray] = None
        self._last_map: Optional[np.ndarray] = None

        self._timer = QTimer(self)
        self._timer.setInterval(1500)  # 1.5 с — эмуляция конвейера
        self._timer.timeout.connect(self._on_timer_tick)

        self._build_ui()
        self._apply_dark_theme()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = self._build_left_panel()
        center = self._build_center_panel()
        right = self._build_right_panel()
        splitter.addWidget(left)
        splitter.addWidget(center)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        outer.addWidget(splitter, stretch=1)

        outer.addWidget(self._build_log_panel())

    def _frame(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setFrameShadow(QFrame.Shadow.Raised)
        lay = QVBoxLayout(frame)
        lab = QLabel(title)
        lab.setProperty("role", "section")
        lay.addWidget(lab)
        return frame, lay

    def _build_left_panel(self) -> QFrame:
        frame, lay = self._frame("Управление")
        self._model_label = QLabel("Модель: не выбрана")
        self._model_label.setWordWrap(True)
        btn_model = QPushButton("Обзор… (.pt)")
        btn_model.clicked.connect(self._choose_model)
        self._folder_label = QLabel("Папка: не выбрана")
        self._folder_label.setWordWrap(True)
        btn_folder = QPushButton("Папка с изображениями")
        btn_folder.clicked.connect(self._choose_folder)

        self._btn_start = QPushButton("▶ СТАРТ")
        self._btn_stop = QPushButton("⏹ СТОП")
        self._btn_start.clicked.connect(self._start_conveyor)
        self._btn_stop.clicked.connect(self._stop_conveyor)
        self._btn_stop.setEnabled(False)
        self._btn_start.setProperty("role", "start")
        self._btn_stop.setProperty("role", "stop")

        lay.addWidget(self._model_label)
        lay.addWidget(btn_model)
        lay.addWidget(self._folder_label)
        lay.addWidget(btn_folder)
        lay.addStretch()
        lay.addWidget(self._btn_start)
        lay.addWidget(self._btn_stop)
        return frame

    def _build_center_panel(self) -> QFrame:
        frame, lay = self._frame("Визуализация")
        self._image_view = ScaledImageLabel()
        lay.addWidget(self._image_view, stretch=1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Режим:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Оригинал", "Тепловая карта", "Наложение (Overlay)"])
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row.addWidget(self._mode_combo)
        row.addStretch()
        lay.addLayout(row)
        return frame

    def _build_right_panel(self) -> QFrame:
        frame, lay = self._frame("Результаты")

        self._verdict_label = QLabel("—")
        self._verdict_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vf = QFont()
        vf.setPointSize(22)
        vf.setBold(True)
        self._verdict_label.setFont(vf)
        self._verdict_label.setMinimumHeight(100)
        lay.addWidget(self._verdict_label)

        self._score_value = QLabel("Score: —")
        self._time_value = QLabel("Время: — мс")
        lay.addWidget(self._score_value)
        lay.addWidget(self._time_value)

        lay.addWidget(QLabel("Порог (нормированный, 0…1):"))
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setMinimum(0)
        self._threshold_slider.setMaximum(100)
        self._threshold_slider.setSingleStep(1)
        self._threshold_slider.setPageStep(5)
        self._threshold_slider.setValue(50)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        lay.addWidget(self._threshold_slider)

        self._threshold_caption = QLabel("Порог: 0.50")
        lay.addWidget(self._threshold_caption)
        lay.addStretch()
        return frame

    def _build_log_panel(self) -> QFrame:
        frame, flay = self._frame("Журнал")
        self._log_table = QTableWidget(0, 4)
        self._log_table.setHorizontalHeaderLabels(["Время", "Имя файла", "Score", "Статус"])
        self._log_table.horizontalHeader().setStretchLastSection(True)
        self._log_table.setAlternatingRowColors(True)
        self._log_table.setEditTriggers(self._log_table.EditTrigger.NoEditTriggers)
        self._log_table.setSelectionBehavior(self._log_table.SelectionBehavior.SelectRows)
        flay.addWidget(self._log_table)
        return frame

    def _apply_dark_theme(self) -> None:
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, QColor(45, 45, 48))
        pal.setColor(QPalette.ColorRole.WindowText, QColor(230, 230, 230))
        pal.setColor(QPalette.ColorRole.Base, QColor(37, 37, 40))
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(50, 50, 54))
        pal.setColor(QPalette.ColorRole.Text, QColor(230, 230, 230))
        pal.setColor(QPalette.ColorRole.Button, QColor(60, 60, 65))
        pal.setColor(QPalette.ColorRole.ButtonText, QColor(240, 240, 240))
        self.setPalette(pal)

        self.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #2d2d30; color: #e4e4e4; }
            QFrame { background-color: #323238; border: 1px solid #3f3f46; border-radius: 6px; }
            QLabel[role="section"] { font-weight: 600; color: #c0c0c0; }
            QPushButton { padding: 8px 14px; border-radius: 4px; border: 1px solid #555; min-height: 24px; }
            QPushButton:hover { background-color: #3c3c44; }
            QPushButton[role="start"] { background-color: #1b6b3a; color: white; font-weight: bold; border: 1px solid #2a8f52; }
            QPushButton[role="start"]:hover { background-color: #218c48; }
            QPushButton[role="stop"] { background-color: #8b2020; color: white; font-weight: bold; border: 1px solid #a82e2e; }
            QPushButton[role="stop"]:hover { background-color: #a22828; }
            QComboBox { padding: 4px 8px; background-color: #3c3c44; border: 1px solid #555; border-radius: 4px; }
            QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; margin: -5px 0; background: #6ba3d6; border-radius: 8px; }
            QTableWidget { gridline-color: #444; background-color: #252526; alternate-background-color: #2a2a2e; }
            QHeaderView::section { background-color: #3c3c44; padding: 4px; border: 1px solid #555; }
            """
        )

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбор модели", "", "PyTorch (*.pt)")
        if path:
            self._model_path = path
            self._model_label.setText(f"Модель:\n{path}")

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка с изображениями")
        if path:
            self._image_dir = path
            self._folder_label.setText(f"Папка:\n{path}")

    def _start_conveyor(self) -> None:
        if self._running:
            return
        if not self._model_path:
            QMessageBox.warning(self, "Нет модели", "Укажите файл весов .pt.")
            return
        if not self._image_dir:
            QMessageBox.warning(self, "Нет папки", "Укажите папку с изображениями.")
            return
        paths = list_image_paths(self._image_dir)
        if not paths:
            QMessageBox.warning(self, "Пусто", "В папке нет поддерживаемых изображений.")
            return
        self._image_paths = paths
        self._conveyor_index = 0
        self._processed_count = 0

        device = select_device(self._device_pref)
        self._worker = InferenceWorker(self._model_path, device)
        self._worker.model_ready.connect(self._on_model_ready)
        self._worker.inference_done.connect(self._on_inference_done)
        self._worker.inference_failed.connect(self._on_inference_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

        self._running = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._timer.start()

    def _stop_conveyor(self) -> None:
        self._timer.stop()
        self._running = False
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if self._worker is not None:
            self._worker.request_stop()
            self._worker.wait(10_000)

    def _on_worker_finished(self) -> None:
        self._worker = None

    def _on_model_ready(self, score_min: float, score_max: float, threshold_raw: float) -> None:
        self._score_min = score_min
        self._score_max = score_max
        nthr = normalize_score(threshold_raw, score_min, score_max)
        v = int(round(nthr * 100))
        self._threshold_slider.blockSignals(True)
        self._threshold_slider.setValue(max(0, min(100, v)))
        self._threshold_slider.blockSignals(False)
        self._threshold_caption.setText(f"Порог: {nthr:.2f} (из модели)")

    def _on_timer_tick(self) -> None:
        if not self._running or self._worker is None:
            return
        if self._conveyor_index >= len(self._image_paths):
            self._timer.stop()
            self._try_finalize_conveyor()
            return
        path = self._image_paths[self._conveyor_index]
        self._conveyor_index += 1
        self._worker.enqueue_path(path)

    def _on_inference_done(
        self,
        path: str,
        raw_score: float,
        norm_score: float,
        anomaly_map: np.ndarray,
        elapsed_ms: float,
    ) -> None:
        self._last_path = path
        self._last_raw_score = raw_score
        self._last_norm_score = norm_score
        self._last_elapsed_ms = elapsed_ms
        self._last_rgb = load_display_rgb_224(path)
        self._last_map = np.asarray(anomaly_map, dtype=np.float32)
        short_name = Path(path).name
        self._score_value.setText(f"Score: {raw_score:.4f} (норм: {norm_score:.2f})")
        self._time_value.setText(f"Время: {elapsed_ms:.0f} мс")
        self._refresh_verdict()
        self._refresh_image_view()

        thr = self._threshold_slider.value() / 100.0
        status = "БРАК" if norm_score > thr else "НОРМА"
        self._append_log_row(short_name, raw_score, status)

        self._processed_count += 1
        self._try_finalize_conveyor()

    def _try_finalize_conveyor(self) -> None:
        """Завершает сессию, когда все кадры выданы в очередь и обработаны воркером."""
        if not self._running or not self._image_paths:
            return
        if self._conveyor_index < len(self._image_paths):
            return
        if self._processed_count < len(self._image_paths):
            return
        self._stop_conveyor()
        QMessageBox.information(self, "Конвейер", "Все изображения обработаны.")

    def _on_inference_failed(self, path: str, message: str) -> None:
        QMessageBox.critical(self, "Ошибка", f"{path}\n{message}")
        self._stop_conveyor()

    def _on_threshold_changed(self, _value: int) -> None:
        t = self._threshold_slider.value() / 100.0
        self._threshold_caption.setText(f"Порог: {t:.2f}")
        self._refresh_verdict()

    def _refresh_verdict(self) -> None:
        if self._last_rgb is None:
            self._verdict_label.setText("—")
            self._verdict_label.setStyleSheet("background-color: #444; color: #aaa; border-radius: 6px;")
            return
        thr = self._threshold_slider.value() / 100.0
        is_defect = self._last_norm_score > thr
        if is_defect:
            self._verdict_label.setText("БРАК")
            self._verdict_label.setStyleSheet(
                "background-color: #8b2020; color: white; border-radius: 6px; padding: 12px;"
            )
        else:
            self._verdict_label.setText("НОРМА")
            self._verdict_label.setStyleSheet(
                "background-color: #1b6b3a; color: white; border-radius: 6px; padding: 12px;"
            )

    def _refresh_image_view(self) -> None:
        if self._last_rgb is None or self._last_map is None:
            return
        mode = ViewMode(self._mode_combo.currentIndex())
        pm = self._compose_view_pixmap(mode)
        self._image_view.set_source_pixmap(pm)

    def _compose_view_pixmap(self, mode: ViewMode) -> QPixmap:
        rgb = self._last_rgb
        m = self._last_map
        if rgb is None or m is None:
            return numpy_rgb_to_qpixmap(np.zeros((224, 224, 3), dtype=np.uint8))
        heat_bgr = anomaly_map_to_bgr_heatmap(m)
        if mode == ViewMode.ORIGINAL:
            out = rgb
        elif mode == ViewMode.HEATMAP:
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
            out = heat_rgb
        else:
            out = blend_rgb_with_heat_bgr(rgb, heat_bgr, alpha=0.45)
        return numpy_rgb_to_qpixmap(out)

    def _on_mode_changed(self, _index: int) -> None:
        self._refresh_image_view()

    def _append_log_row(self, filename: str, score: float, status: str) -> None:
        self._log_table.insertRow(0)
        time_s = datetime.now().strftime("%H:%M:%S")
        items = [
            QTableWidgetItem(time_s),
            QTableWidgetItem(filename),
            QTableWidgetItem(f"{score:.4f}"),
            QTableWidgetItem(status),
        ]
        for col, it in enumerate(items):
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._log_table.setItem(0, col, it)

"""
Главное окно приложения контроля качества: тёмная тема, четыре зоны UI, конвейер по таймеру.
"""

from __future__ import annotations

import sys
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
    QDialogButtonBox,
)

from patchcore_gui.utils import (
    anomaly_map_to_bgr_heatmap,
    blend_rgb_with_heat_bgr,
    load_display_rgb_224,
    list_image_paths,
    numpy_rgb_to_qpixmap,
    scaled_pixmap,
)
from patchcore_gui.history_types import InferenceHistoryEntry
from patchcore_gui.settings_dialog import SettingsDialog, TrainingSettings
from patchcore_gui.workers import InferenceWorker, TrainingWorker, normalize_score, select_device


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
        if pixmap.isNull():
            super().clear()
            return
        self._apply_scale()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._apply_scale()

    def _apply_scale(self) -> None:
        if self._source is None or self._source.isNull():
            return
        scaled = scaled_pixmap(self._source, max(1, self.width()), max(1, self.height()))
        super().setPixmap(scaled)


class _QtLogSignaller(QObject):
    """Вспомогательный объект для потокобезопасной передачи сообщений в GUI через сигнал."""
    message = pyqtSignal(str)


class _StdoutRedirector:
    """
    Перехватывает sys.stdout / sys.stderr и дублирует вывод в QTextEdit,
    при этом сохраняя оригинальный поток (терминал).
    """

    def __init__(self, original, signal: "_QtLogSignaller") -> None:
        self._original = original
        self._signal = signal

    def write(self, text: str) -> None:
        if self._original:
            self._original.write(text)
        stripped = text.rstrip("\n")
        if stripped:
            self._signal.message.emit(stripped)

    def flush(self) -> None:
        if self._original:
            self._original.flush()

    def fileno(self):
        raise OSError("fileno not supported on redirected stream")


class ModelInfoDialog(QDialog):
    """
    Диалог просмотра параметров обученной модели PatchCore, считанных из .pt файла.
    Отображает все метаданные, сохранённые в state dict через PatchCore.save().
    """

    def __init__(self, state: dict, model_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Параметры модели")
        self.setModal(True)
        self.resize(520, 480)
        self._build_ui(state, model_path)

    def _build_ui(self, state: dict, model_path: str) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # Path header
        path_label = QLabel(f"<b>Файл:</b> {model_path}")
        path_label.setWordWrap(True)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(path_label)

        # Parameters table
        table = QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(table.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(table.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)

        rows = ModelInfoDialog._build_rows(state)
        table.setRowCount(len(rows))
        section_font = QFont()
        section_font.setBold(True)
        section_color = QColor("#8fc3f0")

        for i, (param, value) in enumerate(rows):
            p_item = QTableWidgetItem(param)
            v_item = QTableWidgetItem(value)
            is_section = param.startswith("──")
            if is_section:
                p_item.setFont(section_font)
                p_item.setForeground(section_color)
                v_item.setForeground(section_color)
                flags = p_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                p_item.setFlags(flags)
                v_item.setFlags(flags)
            p_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            v_item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            table.setItem(i, 0, p_item)
            table.setItem(i, 1, v_item)

        table.resizeColumnToContents(0)
        root.addWidget(table)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.accept)
        root.addWidget(buttons)

    @staticmethod
    def _build_rows(state: dict) -> list[tuple[str, str]]:
        """Формирует список (параметр, значение) из state dict модели."""
        rows: list[tuple[str, str]] = []

        # Architecture
        rows.append(("── Архитектура ──", ""))
        rows.append(("Backbone", str(state.get("backbone_name", "—"))))
        layers = state.get("layers", ())
        rows.append(("Слои (layers)", ", ".join(layers) if layers else "—"))
        rows.append(("Размер патча (patch_size)", str(state.get("patch_size", "—"))))

        # Coreset / index
        rows.append(("── Coreset и индекс ──", ""))
        coreset_ratio = state.get("coreset_ratio", None)
        rows.append(("Coreset ratio", f"{coreset_ratio * 100:.2f} %" if coreset_ratio is not None else "—"))
        memory_bank = state.get("memory_bank", None)
        if memory_bank is not None:
            rows.append(("Размер M_C (банк памяти)", str(tuple(memory_bank.shape))))
        else:
            rows.append(("Размер M_C (банк памяти)", "—"))
        spatial = state.get("spatial_size", None)
        rows.append(("Карта признаков (spatial_size)", str(spatial) if spatial else "—"))

        # Scoring
        rows.append(("── Скоры и порог ──", ""))
        rows.append(("score_min", f"{float(state['score_min']):.6f}" if "score_min" in state else "—"))
        rows.append(("score_max", f"{float(state['score_max']):.6f}" if "score_max" in state else "—"))
        rows.append(("threshold", f"{float(state['threshold']):.6f}" if "threshold" in state else "—"))

        # Hyperparams
        rows.append(("── Гиперпараметры обучения ──", ""))
        rows.append(("n_reweight_nn", str(state.get("n_reweight_nn", "—"))))
        rows.append(("gaussian_sigma", str(state.get("gaussian_sigma", "—"))))

        return rows


class MainWindow(QMainWindow):
    """Основное окно: управление, визуализация, вердикт и журнал."""

    def __init__(self, device_preference: str = "auto") -> None:
        super().__init__()
        self.setWindowTitle("PatchCore — визуальный контроль качества")
        self.resize(1280, 800)

        self._device_pref = device_preference
        self._model_path: str = ""
        self._model_state: dict | None = None
        self._image_dir: str = ""
        self._image_paths: list[str] = []
        self._conveyor_index: int = 0
        self._processed_count: int = 0
        self._running: bool = False

        self._worker: Optional[InferenceWorker] = None
        self._training_worker: Optional[TrainingWorker] = None
        self._training_progress: Optional[QProgressDialog] = None
        self._training_busy: bool = False
        self._score_min: float = 0.0
        self._score_max: float = 1.0
        self._model_auto_threshold_raw: float = 0.0

        self._history: list[InferenceHistoryEntry] = []
        self._current_history_idx: int = -1

        self._train_image_dir: str = ""
        self._train_save_path: str = ""
        self._training_settings: TrainingSettings = TrainingSettings()

        self._last_path: str = ""
        self._last_raw_score: float = 0.0
        self._last_elapsed_ms: float = 0.0
        self._last_rgb: Optional[np.ndarray] = None
        self._last_map: Optional[np.ndarray] = None

        self._timer = QTimer(self)
        self._timer.setInterval(1500)  # 1.5 с — эмуляция конвейера
        self._timer.timeout.connect(self._on_timer_tick)

        self._build_ui()
        self._apply_dark_theme()
        self._on_role_changed(self._role_combo.currentIndex())
        self._setup_log_capture()

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_role_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = self._build_left_column()
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

    def _build_role_bar(self) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(QLabel("Режим:"))
        self._role_combo = QComboBox()
        self._role_combo.addItems(["Оператор", "Инженер"])
        self._role_combo.currentIndexChanged.connect(self._on_role_changed)
        h.addWidget(self._role_combo)
        h.addStretch()
        return row

    def _build_left_column(self) -> QTabWidget:
        """Один навигатор: у оператора полоска вкладок скрыта, у инженера — Инференс / Обучение."""
        self._left_tabs = QTabWidget()
        self._left_tabs.addTab(self._build_inference_tab(), "Инференс")
        self._left_tabs.addTab(self._build_training_tab(), "Обучение (Fit)")
        self._left_tabs.tabBar().setVisible(False)
        return self._left_tabs

    def _build_inference_tab(self) -> QFrame:
        frame, lay = self._frame("Управление")
        self._model_label = QLabel("Модель: не выбрана")
        self._model_label.setWordWrap(True)
        self._btn_choose_model = QPushButton("Обзор… (.pt)")
        self._btn_choose_model.clicked.connect(self._choose_model)
        self._folder_label = QLabel("Папка: не выбрана")
        self._folder_label.setWordWrap(True)
        self._btn_choose_folder = QPushButton("Папка с изображениями")
        self._btn_choose_folder.clicked.connect(self._choose_folder)

        self._btn_start = QPushButton("▶ СТАРТ")
        self._btn_stop = QPushButton("⏹ СТОП")
        self._btn_start.clicked.connect(self._start_conveyor)
        self._btn_stop.clicked.connect(self._stop_conveyor)
        self._btn_stop.setEnabled(False)
        self._btn_start.setProperty("role", "start")
        self._btn_stop.setProperty("role", "stop")

        self._btn_model_info = QPushButton("ℹ Параметры модели")
        self._btn_model_info.clicked.connect(self._show_model_info)
        self._btn_model_info.setEnabled(False)

        lay.addWidget(self._model_label)
        lay.addWidget(self._btn_choose_model)
        lay.addWidget(self._btn_model_info)
        lay.addWidget(self._folder_label)
        lay.addWidget(self._btn_choose_folder)
        lay.addStretch()
        lay.addWidget(self._btn_start)
        lay.addWidget(self._btn_stop)
        return frame

    def _build_training_tab(self) -> QFrame:
        frame, lay = self._frame("Обучение модели (Fit)")
        note = QLabel(
            "Используйте только изображения нормального класса (без дефектов). "
            "Алгоритм строит банк памяти исключительно из эталонов."
        )
        note.setWordWrap(True)
        note.setProperty("role", "hint")
        lay.addWidget(note)

        self._train_dir_label = QLabel("Папка НОРМЫ: не выбрана")
        self._train_dir_label.setWordWrap(True)
        self._btn_choose_train_dir = QPushButton("Выбрать папку с НОРМОЙ")
        self._btn_choose_train_dir.clicked.connect(self._choose_train_dir)

        self._train_save_label = QLabel("Файл модели: не выбран")
        self._train_save_label.setWordWrap(True)
        self._btn_save_model_as = QPushButton("Сохранить модель как…")
        self._btn_save_model_as.clicked.connect(self._choose_train_save_path)

        self._btn_train = QPushButton("▶ ОБУЧИТЬ")
        self._btn_train.setProperty("role", "train")
        self._btn_train.clicked.connect(self._start_training)
        self._btn_training_settings = QPushButton("⚙ Настройки обучения")
        self._btn_training_settings.clicked.connect(self._open_training_settings)

        lay.addWidget(self._train_dir_label)
        lay.addWidget(self._btn_choose_train_dir)
        lay.addWidget(self._train_save_label)
        lay.addWidget(self._btn_save_model_as)
        lay.addWidget(self._btn_training_settings)
        lay.addStretch()
        lay.addWidget(self._btn_train)
        return frame

    def _build_center_panel(self) -> QFrame:
        frame, lay = self._frame("Визуализация")
        self._image_view = ScaledImageLabel()
        lay.addWidget(self._image_view, stretch=1)

        gal = QHBoxLayout()
        self._btn_hist_prev = QPushButton("⬅️ Предыдущее")
        self._btn_hist_next = QPushButton("Следующее ➡️")
        self._btn_hist_prev.clicked.connect(self._on_history_prev)
        self._btn_hist_next.clicked.connect(self._on_history_next)
        self._gallery_label = QLabel("Нет результатов")
        self._gallery_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gal.addWidget(self._btn_hist_prev)
        gal.addWidget(self._gallery_label, stretch=1)
        gal.addWidget(self._btn_hist_next)
        self._update_gallery_buttons_state()
        lay.addLayout(gal)

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

        self._threshold_auto_check = QCheckBox("Автоматический порог (из модели)")
        self._threshold_auto_check.setChecked(True)
        self._threshold_auto_check.toggled.connect(self._on_auto_threshold_toggled)
        lay.addWidget(self._threshold_auto_check)

        lay.addWidget(QLabel("Ручной порог (шкала score_min … score_max модели):"))
        self._threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self._threshold_slider.setMinimum(0)
        self._threshold_slider.setMaximum(100)
        self._threshold_slider.setSingleStep(1)
        self._threshold_slider.setPageStep(5)
        self._threshold_slider.setValue(50)
        self._threshold_slider.setEnabled(False)
        self._threshold_slider.valueChanged.connect(self._on_threshold_changed)
        lay.addWidget(self._threshold_slider)

        self._threshold_caption = QLabel("Порог: — (выберите модель .pt)")
        lay.addWidget(self._threshold_caption)
        lay.addStretch()
        return frame

    def _build_log_panel(self) -> QFrame:
        frame, flay = self._frame("Журнал / Лог")
        frame.setMaximumHeight(220)

        # --- Tab bar row: tabs on the left, clear button on the right ---
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(6)

        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.setDocumentMode(True)
        top_row.addWidget(self._bottom_tabs, stretch=1)

        self._btn_clear_journal = QPushButton("🗑 Очистить журнал")
        self._btn_clear_journal.setFixedWidth(160)
        self._btn_clear_journal.setProperty("role", "clear")
        self._btn_clear_journal.clicked.connect(self._clear_journal)
        top_row.addWidget(self._btn_clear_journal, alignment=Qt.AlignmentFlag.AlignTop)

        flay.addLayout(top_row)

        # --- Tab 1: Inference journal ---
        journal_widget = QWidget()
        journal_layout = QVBoxLayout(journal_widget)
        journal_layout.setContentsMargins(0, 0, 0, 0)

        self._log_table = QTableWidget(0, 4)
        self._log_table.setHorizontalHeaderLabels(["Время", "Имя файла", "Score", "Статус"])
        self._log_table.horizontalHeader().setStretchLastSection(True)
        self._log_table.setAlternatingRowColors(True)
        self._log_table.setEditTriggers(self._log_table.EditTrigger.NoEditTriggers)
        self._log_table.setSelectionBehavior(self._log_table.SelectionBehavior.SelectRows)
        journal_layout.addWidget(self._log_table)
        self._bottom_tabs.addTab(journal_widget, "📋 Журнал")

        # --- Tab 2: Terminal/stdout log ---
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._log_text.setFont(QFont("Courier New", 9))
        self._log_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a1c; color: #b0d0b0; border: none; }"
        )
        self._bottom_tabs.addTab(self._log_text, "🖥 Лог")

        # Sync clear button visibility with active tab
        self._bottom_tabs.currentChanged.connect(self._on_bottom_tab_changed)
        self._on_bottom_tab_changed(0)

        return frame

    def _on_bottom_tab_changed(self, index: int) -> None:
        self._btn_clear_journal.setVisible(index == 0)

    def _clear_journal(self) -> None:
        self._log_table.setRowCount(0)

    def _setup_log_capture(self) -> None:
        """Перехватывает sys.stdout и sys.stderr — только print() и явный вывод программы."""
        signaller = _QtLogSignaller()
        signaller.message.connect(self._append_log_text)
        sys.stdout = _StdoutRedirector(sys.__stdout__, signaller)
        sys.stderr = _StdoutRedirector(sys.__stderr__, signaller)

    def _append_log_text(self, text: str) -> None:
        self._log_text.append(text)
        # Auto-scroll to bottom
        sb = self._log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

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
            QPushButton[role="train"] { background-color: #1e5a8a; color: white; font-weight: bold; border: 1px solid #2d7ab8; }
            QPushButton[role="train"]:hover { background-color: #256ba5; }
            QComboBox { padding: 4px 8px; background-color: #3c3c44; border: 1px solid #555; border-radius: 4px; }
            QTabWidget::pane { border: 1px solid #3f3f46; border-radius: 6px; background: #323238; }
            QTabBar::tab { background: #3c3c44; padding: 8px 14px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #4a4a52; }
            QCheckBox { spacing: 8px; }
            QLabel[role="hint"] { color: #a0a0a8; font-size: 11px; }
            QSlider::groove:horizontal { height: 6px; background: #444; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; margin: -5px 0; background: #6ba3d6; border-radius: 8px; }
            QTableWidget { gridline-color: #444; background-color: #252526; alternate-background-color: #2a2a2e; }
            QHeaderView::section { background-color: #3c3c44; padding: 4px; border: 1px solid #555; }
            QTextEdit { background-color: #1a1a1c; color: #b0d0b0; border: none; }
            QPushButton[role="clear"] { background-color: #3a2a2a; border: 1px solid #6a3a3a; color: #e08080; }
            QPushButton[role="clear"]:hover { background-color: #4a2e2e; }
            """
        )

    def _on_role_changed(self, index: int) -> None:
        is_engineer = index == 1
        self._left_tabs.tabBar().setVisible(is_engineer)
        if not is_engineer:
            self._left_tabs.setCurrentIndex(0)

    def _current_threshold_raw(self) -> float:
        """
        Активный порог в сырой шкале скоров модели.

        Авто: ``threshold`` из чекпоинта. Ручной: линейная интерполяция между
        ``score_min`` и ``score_max`` по положению слайдера 0…100.
        """
        if self._threshold_auto_check.isChecked():
            return float(self._model_auto_threshold_raw)
        span = self._score_max - self._score_min
        t = self._threshold_slider.value() / 100.0
        return self._score_min + t * span

    def _slider_pos_from_raw_threshold(self, raw_thr: float) -> int:
        """Позиция слайдера 0…100, соответствующая сырому порогу (для отображения)."""
        span = self._score_max - self._score_min
        if span <= 1e-12:
            return 50
        pos = (raw_thr - self._score_min) / span
        return int(round(float(np.clip(pos, 0.0, 1.0)) * 100.0))

    def _sync_threshold_ui_from_metadata(self) -> None:
        """Подпись и положение слайдера после чтения .pt / обучения / model_ready."""
        if self._threshold_auto_check.isChecked():
            self._threshold_slider.blockSignals(True)
            self._threshold_slider.setValue(self._slider_pos_from_raw_threshold(self._model_auto_threshold_raw))
            self._threshold_slider.blockSignals(False)
            self._threshold_caption.setText(
                f"Порог: {self._model_auto_threshold_raw:.4f} (авто, из модели)"
            )
        else:
            self._refresh_threshold_caption_manual()
        self._refresh_verdict()

    def _refresh_threshold_caption_manual(self) -> None:
        cur = self._current_threshold_raw()
        self._threshold_caption.setText(f"Порог: {cur:.4f} (вручную)")

    def _update_gallery_buttons_state(self) -> None:
        n = len(self._history)
        idx = self._current_history_idx
        self._btn_hist_prev.setEnabled(n > 0 and idx > 0)
        self._btn_hist_next.setEnabled(n > 0 and idx < n - 1)

    def _update_gallery_label(self) -> None:
        n = len(self._history)
        if n == 0 or self._current_history_idx < 0:
            self._gallery_label.setText("Нет результатов")
            return
        self._gallery_label.setText(
            f"Изображение {self._current_history_idx + 1} из {n}"
        )

    def _apply_history_index(self) -> None:
        """Подставляет текущую запись истории в поля отображения и перерисовывает UI."""
        if not self._history or self._current_history_idx < 0:
            self._last_path = ""
            self._last_raw_score = 0.0
            self._last_elapsed_ms = 0.0
            self._last_rgb = None
            self._last_map = None
            self._score_value.setText("Score: —")
            self._time_value.setText("Время: — мс")
            self._update_gallery_label()
            self._update_gallery_buttons_state()
            self._refresh_verdict()
            self._image_view.set_source_pixmap(QPixmap())
            return

        e = self._history[self._current_history_idx]
        self._last_path = e.path
        self._last_raw_score = e.raw_score
        self._last_elapsed_ms = e.elapsed_ms
        self._last_rgb = e.rgb
        self._last_map = e.anomaly_map
        norm = normalize_score(e.raw_score, self._score_min, self._score_max)
        self._score_value.setText(
            f"Score: {e.raw_score:.4f} (норм: {norm:.2f})"
        )
        self._time_value.setText(f"Время: {e.elapsed_ms:.0f} мс")
        self._update_gallery_label()
        self._update_gallery_buttons_state()
        self._refresh_verdict()
        self._refresh_image_view()

    def _on_history_prev(self) -> None:
        if self._current_history_idx > 0:
            self._current_history_idx -= 1
            self._apply_history_index()

    def _on_history_next(self) -> None:
        if self._current_history_idx < len(self._history) - 1:
            self._current_history_idx += 1
            self._apply_history_index()

    def _on_auto_threshold_toggled(self, checked: bool) -> None:
        self._threshold_slider.setEnabled(not checked)
        if checked:
            self._sync_threshold_ui_from_metadata()
        else:
            self._refresh_threshold_caption_manual()
        self._refresh_verdict()

    def _set_training_locked(self, locked: bool) -> None:
        """Блокирует UI на время обучения (кроме закрытия окна)."""
        self._training_busy = locked
        if locked:
            self._role_combo.setEnabled(False)
            self._btn_choose_model.setEnabled(False)
            self._btn_choose_folder.setEnabled(False)
            self._btn_choose_train_dir.setEnabled(False)
            self._btn_save_model_as.setEnabled(False)
            self._btn_training_settings.setEnabled(False)
            self._btn_train.setEnabled(False)
            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(False)
            self._mode_combo.setEnabled(False)
            self._threshold_auto_check.setEnabled(False)
            self._threshold_slider.setEnabled(False)
        else:
            self._role_combo.setEnabled(True)
            self._btn_choose_model.setEnabled(True)
            self._btn_choose_folder.setEnabled(True)
            self._btn_choose_train_dir.setEnabled(True)
            self._btn_save_model_as.setEnabled(True)
            self._btn_training_settings.setEnabled(True)
            self._btn_train.setEnabled(True)
            self._mode_combo.setEnabled(True)
            self._threshold_auto_check.setEnabled(True)
            self._threshold_slider.setEnabled(not self._threshold_auto_check.isChecked())
            self._btn_stop.setEnabled(self._running)
            self._btn_start.setEnabled(not self._running)

    def _open_training_progress(self) -> None:
        dlg = QProgressDialog(self)
        dlg.setLabelText("Идёт обучение…")
        dlg.setWindowTitle("Обучение PatchCore")
        dlg.setRange(0, 0)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        dlg.show()
        self._training_progress = dlg

    def _close_training_progress(self) -> None:
        if self._training_progress is not None:
            self._training_progress.close()
            self._training_progress.deleteLater()
            self._training_progress = None

    def _append_training_status_log(self) -> None:
        self._log_table.insertRow(0)
        items = [
            QTableWidgetItem(datetime.now().strftime("%H:%M:%S")),
            QTableWidgetItem("—"),
            QTableWidgetItem("—"),
            QTableWidgetItem("Идёт обучение…"),
        ]
        for col, it in enumerate(items):
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._log_table.setItem(0, col, it)

    def _append_training_success_log(self, threshold: float) -> None:
        self._log_table.insertRow(0)
        items = [
            QTableWidgetItem(datetime.now().strftime("%H:%M:%S")),
            QTableWidgetItem("ОБУЧЕНИЕ ЗАВЕРШЕНО"),
            QTableWidgetItem(f"{threshold:.6f}"),
            QTableWidgetItem("УСПЕХ"),
        ]
        for col, it in enumerate(items):
            it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self._log_table.setItem(0, col, it)

    def _choose_train_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Папка только с нормальными изображениями (без дефектов)",
        )
        if path:
            self._train_image_dir = path
            self._train_dir_label.setText(f"Папка НОРМЫ:\n{path}")

    def _choose_train_save_path(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить обученную модель",
            "",
            "PyTorch (*.pt)",
        )
        if path:
            if not path.lower().endswith(".pt"):
                path += ".pt"
            self._train_save_path = path
            self._train_save_label.setText(f"Файл модели:\n{path}")

    def _start_training(self) -> None:
        if self._training_busy:
            return
        if self._running:
            QMessageBox.warning(
                self,
                "Конвейер активен",
                "Остановите конвейер перед запуском обучения.",
            )
            return
        if not self._train_image_dir:
            QMessageBox.warning(
                self,
                "Нет данных",
                "Выберите папку с изображениями нормального класса.",
            )
            return
        if not self._train_save_path:
            QMessageBox.warning(self, "Нет пути", "Укажите файл для сохранения .pt.")
            return
        if (
            self._training_settings.threshold_mode == "f1_optimal"
            and not self._training_settings.validation_dir
        ):
            QMessageBox.warning(
                self,
                "Validation",
                "Для F1-оптимального порога сначала укажите папку Validation в настройках обучения.",
            )
            return
        train_files = list_image_paths(self._train_image_dir)
        if not train_files:
            QMessageBox.warning(
                self,
                "Пустая папка",
                "В выбранной папке нет поддерживаемых изображений.",
            )
            return

        self._append_training_status_log()
        self._set_training_locked(True)
        self._open_training_progress()

        device = select_device(self._device_pref)
        self._training_worker = TrainingWorker(
            self._train_image_dir,
            self._train_save_path,
            device,
            self._training_settings,
        )
        self._training_worker.training_success.connect(self._on_training_success)
        self._training_worker.training_failed.connect(self._on_training_failed)
        self._training_worker.finished.connect(self._on_training_worker_finished)
        self._training_worker.start()

    def _open_training_settings(self) -> None:
        dlg = SettingsDialog(self._training_settings, self)
        if dlg.exec():
            if dlg.settings is not None:
                self._training_settings = dlg.settings

    def _on_training_success(self, threshold: float, score_min: float, score_max: float) -> None:
        self._close_training_progress()
        self._set_training_locked(False)
        self._model_auto_threshold_raw = float(threshold)
        self._score_min = float(score_min)
        self._score_max = float(score_max)
        self._sync_threshold_ui_from_metadata()
        QMessageBox.information(
            self,
            "Обучение завершено",
            f"Порог (threshold): {threshold:.6f}\n"
            f"score_min: {score_min:.6f}\n"
            f"score_max: {score_max:.6f}\n\n"
            f"Файл: {self._train_save_path}",
        )
        self._append_training_success_log(threshold)

    def _on_training_failed(self, message: str) -> None:
        self._close_training_progress()
        self._set_training_locked(False)
        QMessageBox.critical(self, "Ошибка обучения", message)

    def _on_training_worker_finished(self) -> None:
        self._training_worker = None

    def _choose_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Выбор модели", "", "PyTorch (*.pt)")
        if not path:
            return
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("Ожидался словарь состояния PatchCore.")
            self._score_min = float(state.get("score_min", 0.0))
            self._score_max = float(state.get("score_max", 1.0))
            self._model_auto_threshold_raw = float(state.get("threshold", 0.5))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Файл модели",
                f"Не удалось прочитать метаданные:\n{path}\n\n{exc}",
            )
            return
        self._model_path = path
        self._model_state = state
        self._model_label.setText(f"Модель:\n{path}")
        self._btn_model_info.setEnabled(True)
        self._sync_threshold_ui_from_metadata()

    def _show_model_info(self) -> None:
        if self._model_state is None:
            return
        dlg = ModelInfoDialog(self._model_state, self._model_path, self)
        dlg.exec()

    def _choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Папка с изображениями")
        if path:
            self._image_dir = path
            self._folder_label.setText(f"Папка:\n{path}")

    def _start_conveyor(self) -> None:
        if self._running:
            return
        if self._training_busy:
            QMessageBox.warning(
                self,
                "Обучение",
                "Дождитесь завершения обучения перед запуском конвейера.",
            )
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
        self._history.clear()
        self._current_history_idx = -1
        self._apply_history_index()

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
        self._score_min = float(score_min)
        self._score_max = float(score_max)
        self._model_auto_threshold_raw = float(threshold_raw)
        self._sync_threshold_ui_from_metadata()

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
        rgb = load_display_rgb_224(path)
        amap = np.asarray(anomaly_map, dtype=np.float32)
        entry = InferenceHistoryEntry(
            path=path,
            raw_score=float(raw_score),
            rgb=np.copy(rgb),
            anomaly_map=np.copy(amap),
            elapsed_ms=float(elapsed_ms),
        )
        self._history.append(entry)
        self._current_history_idx = len(self._history) - 1
        self._apply_history_index()

        short_name = Path(path).name
        thr = self._current_threshold_raw()
        status = "БРАК" if raw_score >= thr else "НОРМА"
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
        if self._threshold_auto_check.isChecked():
            return
        self._refresh_threshold_caption_manual()
        self._refresh_verdict()

    def _refresh_verdict(self) -> None:
        if self._last_rgb is None:
            self._verdict_label.setText("—")
            self._verdict_label.setStyleSheet("background-color: #444; color: #aaa; border-radius: 6px;")
            return
        thr = self._current_threshold_raw()
        is_defect = self._last_raw_score >= thr
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
        # Для визуализации не даём диапазону схлопнуться ниже порога:
        # иначе даже "нормальные" кадры быстро насыщаются в красный.
        vis_max = max(float(self._score_max), float(self._model_auto_threshold_raw))
        heat_bgr = anomaly_map_to_bgr_heatmap(m, self._score_min, vis_max)
        if mode == ViewMode.ORIGINAL:
            out = rgb
        elif mode == ViewMode.HEATMAP:
            heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
            out = heat_rgb
        else:
            span = vis_max - float(self._score_min)
            if span > 1e-12:
                intensity = np.clip((m - float(self._score_min)) / span, 0.0, 1.0)
            else:
                intensity = np.zeros_like(m, dtype=np.float32)
            out = blend_rgb_with_heat_bgr(rgb, heat_bgr, alpha=0.55, intensity_map=intensity)
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

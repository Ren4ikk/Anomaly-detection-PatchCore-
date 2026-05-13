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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, QSettings
from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
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
from patchcore_gui.workers import InferenceWorker, TrainingWorker, select_device

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


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

        # Metrics
        m = state.get("metrics", {})
        if m:
            rows.append(("── Метрики качества ──", ""))
            if "image_auroc" in m:
                rows.append(("Image AUROC", f"{float(m['image_auroc']):.4f}"))
            if "pixel_auroc" in m:
                rows.append(("Pixel AUROC", f"{float(m['pixel_auroc']):.4f}"))
            if "pro_score" in m:
                rows.append(("PRO Score", f"{float(m['pro_score']):.4f}"))

        return rows



class TrainingResultDialog(QDialog):
    """
    Диалог результатов обучения: параметры модели + метрики + ROC-кривые.
    Показывается сразу после успешного завершения TrainingWorker.
    """

    def __init__(
        self,
        threshold: float,
        score_min: float,
        score_max: float,
        save_path: str,
        metrics: dict,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Обучение завершено")
        self.setModal(True)
        self.resize(640, 520)
        self._metrics = metrics
        self._build_ui(threshold, score_min, score_max, save_path, metrics)

    def _build_ui(
        self,
        threshold: float,
        score_min: float,
        score_max: float,
        save_path: str,
        metrics: dict,
    ) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Header
        header = QLabel(f"<b>Модель сохранена:</b> {save_path}")
        header.setWordWrap(True)
        header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(header)

        # Tabs: параметры | метрики и графики
        tabs = QTabWidget(self)
        tabs.addTab(self._build_params_tab(threshold, score_min, score_max), "Параметры")
        if metrics:
            tabs.addTab(self._build_metrics_tab(metrics), "Метрики")
            if _MATPLOTLIB_AVAILABLE:
                charts_tab = self._build_charts_tab(metrics)
                if charts_tab is not None:
                    tabs.addTab(charts_tab, "Графики ROC")
        root.addWidget(tabs)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.accept)
        root.addWidget(buttons)

    def _build_params_tab(
        self, threshold: float, score_min: float, score_max: float
    ) -> QWidget:
        page = QWidget(self)
        lay = QVBoxLayout(page)
        table = QTableWidget(3, 2, page)
        table.setHorizontalHeaderLabels(["Параметр", "Значение"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(table.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        rows = [
            ("Порог (threshold)", f"{threshold:.6f}"),
            ("score_min", f"{score_min:.6f}"),
            ("score_max", f"{score_max:.6f}"),
        ]
        for i, (k, v) in enumerate(rows):
            table.setItem(i, 0, QTableWidgetItem(k))
            table.setItem(i, 1, QTableWidgetItem(v))
        table.resizeColumnToContents(0)
        lay.addWidget(table)
        return page

    def _build_metrics_tab(self, metrics: dict) -> QWidget:
        page = QWidget(self)
        lay = QVBoxLayout(page)

        rows = []
        if "image_auroc" in metrics:
            rows.append(("Image AUROC", f"{float(metrics['image_auroc']):.4f}"))
        if "pixel_auroc" in metrics:
            rows.append(("Pixel AUROC", f"{float(metrics['pixel_auroc']):.4f}"))
        if "pro_score" in metrics:
            rows.append(("PRO Score", f"{float(metrics['pro_score']):.4f}"))

        table = QTableWidget(len(rows), 2, page)
        table.setHorizontalHeaderLabels(["Метрика", "Значение"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(table.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        for i, (k, v) in enumerate(rows):
            k_item = QTableWidgetItem(k)
            v_item = QTableWidgetItem(v)
            v_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(i, 0, k_item)
            table.setItem(i, 1, v_item)
        table.resizeColumnToContents(0)
        lay.addWidget(table)

        if not _MATPLOTLIB_AVAILABLE:
            lay.addWidget(QLabel("Установите matplotlib для отображения графиков."))
        return page

    def _build_charts_tab(self, metrics: dict) -> "QWidget | None":
        """Строит вкладку с графиками ROC/PRO через matplotlib → PNG → QLabel."""
        import io
        import numpy as np
        from PyQt6.QtGui import QPixmap

        curves = []
        # Собираем все доступные графики
        if "image_fpr" in metrics and "image_tpr" in metrics:
            auroc = metrics.get("image_auroc", 0.0)
            curves.append(("Image ROC", metrics["image_fpr"], metrics["image_tpr"], f"AUC = {auroc:.4f}"))

        if "pixel_fpr" in metrics and "pixel_tpr" in metrics:
            auroc = metrics.get("pixel_auroc", 0.0)
            curves.append(("Pixel ROC", metrics["pixel_fpr"], metrics["pixel_tpr"], f"AUC = {auroc:.4f}"))

        if "pro_fpr" in metrics and "pro_tpr" in metrics:
            pro_score = metrics.get("pro_score", 0.0)
            curves.append(("PRO Curve", metrics["pro_fpr"], metrics["pro_tpr"], f"PRO = {pro_score:.4f}"))

        if not curves:
            return None

        # Динамически создаем сабплоты в зависимости от количества графиков
        fig, axes = plt.subplots(1, len(curves), figsize=(5 * len(curves), 4), dpi=96)
        if len(curves) == 1:
            axes = [axes]
        fig.patch.set_facecolor("#1e1e20")

        # Добавили третий цвет (зеленоватый) для графика PRO
        colors = ["#6ba3d6", "#f0a060", "#a0d66b"]

        for ax, (title, fpr, tpr, label_text), color in zip(axes, curves, colors):
            fpr_arr = np.asarray(fpr, dtype=np.float32)
            tpr_arr = np.asarray(tpr, dtype=np.float32)

            ax.set_facecolor("#252526")
            ax.plot(fpr_arr, tpr_arr, color=color, lw=2, label=label_text)
            ax.plot([0, 1], [0, 1], ":", color="#666666", lw=1)

            ax.set_xlabel("FPR", color="#c0c0c0")
            ax.set_ylabel("TPR / Overlap", color="#c0c0c0")
            ax.set_title(title, color="#e0e0e0")
            ax.tick_params(colors="#a0a0a0")

            for spine in ax.spines.values():
                spine.set_edgecolor("#555555")

            ax.legend(facecolor="#2e2e32", edgecolor="#555555", labelcolor="#e0e0e0")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)

        plt.tight_layout(pad=1.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        page = QWidget(self)
        lay = QVBoxLayout(page)
        img_label = QLabel(page)
        img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pm = QPixmap()
        pm.loadFromData(buf.read())
        img_label.setPixmap(pm)
        lay.addWidget(img_label)

        return page


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
        self._preload_worker: Optional[InferenceWorker] = None
        self._preload_model_path: str = ""
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

        self._load_ui_state()

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
        # Вертикальный сплиттер: верхняя зона (изображение/панели) + нижняя (журнал/лог)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(splitter)
        v_splitter.addWidget(top_widget)
        v_splitter.addWidget(self._build_log_panel())
        v_splitter.setStretchFactor(0, 1)
        v_splitter.setStretchFactor(1, 0)
        v_splitter.setSizes([600, 200])
        outer.addWidget(v_splitter, stretch=1)

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

        self._btn_start = QPushButton("СТАРТ")
        self._btn_stop = QPushButton("СТОП")
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

        self._btn_train = QPushButton("ОБУЧИТЬ")
        self._btn_train.setProperty("role", "train")
        self._btn_train.clicked.connect(self._start_training)
        self._btn_training_settings = QPushButton("Настройки обучения")
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
        self._btn_hist_prev = QPushButton("Предыдущее")
        self._btn_hist_next = QPushButton("Следующее")
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

        lay.addWidget(QLabel("Ручной порог:"))
        self._threshold_spinbox = QDoubleSpinBox()
        self._threshold_spinbox.setDecimals(4)
        self._threshold_spinbox.setMinimum(0.0)
        self._threshold_spinbox.setMaximum(1e9)
        self._threshold_spinbox.setSingleStep(0.0001)
        self._threshold_spinbox.setValue(0.5)
        self._threshold_spinbox.setEnabled(False)
        self._threshold_spinbox.setToolTip(
            "Сырое значение порога в единицах L2-расстояния модели.\n"
            "Используйте стрелки (шаг 0.0001) или введите значение вручную."
        )
        self._threshold_spinbox.valueChanged.connect(self._on_threshold_changed)
        lay.addWidget(self._threshold_spinbox)

        self._threshold_caption = QLabel("Порог: — (выберите модель .pt)")
        self._threshold_caption.setProperty("role", "hint")
        lay.addWidget(self._threshold_caption)
        lay.addStretch()
        return frame

    def _build_log_panel(self) -> QFrame:
        frame, flay = self._frame("Журнал / Лог")

        # --- Tab bar row: menu button on the left, then tabs ---
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(4)

        # Кнопка-меню (⋯) с действиями
        self._log_menu_btn = QToolButton()
        self._log_menu_btn.setText("☰")
        self._log_menu_btn.setToolTip("Действия с журналом")
        self._log_menu_btn.setFixedSize(32, 32)
        self._log_menu_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)

        log_menu = QMenu(self._log_menu_btn)
        self._action_clear_journal = log_menu.addAction("Очистить журнал")
        self._action_clear_journal.triggered.connect(self._clear_journal)
        log_menu.addSeparator()
        self._action_export_excel = log_menu.addAction("Экспорт в Excel…")
        self._action_export_excel.triggered.connect(self._export_to_excel)
        self._log_menu_btn.setMenu(log_menu)

        top_row.addWidget(self._log_menu_btn, alignment=Qt.AlignmentFlag.AlignTop)

        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.setDocumentMode(True)
        top_row.addWidget(self._bottom_tabs, stretch=1)

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
        self._bottom_tabs.addTab(journal_widget, "Журнал")

        # --- Tab 2: Terminal/stdout log ---
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._log_text.setFont(QFont("Courier New", 9))
        self._log_text.setStyleSheet(
            "QTextEdit { background-color: #1a1a1c; color: #b0d0b0; border: none; }"
        )
        self._bottom_tabs.addTab(self._log_text, "Лог")

        # Sync clear button visibility with active tab
        self._bottom_tabs.currentChanged.connect(self._on_bottom_tab_changed)
        self._on_bottom_tab_changed(0)

        return frame

    def _on_bottom_tab_changed(self, index: int) -> None:
        is_journal = index == 0
        self._log_menu_btn.setVisible(is_journal)

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
            QToolButton { padding: 4px 6px; border-radius: 4px; border: 1px solid #555; background-color: #3c3c44; font-size: 14px; }
            QToolButton:hover { background-color: #4a4a54; }
            QToolButton::menu-indicator { image: none; width: 0; }
            QMenu { background-color: #2d2d30; border: 1px solid #555; padding: 4px 0; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background-color: #3c3c44; }
            QMenu::item:disabled { color: #666; }
            QMenu::separator { height: 1px; background: #444; margin: 4px 8px; }
            """
        )

    def _on_role_changed(self, index: int) -> None:
        is_engineer = index == 1
        self._left_tabs.tabBar().setVisible(is_engineer)
        if not is_engineer:
            self._left_tabs.setCurrentIndex(0)

    def _current_threshold_raw(self) -> float:
        """Активный порог в сырой шкале скоров модели."""
        if self._threshold_auto_check.isChecked():
            return float(self._model_auto_threshold_raw)
        return float(self._threshold_spinbox.value())

    def _sync_threshold_ui_from_metadata(self) -> None:
        """Обновляет SpinBox и подпись после загрузки модели / обучения / model_ready."""
        # Подстраиваем шаг под масштаб диапазона — удобнее крутить стрелками
        span = self._score_max - self._score_min
        if span > 0:
            magnitude = span / 1000.0
            # Округляем шаг до 1, 2 или 5 × 10^n
            import math
            exp = math.floor(math.log10(magnitude)) if magnitude > 0 else -4
            step = 10 ** exp
        else:
            step = 0.0001
        self._threshold_spinbox.setSingleStep(step)

        if self._threshold_auto_check.isChecked():
            self._threshold_spinbox.blockSignals(True)
            self._threshold_spinbox.setValue(self._model_auto_threshold_raw)
            self._threshold_spinbox.blockSignals(False)
            self._threshold_caption.setText(
                f"Авто (из модели): {self._model_auto_threshold_raw:.4f}"
            )
        else:
            self._refresh_threshold_caption_manual()
        self._refresh_verdict()

    def _refresh_threshold_caption_manual(self) -> None:
        cur = self._current_threshold_raw()
        self._threshold_caption.setText(f"Вручную: {cur:.4f}")

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
        self._score_value.setText(f"Score: {e.raw_score:.4f}")
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
        self._threshold_spinbox.setEnabled(not checked)
        if checked:
            self._sync_threshold_ui_from_metadata()
        else:
            # Pre-fill spinbox with current auto value so user starts from a sensible number
            self._threshold_spinbox.blockSignals(True)
            self._threshold_spinbox.setValue(self._model_auto_threshold_raw)
            self._threshold_spinbox.blockSignals(False)
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
            self._threshold_spinbox.setEnabled(False)
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
            self._threshold_spinbox.setEnabled(not self._threshold_auto_check.isChecked())
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
            metrics_val_dir=self._training_settings.metrics_val_dir,
            metrics_mask_dir=self._training_settings.metrics_mask_dir,
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

    def _on_training_success(
        self, threshold: float, score_min: float, score_max: float, metrics: dict
    ) -> None:
        self._close_training_progress()
        self._set_training_locked(False)

        if self._model_state is None:
            self._model_state = {}
        self._model_state["metrics"] = metrics

        self._model_auto_threshold_raw = float(threshold)
        self._score_min = float(score_min)
        self._score_max = float(score_max)
        self._sync_threshold_ui_from_metadata()
        self._append_training_success_log(threshold)
        dlg = TrainingResultDialog(
            threshold=threshold,
            score_min=score_min,
            score_max=score_max,
            save_path=self._train_save_path,
            metrics=metrics,
            parent=self,
        )
        dlg.exec()

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
        self._start_preload_worker(path)

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

        self._running = True
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

        device = select_device(self._device_pref)
        if (
            self._preload_worker is not None
            and self._preload_worker.isRunning()
            and self._preload_model_path == self._model_path
        ):
            # Берём предзагрузочный воркер — модель уже загружается или загружена
            self._worker = self._preload_worker
            self._preload_worker = None
            self._preload_model_path = ""
            self._worker.inference_done.connect(self._on_inference_done)
            self._worker.inference_failed.connect(self._on_inference_failed)
            self._worker.finished.connect(self._on_worker_finished)
            if self._worker.is_model_loaded:
                # Модель уже готова — немедленно стартуем конвейер
                self._on_timer_tick()
                self._timer.start()
            # Иначе _on_model_ready придёт чуть позже и запустит таймер
        else:
            # Предзагрузка недоступна — создаём воркер как обычно
            self._discard_preload_worker()
            self._worker = InferenceWorker(self._model_path, device)
            self._worker.is_model_loaded = False
            self._worker.model_ready.connect(self._on_model_ready)
            self._worker.inference_done.connect(self._on_inference_done)
            self._worker.inference_failed.connect(self._on_inference_failed)
            self._worker.finished.connect(self._on_worker_finished)
            self._worker.start()
            # Таймер запустится в _on_model_ready

    def _start_preload_worker(self, model_path: str) -> None:
        """Запускает фоновую загрузку модели сразу при её выборе."""
        self._discard_preload_worker()
        device = select_device(self._device_pref)
        w = InferenceWorker(model_path, device)
        w.is_model_loaded = False
        w.model_ready.connect(self._on_model_ready)
        self._preload_worker = w
        self._preload_model_path = model_path
        w.start()
        print(f"[Preload] Фоновая загрузка модели: {model_path}")

    def _discard_preload_worker(self) -> None:
        """Останавливает и удаляет предзагрузочный воркер."""
        if self._preload_worker is not None:
            self._preload_worker.request_stop()
            self._preload_worker.wait(5_000)
            self._preload_worker = None
            self._preload_model_path = ""

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
        # Если конвейер уже запущен и ждёт загрузки — стартуем таймер.
        # Если это предзагрузка в фоне — ничего не делаем, запомнит is_model_loaded.
        if self._running and self._worker is not None:
            self._worker.is_model_loaded = True
            self._on_timer_tick()
            self._timer.start()
        elif self._preload_worker is not None:
            self._preload_worker.is_model_loaded = True

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
        _norm_score: float,
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

    def _on_threshold_changed(self, _value: float) -> None:
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

    def closeEvent(self, event) -> None:
        """Перехватываем закрытие окна для сохранения UI и остановки процессов."""
        if self._running:
            self._stop_conveyor()
        self._discard_preload_worker()
        self._save_ui_state()
        super().closeEvent(event)

    def _save_ui_state(self) -> None:
        """Сохраняет состояние интерфейса в QSettings."""
        settings = QSettings("VKR", "PatchCoreApp")

        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("role_index", self._role_combo.currentIndex())
        settings.setValue("mode_index", self._mode_combo.currentIndex())

        settings.setValue("model_path", self._model_path)
        settings.setValue("image_dir", self._image_dir)
        settings.setValue("train_image_dir", self._train_image_dir)
        settings.setValue("train_save_path", self._train_save_path)

        settings.setValue("threshold_auto", self._threshold_auto_check.isChecked())
        settings.setValue("threshold_manual", self._threshold_spinbox.value())

    def _load_ui_state(self) -> None:
        """Загружает состояние интерфейса при старте программы."""
        settings = QSettings("VKR", "PatchCoreApp")

        # 1. Восстанавливаем размер окна (если окно ранее было закрыто)
        geom = settings.value("geometry")
        if geom is not None:
            self.restoreGeometry(geom)
        else:
            self.showMaximized()

        # 2. Выпадающие списки и переключатели
        self._role_combo.setCurrentIndex(settings.value("role_index", 0, type=int))
        self._mode_combo.setCurrentIndex(settings.value("mode_index", 0, type=int))
        self._threshold_auto_check.setChecked(settings.value("threshold_auto", True, type=bool))
        self._threshold_spinbox.setValue(settings.value("threshold_manual", 0.5, type=float))

        # 3. Восстанавливаем обычные пути к папкам и обновляем их ярлыки (Label)
        self._image_dir = settings.value("image_dir", "", type=str)
        if self._image_dir:
            self._folder_label.setText(f"Папка:\n{self._image_dir}")

        self._train_image_dir = settings.value("train_image_dir", "", type=str)
        if self._train_image_dir:
            self._train_dir_label.setText(f"Папка НОРМЫ:\n{self._train_image_dir}")

        self._train_save_path = settings.value("train_save_path", "", type=str)
        if self._train_save_path:
            self._train_save_label.setText(f"Файл модели:\n{self._train_save_path}")

        # 4. Восстанавливаем модель только если файл всё ещё существует на диске
        saved_model = settings.value("model_path", "", type=str)
        if saved_model and Path(saved_model).is_file():
            self._model_path = saved_model
            self._model_label.setText(f"Модель:\n{self._model_path}")
            self._try_restore_model_metadata(self._model_path)

    def _try_restore_model_metadata(self, path: str) -> None:
        """Тихо подгружает метаданные из сохраненного файла без блокировки."""
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(state, dict):
                self._model_state = state
                self._score_min = float(state.get("score_min", 0.0))
                self._score_max = float(state.get("score_max", 1.0))
                self._model_auto_threshold_raw = float(state.get("threshold", 0.5))
                self._btn_model_info.setEnabled(True)
                self._sync_threshold_ui_from_metadata()
                self._start_preload_worker(path)
        except Exception as e:
            print(f"[UI State] Не удалось восстановить метаданные модели: {e}")

    def _export_to_excel(self) -> None:
        """Собирает параметры модели, результаты инференса и метрики качества в Excel."""
        if not self._history:
            QMessageBox.warning(self, "Пусто", "Нет данных для экспорта. Сначала запустите конвейер.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить отчет",
            "PatchCore_Report.xlsx",
            "Excel Files (*.xlsx)"
        )
        if not path:
            return

        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"

        try:
            import pandas as pd
        except ImportError:
            QMessageBox.critical(
                self,
                "Ошибка",
                "Для экспорта в Excel необходимы библиотеки pandas и openpyxl.\nУстановите их командой: pip install pandas openpyxl"
            )
            return

        try:
            # --- 1. Лист: Параметры модели ---
            model_info = {
                "Параметр": ["Путь к модели"],
                "Значение": [self._model_path]
            }
            if self._model_state:
                for k, v in self._model_state.items():
                    # Исключаем словарь метрик (он пойдет на отдельный лист)
                    # и сложные объекты, оставляя только базовые параметры
                    if k != "metrics" and isinstance(v, (int, float, str, tuple, list)):
                        model_info["Параметр"].append(k)
                        model_info["Значение"].append(str(v))
            df_model = pd.DataFrame(model_info)

            # --- 2. Лист: Метрики качества (НОВОЕ) ---
            metrics_data = {"Показатель": [], "Значение": []}
            if self._model_state and "metrics" in self._model_state:
                m = self._model_state["metrics"]

                # Собираем только скалярные значения (основные метрики)
                mapping = {
                    "image_auroc": "Image AUROC (Global)",
                    "pixel_auroc": "Pixel AUROC (Localization)",
                    "pro_score": "PRO Score",
                }

                for key, label in mapping.items():
                    if key in m:
                        metrics_data["Показатель"].append(label)
                        metrics_data["Значение"].append(round(float(m[key]), 4))

            # Если метрик нет (модель не валидировалась), добавим пояснение
            if not metrics_data["Показатель"]:
                metrics_data["Показатель"].append("Статус")
                metrics_data["Значение"].append("Метрики не рассчитывались")

            df_metrics = pd.DataFrame(metrics_data)

            # --- 3. Лист: Результаты инференса ---
            results_data = {
                "Имя файла": [],
                "Score (сырой)": [],
                "Вердикт": [],
                "Время (мс)": [],
                "Полный путь": []
            }

            thr = self._current_threshold_raw()
            for entry in self._history:
                is_defect = entry.raw_score >= thr
                results_data["Имя файла"].append(Path(entry.path).name)
                results_data["Score (сырой)"].append(round(entry.raw_score, 6))
                results_data["Вердикт"].append("БРАК" if is_defect else "НОРМА")
                results_data["Время (мс)"].append(round(entry.elapsed_ms, 2))
                results_data["Полный путь"].append(entry.path)

            df_results = pd.DataFrame(results_data)

            # --- Запись в файл ---
            with pd.ExcelWriter(path, engine='openpyxl') as writer:
                df_model.to_excel(writer, sheet_name="Параметры модели", index=False)
                df_metrics.to_excel(writer, sheet_name="Метрики качества", index=False)
                df_results.to_excel(writer, sheet_name="Результаты", index=False)

            QMessageBox.information(self, "Успех", f"Отчет успешно сохранен в файл:\n{path}")

        except Exception as e:
            QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить файл Excel:\n{e}")

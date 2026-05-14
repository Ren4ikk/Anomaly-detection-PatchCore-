from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class TrainingSettings:
    backbone_name: str = "wide_resnet50_2"
    layers: tuple[str, ...] = ("layer2", "layer3")
    coreset_ratio: float = 0.1
    n_reweight_nn: int = 9
    gaussian_sigma: float = 4.0
    patch_size: int = 3
    device: str = "auto"
    use_gpu_faiss: bool = False
    threshold_mode: str = "three_sigma"
    validation_dir: str | None = None
    gt_mask_dir: str | None = None
    threshold_objective: str = "image_f1"
    # Директории для вычисления метрик (независимо от настроек порога)
    metrics_val_dir: str | None = None
    metrics_mask_dir: str | None = None


class SettingsDialog(QDialog):
    def __init__(self, initial: TrainingSettings | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки формирования эталонного банка памяти")
        self.setModal(True)
        self.resize(760, 520)
        self._initial = initial or TrainingSettings()
        self._settings: TrainingSettings | None = None

        self._build_ui()
        self._apply_initial_settings()
        self._on_threshold_type_changed()
        self._on_f1_target_changed()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget(self)
        tabs.addTab(self._build_arch_tab(), "Архитектура и Гиперпараметры")
        tabs.addTab(self._build_threshold_tab(), "Настройки порога")
        tabs.addTab(self._build_metrics_tab(), "Оценка качества")
        root.addWidget(tabs)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _build_arch_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)

        form = QFormLayout()
        self._backbone_combo = QComboBox(page)
        self._backbone_combo.addItems([
            "wide_resnet50_2",
            "wide_resnet101_2",
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnext50_32x4d",
            "resnext101_32x8d"
        ])
        form.addRow("Backbone:", self._backbone_combo)

        layers_group = QGroupBox("Используемые слои (Feature Hierarchy)", page)
        layers_layout = QHBoxLayout(layers_group)
        self._layer_checks: dict[str, QCheckBox] = {}
        for layer_name in ("layer1", "layer2", "layer3", "layer4"):
            cb = QCheckBox(layer_name, layers_group)
            self._layer_checks[layer_name] = cb
            layers_layout.addWidget(cb)
        layers_layout.addStretch()

        self._coreset_spin = QDoubleSpinBox(page)
        self._coreset_spin.setRange(0.1, 100.0)
        self._coreset_spin.setDecimals(2)
        self._coreset_spin.setSuffix(" %")
        self._coreset_spin.setSingleStep(0.1)
        form.addRow("Размер сжатия Coreset (%):", self._coreset_spin)

        self._neighbors_spin = QSpinBox(page)
        self._neighbors_spin.setRange(1, 128)
        form.addRow("Соседи для Re-weighting (b):", self._neighbors_spin)

        self._sigma_spin = QDoubleSpinBox(page)
        self._sigma_spin.setRange(0.1, 20.0)
        self._sigma_spin.setDecimals(2)
        self._sigma_spin.setSingleStep(0.1)
        form.addRow("Gaussian Blur Sigma:", self._sigma_spin)

        self._patch_spin = QSpinBox(page)
        self._patch_spin.setRange(1, 31)
        self._patch_spin.setSingleStep(2)
        form.addRow("Размер окрестности патча (Patch Size):", self._patch_spin)

        device_row = QWidget(page)
        device_layout = QHBoxLayout(device_row)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(10)
        device_layout.addWidget(QLabel("Устройство (PyTorch):", device_row))
        self._device_combo = QComboBox(device_row)
        self._device_combo.addItems(["auto", "cpu", "cuda"])
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        device_layout.addWidget(self._device_combo)
        self._faiss_gpu_check = QCheckBox("Ускорение FAISS (GPU)", device_row)
        self._faiss_gpu_check.setChecked(False)
        device_layout.addWidget(self._faiss_gpu_check)
        device_layout.addStretch()
        form.addRow("Вычисления:", device_row)

        layout.addLayout(form)
        layout.addWidget(layers_group)
        layout.addStretch()
        return page

    def _build_threshold_tab(self) -> QWidget:
        page = QWidget(self)
        root = QVBoxLayout(page)

        self.radio_3sigma = QRadioButton("Авто-порог по правилу 3-х сигм (по эталонным данным)", page)
        self.radio_f1 = QRadioButton("F1-оптимальный порог (требуется валидационный датасет)", page)
        self.radio_3sigma.setChecked(True)
        self.radio_3sigma.setStyleSheet(
            """
            QRadioButton { spacing: 8px; color: #e4e4e4; }
            QRadioButton::indicator { width: 14px; height: 14px; border-radius: 7px; border: 1px solid #707070; background: #2f2f33; }
            QRadioButton::indicator:checked { background: #6ba3d6; border: 1px solid #8fc3f0; }
            """
        )
        self.radio_f1.setStyleSheet(self.radio_3sigma.styleSheet())
        group = QButtonGroup(page)
        group.addButton(self.radio_3sigma)
        group.addButton(self.radio_f1)
        self.radio_3sigma.toggled.connect(self._on_threshold_type_changed)
        self.radio_f1.toggled.connect(self._on_threshold_type_changed)
        root.addWidget(self.radio_3sigma)
        root.addWidget(self.radio_f1)

        self._f1_box = QGroupBox("Параметры F1-оптимизации", page)
        self._f1_box.setVisible(False)
        form = QFormLayout(self._f1_box)

        self._btn_choose_val = QPushButton("Выбрать папку Validation", self._f1_box)
        self._btn_choose_val.clicked.connect(self._choose_validation_dir)
        self._val_path_label = QLabel("Не выбрано", self._f1_box)
        self._val_path_label.setWordWrap(True)
        val_row = QVBoxLayout()
        val_row.addWidget(self._btn_choose_val)
        val_row.addWidget(self._val_path_label)
        form.addRow("Validation:", self._wrap_layout(val_row, self._f1_box))

        self.gt_mask_container = QWidget(self._f1_box)
        gt_layout = QVBoxLayout(self.gt_mask_container)
        gt_layout.setContentsMargins(0, 0, 0, 0)
        self._gt_mask_title = QLabel("GT Маски:", self.gt_mask_container)
        self._btn_choose_mask = QPushButton(
            "Выбрать папку GT Масок (опционально)", self.gt_mask_container
        )
        self._btn_choose_mask.clicked.connect(self._choose_gt_mask_dir)
        self._mask_path_label = QLabel("Не выбрано", self.gt_mask_container)
        self._mask_path_label.setWordWrap(True)
        gt_layout.addWidget(self._gt_mask_title)
        gt_layout.addWidget(self._btn_choose_mask)
        gt_layout.addWidget(self._mask_path_label)
        form.addRow(self.gt_mask_container)

        self._objective_combo = QComboBox(self._f1_box)
        self._objective_combo.addItems(["Image-level F1", "Pixel-level F1"])
        self._objective_combo.currentIndexChanged.connect(self._on_f1_target_changed)
        self._objective_combo.setVisible(False)  # временно скрыт
        self._objective_label = QLabel("Image-level F1", self._f1_box)
        form.addRow("Цель оптимизации:", self._objective_label)

        root.addWidget(self._f1_box)
        root.addStretch()
        return page

    def _build_metrics_tab(self) -> QWidget:
        """
        Вкладка настройки вычисления метрик после формирования банка.
        Image AUROC — всегда. Pixel AUROC и PRO — только с GT-масками.
        """
        page = QWidget(self)
        root = QVBoxLayout(page)
        root.setSpacing(8)

        info = QLabel("Image AUROC (always). Pixel AUROC & PRO Score - with GT masks.", page)
        info.setWordWrap(True)
        info.setStyleSheet("color: #a0b0c0; font-size: 11px;")
        root.addWidget(info)

        form = QFormLayout()
        form.setSpacing(8)

        # Validation dir
        self._btn_metrics_val = QPushButton("Выбрать папку Validation", page)
        self._btn_metrics_val.clicked.connect(self._choose_metrics_val_dir)
        self._metrics_val_label = QLabel("Не выбрано", page)
        self._metrics_val_label.setWordWrap(True)
        val_col = QVBoxLayout()
        val_col.addWidget(self._btn_metrics_val)
        val_col.addWidget(self._metrics_val_label)
        form.addRow("Validation:", self._wrap_layout(val_col, page))

        # GT masks dir (optional)
        self._btn_metrics_mask = QPushButton("Выбрать папку GT Масок (опционально)", page)
        self._btn_metrics_mask.clicked.connect(self._choose_metrics_mask_dir)
        self._metrics_mask_label = QLabel("Не выбрано", page)
        self._metrics_mask_label.setWordWrap(True)
        mask_col = QVBoxLayout()
        mask_col.addWidget(self._btn_metrics_mask)
        mask_col.addWidget(self._metrics_mask_label)
        form.addRow("GT Маски:", self._wrap_layout(mask_col, page))

        root.addLayout(form)
        root.addStretch()
        return page

    def _choose_metrics_val_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите Validation папку для метрик")
        if path:
            self._set_path_label(self._metrics_val_label, path)

    def _choose_metrics_mask_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите папку GT масок для метрик")
        if path:
            self._set_path_label(self._metrics_mask_label, path)

    @staticmethod
    def _wrap_layout(layout: QVBoxLayout, parent: QWidget) -> QWidget:
        w = QWidget(parent)
        w.setLayout(layout)
        return w

    def _apply_initial_settings(self) -> None:
        s = self._initial
        self._backbone_combo.setCurrentText(s.backbone_name)
        for name, cb in self._layer_checks.items():
            cb.setChecked(name in s.layers)
        self._coreset_spin.setValue(s.coreset_ratio * 100.0)
        self._neighbors_spin.setValue(s.n_reweight_nn)
        self._sigma_spin.setValue(s.gaussian_sigma)
        self._patch_spin.setValue(s.patch_size if s.patch_size % 2 == 1 else s.patch_size + 1)
        self._device_combo.setCurrentText(s.device if s.device in {"auto", "cpu", "cuda"} else "auto")
        self._faiss_gpu_check.setChecked(s.use_gpu_faiss)
        self._on_device_changed()

        is_f1 = s.threshold_mode == "f1_optimal"
        self.radio_f1.setChecked(is_f1)
        self.radio_3sigma.setChecked(not is_f1)
        self._set_path_label(self._val_path_label, s.validation_dir or "")
        self._set_path_label(self._mask_path_label, s.gt_mask_dir or "")
        self._set_path_label(self._metrics_val_label, s.metrics_val_dir or "")
        self._set_path_label(self._metrics_mask_label, s.metrics_mask_dir or "")
        self._objective_combo.setCurrentIndex(0 if s.threshold_objective == "image_f1" else 1)
        self._on_threshold_type_changed()
        self._on_f1_target_changed()

    def _choose_validation_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите Validation папку")
        if path:
            self._set_path_label(self._val_path_label, path)

    def _choose_gt_mask_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Выберите папку GT масок")
        if path:
            self._set_path_label(self._mask_path_label, path)

    @staticmethod
    def _set_path_label(label: QLabel, path: str) -> None:
        label.setText(path if path else "Не выбрано")
        label.setToolTip(path)

    def _on_device_changed(self, _index: int = 0) -> None:
        is_cuda = self._device_combo.currentText() == "cuda"
        self._faiss_gpu_check.setVisible(is_cuda)
        if not is_cuda:
            self._faiss_gpu_check.setChecked(False)

    def _on_threshold_type_changed(self) -> None:
        self._f1_box.setVisible(self.radio_f1.isChecked())

    def _on_f1_target_changed(self) -> None:
        is_pixel_f1 = self._objective_combo.currentIndex() == 1
        self.gt_mask_container.setVisible(is_pixel_f1)

    def _collect_layers(self) -> tuple[str, ...]:
        return tuple(name for name, cb in self._layer_checks.items() if cb.isChecked())

    def _validate_f1_dirs(self, validation_dir: str) -> bool:
        val_path = Path(validation_dir)
        if not val_path.is_dir():
            return False
        good_ok = (val_path / "good").is_dir()
        defect_dirs = [
            p for p in val_path.iterdir() if p.is_dir() and p.name != "good"
        ]
        return good_ok and len(defect_dirs) > 0

    def _on_accept(self) -> None:
        layers = self._collect_layers()
        if not layers:
            QMessageBox.warning(self, "Слои", "Выберите минимум один слой признаков.")
            return

        validation_dir_text = (
            "" if self._val_path_label.text() == "Не выбрано" else self._val_path_label.text()
        )
        gt_mask_dir_text = (
            "" if self._mask_path_label.text() == "Не выбрано" else self._mask_path_label.text()
        )
        threshold_mode = "f1_optimal" if self.radio_f1.isChecked() else "three_sigma"
        objective = "image_f1" if self._objective_combo.currentIndex() == 0 else "pixel_f1"
        validation_dir: str | None = validation_dir_text or None
        gt_mask_dir: str | None = gt_mask_dir_text or None

        if threshold_mode == "f1_optimal":
            if validation_dir is None:
                QMessageBox.warning(self, "Validation", "Для F1-порога выберите папку Validation.")
                return
            if not self._validate_f1_dirs(validation_dir):
                QMessageBox.warning(
                    self,
                    "Validation",
                    "Папка Validation должна содержать подпапку 'good' и хотя бы одну папку с дефектами.",
                )
                return

            if objective == "image_f1":
                gt_mask_dir = None
        else:
            validation_dir = None
            gt_mask_dir = None

        metrics_val = (
            None if self._metrics_val_label.text() == "Не выбрано"
            else self._metrics_val_label.text() or None
        )
        metrics_mask = (
            None if self._metrics_mask_label.text() == "Не выбрано"
            else self._metrics_mask_label.text() or None
        )

        # Валидация папки метрик: структура должна совпадать с validation (good + дефекты)
        if metrics_val is not None:
            if not self._validate_f1_dirs(metrics_val):
                QMessageBox.warning(
                    self,
                    "Оценка качества — Validation",
                    "Папка Validation для метрик должна содержать подпапку 'good' "
                    "и хотя бы одну папку с дефектами.\n\n"
                    "Ожидаемая структура:\n"
                    "  <папка>/good/*.png\n"
                    "  <папка>/<дефект>/*.png",
                )
                return

        # Валидация папки GT-масок: должна существовать и быть директорией
        if metrics_mask is not None:
            if not Path(metrics_mask).is_dir():
                QMessageBox.warning(
                    self,
                    "Оценка качества — GT-маски",
                    f"Папка GT-масок не найдена:\n{metrics_mask}\n\n"
                    "Убедитесь, что путь указан верно.",
                )
                return

        self._settings = TrainingSettings(
            backbone_name=self._backbone_combo.currentText(),
            layers=layers,
            coreset_ratio=float(self._coreset_spin.value()) / 100.0,
            n_reweight_nn=int(self._neighbors_spin.value()),
            gaussian_sigma=float(self._sigma_spin.value()),
            patch_size=int(self._patch_spin.value()),
            device=self._device_combo.currentText(),
            use_gpu_faiss=self._faiss_gpu_check.isChecked(),
            threshold_mode=threshold_mode,
            validation_dir=validation_dir,
            gt_mask_dir=gt_mask_dir,
            threshold_objective=objective,
            metrics_val_dir=metrics_val,
            metrics_mask_dir=metrics_mask,
        )
        self.accept()

    @property
    def settings(self) -> TrainingSettings | None:
        return self._settings

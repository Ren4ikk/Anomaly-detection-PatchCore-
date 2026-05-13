"""
Этап 1 — Инфраструктура: Метрики качества PatchCore.

Реализует три метрики из оригинальной статьи:
  • Image-level AUROC  — основная метрика обнаружения аномалий
  • Pixel-level AUROC  — метрика точности сегментации (локализации)
  • PRO Score          — Per-Region Overlap, критически важна для
                         промышленных задач (оценивает каждый
                         связный компонент отдельно)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label as scipy_label
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# Датакласс для хранения результатов одного прогона

@dataclass
class MetricResults:
    """Хранит все метрики после вызова Metrics.compute()."""

    image_auroc: float = 0.0
    pixel_auroc: float = 0.0
    pro_score: float = 0.0

    # Вспомогательные данные для построения кривых
    image_fpr: NDArray = field(default_factory=lambda: np.array([]))
    image_tpr: NDArray = field(default_factory=lambda: np.array([]))
    pixel_fpr: NDArray = field(default_factory=lambda: np.array([]))
    pixel_tpr: NDArray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        return (
            f"Image-AUROC : {self.image_auroc:.4f}\n"
            f"Pixel-AUROC : {self.pixel_auroc:.4f}\n"
            f"PRO Score   : {self.pro_score:.4f}"
        )


# Вспомогательные функции

def _compute_pro(
        anomaly_maps: NDArray,
        gt_masks: NDArray,
        num_thresh: int = 100,
) -> tuple[float, NDArray, NDArray]:
    gt_masks = gt_masks.astype(bool)
    normal_masks = ~gt_masks
    total_normal_pixels = int(normal_masks.sum())

    # 1. Оптимизация по памяти: берем подвыборку пикселей для расчета перцентилей.
    # Шага 10 или 100 достаточно, чтобы идеально восстановить распределение.
    sampled_maps = anomaly_maps.flatten()[::10]
    thresholds = np.percentile(sampled_maps, np.linspace(0, 100, num=num_thresh))
    thresholds = np.unique(thresholds)

    # 2. Оптимизация по скорости: предвычисляем связные компоненты для ВСЕХ масок ОДИН РАЗ
    image_components = []
    for gt_mask in gt_masks:
        labeled, n_components = scipy_label(gt_mask)
        # Сохраняем маску для каждой компоненты
        comps = [labeled == i for i in range(1, n_components + 1)]
        image_components.append(comps)

    all_fprs: list[float] = []
    all_pros: list[float] = []

    # 3. Основной цикл по порогам
    for thresh in thresholds:
        binary_maps = anomaly_maps >= thresh  # Векторизовано: (N, H, W)

        # Подсчет FP сразу по всему батчу (супер быстро)
        fp_pixels = int((binary_maps & normal_masks).sum())
        fpr = fp_pixels / max(total_normal_pixels, 1)

        # Подсчет PRO по предвычисленным компонентам
        pro_values: list[float] = []
        for pred_map, comps in zip(binary_maps, image_components):
            for comp_mask in comps:
                overlap = (pred_map & comp_mask).sum() / comp_mask.sum()
                pro_values.append(float(overlap))

        pro = float(np.mean(pro_values)) if pro_values else 0.0

        all_fprs.append(fpr)
        all_pros.append(pro)

    fprs = np.array(all_fprs)
    pros = np.array(all_pros)

    # Сортируем по возрастанию FPR для интегрирования
    sort_idx = np.argsort(fprs)
    fprs, pros = fprs[sort_idx], pros[sort_idx]

    # Интегрируем до FPR = 0.3, нормируем
    fpr_limit = 0.3
    mask = fprs <= fpr_limit
    if mask.sum() > 1:
        pro_auc = float(np.trapezoid(pros[mask], fprs[mask]) / fpr_limit)
    else:
        pro_auc = 0.0

    return pro_auc, fprs, pros


# Основной класс метрик

class Metrics:
    """
    Вычисляет метрики качества модели обнаружения аномалий.

    Поддерживаемые метрики:
    Image-level AUROC — AUC ROC-кривой по скорам на уровне
    изображения. Основная метрика детекции.
    Pixel-level AUROC — AUC ROC-кривой по скорам на уровне
    пикселей. Метрика точности локализации (сегментации).
    PRO Score (Per-Region Overlap) — AUC кривой overlap/FPR
    до FPR=0.3, нормированной на 0.3. Критически важна для
    промышленных задач: оценивает каждый связный компонент
    аномалии отдельно, не завышая скор за счёт крупных дефектов.

    Пример использования::

        metrics = Metrics()
        results = metrics.compute(
            image_scores=scores_1d,   # (N,)
            gt_labels=labels_1d,      # (N,)
            anomaly_maps=maps_nhw,    # (N, H, W)
            gt_masks=masks_nhw,       # (N, H, W)
        )
        print(results)
    """

    def compute(
        self,
        image_scores: NDArray,
        gt_labels: NDArray,
        anomaly_maps: Optional[NDArray] = None,
        gt_masks: Optional[NDArray] = None,
        pro_num_thresh: int = 100,
    ) -> MetricResults:
        """
        Вычисляет все доступные метрики.

        Args:
            image_scores:   Скоры аномальности на уровне изображений, (N,).
                            Больший скор - более аномально.
            gt_labels:      Бинарные GT-метки изображений, (N,).
                            0 = нормальное, 1 = аномальное.
            anomaly_maps:   Тепловые карты аномальности, (N, H, W).
                            Если None — pixel AUROC и PRO не вычисляются.
            gt_masks:       Бинарные GT-маски пикселей, (N, H, W).
                            Если None — pixel AUROC и PRO не вычисляются.
            pro_num_thresh: Число порогов для PRO-интегрирования.

        Returns:
            MetricResults с заполненными полями.
        """
        image_scores = np.asarray(image_scores, dtype=np.float32)
        gt_labels = np.asarray(gt_labels, dtype=np.int32)

        results = MetricResults()

        # -- Image-level AUROC --------------------------------------
        results.image_auroc = float(roc_auc_score(gt_labels, image_scores))
        image_fpr, image_tpr, _ = roc_curve(gt_labels, image_scores)
        results.image_fpr = image_fpr
        results.image_tpr = image_tpr

        # -- Pixel-level AUROC и PRO --------------------------------
        if anomaly_maps is not None and gt_masks is not None:
            anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
            gt_masks = np.asarray(gt_masks, dtype=np.uint8)

            self._validate_pixel_inputs(anomaly_maps, gt_masks, gt_labels)

            # Pixel AUROC — сплющиваем всё в 1D
            flat_maps = anomaly_maps.flatten()
            flat_masks = gt_masks.flatten()
            results.pixel_auroc = float(roc_auc_score(flat_masks, flat_maps))
            pixel_fpr, pixel_tpr, _ = roc_curve(flat_masks, flat_maps)
            results.pixel_fpr = pixel_fpr
            results.pixel_tpr = pixel_tpr

            # PRO Score.
            # Передаём ВСЕ изображения: нормальные вносят свои пиксели
            # в знаменатель FPR, что соответствует определению метрики.
            # У нормальных изображений gt_mask нулевая — scipy_label
            # не находит компонентов, поэтому overlap не затрагивается.
            anomaly_idx = gt_labels == 1
            if anomaly_idx.sum() > 0:
                results.pro_score, _, _ = _compute_pro(
                    anomaly_maps,
                    gt_masks,
                    num_thresh=pro_num_thresh,
                )

        return results

    @staticmethod
    def compute_f1_optimal_threshold(
        y_true: NDArray,
        y_scores: NDArray,
    ) -> tuple[float, float]:
        y_true_arr = np.asarray(y_true, dtype=np.int32).reshape(-1)
        y_scores_arr = np.asarray(y_scores, dtype=np.float32).reshape(-1)
        if y_true_arr.shape[0] != y_scores_arr.shape[0]:
            raise ValueError("Размерности y_true и y_scores должны совпадать.")
        if y_true_arr.shape[0] == 0:
            raise ValueError("Пустые массивы для расчета F1-порога.")
        if np.unique(y_true_arr).size < 2:
            raise ValueError("Для F1-оптимизации нужны оба класса (0 и 1).")

        precision, recall, thresholds = precision_recall_curve(y_true_arr, y_scores_arr)
        if thresholds.size == 0:
            raise ValueError("Не удалось вычислить пороги для precision-recall кривой.")

        # precision/recall на 1 элемент длиннее thresholds, берем согласованные точки.
        p = precision[:-1]
        r = recall[:-1]
        denom = p + r
        f1 = np.where(denom > 0.0, 2.0 * p * r / denom, 0.0)
        best_idx = int(np.argmax(f1))
        return float(thresholds[best_idx]), float(f1[best_idx])

    # Валидация входных данных

    @staticmethod
    def _validate_pixel_inputs(
        anomaly_maps: NDArray,
        gt_masks: NDArray,
        gt_labels: NDArray,
    ) -> None:
        if anomaly_maps.ndim != 3:
            raise ValueError(
                f"anomaly_maps должен быть (N, H, W), получено: {anomaly_maps.shape}"
            )
        if gt_masks.ndim != 3:
            raise ValueError(
                f"gt_masks должен быть (N, H, W), получено: {gt_masks.shape}"
            )
        if anomaly_maps.shape != gt_masks.shape:
            raise ValueError(
                f"Форма anomaly_maps {anomaly_maps.shape} != gt_masks {gt_masks.shape}"
            )
        if anomaly_maps.shape[0] != len(gt_labels):
            raise ValueError(
                f"N в anomaly_maps ({anomaly_maps.shape[0]}) != "
                f"len(gt_labels) ({len(gt_labels)})"
            )
        if gt_masks.max() > 1:
            raise ValueError("gt_masks должен быть бинарным (0 или 1).")

"""
Этап 1 — Инфраструктура: Метрики качества PatchCore.

Реализует три метрики из оригинальной статьи (Roth et al., 2021):
  • Image-level AUROC  — основная метрика обнаружения аномалий
  • Pixel-level AUROC  — метрика точности сегментации (локализации)
  • PRO Score          — Per-Region Overlap, критически важна для
                         промышленных задач (оценивает каждый
                         связный компонент отдельно)

Ссылки:
  Статья:             https://arxiv.org/pdf/2106.08265  (Sec. 4, App. C)
  Реализация авторов: https://github.com/amazon-science/patchcore-inspection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import label as scipy_label
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# ──────────────────────────────────────────────────────
# Датакласс для хранения результатов одного прогона
# ──────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────

def _compute_pro(
    anomaly_maps: NDArray,  # (N, H, W) float — карты аномальности
    gt_masks: NDArray,       # (N, H, W) bool/uint8 — бинарные GT-маски
    num_thresh: int = 100,
) -> tuple[float, NDArray, NDArray]:
    """
    Вычисляет PRO (Per-Region Overlap) Score.

    Алгоритм (Appendix C статьи):
      1. Для каждого порога t пробегаем по всем изображениям.
      2. На каждом изображении находим связные компоненты в GT-маске.
      3. Для каждого компонента вычисляем долю пикселей, покрытых
         бинаризованной картой аномальности (overlap).
      4. Усредняем overlap по всем компонентам → TPR при данном пороге.
      5. Вычисляем FPR как долю ложно-положительных нормальных пикселей.
      6. Интегрируем TPR по FPR ∈ [0, 0.3] и нормируем на 0.3
         (авторский протокол).

    Args:
        anomaly_maps: Массив тепловых карт аномальности, форма (N, H, W).
        gt_masks:     Бинарные GT-маски, форма (N, H, W). 1 = аномалия.
        num_thresh:   Число равномерно распределённых порогов.

    Returns:
        Кортеж (pro_auc, all_fprs, all_pros).
    """
    gt_masks = gt_masks.astype(bool)

    # Линейное пространство порогов в диапазоне реальных значений карт
    min_val, max_val = anomaly_maps.min(), anomaly_maps.max()
    thresholds = np.linspace(min_val, max_val, num=num_thresh)

    all_fprs: list[float] = []
    all_pros: list[float] = []

    for thresh in thresholds:
        binary_maps = anomaly_maps >= thresh  # (N, H, W)

        pro_values: list[float] = []
        fp_pixels: int = 0
        total_normal_pixels: int = 0

        for pred_map, gt_mask in zip(binary_maps, gt_masks):
            # FP / нормальные пиксели
            normal_mask = ~gt_mask
            fp_pixels += int((pred_map & normal_mask).sum())
            total_normal_pixels += int(normal_mask.sum())

            # Связные компоненты GT-аномалий
            labeled, n_components = scipy_label(gt_mask)
            for comp_idx in range(1, n_components + 1):
                component_mask = labeled == comp_idx
                overlap = (pred_map & component_mask).sum() / component_mask.sum()
                pro_values.append(float(overlap))

        fpr = fp_pixels / max(total_normal_pixels, 1)
        pro = float(np.mean(pro_values)) if pro_values else 0.0

        all_fprs.append(fpr)
        all_pros.append(pro)

    fprs = np.array(all_fprs)
    pros = np.array(all_pros)

    # Сортируем по возрастанию FPR для интегрирования
    sort_idx = np.argsort(fprs)
    fprs, pros = fprs[sort_idx], pros[sort_idx]

    # Интегрируем до FPR = 0.3, нормируем (авторский протокол)
    fpr_limit = 0.3
    mask = fprs <= fpr_limit
    if mask.sum() > 1:
        pro_auc = float(np.trapz(pros[mask], fprs[mask]) / fpr_limit)
    else:
        pro_auc = 0.0

    return pro_auc, fprs, pros


# ──────────────────────────────────────────────────────
# Основной класс метрик
# ──────────────────────────────────────────────────────

class Metrics:
    """
    Вычисляет метрики качества модели обнаружения аномалий.

    Поддерживаемые метрики:
      • **Image-level AUROC** — AUC ROC-кривой по скорам на уровне
        изображения. Основная метрика детекции (Section 4 статьи).
      • **Pixel-level AUROC** — AUC ROC-кривой по скорам на уровне
        пикселей. Метрика точности локализации (сегментации).
      • **PRO Score (Per-Region Overlap)** — AUC кривой overlap/FPR
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
                            Больший скор → более аномально.
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

        # ── Image-level AUROC ──────────────────────────────────────
        results.image_auroc = float(roc_auc_score(gt_labels, image_scores))
        image_fpr, image_tpr, _ = roc_curve(gt_labels, image_scores)
        results.image_fpr = image_fpr
        results.image_tpr = image_tpr

        # ── Pixel-level AUROC и PRO ────────────────────────────────
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

            # PRO Score (только для изображений с аномалиями)
            anomaly_idx = gt_labels == 1
            if anomaly_idx.sum() > 0:
                results.pro_score, _, _ = _compute_pro(
                    anomaly_maps[anomaly_idx],
                    gt_masks[anomaly_idx],
                    num_thresh=pro_num_thresh,
                )

        return results

    # ------------------------------------------------------------------
    # Валидация входных данных
    # ------------------------------------------------------------------

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

"""
tests/test_metrics.py — тесты для patchcore/metrics.py

Покрываем:
  • MetricResults:         поля, строковое представление
  • _compute_pro:          алгоритмические свойства PRO-скора
  • Metrics.compute():     image AUROC, pixel AUROC, PRO Score,
                           обработка ошибок и крайних случаев
"""

from __future__ import annotations

import numpy as np
import pytest

from patchcore.metrics import Metrics, MetricResults, _compute_pro


# -----------------------------------------------------------------------------
# MetricResults
# -----------------------------------------------------------------------------

class TestMetricResults:
    """Тесты датакласса MetricResults."""

    def test_default_values(self):
        """Все поля по умолчанию должны быть нулями/пустыми массивами."""
        r = MetricResults()
        assert r.image_auroc == 0.0
        assert r.pixel_auroc == 0.0
        assert r.pro_score == 0.0

    def test_str_contains_metric_names(self):
        """Строковое представление должно содержать названия метрик."""
        r = MetricResults(image_auroc=0.95, pixel_auroc=0.90, pro_score=0.85)
        s = str(r)
        assert "Image-AUROC" in s
        assert "Pixel-AUROC" in s
        assert "PRO Score" in s
        assert "0.9500" in s


# -----------------------------------------------------------------------------
# _compute_pro (внутренняя функция)
# -----------------------------------------------------------------------------

class TestComputePRO:
    """Тесты функции _compute_pro."""

    def _make_data(self, H=16, W=16, n=4):
        """
        Создаёт синтетические карты аномальности и маски.

        Ключевое требование для _compute_pro:
        Функция интегрирует PRO по FPR ∈ [0, 0.3] и возвращает 0.0
        если в этом диапазоне меньше двух точек (условие mask.sum() > 1).

        FPR определяется долей ложно-позитивных нормальных пикселей.
        Чтобы FPR плавно проходил через [0, 0.3], нормальные пиксели
        должны иметь РАЗНЫЕ значения — тогда при разных порогах разное
        число нормальных пикселей будет детектироваться.

        Решение: нормальные пиксели = равномерный шум в [0.0, 0.4],
        аномальные = высокое значение 0.9. При порогах из [0.0, 0.9]
        FPR будет плавно расти от 0 до 1, давая достаточно точек
        в диапазоне [0, 0.3] для интегрирования.
        """
        rng = np.random.default_rng(seed=42)

        gt_masks = np.zeros((n, H, W), dtype=np.uint8)
        gt_masks[:, :H//2, :W//2] = 1  # квадрант — аномалия

        # Нормальные пиксели: равномерный шум [0.0, 0.4] — FPR плавно растёт
        # Аномальные пиксели: фиксированное высокое значение 0.9
        anomaly_maps = rng.uniform(0.0, 0.4, size=(n, H, W)).astype(np.float32)
        anomaly_maps[:, :H//2, :W//2] = 0.9

        return anomaly_maps, gt_masks

    def test_perfect_predictor_gives_high_pro(self):
        """Идеальный предиктор должен давать PRO близкий к 1.0."""
        maps, masks = self._make_data()
        pro, _, _ = _compute_pro(maps, masks, num_thresh=20)
        assert pro > 0.8, f"PRO идеального предиктора = {pro:.3f}"

    def test_zero_predictor_gives_low_pro(self):
        """Нулевые карты аномальности - PRO должен быть низким."""
        _, masks = self._make_data()
        zero_maps = np.zeros_like(masks, dtype=np.float32)
        pro, _, _ = _compute_pro(zero_maps, masks, num_thresh=20)
        # При нулевых картах все патчи = 0, ни одна аномалия не покрыта
        assert pro < 0.2, f"PRO нулевых карт = {pro:.3f}"

    def test_returns_tuple_of_correct_types(self):
        """_compute_pro должна возвращать (float, ndarray, ndarray)."""
        maps, masks = self._make_data()
        pro, fprs, pros = _compute_pro(maps, masks, num_thresh=10)
        assert isinstance(pro, float)
        assert isinstance(fprs, np.ndarray)
        assert isinstance(pros, np.ndarray)

    def test_fprs_in_0_1_range(self):
        """FPR должен быть в диапазоне [0, 1]."""
        maps, masks = self._make_data()
        _, fprs, _ = _compute_pro(maps, masks, num_thresh=20)
        assert (fprs >= 0).all() and (fprs <= 1).all()

    def test_pros_in_0_1_range(self):
        """PRO-значения должны быть в [0, 1]."""
        maps, masks = self._make_data()
        _, _, pros = _compute_pro(maps, masks, num_thresh=20)
        assert (pros >= 0).all() and (pros <= 1).all()


# -----------------------------------------------------------------------------
# Metrics.compute()
# -----------------------------------------------------------------------------

class TestMetricsCompute:
    """Тесты класса Metrics."""

    def test_perfect_image_auroc(self, perfect_detector_data):
        """Идеальный детектор - image AUROC = 1.0."""
        gt_labels, image_scores = perfect_detector_data
        results = Metrics().compute(image_scores=image_scores, gt_labels=gt_labels)
        assert abs(results.image_auroc - 1.0) < 1e-6

    def test_random_image_auroc_near_half(self, random_detector_data):
        """Случайный детектор - image AUROC ≈ 0.5 (±0.15 для N=100)."""
        gt_labels, image_scores = random_detector_data
        results = Metrics().compute(image_scores=image_scores, gt_labels=gt_labels)
        assert 0.35 <= results.image_auroc <= 0.65, (
            f"AUROC случайного детектора = {results.image_auroc:.3f}"
        )

    def test_image_auroc_in_0_1(self, random_detector_data):
        """image AUROC всегда в [0, 1]."""
        gt_labels, image_scores = random_detector_data
        results = Metrics().compute(image_scores=image_scores, gt_labels=gt_labels)
        assert 0.0 <= results.image_auroc <= 1.0

    def test_pixel_auroc_computed_with_maps(self, pixel_metric_data):
        """Pixel AUROC должен вычисляться при наличии карт и масок."""
        gt_labels, image_scores, anomaly_maps, gt_masks = pixel_metric_data
        results = Metrics().compute(
            image_scores=image_scores,
            gt_labels=gt_labels,
            anomaly_maps=anomaly_maps,
            gt_masks=gt_masks,
        )
        assert results.pixel_auroc > 0.5, (
            f"Pixel AUROC = {results.pixel_auroc:.3f}"
        )

    def test_pixel_auroc_zero_without_maps(self, perfect_detector_data):
        """Без карт аномальности pixel AUROC должен остаться 0.0."""
        gt_labels, image_scores = perfect_detector_data
        results = Metrics().compute(image_scores=image_scores, gt_labels=gt_labels)
        assert results.pixel_auroc == 0.0

    def test_pro_score_computed(self, pixel_metric_data):
        """PRO Score должен вычисляться и быть в [0, 1]."""
        gt_labels, image_scores, anomaly_maps, gt_masks = pixel_metric_data
        results = Metrics().compute(
            image_scores=image_scores,
            gt_labels=gt_labels,
            anomaly_maps=anomaly_maps,
            gt_masks=gt_masks,
        )
        assert 0.0 <= results.pro_score <= 1.0

    def test_good_pixel_auroc_for_good_predictor(self, pixel_metric_data):
        """Хороший предиктор должен давать pixel AUROC > 0.8."""
        gt_labels, image_scores, anomaly_maps, gt_masks = pixel_metric_data
        results = Metrics().compute(
            image_scores=image_scores,
            gt_labels=gt_labels,
            anomaly_maps=anomaly_maps,
            gt_masks=gt_masks,
        )
        assert results.pixel_auroc > 0.8

    def test_roc_curves_returned(self, perfect_detector_data):
        """ROC-кривые (fpr, tpr) должны быть непустыми массивами."""
        gt_labels, image_scores = perfect_detector_data
        results = Metrics().compute(image_scores=image_scores, gt_labels=gt_labels)
        assert len(results.image_fpr) > 0
        assert len(results.image_tpr) > 0

    def test_pixel_roc_curves_returned(self, pixel_metric_data):
        """Pixel ROC-кривые должны быть непустыми при наличии карт."""
        gt_labels, image_scores, anomaly_maps, gt_masks = pixel_metric_data
        results = Metrics().compute(
            image_scores=image_scores,
            gt_labels=gt_labels,
            anomaly_maps=anomaly_maps,
            gt_masks=gt_masks,
        )
        assert len(results.pixel_fpr) > 0
        assert len(results.pixel_tpr) > 0

    def test_mismatched_anomaly_maps_and_masks_raises(self):
        """anomaly_maps и gt_masks несовместимой формы - ValueError."""
        gt_labels = np.array([0, 1], dtype=np.int32)
        image_scores = np.array([0.1, 0.9], dtype=np.float32)
        anomaly_maps = np.zeros((2, 16, 16), dtype=np.float32)
        gt_masks = np.zeros((2, 32, 32), dtype=np.uint8)  # другая форма!

        with pytest.raises(ValueError):
            Metrics().compute(
                image_scores=image_scores,
                gt_labels=gt_labels,
                anomaly_maps=anomaly_maps,
                gt_masks=gt_masks,
            )

    def test_wrong_anomaly_maps_ndim_raises(self):
        """2D anomaly_maps (N, H*W) должен вызывать ValueError."""
        gt_labels = np.array([0, 1], dtype=np.int32)
        image_scores = np.array([0.1, 0.9], dtype=np.float32)
        anomaly_maps_2d = np.zeros((2, 256), dtype=np.float32)  # должен быть 3D
        gt_masks = np.zeros((2, 16, 16), dtype=np.uint8)

        with pytest.raises(ValueError):
            Metrics().compute(
                image_scores=image_scores,
                gt_labels=gt_labels,
                anomaly_maps=anomaly_maps_2d,
                gt_masks=gt_masks,
            )

    def test_nonbinary_mask_raises(self):
        """gt_masks с значениями > 1 должен вызывать ValueError."""
        gt_labels = np.array([0, 1], dtype=np.int32)
        image_scores = np.array([0.1, 0.9], dtype=np.float32)
        anomaly_maps = np.zeros((2, 16, 16), dtype=np.float32)
        gt_masks = np.full((2, 16, 16), 2, dtype=np.uint8)  # значения 2, не 0/1

        with pytest.raises(ValueError, match="бинарным"):
            Metrics().compute(
                image_scores=image_scores,
                gt_labels=gt_labels,
                anomaly_maps=anomaly_maps,
                gt_masks=gt_masks,
            )

    def test_metric_results_str(self, perfect_detector_data):
        """str(MetricResults) должен содержать числовые значения."""
        gt_labels, image_scores = perfect_detector_data
        results = Metrics().compute(image_scores=image_scores, gt_labels=gt_labels)
        s = str(results)
        assert "1.0000" in s or "1.000" in s  # image AUROC = 1.0

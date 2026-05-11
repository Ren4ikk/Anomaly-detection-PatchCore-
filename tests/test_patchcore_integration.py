"""
tests/test_patchcore_integration.py — интеграционные тесты для patchcore/patchcore.py

Проверяем сквозной пайплайн:
  fit() - predict() - save() - load() - predict()

Эти тесты медленнее unit-тестов, так как запускают полный forward через
WideResNet-50. Помечены маркером @pytest.mark.integration — можно
запускать отдельно: pytest -m integration

Для CI используется минимальный датасет (5 изображений, batch_size=2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from patchcore.patchcore import PatchCore, PredictionResult


pytestmark = pytest.mark.integration


# -----------------------------------------------------------------------------
# Фикстуры
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted_model(tmp_path_factory) -> tuple[PatchCore, Path]:
    """
    Обученная модель PatchCore на синтетических изображениях.
    scope="module" — fit() запускается только один раз для всех тестов модуля.

    Возвращает (model, train_dir) — train_dir нужен для проверок.
    """
    import numpy as np
    from PIL import Image

    # Создаём временную директорию с 5 PNG-изображениями
    train_dir = tmp_path_factory.mktemp("train_data")
    for i in range(5):
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        Image.fromarray(arr, "RGB").save(train_dir / f"img_{i:03d}.png")

    model = PatchCore(
        device="cpu",
        coreset_ratio=0.5,   # большой ratio для маленького датасета
        batch_size=2,
        num_workers=0,       # num_workers=0 обязателен в pytest (форки процессов)
    )
    model.fit(str(train_dir))
    # Вычисляем диапазон скоров и порог сразу после fit() —
    # score_min, score_max, threshold заполняются реальными значениями.
    model.compute_score_range(str(train_dir))
    return model, train_dir


@pytest.fixture
def test_image() -> torch.Tensor:
    """Одно тестовое изображение (3, 224, 224)."""
    torch.manual_seed(99)
    return torch.randn(1, 3, 224, 224)


# -----------------------------------------------------------------------------
# После fit()
# -----------------------------------------------------------------------------

class TestPatchCoreFit:
    """Тесты состояния модели после fit() и compute_score_range()."""

    def test_is_fitted_after_fit(self, fitted_model):
        """После fit() индекс должен быть построен."""
        model, _ = fitted_model
        assert model.nn_index.is_fitted

    def test_spatial_size_set(self, fitted_model):
        """_spatial_size должен быть заполнен и равен (28, 28)."""
        model, _ = fitted_model
        assert model._spatial_size is not None
        assert model._spatial_size == (28, 28)

    def test_score_range_valid(self, fitted_model):
        """
        После compute_score_range() score_min < score_max.
        Оба значения — сырые L2-расстояния, заведомо > 0.
        """
        model, _ = fitted_model
        assert model.score_min < model.score_max, (
            f"score_min={model.score_min:.4f} должен быть < score_max={model.score_max:.4f}"
        )
        assert model.score_min >= 0.0, "score_min — L2-расстояние, не может быть < 0"
        assert model.score_max > 0.0

    def test_threshold_above_score_min(self, fitted_model):
        """threshold должен быть выше score_min (порог = p99 + 3σ по train-скорам)."""
        model, _ = fitted_model
        assert model.threshold > model.score_min, (
            f"threshold={model.threshold:.4f} <= score_min={model.score_min:.4f}"
        )

    def test_score_range_finite(self, fitted_model):
        """score_min, score_max, threshold должны быть конечными числами."""
        model, _ = fitted_model
        assert np.isfinite(model.score_min)
        assert np.isfinite(model.score_max)
        assert np.isfinite(model.threshold)

    def test_repr_shows_fitted(self, fitted_model):
        """__repr__ должен показывать 'fitted'."""
        model, _ = fitted_model
        assert "fitted" in repr(model)
        assert "not fitted" not in repr(model)


# -----------------------------------------------------------------------------
# predict()
# -----------------------------------------------------------------------------

class TestPatchCorePredict:
    """Тесты инференса predict() и predict_single()."""

    def test_predict_returns_list_of_results(self, fitted_model, test_image):
        """predict() должен возвращать список PredictionResult."""
        model, _ = fitted_model
        results = model.predict(test_image)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], PredictionResult)

    def test_predict_single_returns_result(self, fitted_model, test_image):
        """predict_single() должен возвращать один PredictionResult."""
        model, _ = fitted_model
        result = model.predict_single(test_image.squeeze(0))
        assert isinstance(result, PredictionResult)

    def test_image_score_is_finite(self, fitted_model, test_image):
        """
        image_score — re-weighted L2-расстояние наиболее аномального патча.
        Должен быть конечным и неотрицательным (weight ∈ [0,1], s_star ≥ 0).
        """
        model, _ = fitted_model
        result = model.predict_single(test_image.squeeze(0))
        assert np.isfinite(result.image_score)
        assert result.image_score >= 0.0

    def test_anomaly_map_shape(self, fitted_model, test_image):
        """anomaly_map должна быть (224, 224)."""
        model, _ = fitted_model
        result = model.predict_single(test_image.squeeze(0))
        assert result.anomaly_map.shape == (224, 224)

    def test_anomaly_map_dtype(self, fitted_model, test_image):
        """
        anomaly_map должна быть float32.
        Значения — сырые L2-расстояния, нормализация намеренно убрана
        (нормализация per-image нарушает сравнимость скоров между кадрами).
        """
        model, _ = fitted_model
        result = model.predict_single(test_image.squeeze(0))
        assert result.anomaly_map.dtype == np.float32

    def test_anomaly_map_finite(self, fitted_model, test_image):
        """
        anomaly_map содержит сырые L2-расстояния (без нормализации в [0,1]):
        не должно быть NaN/Inf, все значения >= 0.
        """
        model, _ = fitted_model
        result = model.predict_single(test_image.squeeze(0))
        assert np.isfinite(result.anomaly_map).all()
        assert (result.anomaly_map >= 0).all(), "L2-расстояния всегда неотрицательны"

    def test_patch_scores_shape(self, fitted_model, test_image):
        """patch_scores должны иметь форму (28*28,) = (784,)."""
        model, _ = fitted_model
        result = model.predict_single(test_image.squeeze(0))
        assert result.patch_scores.shape == (28 * 28,)

    def test_spatial_size_in_result(self, fitted_model, test_image):
        """spatial_size в результате должен быть (28, 28)."""
        model, _ = fitted_model
        result = model.predict_single(test_image.squeeze(0))
        assert result.spatial_size == (28, 28)

    def test_predict_batch_of_2(self, fitted_model):
        """Батч из 2 изображений - 2 результата."""
        model, _ = fitted_model
        images = torch.randn(2, 3, 224, 224)
        results = model.predict(images)
        assert len(results) == 2

    def test_predict_without_fit_raises(self):
        """predict() без fit() должен вызывать RuntimeError."""
        model = PatchCore(device="cpu")
        image = torch.randn(1, 3, 224, 224)
        with pytest.raises(RuntimeError):
            model.predict(image)

    def test_predict_3d_input(self, fitted_model, test_image):
        """predict() с 3D входом (без batch-размерности) должен работать."""
        model, _ = fitted_model
        # predict_single принимает (3, H, W)
        result = model.predict_single(test_image[0])  # убираем batch dim
        assert isinstance(result, PredictionResult)


# -----------------------------------------------------------------------------
# save() / load()
# -----------------------------------------------------------------------------

class TestPatchCoreSaveLoad:
    """Тесты сохранения и загрузки модели."""

    def test_save_creates_file(self, fitted_model, tmp_path):
        """save() должен создать файл .pt."""
        model, _ = fitted_model
        save_path = tmp_path / "model.pt"
        model.save(str(save_path))
        assert save_path.exists()

    def test_save_without_fit_raises(self, tmp_path):
        """save() без fit() должен вызывать RuntimeError."""
        model = PatchCore(device="cpu")
        with pytest.raises(RuntimeError):
            model.save(str(tmp_path / "model.pt"))

    def test_load_restores_fitted_state(self, fitted_model, tmp_path):
        """После load() модель должна быть в состоянии fitted."""
        model, _ = fitted_model
        save_path = str(tmp_path / "model.pt")
        model.save(save_path)

        new_model = PatchCore(device="cpu")
        assert not new_model.nn_index.is_fitted
        new_model.load(save_path)
        assert new_model.nn_index.is_fitted

    def test_load_restores_spatial_size(self, fitted_model, tmp_path):
        """После load() _spatial_size должен восстановиться."""
        model, _ = fitted_model
        save_path = str(tmp_path / "spatial_test.pt")
        model.save(save_path)

        new_model = PatchCore(device="cpu")
        new_model.load(save_path)
        assert new_model._spatial_size == (28, 28)

    def test_save_load_predict_consistent(self, fitted_model, tmp_path, test_image):
        """
        Результаты predict() до и после save()/load() должны совпадать.
        Это критически важно: загруженная модель должна давать те же скоры.
        """
        model, _ = fitted_model
        save_path = str(tmp_path / "consistency.pt")
        model.save(save_path)

        # Результат оригинальной модели
        result_orig = model.predict_single(test_image[0])

        # Загружаем в новую модель
        new_model = PatchCore(device="cpu", num_workers=0)
        new_model.load(save_path)
        result_loaded = new_model.predict_single(test_image[0])

        # image_score должен совпасть (допускаем незначительную погрешность float)
        assert abs(result_orig.image_score - result_loaded.image_score) < 1e-4, (
            f"image_score: оригинал={result_orig.image_score:.6f}, "
            f"загруженный={result_loaded.image_score:.6f}"
        )
        # anomaly_map должна совпасть
        assert np.allclose(
            result_orig.anomaly_map,
            result_loaded.anomaly_map,
            atol=1e-5,
        ), "anomaly_map отличается после save/load"


# -----------------------------------------------------------------------------
# _build_anomaly_map (приватный, но важный)
# -----------------------------------------------------------------------------

class TestBuildAnomalyMap:
    """Тесты приватного метода _build_anomaly_map."""

    def test_output_shape(self, fitted_model):
        """Карта аномальности всегда (224, 224)."""
        model, _ = fitted_model
        patch_scores = np.ones(28 * 28, dtype=np.float32)
        amap = model._build_anomaly_map(patch_scores, spatial_size=(28, 28))
        assert amap.shape == (224, 224)

    def test_output_dtype(self, fitted_model):
        """Тип данных должен быть float32."""
        model, _ = fitted_model
        patch_scores = np.random.rand(28 * 28).astype(np.float32)
        amap = model._build_anomaly_map(patch_scores, spatial_size=(28, 28))
        assert amap.dtype == np.float32

    def test_gaussian_smoothing_applied(self, fitted_model):
        """
        Гауссово сглаживание должно уменьшать дисперсию карты.
        Импульс в центре после сглаживания должен размазаться.
        """
        model, _ = fitted_model
        # Импульс в центре 28×28
        patch_scores = np.zeros(28 * 28, dtype=np.float32)
        patch_scores[28 * 14 + 14] = 1.0  # центральный патч

        amap = model._build_anomaly_map(patch_scores, spatial_size=(28, 28))

        # После сглаживания максимум должен остаться в центре
        max_pos = np.unravel_index(amap.argmax(), amap.shape)
        center = (112, 112)  # центр 224×224
        dist = ((max_pos[0] - center[0]) ** 2 + (max_pos[1] - center[1]) ** 2) ** 0.5
        assert dist < 20, f"Максимум после сглаживания далеко от центра: {max_pos}"

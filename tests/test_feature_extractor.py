"""
tests/test_feature_extractor.py — тесты для patchcore/feature_extractor.py

Покрываем:
  • _LocalAggregation:  форма выхода, сохранение разрешения при stride=1
  • FeatureExtractor:   форма выхода, dtype, детерминированность,
                        корректность spatial_size, работа с разными batch_size
"""

from __future__ import annotations

import pytest
import torch

from patchcore.feature_extractor import (
    FeatureExtractor,
    _LocalAggregation,
    _PATCH_SIZE,
    _STRIDE,
    _TARGET_DIM,
)

# Пространственный размер карты признаков для входа 224×224
EXPECTED_SPATIAL = 28
EXPECTED_FEAT_DIM = _TARGET_DIM  # 1024


# -----------------------------------------------------------------------------
# _LocalAggregation
# -----------------------------------------------------------------------------

class TestLocalAggregation:
    """Тесты модуля локальной агрегации патчей."""

    def test_preserves_spatial_resolution(self):
        """
        При patch_size=3, stride=1 разрешение H×W должно сохраняться.
        Это критично: патч-карта должна быть той же размерности, что вход.
        """
        agg = _LocalAggregation(patch_size=3, stride=1)
        x = torch.randn(2, 8, 14, 14)
        out = agg(x)
        assert out.shape == x.shape, f"Ожидали {x.shape}, получили {out.shape}"

    def test_output_dtype(self):
        """Тип данных не должен меняться при агрегации."""
        agg = _LocalAggregation()
        x = torch.randn(1, 4, 10, 10)
        out = agg(x)
        assert out.dtype == x.dtype

    def test_different_channel_sizes(self):
        """Агрегация должна работать с любым числом каналов."""
        agg = _LocalAggregation(patch_size=3, stride=1)
        for C in [64, 128, 512, 1024]:
            x = torch.randn(1, C, 14, 14)
            out = agg(x)
            assert out.shape == x.shape

    def test_stride_2_reduces_resolution(self):
        """При stride=2 разрешение должно уменьшиться примерно вдвое."""
        agg = _LocalAggregation(patch_size=3, stride=2)
        x = torch.randn(1, 8, 28, 28)
        out = agg(x)
        # При padding=1, stride=2: H_out = (28 + 2*1 - 3)//2 + 1 = 14
        assert out.shape[2] == 14
        assert out.shape[3] == 14

    def test_uniform_input_produces_uniform_output(self):
        """
        Для постоянного входа агрегация (среднее) должна давать тот же результат.
        """
        agg = _LocalAggregation(patch_size=3, stride=1)
        x = torch.ones(1, 4, 10, 10) * 5.0
        out = agg(x)
        # Среднее константного поля = та же константа (без учёта граничных эффектов)
        # Проверяем центральный элемент (без padding-влияния)
        center = out[0, :, 5, 5]
        assert torch.allclose(center, torch.ones(4) * 5.0, atol=1e-5)


# -----------------------------------------------------------------------------
# FeatureExtractor — fixtures на уровне модуля (scope="module") для скорости
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    """
    Инициализируем FeatureExtractor один раз на весь модуль тестов.
    WideResNet-50 скачивается с torchvision (≈270 MB) — кешируется локально.
    В CI с первой загрузкой этот тест будет медленнее; последующие — быстро.
    """
    return FeatureExtractor(device="cpu")


@pytest.fixture(scope="module")
def sample_images() -> torch.Tensor:
    """2 изображения 224×224 — минимальный батч для тестирования."""
    torch.manual_seed(42)
    return torch.randn(2, 3, 224, 224)


class TestFeatureExtractor:
    """Тесты класса FeatureExtractor."""

    def test_extract_output_shape(self, extractor: FeatureExtractor, sample_images: torch.Tensor):
        """
        extract() должен вернуть матрицу (B * H_out * W_out, target_dim).
        Для B=2, H_out=W_out=28: 2 * 784 = 1568 строк, 1024 столбца.
        """
        features = extractor.extract(sample_images)
        B = sample_images.shape[0]
        expected_rows = B * EXPECTED_SPATIAL * EXPECTED_SPATIAL
        assert features.shape == (expected_rows, EXPECTED_FEAT_DIM), (
            f"Ожидали ({expected_rows}, {EXPECTED_FEAT_DIM}), "
            f"получили {features.shape}"
        )

    def test_extract_dtype_float32(self, extractor: FeatureExtractor, sample_images: torch.Tensor):
        """Признаки должны быть float32."""
        features = extractor.extract(sample_images)
        assert features.dtype == torch.float32

    def test_extract_no_nan_inf(self, extractor: FeatureExtractor, sample_images: torch.Tensor):
        """В матрице признаков не должно быть NaN или Inf."""
        features = extractor.extract(sample_images)
        assert torch.isfinite(features).all(), "Найдены NaN или Inf в признаках"

    def test_extract_with_spatial_info_shapes(
        self, extractor: FeatureExtractor, sample_images: torch.Tensor
    ):
        """
        extract_with_spatial_info() должен возвращать признаки
        и корректный spatial_size = (28, 28).
        """
        features, spatial_size = extractor.extract_with_spatial_info(sample_images)
        assert spatial_size == (EXPECTED_SPATIAL, EXPECTED_SPATIAL)
        assert features.shape == (
            sample_images.shape[0] * EXPECTED_SPATIAL * EXPECTED_SPATIAL,
            EXPECTED_FEAT_DIM,
        )

    def test_extract_deterministic(self, extractor: FeatureExtractor):
        """
        Для одного и того же входа extract() должен давать одинаковый результат.
        (backbone заморожен, нет dropout/BN в train mode)
        """
        extractor.eval()
        torch.manual_seed(1)
        imgs = torch.randn(1, 3, 224, 224)
        f1 = extractor.extract(imgs)
        f2 = extractor.extract(imgs)
        assert torch.allclose(f1, f2), "Результаты extract() недетерминированы"

    def test_extract_batch_size_1(self, extractor: FeatureExtractor):
        """Одно изображение в батче должно обрабатываться корректно."""
        img = torch.randn(1, 3, 224, 224)
        features = extractor.extract(img)
        assert features.shape == (EXPECTED_SPATIAL * EXPECTED_SPATIAL, EXPECTED_FEAT_DIM)

    def test_extract_batch_size_4(self, extractor: FeatureExtractor):
        """Батч из 4 изображений: 4 * 784 = 3136 строк признаков."""
        imgs = torch.randn(4, 3, 224, 224)
        features = extractor.extract(imgs)
        assert features.shape == (4 * EXPECTED_SPATIAL * EXPECTED_SPATIAL, EXPECTED_FEAT_DIM)

    def test_features_differ_for_different_images(self, extractor: FeatureExtractor):
        """Разные изображения должны давать разные признаки."""
        img1 = torch.randn(1, 3, 224, 224)
        img2 = torch.randn(1, 3, 224, 224)
        f1 = extractor.extract(img1)
        f2 = extractor.extract(img2)
        assert not torch.allclose(f1, f2), (
            "Признаки двух случайных изображений совпали — что-то пошло не так"
        )

    def test_repr(self, extractor: FeatureExtractor):
        """__repr__ должен содержать backbone_name, слои и target_dim."""
        r = repr(extractor)
        # repr возвращает backbone_name как строку: "backbone=wide_resnet50_2, layers=..."
        assert "wide_resnet50_2" in r
        assert "layer2" in r
        assert "layer3" in r
        assert "1024" in r  # target_dim

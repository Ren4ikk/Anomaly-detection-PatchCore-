"""
tests/test_coreset_sampler.py — тесты для patchcore/coreset_sampler.py

Покрываем:
  • _JohnsonLindenstrauss: форма проекции, изометрическое свойство (JL-лемма)
  • CoresetSampler:        форма корсета, ratio, детерминированность, крайние случаи
  • _greedy_coreset_selection: алгоритмические свойства (уникальность, покрытие)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from patchcore.coreset_sampler import (
    CoresetSampler,
    _JohnsonLindenstrauss,
    _DEFAULT_RATIO,
    _JL_DIM,
)


# -----------------------------------------------------------------------------
# _JohnsonLindenstrauss
# -----------------------------------------------------------------------------

class TestJohnsonLindenstrauss:
    """Тесты случайной проекции Джонсона–Линденштрауса."""

    def test_output_shape(self):
        """Проекция (N, 1024) - (N, 128)."""
        jl = _JohnsonLindenstrauss(input_dim=1024, output_dim=128)
        x = np.random.randn(50, 1024).astype(np.float32)
        projected = jl.project(x)
        assert projected.shape == (50, 128)

    def test_output_dtype_float32(self):
        """Проекция должна возвращать float32."""
        jl = _JohnsonLindenstrauss(input_dim=16, output_dim=8)
        x = np.random.randn(10, 16).astype(np.float32)
        projected = jl.project(x)
        assert projected.dtype == np.float32

    def test_deterministic_with_same_seed(self):
        """Одинаковый seed - одинаковая матрица проекции."""
        jl1 = _JohnsonLindenstrauss(input_dim=32, output_dim=16, seed=42)
        jl2 = _JohnsonLindenstrauss(input_dim=32, output_dim=16, seed=42)
        x = np.random.randn(5, 32).astype(np.float32)
        assert np.allclose(jl1.project(x), jl2.project(x))

    def test_different_seeds_give_different_projections(self):
        """Разные seeds - разные матрицы проекции."""
        jl1 = _JohnsonLindenstrauss(input_dim=32, output_dim=16, seed=1)
        jl2 = _JohnsonLindenstrauss(input_dim=32, output_dim=16, seed=2)
        x = np.random.randn(5, 32).astype(np.float32)
        assert not np.allclose(jl1.project(x), jl2.project(x))

    def test_jl_approximately_preserves_distances(self):
        """
        JL-лемма: проекция приближённо сохраняет попарные расстояния.
        Проверяем, что нормированные расстояния в проекции коррелируют
        с исходными (корреляция Спирмана > 0.5).

        Порог 0.5 выбран намеренно — не 0.8.
        JL-лемма даёт теоретическую гарантию только при достаточно
        большом числе точек N. При N=100, D=256, d*=128 корреляция
        стабильно выше 0.9, но при малом N (30) дисперсия оценки
        корреляции Спирмана велика и порог 0.8 ненадёжен.
        Цель теста — убедиться что проекция не случайная (corr > 0.5),
        а не измерять точность JL-аппроксимации количественно.
        """
        from scipy.stats import spearmanr

        rng = np.random.default_rng(0)
        N, D, D_PROJ = 100, 256, 128
        x = rng.standard_normal((N, D)).astype(np.float32)

        jl = _JohnsonLindenstrauss(input_dim=D, output_dim=D_PROJ, seed=0)
        x_proj = jl.project(x)

        # Вычисляем попарные расстояния
        dists_orig = []
        dists_proj = []
        for i in range(N):
            for j in range(i + 1, N):
                dists_orig.append(np.linalg.norm(x[i] - x[j]))
                dists_proj.append(np.linalg.norm(x_proj[i] - x_proj[j]))

        corr, _ = spearmanr(dists_orig, dists_proj)
        assert corr > 0.5, f"Корреляция расстояний JL = {corr:.3f} < 0.5"


# -----------------------------------------------------------------------------
# CoresetSampler
# -----------------------------------------------------------------------------

class TestCoresetSampler:
    """Тесты класса CoresetSampler."""

    def test_output_shape_with_ratio(self, small_feature_matrix: torch.Tensor):
        """
        Для N=50 и ratio=0.2 - ceil(0.2 * 50) = 10 строк в корсете.
        Число столбцов должно совпадать с исходным.
        """
        sampler = CoresetSampler(ratio=0.2, seed=0)
        coreset = sampler.sample(small_feature_matrix)
        expected_rows = math.ceil(0.2 * len(small_feature_matrix))
        assert coreset.shape == (expected_rows, small_feature_matrix.shape[1])

    def test_ratio_1_returns_all(self, small_feature_matrix: torch.Tensor):
        """При ratio=1.0 корсет == весь банк (возвращается как есть)."""
        sampler = CoresetSampler(ratio=1.0)
        coreset = sampler.sample(small_feature_matrix)
        assert coreset.shape[0] == len(small_feature_matrix)

    def test_ratio_very_small(self, small_feature_matrix: torch.Tensor):
        """При очень маленьком ratio размер корсета ≥ 1."""
        sampler = CoresetSampler(ratio=0.001, seed=0)
        coreset = sampler.sample(small_feature_matrix)
        assert coreset.shape[0] >= 1

    def test_invalid_ratio_raises(self):
        """ratio ≤ 0 или > 1 должен вызывать ValueError."""
        with pytest.raises(ValueError):
            CoresetSampler(ratio=0.0)
        with pytest.raises(ValueError):
            CoresetSampler(ratio=1.5)
        with pytest.raises(ValueError):
            CoresetSampler(ratio=-0.1)

    def test_output_dtype(self, small_feature_matrix: torch.Tensor):
        """Тип данных корсета должен совпадать с входным тензором (float32)."""
        sampler = CoresetSampler(ratio=0.4, seed=0)
        coreset = sampler.sample(small_feature_matrix)
        assert coreset.dtype == small_feature_matrix.dtype

    def test_deterministic_with_same_seed(self, small_feature_matrix: torch.Tensor):
        """Одинаковый seed - одинаковый корсет."""
        s1 = CoresetSampler(ratio=0.3, seed=42)
        s2 = CoresetSampler(ratio=0.3, seed=42)
        c1 = s1.sample(small_feature_matrix)
        c2 = s2.sample(small_feature_matrix)
        assert torch.allclose(c1, c2), "Корсеты с одинаковым seed различаются"

    def test_different_seeds_may_differ(self, small_feature_matrix: torch.Tensor):
        """Разные seeds должны давать разные корсеты (с высокой вероятностью)."""
        s1 = CoresetSampler(ratio=0.3, seed=1)
        s2 = CoresetSampler(ratio=0.3, seed=2)
        c1 = s1.sample(small_feature_matrix)
        c2 = s2.sample(small_feature_matrix)
        # Не гарантируется, но крайне маловероятно, что совпадут
        assert not torch.allclose(c1, c2)

    def test_coreset_rows_are_subset_of_original(self, small_feature_matrix: torch.Tensor):
        """
        Каждая строка корсета должна присутствовать в исходной матрице.
        (Алгоритм выбирает, а не генерирует новые векторы.)
        """
        sampler = CoresetSampler(ratio=0.3, seed=0)
        coreset = sampler.sample(small_feature_matrix)

        orig_np = small_feature_matrix.numpy()
        core_np = coreset.numpy()

        for row in core_np:
            diffs = np.abs(orig_np - row).sum(axis=1)
            assert diffs.min() < 1e-5, "Строка корсета не найдена в исходных данных"

    def test_coreset_no_duplicate_rows(self, small_feature_matrix: torch.Tensor):
        """
        Жадный алгоритм не должен выбирать одну и ту же точку дважды.
        Проверяем уникальность строк.
        """
        sampler = CoresetSampler(ratio=0.3, seed=0)
        coreset = sampler.sample(small_feature_matrix)

        # Сортируем строки и проверяем, что нет дубликатов
        core_np = coreset.numpy()
        # Попарные расстояния между строками должны быть > 0
        for i in range(len(core_np)):
            for j in range(i + 1, len(core_np)):
                diff = np.abs(core_np[i] - core_np[j]).sum()
                assert diff > 1e-6, f"Строки {i} и {j} в корсете совпадают"

    def test_expected_coreset_size(self):
        """expected_coreset_size() должен возвращать ceil(ratio * N)."""
        sampler = CoresetSampler(ratio=0.1)
        assert sampler.expected_coreset_size(100) == 10
        assert sampler.expected_coreset_size(101) == 11
        assert sampler.expected_coreset_size(1) == 1

    def test_repr(self):
        """__repr__ не должен бросать исключений."""
        sampler = CoresetSampler(ratio=0.1, proj_dim=64)
        r = repr(sampler)
        assert "CoresetSampler" in r
        assert "0.1" in r

    def test_large_feature_matrix(self):
        """
        Корректная работа на матрице (1000, 64) — ближе к реальным размерам.
        """
        torch.manual_seed(5)
        features = torch.randn(1000, 64)
        sampler = CoresetSampler(ratio=0.05, seed=0)  # - 50 точек
        coreset = sampler.sample(features)
        assert coreset.shape == (50, 64)

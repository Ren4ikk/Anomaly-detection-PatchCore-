"""
tests/test_nearest_neighbor_index.py — тесты для patchcore/nearest_neighbor_index.py

Покрываем:
  • NearestNeighborIndex.fit():    построение индекса, is_fitted, memory_bank
  • NearestNeighborIndex.search(): форма, dtype, корректность поиска (точные случаи)
  • Граничные случаи:              k > N, вызов без fit(), повторный fit()
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from patchcore.nearest_neighbor_index import NearestNeighborIndex


# -----------------------------------------------------------------------------
# Вспомогательные данные
# -----------------------------------------------------------------------------

@pytest.fixture
def fitted_index(memory_bank_for_index: torch.Tensor) -> NearestNeighborIndex:
    """Индекс, уже заполненный тестовым банком памяти."""
    idx = NearestNeighborIndex(use_gpu=False)
    idx.fit(memory_bank_for_index)
    return idx


# -----------------------------------------------------------------------------
# fit()
# -----------------------------------------------------------------------------

class TestNearestNeighborIndexFit:
    """Тесты метода fit()."""

    def test_is_fitted_after_fit(self, memory_bank_for_index: torch.Tensor):
        """После fit() is_fitted должен быть True."""
        idx = NearestNeighborIndex()
        assert not idx.is_fitted
        idx.fit(memory_bank_for_index)
        assert idx.is_fitted

    def test_memory_bank_stored(self, memory_bank_for_index: torch.Tensor):
        """Банк памяти должен сохраняться и быть доступен через .memory_bank."""
        idx = NearestNeighborIndex()
        idx.fit(memory_bank_for_index)
        stored = idx.memory_bank
        assert stored.shape == memory_bank_for_index.shape
        assert torch.allclose(stored.cpu(), memory_bank_for_index.cpu())

    def test_memory_bank_before_fit_raises(self):
        """Обращение к memory_bank до fit() должно вызывать RuntimeError."""
        idx = NearestNeighborIndex()
        with pytest.raises(RuntimeError):
            _ = idx.memory_bank

    def test_refit_updates_index(self):
        """Повторный вызов fit() должен заменить старый индекс."""
        torch.manual_seed(0)
        bank1 = torch.randn(20, 8)
        bank2 = torch.randn(50, 8)

        idx = NearestNeighborIndex()
        idx.fit(bank1)
        assert idx.memory_bank.shape[0] == 20

        idx.fit(bank2)
        assert idx.memory_bank.shape[0] == 50


# -----------------------------------------------------------------------------
# search()
# -----------------------------------------------------------------------------

class TestNearestNeighborIndexSearch:
    """Тесты метода search()."""

    def test_search_without_fit_raises(self):
        """Вызов search() без fit() должен вызывать RuntimeError."""
        idx = NearestNeighborIndex()
        queries = torch.randn(5, 32)
        with pytest.raises(RuntimeError):
            idx.search(queries, k=1)

    def test_search_output_shapes(
        self,
        fitted_index: NearestNeighborIndex,
        memory_bank_for_index: torch.Tensor,
    ):
        """
        search() с k=3 должен возвращать (N_queries, 3) для distances и indices.
        """
        queries = torch.randn(10, memory_bank_for_index.shape[1])
        distances, indices = fitted_index.search(queries, k=3)
        assert distances.shape == (10, 3)
        assert indices.shape == (10, 3)

    def test_search_k1_distances_nonnegative(
        self,
        fitted_index: NearestNeighborIndex,
        memory_bank_for_index: torch.Tensor,
    ):
        """L2-расстояния должны быть неотрицательными."""
        queries = torch.randn(20, memory_bank_for_index.shape[1])
        distances, _ = fitted_index.search(queries, k=1)
        assert (distances >= 0).all(), "Найдены отрицательные расстояния"

    def test_search_exact_match_gives_zero_distance(
        self,
        memory_bank_for_index: torch.Tensor,
    ):
        """
        Поиск по вектору, который уже есть в банке памяти,
        должен давать нулевое расстояние (точное совпадение).
        """
        idx = NearestNeighborIndex()
        idx.fit(memory_bank_for_index)

        # Берём первый вектор из банка как запрос
        query = memory_bank_for_index[0:1]  # (1, D)
        distances, indices = idx.search(query, k=1)

        assert distances[0, 0] < 1e-5, (
            f"Расстояние для точного совпадения = {distances[0, 0]:.6f}, ожидали ≈0"
        )
        assert indices[0, 0] == 0

    def test_search_nearest_is_truly_nearest(self):
        """
        Явно конструируем пространство: один «близкий» и один «далёкий» вектор.
        Ближайший сосед должен быть правильным.
        """
        # Банк из двух точек: [0, 0] и [10, 0]
        bank = torch.tensor([[0.0, 0.0], [10.0, 0.0]])
        idx = NearestNeighborIndex()
        idx.fit(bank)

        # Запрос [0.5, 0] — должен найти точку [0, 0]
        query = torch.tensor([[0.5, 0.0]])
        distances, indices = idx.search(query, k=1)
        assert indices[0, 0] == 0, f"Ожидали индекс 0, получили {indices[0, 0]}"

        # Запрос [9.5, 0] — должен найти точку [10, 0]
        query2 = torch.tensor([[9.5, 0.0]])
        distances2, indices2 = idx.search(query2, k=1)
        assert indices2[0, 0] == 1, f"Ожидали индекс 1, получили {indices2[0, 0]}"

    def test_search_distances_sorted(
        self,
        fitted_index: NearestNeighborIndex,
        memory_bank_for_index: torch.Tensor,
    ):
        """
        При k > 1 расстояния для каждого запроса должны быть
        отсортированы по возрастанию.
        """
        queries = torch.randn(5, memory_bank_for_index.shape[1])
        distances, _ = fitted_index.search(queries, k=5)
        for i in range(distances.shape[0]):
            row = distances[i]
            assert (row[1:] >= row[:-1]).all(), (
                f"Расстояния для запроса {i} не отсортированы: {row}"
            )

    def test_search_indices_in_valid_range(
        self,
        fitted_index: NearestNeighborIndex,
        memory_bank_for_index: torch.Tensor,
    ):
        """Индексы соседей должны быть в диапазоне [0, N_bank)."""
        N_bank = len(memory_bank_for_index)
        queries = torch.randn(10, memory_bank_for_index.shape[1])
        _, indices = fitted_index.search(queries, k=3)
        assert (indices >= 0).all()
        assert (indices < N_bank).all()

    def test_repr(self):
        """__repr__ работает до и после fit()."""
        idx = NearestNeighborIndex()
        r_before = repr(idx)
        assert "not fitted" in r_before

        bank = torch.randn(10, 8)
        idx.fit(bank)
        r_after = repr(idx)
        assert "not fitted" not in r_after

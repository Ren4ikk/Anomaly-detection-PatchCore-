"""
Этап 4 — Поиск и Инференс: Индекс ближайших соседей.

Класс NearestNeighborIndex — обёртка над FAISS IndexFlatL2,
хранящая корсет M_C и обеспечивающая быстрый поиск k ближайших
соседей при инференсе.
"""

from __future__ import annotations

import numpy as np
import torch
import faiss
from numpy.typing import NDArray


class NearestNeighborIndex:
    """
    Хранит корсет M_C и выполняет быстрый kNN-поиск через FAISS IndexFlatL2.

    IndexFlatL2 — точный брутфорс L2-поиск без аппроксимаций.

    Args:
        use_gpu: Переносить ли индекс на GPU (ускоряет на больших банках).
    """

    def __init__(self, use_gpu: bool = False) -> None:
        self.use_gpu = use_gpu
        self._index: faiss.IndexFlatL2 | None = None
        self._memory_bank: torch.Tensor | None = None

    # Публичный API

    def fit(self, coreset: torch.Tensor) -> None:
        """
        Строит FAISS-индекс из корсета M_C.

        Args:
            coreset: Матрица корсета (N_coreset, D) — выход CoresetSampler.
        """
        # Сохраняем оригинальный тензор для возможного дальнейшего использования
        self._memory_bank = coreset.cpu()

        vectors = coreset.detach().cpu().numpy().astype(np.float32)
        d = vectors.shape[1]

        index = faiss.IndexFlatL2(d)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(np.ascontiguousarray(vectors))
        self._index = index

    def search(
        self,
        queries: torch.Tensor,
        k: int = 1,
    ) -> tuple[NDArray, NDArray]:
        """
        Ищет k ближайших соседей в M_C для каждого запросного вектора.

        Args:
            queries: Матрица запросов (N_queries, D).
            k:       Число ближайших соседей.

        Returns:
            distances: (N_queries, k) — L2-расстояния (не квадраты).
            indices:   (N_queries, k) — индексы соседей в M_C.

        Raises:
            RuntimeError: Если fit() не был вызван.
        """
        if self._index is None:
            raise RuntimeError("Сначала вызовите fit() для построения индекса.")

        queries_np = np.ascontiguousarray(
            queries.detach().cpu().numpy(), dtype=np.float32
        )
        # FAISS возвращает квадраты расстояний
        distances_sq, indices = self._index.search(queries_np, k)
        distances = np.sqrt(distances_sq.clip(min=0.0))
        return distances, indices

    @property
    def memory_bank(self) -> torch.Tensor:
        """Возвращает сохранённый корсет M_C."""
        if self._memory_bank is None:
            raise RuntimeError("Индекс пуст. Вызовите fit() сначала.")
        return self._memory_bank

    @property
    def is_fitted(self) -> bool:
        """True если индекс построен."""
        return self._index is not None

    def __repr__(self) -> str:
        size = self._memory_bank.shape if self._memory_bank is not None else "not fitted"
        return f"{self.__class__.__name__}(memory_bank={size}, use_gpu={self.use_gpu})"

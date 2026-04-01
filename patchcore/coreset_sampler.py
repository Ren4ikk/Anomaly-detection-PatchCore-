"""
Этап 3 — Оптимизация памяти (Coreset Subsampling).

Реализует класс CoresetSampler, который сжимает банк патч-признаков M
до репрезентативного подмножества M_C через жадный алгоритм minimax
facility location (Algorithm 1 из статьи).

Математика из статьи (Section 3.2, формула 5):

  M*_C = argmin_{M_C ⊂ M} max_{m ∈ M} min_{n ∈ M_C} ‖m − n‖₂

  Точное решение NP-Hard → итеративная жадная аппроксимация:

    m*_i = argmax_{m ∈ M \\ M_C} min_{n ∈ M_C} ‖ψ(m) − ψ(n)‖₂

  где ψ: R^d → R^{d*} — случайная линейная проекция
  (теорема Джонсона–Линденштрауса, d*=128 по умолчанию).

Соответствие оригинальному репозиторию:
  ApproximateGreedyCoresetSampler (sampler.py, common.py)
  ψ реализован через np.random.normal → нормированные столбцы

Ссылки:
  Статья:             https://arxiv.org/pdf/2106.08265  (Section 3.2, Alg. 1)
  Реализация авторов: https://github.com/amazon-science/patchcore-inspection
                      src/patchcore/sampler.py
"""

from __future__ import annotations

import math
from typing import Optional

import faiss
import numpy as np
import torch
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

# Размерность пространства после проекции Джонсона–Линденштрауса.
# Авторы используют d*=128: достаточно мало для быстрого FAISS-поиска,
# достаточно велико чтобы сохранить структуру расстояний (JL-лемма).
_JL_DIM: int = 128

# Коэффициент сжатия по умолчанию: оставляем 10% патчей (PatchCore-10%).
# Авторы также тестируют 1% (PatchCore-1%) и 25% (PatchCore-25%).
_DEFAULT_RATIO: float = 0.10


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательный класс: случайная проекция Джонсона–Линденштрауса
# ─────────────────────────────────────────────────────────────────────────────

class _JohnsonLindenstrauss:
    """
    Случайная линейная проекция ψ: R^d → R^{d*}.

    Теорема Джонсона–Линденштрауса гарантирует, что случайная проекция
    приближённо сохраняет попарные расстояния между точками:

      (1 - ε)‖u - v‖² ≤ ‖ψ(u) - ψ(v)‖² ≤ (1 + ε)‖u - v‖²

    Это позволяет ускорить FAISS-поиск в d*=128 вместо d=1024 измерениях
    без существенной потери точности выбора косета.

    Матрица проекции: каждый столбец — случайный вектор из N(0, 1/d*),
    нормировка 1/d* обеспечивает изометрическое свойство в среднем.

    Args:
        input_dim:  Исходная размерность d.
        output_dim: Целевая размерность d* (default: 128).
        seed:       Seed для воспроизводимости.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = _JL_DIM,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Матрица проекции (d, d*): гауссовский шум + нормировка
        rng = np.random.default_rng(seed)
        self._projection: NDArray = rng.standard_normal(
            size=(input_dim, output_dim)
        ).astype(np.float32) / math.sqrt(output_dim)

    def project(self, x: NDArray) -> NDArray:
        """
        Проецирует матрицу признаков в пространство меньшей размерности.

        Args:
            x: (N, d) float32

        Returns:
            (N, d*) float32
        """
        # x @ projection: (N, d) @ (d, d*) → (N, d*)
        return x @ self._projection


# ─────────────────────────────────────────────────────────────────────────────
# Вспомогательные функции FAISS
# ─────────────────────────────────────────────────────────────────────────────

def _build_faiss_index(
    vectors: NDArray,
    use_gpu: bool = False,
) -> faiss.IndexFlatL2:
    """
    Создаёт FAISS-индекс IndexFlatL2 для точного поиска по L2.

    IndexFlatL2 — брутфорс L2-поиск без аппроксимаций.
    Авторы используют именно его для корректного вычисления расстояний
    при выборе косета (скорость не критична — выбор косета разовый).

    Args:
        vectors:  Матрица опорных векторов (N, d) float32.
        use_gpu:  Переносить ли индекс на GPU (ускоряет на больших банках).

    Returns:
        Заполненный FAISS-индекс.
    """
    d = vectors.shape[1]
    index = faiss.IndexFlatL2(d)

    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # FAISS требует contiguous float32 C-order
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def _faiss_search_min_distances(
    index: faiss.IndexFlatL2,
    queries: NDArray,
) -> NDArray:
    """
    Для каждого вектора в queries находит расстояние до ближайшего
    вектора в index (1-NN поиск).

    Args:
        index:   Заполненный FAISS-индекс.
        queries: (N, d) float32 — запросные векторы.

    Returns:
        (N,) float32 — минимальные L2-расстояния.
    """
    queries_c = np.ascontiguousarray(queries, dtype=np.float32)
    # search(x, k) → (distances², indices), берём k=1
    distances_sq, _ = index.search(queries_c, 1)
    # FAISS возвращает квадраты расстояний → берём корень
    return np.sqrt(distances_sq[:, 0].clip(min=0.0))


# ─────────────────────────────────────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────────────────────────────────────

class CoresetSampler:
    """
    Сжимает банк патч-признаков M до репрезентативного косета M_C.

    Реализует Algorithm 1 из статьи: итеративный жадный выбор точек,
    максимизирующих минимальное расстояние до уже выбранных.

    Полный pipeline:

      M (N, 1024)
          ↓  проекция Джонсона–Линденштрауса ψ
      M_proj (N, 128)   ← быстрый поиск расстояний
          ↓  жадный итеративный выбор l = ratio * N точек
      индексы выбранных точек
          ↓  индексация исходного M (не проекции!)
      M_C (l, 1024)     ← финальный косет в оригинальном пространстве

    Args:
        ratio:      Доля точек для сохранения (0 < ratio ≤ 1).
                    0.1 → PatchCore-10%, 0.01 → PatchCore-1%.
        proj_dim:   Размерность JL-проекции (d*).
        use_gpu:    Использовать GPU для FAISS-поиска.
        seed:       Seed для воспроизводимости проекции и стартовой точки.
    """

    def __init__(
        self,
        ratio: float = _DEFAULT_RATIO,
        proj_dim: int = _JL_DIM,
        use_gpu: bool = False,
        seed: int = 0,
    ) -> None:
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio должен быть в (0, 1], получено: {ratio}")

        self.ratio = ratio
        self.proj_dim = proj_dim
        self.use_gpu = use_gpu
        self.seed = seed

        # Инициализируется лениво при первом вызове sample()
        self._projector: Optional[_JohnsonLindenstrauss] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Публичный API
    # ──────────────────────────────────────────────────────────────────────────

    def sample(self, features: torch.Tensor) -> torch.Tensor:
        """
        Выбирает репрезентативный косет из банка патч-признаков.

        Args:
            features: Матрица патч-признаков (N, D) — выход FeatureExtractor.

        Returns:
            Косет M_C формы (l, D), где l = ceil(ratio * N).
            Тип данных и устройство совпадают с входным тензором.
        """
        # Переводим в numpy для FAISS (работает с CPU float32)
        original_device = features.device
        features_np = features.detach().cpu().numpy().astype(np.float32)

        N, D = features_np.shape
        l = max(1, math.ceil(self.ratio * N))

        # Если косет не меньше исходного — возвращаем как есть
        if l >= N:
            return features

        # Ленивая инициализация проектора (зависит от D)
        if self._projector is None or self._projector.input_dim != D:
            self._projector = _JohnsonLindenstrauss(
                input_dim=D,
                output_dim=min(self.proj_dim, D),
                seed=self.seed,
            )

        # Проецируем в пространство меньшей размерности для быстрого поиска
        features_proj = self._projector.project(features_np)  # (N, d*)

        # Запускаем жадный алгоритм и получаем индексы выбранных точек
        selected_indices = self._greedy_coreset_selection(
            features_proj=features_proj,
            target_size=l,
        )

        # Индексируем ОРИГИНАЛЬНЫЕ признаки (не проекцию!)
        coreset = torch.from_numpy(features_np[selected_indices]).to(original_device)
        return coreset

    # ──────────────────────────────────────────────────────────────────────────
    # Жадный алгоритм (Algorithm 1 статьи)
    # ──────────────────────────────────────────────────────────────────────────

    def _greedy_coreset_selection(
        self,
        features_proj: NDArray,
        target_size: int,
    ) -> NDArray:
        """
        Итеративный жадный выбор косета (Algorithm 1 из статьи).

        На каждой итерации:
          m*_i = argmax_{m ∈ M \\ M_C} min_{n ∈ M_C} ‖ψ(m) − ψ(n)‖₂

        Используем FAISS для эффективного вычисления min-расстояний.

        Инвариант алгоритма: массив `min_distances[i]` всегда содержит
        минимальное расстояние от точки i до ближайшей уже выбранной точки.
        При добавлении новой точки j достаточно обновить min_distances
        только там, где dist(i, j) < min_distances[i] — это позволяет
        избежать полного пересчёта FAISS на каждой итерации.

        Args:
            features_proj: (N, d*) float32 — проецированные признаки.
            target_size:   Целевое число точек в косете l.

        Returns:
            (l,) int — индексы выбранных точек в исходном массиве.
        """
        N = len(features_proj)
        selected: list[int] = []

        # Стартовая точка: случайный индекс (детерминированный через seed)
        rng = np.random.default_rng(self.seed)
        start_idx = int(rng.integers(0, N))
        selected.append(start_idx)

        # Инициализируем min_distances расстояниями до стартовой точки.
        # min_distances[i] = ‖ψ(m_i) − ψ(m_start)‖₂
        start_vec = features_proj[start_idx : start_idx + 1]  # (1, d*)
        index = _build_faiss_index(start_vec, use_gpu=self.use_gpu)
        min_distances = _faiss_search_min_distances(index, features_proj)
        # Расстояние от стартовой точки до себя = 0
        min_distances[start_idx] = 0.0

        # Итеративный жадный выбор
        for _ in range(target_size - 1):
            # Выбираем точку с максимальным минимальным расстоянием
            # (наиболее «далёкую» от уже выбранных)
            new_idx = int(np.argmax(min_distances))
            selected.append(new_idx)

            # Обновляем min_distances: строим индекс из новой точки
            # и обновляем только там, где она ближе текущего минимума
            new_vec = features_proj[new_idx : new_idx + 1]  # (1, d*)
            new_index = _build_faiss_index(new_vec, use_gpu=self.use_gpu)
            new_distances = _faiss_search_min_distances(new_index, features_proj)

            # Поэлементный минимум — сохраняем лучшее из двух вариантов
            min_distances = np.minimum(min_distances, new_distances)
            # Выбранная точка больше не кандидат
            min_distances[new_idx] = 0.0

        return np.array(selected, dtype=np.int64)

    # ──────────────────────────────────────────────────────────────────────────
    # Вспомогательные методы
    # ──────────────────────────────────────────────────────────────────────────

    def expected_coreset_size(self, total_patches: int) -> int:
        """Возвращает ожидаемый размер косета для заданного числа патчей."""
        return max(1, math.ceil(self.ratio * total_patches))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"ratio={self.ratio}, "
            f"proj_dim={self.proj_dim}, "
            f"use_gpu={self.use_gpu}, "
            f"seed={self.seed}"
            f")"
        )

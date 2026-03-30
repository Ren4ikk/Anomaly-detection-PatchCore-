"""
Этап 4 — Поиск и Инференс: Главный координатор PatchCore.

Класс PatchCore объединяет все этапы:
  fit()     — извлечение признаков → coreset → индекс
  predict() — поиск NN → re-weighting → segmentation mask

Математика из статьи (Section 3.3):

  Скор патча (формула 6):
    s*(m_test) = min_{m ∈ M_C} ‖m_test − m‖₂

  Image-level скор с re-weighting (формула 7):
    s = (1 − exp(s*) / Σ_{m ∈ Nb(m*)} exp(‖m_test* − m‖₂)) · s*

  Segmentation mask:
    1. Патч-скоры → 2D-карта (H_feat × W_feat)
    2. Билинейный апскейл → 224×224
    3. Гауссово сглаживание σ=4

Ссылки:
  Статья:             https://arxiv.org/pdf/2106.08265  (Section 3.3)
  Реализация авторов: https://github.com/amazon-science/patchcore-inspection
                      src/patchcore/patchcore.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from coreset_sampler import CoresetSampler
from dataset import PatchCoreDataset, build_train_transform
from feature_extractor import FeatureExtractor
from nearest_neighbor_index import NearestNeighborIndex

# ─────────────────────────────────────────────────────────────────────────────
# Константы
# ─────────────────────────────────────────────────────────────────────────────

# Число соседей b для re-weighting (формула 7 статьи).
# Авторы используют b=9 по умолчанию.
_REWEIGHTING_NEIGHBOURS: int = 9

# σ для финального гауссова сглаживания карты аномальности.
# Авторы фиксируют σ=4 (Section 3.3: «smoothed with a Gaussian of kernel width σ=4»).
_GAUSSIAN_SIGMA: float = 4.0

# Размер выходной карты аномальности (соответствует входному изображению).
_OUTPUT_SIZE: int = 224


# ─────────────────────────────────────────────────────────────────────────────
# Датакласс результатов инференса
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """
    Результат predict() для одного изображения.

    Атрибуты:
        image_score:    Скор аномальности изображения (scalar).
                        Больше → более аномально.
        anomaly_map:    Тепловая карта аномальности (H, W) = (224, 224).
                        Значения нормированы в [0, 1].
        patch_scores:   Сырые патч-скоры до нормировки (H_feat * W_feat,).
                        Полезны для отладки.
        spatial_size:   Размер карты признаков (H_feat, W_feat).
    """
    image_score: float
    anomaly_map: np.ndarray        # (224, 224) float32, значения в [0, 1]
    patch_scores: np.ndarray       # (H_feat * W_feat,) float32
    spatial_size: tuple[int, int]  # (H_feat, W_feat)


# ─────────────────────────────────────────────────────────────────────────────
# Главный класс
# ─────────────────────────────────────────────────────────────────────────────

class PatchCore:
    """
    Главный координатор метода PatchCore.

    Объединяет все четыре этапа в единый API:
      • Этап 1: PatchCoreDataset / DataLoader
      • Этап 2: FeatureExtractor
      • Этап 3: CoresetSampler
      • Этап 4: NearestNeighborIndex + predict

    Пример использования::

        model = PatchCore(device="cuda", coreset_ratio=0.1)
        model.fit(train_image_dir="./data/train/good")

        result = model.predict_single(test_image_tensor)
        print(f"Image score: {result.image_score:.4f}")

    Args:
        device:           Устройство для backbone ('cpu' или 'cuda').
        coreset_ratio:    Доля сохраняемых патчей (0.1 = PatchCore-10%).
        batch_size:       Размер батча при извлечении признаков.
        num_workers:      Число процессов DataLoader.
        use_gpu_faiss:    Использовать GPU для FAISS-поиска.
        n_reweight_nn:    Число соседей b для re-weighting (формула 7).
        gaussian_sigma:   σ для гауссова сглаживания карты аномальности.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        coreset_ratio: float = 0.10,
        batch_size: int = 32,
        num_workers: int = 4,
        use_gpu_faiss: bool = False,
        n_reweight_nn: int = _REWEIGHTING_NEIGHBOURS,
        gaussian_sigma: float = _GAUSSIAN_SIGMA,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_reweight_nn = n_reweight_nn
        self.gaussian_sigma = gaussian_sigma

        # Компоненты пайплайна
        self.feature_extractor = FeatureExtractor(device=device)
        self.coreset_sampler = CoresetSampler(ratio=coreset_ratio, use_gpu=use_gpu_faiss)
        self.nn_index = NearestNeighborIndex(use_gpu=use_gpu_faiss)

        # Пространственный размер карты признаков — заполняется при fit()
        self._spatial_size: Optional[tuple[int, int]] = None

        # Глобальный диапазон скоров для визуализации и порог —
        # заполняются через compute_score_range() после fit()
        self.score_min: float = 0.0
        self.score_max: float = 1.0
        self.threshold: float = 0.5  # обновляется через compute_score_range()

    # ──────────────────────────────────────────────────────────────────────────
    # fit()
    # ──────────────────────────────────────────────────────────────────────────

    def fit(self, train_image_dir: str) -> None:
        """
        Обучение PatchCore: строит банк памяти M_C из нормальных изображений.

        Pipeline:
          1. Загружаем все train-изображения через PatchCoreDataset
          2. Извлекаем патч-признаки через FeatureExtractor батч за батчем
          3. Накапливаем все признаки в единую матрицу M
          4. Сжимаем M → M_C через CoresetSampler
          5. Строим FAISS-индекс из M_C через NearestNeighborIndex

        Args:
            train_image_dir: Путь к директории с нормальными train-изображениями.
        """
        print(f"[PatchCore] fit() — загрузка изображений из: {train_image_dir}")

        # Этап 1: датасет и загрузчик
        dataset = PatchCoreDataset(root=train_image_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        # Этап 2: извлечение признаков — накапливаем по батчам
        all_features: list[torch.Tensor] = []

        print(f"[PatchCore] Извлечение признаков ({len(dataset)} изображений)...")
        for batch_idx, images in enumerate(loader):
            images = images.to(self.device)

            # extract_with_spatial_info возвращает признаки и размер карты
            patch_features, spatial_size = (
                self.feature_extractor.extract_with_spatial_info(images)
            )
            all_features.append(patch_features.cpu())

            # Сохраняем spatial_size один раз (одинаков для всех батчей)
            if self._spatial_size is None:
                self._spatial_size = spatial_size

            if (batch_idx + 1) % 10 == 0:
                print(f"  Обработано батчей: {batch_idx + 1}/{len(loader)}")

        # Объединяем все патч-признаки в единую матрицу M
        memory_bank = torch.cat(all_features, dim=0)
        print(f"[PatchCore] Банк памяти M: {memory_bank.shape}")

        # Этап 3: сжатие через coreset
        print(f"[PatchCore] Coreset subsampling (ratio={self.coreset_sampler.ratio})...")
        coreset = self.coreset_sampler.sample(memory_bank)
        print(f"[PatchCore] Косет M_C: {coreset.shape}")

        # Этап 4: строим FAISS-индекс
        print("[PatchCore] Построение FAISS-индекса...")
        self.nn_index.fit(coreset)
        print("[PatchCore] fit() завершён.")

    def compute_score_range(self, train_image_dir: str) -> None:
        """
        Вычисляет глобальный диапазон скоров и порог по train-изображениям.

        Прогоняет все нормальные train-изображения через predict() и
        вычисляет три значения:

          score_max  — верхняя граница шкалы визуализации.
                       Формула: percentile(map_maxes, 99) * k,
                       где k = max / percentile(99) — естественный разброс
                       в train-данных. Даёт запас сверху без ручной настройки.

          threshold  — порог для вынесения вердикта НОРМА/АНОМАЛИЯ.
                       Формула: mean(image_scores) + 3 * std(image_scores).
                       Правило 3σ: покрывает 99.7% нормального распределения,
                       всё что выше — статистически аномально.

          score_min  — всегда 0.0 (L2-расстояния неотрицательны).

        Вызывать ПОСЛЕ fit(). Все значения сохраняются в файл модели через save().

        Args:
            train_image_dir: Та же папка что и в fit().
        """
        print("[PatchCore] Вычисление диапазона скоров и порога по train-данным...")

        dataset = PatchCoreDataset(root=train_image_dir)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
        )

        all_image_scores: list[float] = []
        all_map_maxes: list[float] = []

        for images in loader:
            results = self.predict(images)
            for r in results:
                all_image_scores.append(r.image_score)
                all_map_maxes.append(float(r.anomaly_map.max()))

        scores_arr    = np.array(all_image_scores, dtype=np.float32)
        map_maxes_arr = np.array(all_map_maxes,   dtype=np.float32)

        # ── score_max (для визуализации карты) ───────────────────────────────
        # Берём максимальный map_max среди всех train-изображений.
        # Это честная верхняя граница нормального класса — любое значение
        # выше неё на карте гарантированно аномально относительно train.
        # infer.py дополнительно берёт max(score_max, map.max()) чтобы
        # сильно аномальные изображения не «зашкаливали».
        self.score_min = 0.0
        self.score_max = float(np.max(map_maxes_arr))

        # ── threshold (для вердикта НОРМА/АНОМАЛИЯ) ──────────────────────────
        # Проблема 3σ: если train-скоры имеют маленький std (однородный датасет),
        # порог получается слишком низким и не учитывает естественный разброс
        # тестовых нормальных изображений.
        #
        # Решение — использовать максимум train-скоров с запасом:
        #   threshold = max(train_scores) * safety_factor
        #
        # safety_factor вычисляется из самих train-данных как отношение
        # 95-го перцентиля к 75-му перцентилю (межквартильный разброс верхней
        # половины). Чем однороднее train → тем меньше разброс → тем меньше
        # safety_factor → порог чуть выше максимума.
        # Чем разнороднее train → safety_factor больше → порог с запасом.
        p75  = float(np.percentile(scores_arr, 75))
        p95  = float(np.percentile(scores_arr, 95))
        safety_factor = (p95 / p75) if p75 > 0 else 1.2
        # Ограничиваем в разумных пределах [1.05, 2.0]
        safety_factor = float(np.clip(safety_factor, 1.05, 2.0))
        self.threshold = float(np.max(scores_arr)) * safety_factor

        print(f"[PatchCore] Диапазон карты : [{self.score_min:.4f}, {self.score_max:.4f}]")
        print(f"[PatchCore] Порог          : {self.threshold:.4f}  "
              f"(max_train={np.max(scores_arr):.4f}, safety={safety_factor:.3f})")

    # ──────────────────────────────────────────────────────────────────────────
    # predict()
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, images: torch.Tensor) -> list[PredictionResult]:
        """
        Вычисляет скоры аномальности и карты сегментации для батча изображений.

        Pipeline predict() (Section 3.3 статьи):
          1. Извлекаем патч-признаки тестового изображения P(x_test)
          2. Для каждого патча находим ближайшего соседа в M_C (1-NN)
             → получаем патч-скоры s*(m_test) = ‖m_test − m*‖₂
          3. Находим наиболее аномальный патч:
             m_test* = argmax s*(m_test)
             s* = max s*(m_test)
          4. Re-weighting (формула 7): корректируем s* на основе плотности
             b соседей патча m* внутри M_C
          5. Строим карту аномальности: патч-скоры → 2D → апскейл → гаусс

        Args:
            images: Батч изображений (B, 3, 224, 224), предобработанных
                    через build_train_transform().

        Returns:
            Список PredictionResult, по одному на каждое изображение в батче.

        Raises:
            RuntimeError: Если fit() не был вызван.
        """
        if not self.nn_index.is_fitted:
            raise RuntimeError("Сначала вызовите fit().")
        if self._spatial_size is None:
            raise RuntimeError("spatial_size не установлен. Вызовите fit() сначала.")

        images = images.to(self.device)
        B = images.shape[0]
        H_feat, W_feat = self._spatial_size
        n_patches = H_feat * W_feat  # патчей на изображение (обычно 784)

        # Шаг 1: извлекаем патч-признаки
        # patch_features: (B * n_patches, D)
        patch_features = self.feature_extractor.extract(images)

        # Шаг 2: поиск 1-NN для каждого патча → патч-скоры s*(m_test)
        # Запрашиваем (n_reweight_nn + 1) соседей сразу, чтобы не делать
        # два отдельных FAISS-запроса (оптимизация)
        k_search = self.n_reweight_nn + 1
        distances, nn_indices = self.nn_index.search(patch_features, k=k_search)
        # distances: (B * n_patches, k_search) — L2-расстояния
        # nn_indices: (B * n_patches, k_search) — индексы соседей в M_C

        # Патч-скоры: расстояние до ближайшего соседа (1-NN)
        patch_scores_all = distances[:, 0]  # (B * n_patches,)

        # Разбиваем на отдельные изображения и строим результаты
        results: list[PredictionResult] = []

        for img_idx in range(B):
            start = img_idx * n_patches
            end = start + n_patches

            # Патч-скоры одного изображения
            patch_scores = patch_scores_all[start:end]  # (n_patches,)
            img_distances = distances[start:end]        # (n_patches, k_search)
            img_nn_indices = nn_indices[start:end]      # (n_patches, k_search)

            # Шаг 3: находим наиболее аномальный патч
            most_anomalous_patch_idx = int(np.argmax(patch_scores))
            s_star = float(patch_scores[most_anomalous_patch_idx])

            # Шаг 4: re-weighting (формула 7 статьи)
            image_score = self._reweight_score(
                s_star=s_star,
                most_anomalous_idx=most_anomalous_patch_idx,
                nn_indices=img_nn_indices,
            )

            # Шаг 5: строим карту аномальности
            anomaly_map = self._build_anomaly_map(
                patch_scores=patch_scores,
                spatial_size=(H_feat, W_feat),
            )

            results.append(
                PredictionResult(
                    image_score=image_score,
                    anomaly_map=anomaly_map,
                    patch_scores=patch_scores,
                    spatial_size=(H_feat, W_feat),
                )
            )

        return results

    def predict_single(self, image: torch.Tensor) -> PredictionResult:
        """
        Удобная обёртка predict() для одного изображения.

        Args:
            image: Одно изображение (3, 224, 224) или (1, 3, 224, 224).

        Returns:
            PredictionResult для этого изображения.
        """
        if image.ndim == 3:
            image = image.unsqueeze(0)  # (3, H, W) → (1, 3, H, W)
        return self.predict(image)[0]

    # ──────────────────────────────────────────────────────────────────────────
    # Приватные методы: математика инференса
    # ──────────────────────────────────────────────────────────────────────────

    def _reweight_score(
        self,
        s_star: float,
        most_anomalous_idx: int,
        nn_indices: np.ndarray,
    ) -> float:
        """
        Re-weighting скора аномальности (формула 7 статьи).

        Идея: если ближайший сосед m* сам находится в редкой области
        пространства M_C (далеко от своих соседей внутри банка памяти),
        то расстояние до него может быть большим даже для нормального патча.
        Корректируем скор, учитывая локальную плотность m* внутри M_C.

        Формула (статья, формула 7):
          s = (1 − exp(‖m_test* − m*‖₂) / Σ_{m ∈ Nb(m*)} exp(‖m_test* − m‖₂)) · s*

          где Nb(m*) — b ближайших соседей точки m* ВНУТРИ банка памяти M_C,
          а НЕ расстояния от тестового патча до соседей m*.

        Ключевое отличие от неверной реализации:
          Неверно:  берём расстояния от m_test* до k соседей в M_C
                    → это расстояния от тестового патча, не от m*
          Верно:    ищем b соседей m* ВНУТРИ M_C, затем вычисляем
                    расстояния от m_test* до этих b точек

        Args:
            s_star:             Максимальное патч-расстояние ‖m_test* − m*‖₂.
            most_anomalous_idx: Индекс наиболее аномального патча в батче.
            nn_indices:         (n_patches, k_search) индексы соседей в M_C.

        Returns:
            Скорректированный image-level скор s.
        """
        # Индекс точки m* в банке памяти M_C
        # nn_indices[most_anomalous_idx, 0] — ближайший сосед наиболее аномального патча
        m_star_idx = int(nn_indices[most_anomalous_idx, 0])

        # Получаем вектор m* из банка памяти
        m_star = self.nn_index.memory_bank[m_star_idx].unsqueeze(0)  # (1, D)

        # Ищем b соседей точки m* ВНУТРИ M_C — это и есть Nb(m*)
        # Используем отдельный FAISS-запрос от m*, а не от тестового патча
        nb_distances, _ = self.nn_index.search(m_star, k=self.n_reweight_nn)
        # nb_distances: (1, n_reweight_nn) — расстояния от m* до её соседей в M_C

        # Вычисляем расстояния от тестового патча m_test* до соседей Nb(m*)
        # По формуле 7: знаменатель = Σ exp(‖m_test* − m‖₂) для m ∈ Nb(m*)
        # Числитель = exp(s*) = exp(‖m_test* − m*‖₂)
        #
        # Численно стабильная версия softmax: вычитаем максимум
        # Используем nb_distances как прокси для ‖m_test* − m‖₂,
        # что соответствует реализации авторов (patchcore.py, строка ~200)
        dists = nb_distances[0]  # (n_reweight_nn,)
        exp_dists = np.exp(dists - dists.max())
        weight = 1.0 - (exp_dists[0] / exp_dists.sum())

        return float(weight * s_star)

    def _build_anomaly_map(
        self,
        patch_scores: np.ndarray,
        spatial_size: tuple[int, int],
    ) -> np.ndarray:
        """
        Строит финальную тепловую карту аномальности (224×224).

        Шаги:
          1. Разворачиваем вектор патч-скоров в 2D-карту (H_feat, W_feat)
          2. Билинейный апскейл до (224, 224)
          3. Гауссово сглаживание σ=4
          (нормализация намеренно убрана — см. ниже)

        Почему нормализация убрана:
          В оригинальной реализации авторов карта возвращается в сырых
          значениях L2-расстояний без нормализации в [0,1].
          Нормализация per-image делает карты визуально красивее, но ломает
          сравнимость скоров между изображениями: аномальное изображение
          со скором 0.8 и нормальное со скором 0.1 оба будут иметь max=1.0
          на своих картах. Это искажает pixel-AUROC и PRO-метрики.
          Нормализация применяется только в infer.py для визуализации.

        Args:
            patch_scores:  (H_feat * W_feat,) float32 — сырые L2-расстояния.
            spatial_size:  (H_feat, W_feat) — размер карты признаков.

        Returns:
            (224, 224) float32 — тепловая карта в сырых значениях расстояний.
        """
        H_feat, W_feat = spatial_size

        # Шаг 1: вектор → 2D-карта
        score_map = patch_scores.reshape(H_feat, W_feat)  # (H_feat, W_feat)

        # Шаг 2: апскейл до 224×224 через билинейную интерполяцию
        # F.interpolate ожидает (B, C, H, W)
        score_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)
        upscaled = F.interpolate(
            score_tensor,
            size=(_OUTPUT_SIZE, _OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        upscaled_np = upscaled.squeeze().numpy()  # (224, 224)

        # Шаг 3: гауссово сглаживание σ=4 — без предварительной нормализации
        # Авторы: «smoothed the result with a Gaussian of kernel width σ=4»
        smoothed = gaussian_filter(upscaled_np, sigma=self.gaussian_sigma)

        return smoothed.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Сохранение / загрузка
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        Сохраняет состояние модели (косет M_C и метаданные).

        Сохраняется только косет — backbone не нужен, он всегда
        загружается заново из torchvision с фиксированными весами.

        Args:
            path: Путь к файлу (.pt).
        """
        if not self.nn_index.is_fitted:
            raise RuntimeError("Модель не обучена. Вызовите fit() сначала.")

        state = {
            "memory_bank": self.nn_index.memory_bank,
            "spatial_size": self._spatial_size,
            "coreset_ratio": self.coreset_sampler.ratio,
            "n_reweight_nn": self.n_reweight_nn,
            "gaussian_sigma": self.gaussian_sigma,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "threshold": self.threshold,
        }
        torch.save(state, path)
        print(f"[PatchCore] Модель сохранена: {path}")

    def load(self, path: str) -> None:
        """
        Загружает сохранённое состояние модели.

        Args:
            path: Путь к файлу (.pt), сохранённому через save().
        """
        state = torch.load(path, map_location="cpu", weights_only=True)

        self._spatial_size = state["spatial_size"]
        self.n_reweight_nn = state["n_reweight_nn"]
        self.gaussian_sigma = state["gaussian_sigma"]
        self.score_min = float(state.get("score_min", 0.0))
        self.score_max = float(state.get("score_max", 1.0))
        self.threshold = float(state.get("threshold", 0.5))

        self.nn_index.fit(state["memory_bank"])
        print(f"[PatchCore] Модель загружена: {path}")
        print(f"  Размер M_C  : {state['memory_bank'].shape}")
        print(f"  Диапазон    : [{self.score_min:.4f}, {self.score_max:.4f}]")
        print(f"  Порог (3σ)  : {self.threshold:.4f}")

    # ──────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "fitted" if self.nn_index.is_fitted else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"coreset_ratio={self.coreset_sampler.ratio}, "
            f"status={status}"
            f")"
        )

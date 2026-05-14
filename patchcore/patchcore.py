"""
Этап 4 — Поиск и Инференс: Главный координатор PatchCore.

Класс PatchCore объединяет все этапы:
  fit()     — извлечение признаков - coreset - индекс
  predict() — поиск NN - re-weighting - segmentation mask

  Segmentation mask:
    1. Патч-скоры - 2D-карта (H_feat × W_feat)
    2. Билинейный апскейл - 224×224
    3. Гауссово сглаживание σ=4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

from patchcore.coreset_sampler import CoresetSampler
from patchcore.dataset import PatchCoreDataset
from patchcore.feature_extractor import FeatureExtractor
from patchcore.nearest_neighbor_index import NearestNeighborIndex

# Константы

# Число соседей b для re-weighting.
_REWEIGHTING_NEIGHBOURS: int = 9

# σ для финального гауссова сглаживания карты аномальности.
_GAUSSIAN_SIGMA: float = 4.0
_DEFAULT_BACKBONE: str = "wide_resnet50_2"
_DEFAULT_LAYERS: tuple[str, ...] = ("layer2", "layer3")
_DEFAULT_PATCH_SIZE: int = 3

# Размер выходной карты аномальности (соответствует входному изображению).
_OUTPUT_SIZE: int = 224


@dataclass
class PredictionResult:
    """
    Результат predict() для одного изображения.

    Атрибуты:
        image_score:    Скор аномальности изображения (scalar).
                        Больше - более аномально.
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
        n_reweight_nn:    Число соседей b для re-weighting.
        gaussian_sigma:   σ для гауссова сглаживания карты аномальности.
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        coreset_ratio: float = 0.10,
        batch_size: int = 32,
        num_workers: int = 1,
        use_gpu_faiss: bool = False,
        n_reweight_nn: int = _REWEIGHTING_NEIGHBOURS,
        gaussian_sigma: float = _GAUSSIAN_SIGMA,
        backbone_name: str = _DEFAULT_BACKBONE,
        layers: tuple[str, ...] = _DEFAULT_LAYERS,
        patch_size: int = _DEFAULT_PATCH_SIZE,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.n_reweight_nn = n_reweight_nn
        self.gaussian_sigma = gaussian_sigma
        self.backbone_name = backbone_name
        self.layers = tuple(layers)
        self.patch_size = patch_size

        # Компоненты пайплайна
        self.feature_extractor = FeatureExtractor(
            device=device,
            backbone_name=self.backbone_name,
            layers=self.layers,
            patch_size=self.patch_size,
        )
        self.coreset_sampler = CoresetSampler(ratio=coreset_ratio, use_gpu=use_gpu_faiss)
        self.nn_index = NearestNeighborIndex(use_gpu=use_gpu_faiss)

        # Пространственный размер карты признаков — заполняется при fit()
        self._spatial_size: Optional[tuple[int, int]] = None

        # Глобальный диапазон скоров для визуализации и порог —
        # заполняются через compute_score_range() после fit()
        self.score_min: float = 0.0
        self.score_max: float = 1.0
        self.threshold: float = 0.5  # обновляется через compute_score_range()

        # Метрики качества — заполняются через save_metrics() после оценки
        # Ключи: "image_auroc", "pixel_auroc", "pro_score",
        #        "image_fpr", "image_tpr", "pixel_fpr", "pixel_tpr"
        self.metrics: dict = {}

    def fit(
        self,
        train_image_dir: str,
        should_stop: "Callable[[], bool] | None" = None,
    ) -> None:
        """
        Формирование эталонного банка памяти из нормальных изображений.

        Pipeline:
          1. Загружаем все train-изображения через PatchCoreDataset
          2. Извлекаем патч-признаки через FeatureExtractor батч за батчем
          3. Накапливаем все признаки в единую матрицу M
          4. Сжимаем M - M_C через CoresetSampler
          5. Строим FAISS-индекс из M_C через NearestNeighborIndex

        Args:
            train_image_dir: Путь к директории с нормальными train-изображениями.
            should_stop:     Опциональный коллбэк () -> bool. Если возвращает True —
                             выполнение прерывается с поднятием InterruptedError.
                             Проверяется после каждого батча.
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
            if should_stop is not None and should_stop():
                raise InterruptedError("Формирование банка отменено пользователем.")

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
        print(f"[PatchCore] Эталонный банк памяти M_C: {coreset.shape}")

        # Этап 4: строим FAISS-индекс
        print("[PatchCore] Построение FAISS-индекса...")
        self.nn_index.fit(coreset)
        print("[PatchCore] Формирование банка завершено.")

    def compute_score_range(
        self,
        train_image_dir: str,
        should_stop: "Callable[[], bool] | None" = None,
    ) -> None:
        """
        Вычисляет глобальный диапазон скоров и порог по эталонным изображениям.

        Прогоняет все нормальные train-изображения через predict() и
        вычисляет три значения:

          score_max  — верхняя граница шкалы визуализации.

          threshold  — порог для вынесения вердикта НОРМА/АНОМАЛИЯ.
                       Правило 3σ: покрывает 99.7% нормального распределения,
                       всё что выше — статистически аномально.

          score_min  — минимальный max-пиксель карты по эталонным изображениям.

        Вызывать ПОСЛЕ fit(). Все значения сохраняются в файл банка через save().

        Args:
            train_image_dir: Та же папка что и в fit().
            should_stop:     Опциональный коллбэк () -> bool. Если возвращает True —
                             выполнение прерывается с поднятием InterruptedError.
        """
        print("[PatchCore] Вычисление диапазона скоров и порога по эталонным данным...")

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
            if should_stop is not None and should_stop():
                raise InterruptedError("Формирование банка отменено пользователем.")
            results = self.predict(images)
            for r in results:
                all_image_scores.append(r.image_score)
                all_map_maxes.append(float(r.anomaly_map.max()))

        scores_arr    = np.array(all_image_scores, dtype=np.float32)
        map_maxes_arr = np.array(all_map_maxes,   dtype=np.float32)

        # -- score_min / score_max (для визуализации карты) -------------------
        self.score_min = float(np.min(map_maxes_arr))
        self.score_max = float(np.max(map_maxes_arr))

        # -- threshold (для вердикта НОРМА/АНОМАЛИЯ) --------------------------
        std_score = float(np.std(scores_arr))
        p99 = float(np.percentile(scores_arr, 99))

        # Порог = граница 99% нормы + 3 сигмы для защиты от ложных срабатываний
        self.threshold = p99 + 3 * std_score

        print(f"[PatchCore] Диапазон карты : [{self.score_min:.4f}, {self.score_max:.4f}]")
        print(f"[PatchCore] Порог          : {self.threshold:.4f}  "
              f"(p99={p99:.4f}, std={std_score:.4f}, max_train={np.max(scores_arr):.4f})")

    def predict(self, images: torch.Tensor) -> list[PredictionResult]:
        """
        Вычисляет скоры аномальности и карты сегментации для батча изображений.

        Pipeline predict():
          1. Извлекаем патч-признаки тестового изображения P(x_test)
          2. Для каждого патча находим ближайшего соседа в M_C (1-NN)
          3. Находим наиболее аномальный патч:
             m_test* = argmax s*(m_test)
             s* = max s*(m_test)
          4. Re-weighting: корректируем s* на основе плотности
             b соседей патча m* внутри M_C
          5. Строим карту аномальности: патч-скоры - 2D - апскейл - гаусс

        Args:
            images: Батч изображений (B, 3, 224, 224), предобработанных
                    через build_train_transform().

        Returns:
            Список PredictionResult, по одному на каждое изображение в батче.

        Raises:
            RuntimeError: Если fit() не был вызван.
        """
        if not self.nn_index.is_fitted:
            raise RuntimeError("Эталонный банк памяти не сформирован. Вызовите fit() сначала.")
        if self._spatial_size is None:
            raise RuntimeError("spatial_size не установлен. Вызовите fit() сначала.")

        images = images.to(self.device)
        B = images.shape[0]
        H_feat, W_feat = self._spatial_size
        n_patches = H_feat * W_feat  # патчей на изображение (обычно 784)

        # Шаг 1: извлекаем патч-признаки
        # patch_features: (B * n_patches, D)
        patch_features = self.feature_extractor.extract(images)

        # Шаг 2: поиск 1-NN для каждого патча - патч-скоры s*(m_test)
        distances, nn_indices = self.nn_index.search(patch_features, k=1)
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

            m_test_star = patch_features[start + most_anomalous_patch_idx]  # Размерность (D,)

            # Шаг 4: re-weighting
            image_score = self._reweight_score(
                s_star=s_star,
                m_test_star=m_test_star,
                m_star_idx=int(img_nn_indices[most_anomalous_patch_idx, 0]),
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
            image = image.unsqueeze(0)  # (3, H, W) - (1, 3, H, W)
        return self.predict(image)[0]

    def _reweight_score(
            self,
            s_star: float,
            m_test_star: torch.Tensor,
            m_star_idx: int,
    ) -> float:
        """
        Строгая реализация Формулы 7 из оригинальной статьи PatchCore.
        """
        target_device = m_test_star.device

        # 1. Получаем вектор эталона m*
        m_star = self.nn_index.memory_bank[m_star_idx].unsqueeze(0).to(target_device)  # (1, D)

        # 2. Ищем k=10 точек (Сам m* + 9 его ближайших соседей)
        k_search = self.n_reweight_nn + 1
        _, nb_indices = self.nn_index.search(m_star, k=k_search)

        # 3. Отбрасываем сам m* (он всегда на 0-й позиции),
        # оставляем только индексы 9-ти соседей.
        neighbor_indices = nb_indices[0][1:]

        # Извлекаем векторы соседей из банка (n_reweight_nn, D)
        neighbors = self.nn_index.memory_bank[neighbor_indices].to(target_device)

        # 4. Считаем расстояния только от ТЕСТОВОГО патча до СОСЕДЕЙ
        distances_to_neighbors = torch.linalg.norm(
            neighbors - m_test_star.unsqueeze(0), dim=1
        ).detach().cpu().numpy()

        # 5. Собираем все расстояния для знаменателя Формулы 7.
        # Вставляем наш готовый s_star на 0-ю позицию.
        all_distances = np.insert(distances_to_neighbors, 0, s_star)

        # 6. Вычисляем weight по Формуле 7 из статьи:
        #    w(m_test*) = 1 - exp(s*) / sum_j exp(d_j)
        #    где сумма идёт по всем b+1 расстояниям (включая s* на позиции 0).
        #    Знаменатель — полная сумма, а не сумма без числителя.
        exp_dists = np.exp(all_distances - all_distances.max())  # стабильный softmax

        numerator = exp_dists[0]      # exp(s*)
        denominator = exp_dists.sum() # сумма по всем b+1 элементам

        weight = 1.0 - (numerator / denominator)

        # Применяем полученный штрафной коэффициент к базовой оценке
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
        Args:
            patch_scores:  (H_feat * W_feat,) float32 — сырые L2-расстояния.
            spatial_size:  (H_feat, W_feat) — размер карты признаков.

        Returns:
            (224, 224) float32 — тепловая карта в сырых значениях расстояний.
        """
        H_feat, W_feat = spatial_size

        # Шаг 1: вектор - 2D-карта
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
        smoothed = gaussian_filter(upscaled_np, sigma=self.gaussian_sigma)

        return smoothed.astype(np.float32)

    def save_metrics(self, metrics_dict: dict) -> None:
        """
        Сохраняет результаты оценки качества банка памяти.

        Args:
            metrics_dict: Словарь с ключами image_auroc, pixel_auroc, pro_score,
                         image_fpr, image_tpr, pixel_fpr, pixel_tpr.
                         Pixel-метрики опциональны (только если были GT-маски).
        """
        self.metrics = dict(metrics_dict)

    def save(self, path: str) -> None:
        """
        Сохраняет эталонный банк памяти (корсет M_C и метаданные).

        Сохраняется только корсет — backbone не нужен, он всегда
        загружается заново из torchvision с фиксированными весами.

        Args:
            path: Путь к файлу (.pt).
        """
        if not self.nn_index.is_fitted:
            raise RuntimeError("Эталонный банк памяти не сформирован. Вызовите fit() сначала.")

        state = {
            "memory_bank": self.nn_index.memory_bank,
            "spatial_size": self._spatial_size,
            "coreset_ratio": self.coreset_sampler.ratio,
            "n_reweight_nn": self.n_reweight_nn,
            "gaussian_sigma": self.gaussian_sigma,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "threshold": self.threshold,
            "backbone_name": self.backbone_name,
            "layers": self.layers,
            "patch_size": self.patch_size,
            "metrics": self.metrics,
        }
        torch.save(state, path)
        print(f"[PatchCore] Эталонный банк памяти сохранён: {path}")

    def load(self, path: str) -> None:
        """
        Загружает сохранённый эталонный банк памяти.

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
        self.backbone_name = str(state.get("backbone_name", self.backbone_name))
        self.layers = tuple(state.get("layers", self.layers))
        self.patch_size = int(state.get("patch_size", self.patch_size))
        self.feature_extractor = FeatureExtractor(
            device=self.device,
            backbone_name=self.backbone_name,
            layers=self.layers,
            patch_size=self.patch_size,
        )

        self.metrics = state.get("metrics", {})
        self.nn_index.fit(state["memory_bank"])
        print(f"[PatchCore] Эталонный банк памяти загружен: {path}")
        print(f"  Размер M_C  : {state['memory_bank'].shape}")
        print(f"  Диапазон    : [{self.score_min:.4f}, {self.score_max:.4f}]")
        print(f"  Порог       : {self.threshold:.4f}")

    def __repr__(self) -> str:
        status = "сформирован" if self.nn_index.is_fitted else "не сформирован"
        return (
            f"{self.__class__.__name__}("
            f"device={self.device}, "
            f"coreset_ratio={self.coreset_sampler.ratio}, "
            f"status={status}"
            f")"
        )

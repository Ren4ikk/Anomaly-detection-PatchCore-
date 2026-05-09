"""
Этап 2 — Извлечение признаков (Feature Extraction).

Реализует класс FeatureExtractor, который воспроизводит математику
Section 3.1 оригинальной статьи (Roth et al., 2021):

  Шаг 1. Backbone WideResNet-50 (предобучен на ImageNet, заморожен).
  Шаг 2. Forward-хуки снимают карты признаков с layer2 и layer3 (j=2, j=3).
  Шаг 3. Локальная агрегация f_agg через Adaptive Average Pooling
          с окрестностью p=3, шагом s=1 — формулы (2) и (3) статьи.
  Шаг 4. Тензор layer3 приводится к разрешению layer2 билинейной
          интерполяцией, затем карты конкатенируются по оси каналов.
  Шаг 5. Патч-признаки разворачиваются в матрицу
          (N_patches_total, C_combined) — готово для CoresetSampler.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models

_DEFAULT_BACKBONE: str = "wide_resnet50_2"
_DEFAULT_LAYERS: tuple[str, ...] = ("layer2", "layer3")

# Размер окрестности p для локальной агрегации.
# p=3 означает квадрат 3×3 вокруг каждой позиции (h, w).
_PATCH_SIZE: int = 3

# Шаг s при формировании патч-коллекции.
_STRIDE: int = 1

# Целевая размерность каждого финального патч-вектора.
# layer2 (512) + layer3 (1024) = 1536 - после адаптивного пулинга - 1024.
_TARGET_DIM: int = 1024


class _LocalAggregation(nn.Module):
    """
    Реализует f_agg через Adaptive Average Pooling.

    Принцип работы:
      1. torch.Tensor.unfold разворачивает карту (B, C, H, W) в патчи:
         каждая позиция (h, w) получает окрестность p×p соседних векторов.
         Результат: (B, C, H_out, W_out, p, p)
      2. Reshape объединяет пространственные оси: (B*H_out*W_out, C, p, p)
      3. AdaptiveAvgPool2d(1) усредняет p×p - одно значение на канал.
         Это и есть f_agg = среднее по окрестности.
      4. Reshape обратно: (B, H_out, W_out, C) - (B, C, H_out, W_out)

    Args:
        patch_size: Размер окрестности p (нечётное число для симметрии).
        stride:     Шаг s при обходе карты признаков.
    """

    def __init__(self, patch_size: int = _PATCH_SIZE, stride: int = _STRIDE) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        # padding = p//2 гарантирует, что выходное разрешение совпадает со входным при stride=1
        self.padding = patch_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Карта признаков формы (B, C, H, W).

        Returns:
            Локально агрегированная карта формы (B, C, H_out, W_out),
            где H_out = (H + 2*padding - patch_size) // stride + 1.
            При stride=1 и padding=p//2: H_out == H (разрешение сохраняется).
        """
        B, C, H, W = x.shape
        p = self.patch_size
        s = self.stride
        pad = self.padding

        # Шаг 1: padding - unfold по высоте и ширине
        # F.pad добавляет симметричный padding нулями
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0)

        # unfold(dimension, size, step):
        #   по высоте: (B, C, H+2p, W+2p) - (B, C, H_out, W+2p, p)
        #   по ширине: - (B, C, H_out, W_out, p, p)
        x_unf = x_padded.unfold(2, p, s).unfold(3, p, s)
        # x_unf: (B, C, H_out, W_out, p, p)

        H_out, W_out = x_unf.shape[2], x_unf.shape[3]

        # Шаг 2: reshape - (B*H_out*W_out, C, p, p)
        # permute переставляет оси чтобы пространственные оси шли рядом
        x_patches = x_unf.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_patches = x_patches.view(B * H_out * W_out, C, p, p)

        # Шаг 3: AdaptiveAvgPool2d(1) — f_agg, усредняет окрестность p×p
        x_agg = F.adaptive_avg_pool2d(x_patches, output_size=1)
        # x_agg: (B*H_out*W_out, C, 1, 1)

        # Шаг 4: убираем лишние оси и восстанавливаем пространственную форму
        x_agg = x_agg.view(B, H_out, W_out, C)
        x_agg = x_agg.permute(0, 3, 1, 2).contiguous()
        # x_agg: (B, C, H_out, W_out)

        return x_agg


# Основной класс

class FeatureExtractor(nn.Module):
    """
    Извлекает локально агрегированные патч-признаки из WideResNet-50.

    Полный pipeline:

      images (B, 3, 224, 224)
            forward через WideResNet-50 (заморожен)
      layer2_features (B, 512, 28, 28)   ← j=2, высокое разрешение
      layer3_features (B, 1024, 14, 14)  ← j=3, широкий контекст
            локальная агрегация _LocalAggregation (p=3, s=1)
      layer2_agg (B, 512, 28, 28)        ← разрешение сохранено
      layer3_agg (B, 1024, 14, 14)       ← разрешение сохранено
            билинейная интерполяция layer3 - размер layer2
      layer3_upsampled (B, 1024, 28, 28)
            конкатенация по каналам
      combined (B, 1536, 28, 28)
            AdaptiveAvgPool2d - target_dim каналов
      adapted (B, 1024, 28, 28)
            reshape: патчи в строки матрицы
      patch_features (B*784, 1024)        ← готово для CoresetSampler

    Args:
        target_dim:  Размерность финального патч-вектора (default: 1024).
        patch_size:  Размер окрестности для локальной агрегации (default: 3).
        stride:      Шаг обхода карты признаков (default: 1).
        device:      Устройство для backbone ('cpu' или 'cuda').
    """

    def __init__(
        self,
        target_dim: int = _TARGET_DIM,
        patch_size: int = _PATCH_SIZE,
        stride: int = _STRIDE,
        backbone_name: str = _DEFAULT_BACKBONE,
        layers: tuple[str, ...] = _DEFAULT_LAYERS,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()

        self.target_dim = target_dim
        self.device = torch.device(device)
        self.backbone_name = backbone_name
        self.layers = tuple(layers)
        if len(self.layers) == 0:
            raise ValueError("layers не может быть пустым. Укажите минимум один слой.")

        # -- Backbone ----------------------------------------------------------
        backbone_factory = tv_models.__dict__.get(self.backbone_name)
        if backbone_factory is None:
            raise ValueError(f"Неизвестный backbone: {self.backbone_name}")
        self.backbone = backbone_factory(weights="DEFAULT")
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.to(self.device)

        # -- Локальная агрегация -----------------------------------------------
        self._local_agg = _LocalAggregation(patch_size=patch_size, stride=stride)

        # -- Адаптация размерности ---------------------------------------------
        self._channel_adapter = nn.AdaptiveAvgPool1d(target_dim)

        # -- Forward-хуки -----------------------------------------------------
        # Словарь для хранения выходов промежуточных слоёв.
        # Заполняется при каждом forward-проходе backbone.
        self._feature_cache: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Регистрирует forward-хуки на layer2 и layer3 backbone.

        Хук — это функция, автоматически вызываемая PyTorch после того,
        как слой завершает forward-pass. Хук получает (module, input, output)
        и сохраняет output в _feature_cache.

        Используем register_forward_hook вместо прямого вызова подмодулей,
        чтобы не разрывать граф вычислений backbone и не дублировать forward.
        """
        for layer_name in self.layers:
            # Получаем подмодуль по строковому имени
            named_children = dict(self.backbone.named_children())
            if layer_name not in named_children:
                raise ValueError(
                    f"Слой '{layer_name}' не найден в backbone '{self.backbone_name}'."
                )
            layer: nn.Module = named_children[layer_name]

            # Замыкание захватывает layer_name для правильного ключа в словаре
            def make_hook(name: str):
                def hook(
                    module: nn.Module,
                    input: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                ) -> None:
                    # .detach() отрезает тензор от графа градиентов —
                    # backbone заморожен, градиенты нам не нужны.
                    self._feature_cache[name] = output.detach()

                return hook

            handle = layer.register_forward_hook(make_hook(layer_name))
            self._hook_handles.append(handle)

    def remove_hooks(self) -> None:
        """Удаляет все зарегистрированные хуки. Вызывать после использования."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @contextmanager
    def feature_extraction_context(self) -> Generator[FeatureExtractor, None, None]:
        """
        Context manager: гарантирует удаление хуков даже при исключении.

        Использование::

            with extractor.feature_extraction_context() as ext:
                patches = ext.extract(images)
        """
        try:
            yield self
        finally:
            self.remove_hooks()


    def _run_backbone(self, images: torch.Tensor) -> None:
        """
        Прогоняет изображения через backbone, заполняя _feature_cache.

        forward backbone запускается в torch.no_grad() — градиенты не нужны,
        это экономит память и ускоряет прогон примерно в 2 раза.
        """
        self._feature_cache.clear()
        with torch.no_grad():
            self.backbone(images)
        # После этого _feature_cache содержит:
        #   "layer2": (B, 512,  28, 28)
        #   "layer3": (B, 1024, 14, 14)

    def _aggregate(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Применяет локальную агрегацию f_agg к карте признаков.
        Разрешение карты сохраняется.
        """
        return self._local_agg(feat)

    def _align_resolutions(
        self,
        feat_high_res: torch.Tensor,
        feat_low_res: torch.Tensor,
    ) -> torch.Tensor:
        """
        Приводит feat_low_res к пространственному размеру feat_high_res
        билинейной интерполяцией.

        Args:
            feat_high_res: Тензор с целевым разрешением (B, C1, H, W).
            feat_low_res:  Тензор, который нужно масштабировать (B, C2, h, w).

        Returns:
            feat_low_res, масштабированный до (B, C2, H, W).
        """
        target_h, target_w = feat_high_res.shape[2], feat_high_res.shape[3]
        return F.interpolate(
            feat_low_res,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,  # современная рекомендация PyTorch
        )

    def _adapt_channels(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Сжимает число каналов с 1536 до target_dim=1024 через AdaptiveAvgPool1d.

        Зачем нужен этот шаг:
            После конкатенации признаков layer2 (512 каналов) и layer3
            (1024 канала) получается тензор с 1536 каналами. Авторы используют
            --target_embed_dimension 1024, поэтому нужно привести 1536 - 1024.
            AdaptiveAvgPool1d делает это усреднением групп соседних каналов —
            без обучаемых параметров, быстро.

        Трансформации формы по шагам (пример: B=32, H=W=28):
            Вход:                    (32, 1536, 28, 28)
            permute(0,2,3,1):        (32,  28,  28, 1536)   каналы в конец
            reshape(B*H*W, 1, C):    (25088, 1, 1536)       каждый патч отдельно
            AdaptiveAvgPool1d(1024): (25088, 1, 1024)        пул по оси каналов
            reshape(B, H, W, 1024):  (32,  28,  28, 1024)   восстанавливаем батч
            permute(0,3,1,2):        (32, 1024, 28, 28)      каналы обратно вперёд

        Ключевой момент — почему (B*H*W, 1, C), а не (B, H*W, C):
            AdaptiveAvgPool1d пулит по оси L (последней). Если подать (B, H*W, C),
            то пул применится по оси C длиной 1536 — верно, но тогда все патчи
            одного изображения смешиваются в один батч, что некорректно.
            Подавая каждый патч как (1, C), мы явно говорим: пулить нужно
            именно эти 1536 чисел, и каждый патч обрабатывается независимо.

        Args:
            feat: Тензор формы (B, C, H, W), где C=1536 после конкатенации.

        Returns:
            Тензор формы (B, target_dim, H, W), где target_dim=1024.
            Пространственное разрешение H x W сохраняется.
        """
        B, C, H, W = feat.shape

        # Разворачиваем пространство в одну ось: (B*H*W, 1, C)
        # AdaptiveAvgPool1d пулит по последней оси C: 1536 - 1024
        feat_2d = feat.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, 1, C)
        # (B*H*W, 1, C) - (B*H*W, 1, target_dim)
        feat_adapted = self._channel_adapter(feat_2d)
        # Восстанавливаем пространственные оси
        feat_adapted = feat_adapted.reshape(B, H, W, self.target_dim)
        feat_adapted = feat_adapted.permute(0, 3, 1, 2).contiguous()  # (B, target_dim, H, W)
        return feat_adapted

    def _to_patch_matrix(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Разворачивает пространственную карту признаков в матрицу патчей.

        (B, C, H, W) - (B*H*W, C)

        Каждая строка матрицы — один патч-вектор.
        Это финальный формат для CoresetSampler и NearestNeighborIndex.
        """
        B, C, H, W = feat.shape
        # permute + reshape: (B, C, H, W) - (B, H, W, C) - (B*H*W, C)
        return feat.permute(0, 2, 3, 1).reshape(B * H * W, C)

    # Публичный API

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Полный pipeline извлечения патч-признаков для батча изображений.

        Args:
            images: Батч нормализованных изображений (B, 3, 224, 224).
                    Должны быть предобработаны build_train_transform() из dataset.py.

        Returns:
            Матрица патч-признаков формы (B * H_out * W_out, target_dim).
            Для входа 224×224: H_out = W_out = 28, итого B*784 строк.

        Raises:
            RuntimeError: Если хуки не зарегистрированы (после remove_hooks()).
        """
        if not self._hook_handles:
            raise RuntimeError(
                "Forward-хуки удалены. Создайте новый экземпляр FeatureExtractor "
                "или не вызывайте remove_hooks() до завершения работы."
            )

        images = images.to(self.device)

        # -- Шаг 1: прогон backbone, заполнение _feature_cache -------------
        self._run_backbone(images)

        aggregated_features: list[torch.Tensor] = []
        for layer_name in self.layers:
            if layer_name not in self._feature_cache:
                raise RuntimeError(f"Не удалось получить признаки слоя: {layer_name}")
            aggregated_features.append(self._aggregate(self._feature_cache[layer_name]))

        # Приводим все карты к наибольшему пространственному разрешению.
        max_idx = max(
            range(len(aggregated_features)),
            key=lambda i: aggregated_features[i].shape[2] * aggregated_features[i].shape[3],
        )
        reference = aggregated_features[max_idx]
        aligned_features = [
            feat if feat.shape[2:] == reference.shape[2:] else self._align_resolutions(reference, feat)
            for feat in aggregated_features
        ]

        # -- Шаг 4: конкатенация по оси каналов ----------------------------
        combined = torch.cat(aligned_features, dim=1)

        # -- Шаг 5: адаптация размерности каналов --------------------------
        adapted = self._adapt_channels(combined)  # (B, 1024, 28, 28)

        # -- Шаг 6: разворачивание в матрицу патчей -------------------------
        patch_features = self._to_patch_matrix(adapted)  # (B*784, 1024)

        return patch_features

    @torch.no_grad()
    def extract_with_spatial_info(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        То же что extract(), но дополнительно возвращает пространственный размер.

        Пространственный размер нужен при инференсе: чтобы перевести
        индексы строк матрицы обратно в (h, w) координаты для построения
        карты аномальности.

        Returns:
            patch_features: (B * H_out * W_out, target_dim)
            spatial_size:   (H_out, W_out) — размер карты признаков
        """
        images = images.to(self.device)
        self._run_backbone(images)

        aggregated_features: list[torch.Tensor] = []
        for layer_name in self.layers:
            if layer_name not in self._feature_cache:
                raise RuntimeError(f"Не удалось получить признаки слоя: {layer_name}")
            aggregated_features.append(self._aggregate(self._feature_cache[layer_name]))

        max_idx = max(
            range(len(aggregated_features)),
            key=lambda i: aggregated_features[i].shape[2] * aggregated_features[i].shape[3],
        )
        reference = aggregated_features[max_idx]
        aligned_features = [
            feat if feat.shape[2:] == reference.shape[2:] else self._align_resolutions(reference, feat)
            for feat in aggregated_features
        ]
        combined = torch.cat(aligned_features, dim=1)
        adapted = self._adapt_channels(combined)

        spatial_size = (adapted.shape[2], adapted.shape[3])  # (H_out, W_out)
        patch_features = self._to_patch_matrix(adapted)

        return patch_features, spatial_size

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"backbone={self.backbone_name}, "
            f"layers={list(self.layers)}, "
            f"patch_size={self._local_agg.patch_size}, "
            f"stride={self._local_agg.stride}, "
            f"target_dim={self.target_dim}, "
            f"device={self.device}"
            f")"
        )

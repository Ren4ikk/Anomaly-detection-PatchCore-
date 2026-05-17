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

import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models._api import WeightsEnum

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


def _get_bundled_weights_dir() -> Path | None:
    """
    Возвращает путь к папке bundled_weights/, упакованной PyInstaller.

    В скомпилированном .exe PyInstaller распаковывает --add-data ресурсы
    во временную папку sys._MEIPASS. В режиме разработки ищем bundled_weights/
    рядом с корнем проекта (две директории вверх от этого файла).

    Returns:
        Path к папке с весами если она существует, иначе None.
    """
    # 1. Сборка PyInstaller: sys._MEIPASS содержит распакованные ресурсы
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        candidate = Path(meipass) / "bundled_weights"
        if candidate.is_dir():
            return candidate

    # 2. Режим разработки: bundled_weights/ в корне репозитория
    #    (два уровня вверх от patchcore/feature_extractor.py)
    dev_candidate = Path(__file__).resolve().parent.parent / "bundled_weights"
    if dev_candidate.is_dir():
        return dev_candidate

    return None


def _load_backbone_weights_offline(
    model: nn.Module,
    backbone_name: str,
    weights_enum: WeightsEnum,
) -> bool:
    """
    Пытается загрузить веса из локальной папки bundled_weights/.

    Алгоритм поиска файла:
      1. Берём URL из weights_enum.url → имя файла (последний сегмент URL).
      2. Ищем этот файл в bundled_weights/.
      3. Если нашли — загружаем через weights_enum.transforms() метаданные
         и state_dict через torch.load, применяем к модели.

    Args:
        model:         Экземпляр backbone (уже создан без весов).
        backbone_name: Имя backbone-а для диагностических сообщений.
        weights_enum:  Объект WeightsEnum (например Wide_ResNet50_2_Weights.DEFAULT).

    Returns:
        True если веса загружены из локального файла, False если файл не найден.
    """
    bundled_dir = _get_bundled_weights_dir()
    if bundled_dir is None:
        return False

    # Имя файла совпадает с именем в URL torchvision
    url: str = weights_enum.url  # type: ignore[attr-defined]
    filename = url.split("/")[-1]
    weight_file = bundled_dir / filename

    if not weight_file.is_file():
        # Попытка поиска по частичному имени на случай расхождения версий
        stem = filename.split("-")[0]  # например "wide_resnet50_2"
        candidates = list(bundled_dir.glob(f"{stem}*.pth"))
        if not candidates:
            return False
        weight_file = candidates[0]

    print(f"[FeatureExtractor] Загрузка весов из локального файла: {weight_file.name}")
    state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return True


def _build_backbone(backbone_name: str) -> nn.Module:
    """
    Создаёт backbone и загружает веса.

    Стратегия (offline-first):
      1. Создаём модель БЕЗ весов (weights=None) — быстро, без сети.
      2. Пытаемся загрузить веса из bundled_weights/ (работает offline).
      3. Если локальный файл не найден — загружаем через torchvision
         стандартным способом (требует интернет, но не упадёт в dev-режиме).

    Args:
        backbone_name: Имя модели из torchvision.models (например 'wide_resnet50_2').

    Returns:
        Инициализированный backbone nn.Module с загруженными весами ImageNet.

    Raises:
        ValueError: Если backbone_name не найден в torchvision.models.
    """
    backbone_factory = tv_models.__dict__.get(backbone_name)
    if backbone_factory is None:
        raise ValueError(f"Неизвестный backbone: {backbone_name}")

    # Шаг 1: создаём модель без весов
    model: nn.Module = backbone_factory(weights=None)

    # Шаг 2: пытаемся загрузить локально
    weights_enum = _get_default_weights_enum(backbone_name)
    if weights_enum is not None:
        loaded = _load_backbone_weights_offline(model, backbone_name, weights_enum)
        if loaded:
            return model
        # Локальный файл не найден — предупреждаем и падаем на онлайн-загрузку
        print(
            f"[FeatureExtractor] Локальные веса для '{backbone_name}' не найдены "
            f"в bundled_weights/. Загрузка из интернета (torchvision)…"
        )

    # Шаг 3: онлайн-загрузка через torchvision (fallback для dev-режима)
    model = backbone_factory(weights="DEFAULT")
    return model


def _get_default_weights_enum(backbone_name: str) -> WeightsEnum | None:
    """
    Возвращает объект WeightsEnum.DEFAULT для заданного backbone-а.

    Используется для получения URL файла весов и последующей локальной загрузки.

    Args:
        backbone_name: Имя backbone-а из torchvision.models.

    Returns:
        WeightsEnum.DEFAULT или None если не удалось определить.
    """
    # Явная таблица соответствий — надёжнее чем интроспекция через getattr
    from torchvision.models import (
        Wide_ResNet50_2_Weights,
        Wide_ResNet101_2_Weights,
        ResNet18_Weights,
        ResNet34_Weights,
        ResNet50_Weights,
        ResNet101_Weights,
        ResNeXt50_32X4D_Weights,
        ResNeXt101_32X8D_Weights,
    )
    _WEIGHTS_MAP: dict[str, WeightsEnum] = {
        "wide_resnet50_2":   Wide_ResNet50_2_Weights.DEFAULT,
        "wide_resnet101_2":  Wide_ResNet101_2_Weights.DEFAULT,
        "resnet18":          ResNet18_Weights.DEFAULT,
        "resnet34":          ResNet34_Weights.DEFAULT,
        "resnet50":          ResNet50_Weights.DEFAULT,
        "resnet101":         ResNet101_Weights.DEFAULT,
        "resnext50_32x4d":   ResNeXt50_32X4D_Weights.DEFAULT,
        "resnext101_32x8d":  ResNeXt101_32X8D_Weights.DEFAULT,
    }
    return _WEIGHTS_MAP.get(backbone_name)


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
        x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0)

        # unfold(dimension, size, step):
        #   по высоте: (B, C, H+2p, W+2p) - (B, C, H_out, W+2p, p)
        #   по ширине: - (B, C, H_out, W_out, p, p)
        x_unf = x_padded.unfold(2, p, s).unfold(3, p, s)

        H_out, W_out = x_unf.shape[2], x_unf.shape[3]

        # Шаг 2: reshape - (B*H_out*W_out, C, p, p)
        x_patches = x_unf.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_patches = x_patches.view(B * H_out * W_out, C, p, p)

        # Шаг 3: AdaptiveAvgPool2d(1) — f_agg, усредняет окрестность p×p
        x_agg = F.adaptive_avg_pool2d(x_patches, output_size=1)

        # Шаг 4: убираем лишние оси и восстанавливаем пространственную форму
        x_agg = x_agg.view(B, H_out, W_out, C)
        x_agg = x_agg.permute(0, 3, 1, 2).contiguous()

        return x_agg


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
        # Offline-first загрузка: сначала из bundled_weights/, потом из интернета
        self.backbone = _build_backbone(self.backbone_name)
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad_(False)
        self.backbone.to(self.device)

        # -- Локальная агрегация -----------------------------------------------
        self._local_agg = _LocalAggregation(patch_size=patch_size, stride=stride)

        # -- Адаптация размерности ---------------------------------------------
        self._channel_adapter = nn.AdaptiveAvgPool1d(target_dim)

        # -- Forward-хуки -----------------------------------------------------
        self._feature_cache: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Регистрирует forward-хуки на layer2 и layer3 backbone.

        Хук — это функция, автоматически вызываемая PyTorch после того,
        как слой завершает forward-pass. Хук получает (module, input, output)
        и сохраняет output в _feature_cache.
        """
        for layer_name in self.layers:
            named_children = dict(self.backbone.named_children())
            if layer_name not in named_children:
                raise ValueError(
                    f"Слой '{layer_name}' не найден в backbone '{self.backbone_name}'."
                )
            layer: nn.Module = named_children[layer_name]

            def make_hook(name: str):
                def hook(
                    module: nn.Module,
                    input: tuple[torch.Tensor, ...],
                    output: torch.Tensor,
                ) -> None:
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
        """
        self._feature_cache.clear()
        with torch.no_grad():
            self.backbone(images)

    def _aggregate(self, feat: torch.Tensor) -> torch.Tensor:
        """Применяет локальную агрегацию f_agg к карте признаков."""
        return self._local_agg(feat)

    def _align_resolutions(
        self,
        feat_high_res: torch.Tensor,
        feat_low_res: torch.Tensor,
    ) -> torch.Tensor:
        """
        Приводит feat_low_res к пространственному размеру feat_high_res
        билинейной интерполяцией.
        """
        target_h, target_w = feat_high_res.shape[2], feat_high_res.shape[3]
        return F.interpolate(
            feat_low_res,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

    def _adapt_channels(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Сжимает число каналов с 1536 до target_dim=1024 через AdaptiveAvgPool1d.
        """
        B, C, H, W = feat.shape
        feat_2d = feat.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, 1, C)
        feat_adapted = self._channel_adapter(feat_2d)
        feat_adapted = feat_adapted.reshape(B, H, W, self.target_dim)
        feat_adapted = feat_adapted.permute(0, 3, 1, 2).contiguous()
        return feat_adapted

    def _to_patch_matrix(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Разворачивает пространственную карту признаков в матрицу патчей.
        (B, C, H, W) - (B*H*W, C)
        """
        B, C, H, W = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B * H * W, C)

    # Публичный API

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Полный pipeline извлечения патч-признаков для батча изображений.

        Args:
            images: Батч нормализованных изображений (B, 3, 224, 224).

        Returns:
            Матрица патч-признаков формы (B * H_out * W_out, target_dim).

        Raises:
            RuntimeError: Если хуки не зарегистрированы (после remove_hooks()).
        """
        if not self._hook_handles:
            raise RuntimeError(
                "Forward-хуки удалены. Создайте новый экземпляр FeatureExtractor "
                "или не вызывайте remove_hooks() до завершения работы."
            )

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
        patch_features = self._to_patch_matrix(adapted)

        return patch_features

    @torch.no_grad()
    def extract_with_spatial_info(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """
        То же что extract(), но дополнительно возвращает пространственный размер.

        Returns:
            patch_features: (B * H_out * W_out, target_dim)
            spatial_size:   (H_out, W_out)
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

        spatial_size = (adapted.shape[2], adapted.shape[3])
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

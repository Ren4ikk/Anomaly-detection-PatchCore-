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
_PATCH_SIZE: int = 3

# Шаг s при формировании патч-коллекции.
_STRIDE: int = 1

# Целевая размерность каждого финального патч-вектора.
# layer2 (512) + layer3 (1024) = 1536 → после адаптивного пулинга → 1024.
_TARGET_DIM: int = 1024


# ---------------------------------------------------------------------------
# Таблица соответствий backbone → WeightsEnum.DEFAULT
# ---------------------------------------------------------------------------
def _get_default_weights_enum(backbone_name: str) -> WeightsEnum | None:
    """
    Возвращает WeightsEnum.DEFAULT для заданного backbone-а.

    Явная таблица надёжнее интроспекции через getattr — не зависит от
    внутренних соглашений об именовании в разных версиях torchvision.

    Returns:
        WeightsEnum.DEFAULT или None если backbone не найден в таблице.
    """
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

    _TABLE: dict[str, WeightsEnum] = {
        "wide_resnet50_2":   Wide_ResNet50_2_Weights.DEFAULT,
        "wide_resnet101_2":  Wide_ResNet101_2_Weights.DEFAULT,
        "resnet18":          ResNet18_Weights.DEFAULT,
        "resnet34":          ResNet34_Weights.DEFAULT,
        "resnet50":          ResNet50_Weights.DEFAULT,
        "resnet101":         ResNet101_Weights.DEFAULT,
        "resnext50_32x4d":   ResNeXt50_32X4D_Weights.DEFAULT,
        "resnext101_32x8d":  ResNeXt101_32X8D_Weights.DEFAULT,
    }
    return _TABLE.get(backbone_name)


# ---------------------------------------------------------------------------
# Поиск папки bundled_weights/
# ---------------------------------------------------------------------------
def _get_bundled_weights_dir() -> Path | None:
    """
    Возвращает Path к папке bundled_weights/ или None если она не найдена.

    Порядок поиска:
      1. sys._MEIPASS — PyInstaller распаковывает --add-data ресурсы сюда.
         Этот атрибут существует ТОЛЬКО в скомпилированном .exe.
      2. Корень репозитория — один уровень выше папки patchcore/.
         Путь: patchcore/feature_extractor.py → patchcore/ → repo_root/.
         Используется при локальной офлайн-сборке или после запуска
         scripts/download_weights.py разработчиком.

    Важно: функция возвращает None если папка не существует физически.
    Наличие sys._MEIPASS без папки bundled_weights/ внутри означает
    что сборка была выполнена без --add-data bundled_weights — это
    обрабатывается явной ошибкой в _build_backbone().
    """
    # 1. Сборка PyInstaller: sys._MEIPASS существует только в .exe
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        candidate = Path(meipass) / "bundled_weights"
        if candidate.is_dir():
            return candidate
        # FIX: папка не найдена внутри .exe — возвращаем специальный sentinel.
        # _build_backbone() увидит _MEIPASS и выдаст понятную ошибку
        # вместо попытки онлайн-загрузки (которая гарантированно упадёт в .exe).
        return None

    # 2. Корень репозитория при запуске из IDE / исходников.
    # FIX: скрипт лежит в patchcore/feature_extractor.py,
    # значит .parent = patchcore/, .parent.parent = repo_root/.
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / "bundled_weights"
    if candidate.is_dir():
        return candidate

    return None


def _is_pyinstaller_bundle() -> bool:
    """True если код выполняется внутри скомпилированного PyInstaller .exe."""
    return getattr(sys, "_MEIPASS", None) is not None


# ---------------------------------------------------------------------------
# Поиск файла весов в папке bundled_weights/
# ---------------------------------------------------------------------------
def _find_weight_file(bundled_dir: Path, weights_enum: WeightsEnum) -> Path | None:
    """
    Ищет .pth файл весов в папке bundled_weights/.

    Сначала ищет по точному имени из URL torchvision, затем по префиксу
    на случай расхождения хеш-суффикса между версиями torchvision.

    Args:
        bundled_dir:  Путь к папке bundled_weights/.
        weights_enum: WeightsEnum для получения URL и имени файла.

    Returns:
        Path к найденному файлу или None.
    """
    url: str = weights_enum.url  # type: ignore[attr-defined]
    # Пример: https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
    filename = url.split("/")[-1]

    # Точное совпадение имени файла
    exact = bundled_dir / filename
    if exact.is_file():
        return exact

    # Частичное совпадение по префиксу до первого дефиса.
    # Защита от расхождения хеш-суффикса между версиями torchvision:
    # "wide_resnet50_2-95faca4d.pth" и "wide_resnet50_2-abcdef12.pth"
    # оба начинаются с "wide_resnet50_2".
    stem = filename.split("-")[0]
    candidates = sorted(bundled_dir.glob(f"{stem}*.pth"))
    if candidates:
        return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Основная функция создания backbone
# ---------------------------------------------------------------------------
def _build_backbone(backbone_name: str) -> nn.Module:
    """
    Создаёт backbone и загружает веса.

    Три сценария работы:

      ┌──────────────────────────┬───────────────────────────┬──────────────────┐
      │ Среда                    │ Источник весов            │ Нужен интернет?  │
      ├──────────────────────────┼───────────────────────────┼──────────────────┤
      │ Скомпилированный .exe    │ sys._MEIPASS/             │ Нет              │
      │ (PyInstaller)            │ bundled_weights/          │                  │
      ├──────────────────────────┼───────────────────────────┼──────────────────┤
      │ IDE / исходники,         │ repo_root/                │ Нет              │
      │ после download_weights   │ bundled_weights/          │                  │
      ├──────────────────────────┼───────────────────────────┼──────────────────┤
      │ IDE / исходники,         │ ~/.cache/torch/           │ Только при       │
      │ без download_weights     │ (torchvision кеш)         │ первом запуске   │
      └──────────────────────────┴───────────────────────────┴──────────────────┘

    Args:
        backbone_name: Имя модели из torchvision.models (например 'wide_resnet50_2').

    Returns:
        Инициализированный nn.Module с загруженными весами ImageNet.

    Raises:
        ValueError:   Если backbone_name не найден в torchvision.models.
        RuntimeError: Если код выполняется внутри .exe но bundled_weights/ не найдена
                      (означает что сборка выполнена без --add-data bundled_weights).
    """
    backbone_factory = tv_models.__dict__.get(backbone_name)
    if backbone_factory is None:
        raise ValueError(f"Неизвестный backbone: {backbone_name}")

    weights_enum = _get_default_weights_enum(backbone_name)
    bundled_dir  = _get_bundled_weights_dir()
    in_exe       = _is_pyinstaller_bundle()

    # ------------------------------------------------------------------
    # Сценарий 1 и 2: локальный файл в bundled_weights/
    # ------------------------------------------------------------------
    if bundled_dir is not None and weights_enum is not None:
        weight_file = _find_weight_file(bundled_dir, weights_enum)
        if weight_file is not None:
            print(
                f"[FeatureExtractor] Загрузка весов из bundled_weights/: "
                f"{weight_file.name}"
            )
            # weights=None — не делаем лишней попытки онлайн-загрузки
            model: nn.Module = backbone_factory(weights=None)
            state_dict = torch.load(
                weight_file, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
            return model

        # bundled_weights/ существует, но файл для этого backbone-а не найден
        print(
            f"[FeatureExtractor] bundled_weights/ найдена, но файл весов "
            f"для '{backbone_name}' отсутствует."
        )

    # ------------------------------------------------------------------
    # FIX: внутри скомпилированного .exe онлайн-загрузка невозможна.
    # Если дошли сюда — значит bundled_weights/ либо отсутствует, либо
    # не содержит нужного файла. Обе ситуации — ошибка сборки.
    # ------------------------------------------------------------------
    if in_exe:
        raise RuntimeError(
            f"Не удалось найти веса backbone '{backbone_name}' в bundled_weights/.\n"
            f"Это означает ошибку сборки: папка bundled_weights/ не была упакована "
            f"в дистрибутив или не содержит нужного .pth файла.\n"
            f"Убедитесь что в PyInstaller передан флаг: "
            f"--add-data 'bundled_weights;bundled_weights'\n"
            f"и что перед сборкой был запущен скрипт: python scripts/download_weights.py"
        )

    # ------------------------------------------------------------------
    # Сценарий 3: IDE / исходники без bundled_weights/.
    # weights="DEFAULT" использует ~/.cache/torch/hub/checkpoints/:
    #   - файл уже есть в кеше → загружает без сети
    #   - файла нет → скачивает автоматически (~100–300 МБ)
    # ------------------------------------------------------------------
    print(
        f"[FeatureExtractor] Загрузка весов через torchvision "
        f"(кеш: ~/.cache/torch/hub/checkpoints/)…"
    )
    model = backbone_factory(weights="DEFAULT")
    return model


# ---------------------------------------------------------------------------
# Вспомогательные модули
# ---------------------------------------------------------------------------

class _LocalAggregation(nn.Module):
    """
    Реализует f_agg через Adaptive Average Pooling.

    Принцип работы:
      1. torch.Tensor.unfold разворачивает карту (B, C, H, W) в патчи:
         каждая позиция (h, w) получает окрестность p×p соседних векторов.
         Результат: (B, C, H_out, W_out, p, p)
      2. Reshape объединяет пространственные оси: (B*H_out*W_out, C, p, p)
      3. AdaptiveAvgPool2d(1) усредняет p×p → одно значение на канал.
         Это и есть f_agg = среднее по окрестности.
      4. Reshape обратно: (B, H_out, W_out, C) → (B, C, H_out, W_out)

    Args:
        patch_size: Размер окрестности p (нечётное число для симметрии).
        stride:     Шаг s при обходе карты признаков.
    """

    def __init__(self, patch_size: int = _PATCH_SIZE, stride: int = _STRIDE) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.padding = patch_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Карта признаков формы (B, C, H, W).

        Returns:
            Локально агрегированная карта формы (B, C, H_out, W_out).
            При stride=1 и padding=p//2: H_out == H (разрешение сохраняется).
        """
        B, C, H, W = x.shape
        p = self.patch_size
        s = self.stride
        pad = self.padding

        x_padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0)
        x_unf = x_padded.unfold(2, p, s).unfold(3, p, s)
        # x_unf: (B, C, H_out, W_out, p, p)

        H_out, W_out = x_unf.shape[2], x_unf.shape[3]

        x_patches = x_unf.permute(0, 2, 3, 1, 4, 5).contiguous()
        x_patches = x_patches.view(B * H_out * W_out, C, p, p)

        x_agg = F.adaptive_avg_pool2d(x_patches, output_size=1)
        # x_agg: (B*H_out*W_out, C, 1, 1)

        x_agg = x_agg.view(B, H_out, W_out, C)
        x_agg = x_agg.permute(0, 3, 1, 2).contiguous()
        # x_agg: (B, C, H_out, W_out)

        return x_agg


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """
    Извлекает локально агрегированные патч-признаки из WideResNet-50.

    Полный pipeline:

      images (B, 3, 224, 224)
            forward через WideResNet-50 (заморожен)
      layer2_features (B, 512, 28, 28)   ← j=2, высокое разрешение
      layer3_features (B, 1024, 14, 14)  ← j=3, широкий контекст
            локальная агрегация _LocalAggregation (p=3, s=1)
      layer2_agg (B, 512, 28, 28)
      layer3_agg (B, 1024, 14, 14)
            билинейная интерполяция layer3 → размер layer2
      layer3_upsampled (B, 1024, 28, 28)
            конкатенация по каналам
      combined (B, 1536, 28, 28)
            AdaptiveAvgPool1d → target_dim каналов
      adapted (B, 1024, 28, 28)
            reshape: патчи в строки матрицы
      patch_features (B*784, 1024)

    Args:
        target_dim:    Размерность финального патч-вектора (default: 1024).
        patch_size:    Размер окрестности для локальной агрегации (default: 3).
        stride:        Шаг обхода карты признаков (default: 1).
        backbone_name: Имя backbone из torchvision.models (default: wide_resnet50_2).
        layers:        Слои для извлечения признаков (default: layer2, layer3).
        device:        Устройство для backbone ('cpu' или 'cuda').
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
        # Offline-first: bundled_weights/ → torchvision кеш → онлайн-загрузка.
        # В скомпилированном .exe третий вариант запрещён (RuntimeError).
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
        # Словарь для хранения выходов промежуточных слоёв.
        # Заполняется при каждом forward-проходе backbone.
        self._feature_cache: dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHook] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Регистрирует forward-хуки на указанные слои backbone.

        Хук сохраняет output слоя в _feature_cache при каждом forward-проходе.
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
        """Прогоняет изображения через backbone, заполняя _feature_cache."""
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

        Трансформации формы по шагам (пример: B=32, H=W=28):
            Вход:                    (32, 1536, 28, 28)
            permute(0,2,3,1):        (32,  28,  28, 1536)
            reshape(B*H*W, 1, C):    (25088, 1, 1536)
            AdaptiveAvgPool1d(1024): (25088, 1, 1024)
            reshape(B, H, W, 1024):  (32,  28,  28, 1024)
            permute(0,3,1,2):        (32, 1024, 28, 28)
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
        (B, C, H, W) → (B*H*W, C)
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

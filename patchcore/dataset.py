"""
Этап 1 — Инфраструктура: Датасет для обучения PatchCore.

Класс PatchCoreDataset реализует загрузку изображений из произвольной папки
с препроцессингом, идентичным оригинальной реализации авторов:
resize 256×256, centre-crop 224×224, ImageNet-нормализация.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Константы препроцессинга

# Шаг 1: resize до 256, затем centre-crop до 224 — стандартный ImageNet pipeline.
_RESIZE_SIZE: int = 256
_CROP_SIZE: int = 224

# ImageNet mean / std — backbone WideResNet-50 предобучен на ImageNet.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Расширения файлов, считаемых изображениями
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


def build_train_transform(
    resize: int = _RESIZE_SIZE,
    crop: int = _CROP_SIZE,
    mean: Sequence[float] = _IMAGENET_MEAN,
    std: Sequence[float] = _IMAGENET_STD,
) -> transforms.Compose:
    """
    Строит стандартный transform для train-изображений PatchCore.

    Pipeline (соответствует оригинальному коду авторов):
      1. Resize до (resize × resize)  — сохраняет контекст патчей
      2. CenterCrop до (crop × crop)  — убирает края, устраняет артефакты resize
      3. ToTensor                     — [0,255] uint8 - [0,1] float32
      4. Normalize(ImageNet)          — приводит к распределению, на котором
                                        обучался backbone

    Args:
        resize: Промежуточный размер перед кропом.
        crop:   Итоговый размер изображения.
        mean:   Per-channel mean для нормализации (ImageNet по умолчанию).
        std:    Per-channel std для нормализации (ImageNet по умолчанию).

    Returns:
        Составной transform torchvision.
    """
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ]
    )


class PatchCoreDataset(Dataset):
    """
    Универсальный датасет для фазы ОБУЧЕНИЯ PatchCore.

    Загружает все изображения из указанной директории (рекурсивно),
    применяет стандартный ImageNet-препроцессинг и возвращает только
    тензоры изображений без масок — обучение PatchCore не требует
    разметки аномалий.

    Структура датасета (MVTec AD как пример):
        root/
          train/
            good/
              *.png

    Класс намеренно не привязан к MVTec: подойдёт любая папка с
    изображениями нормального класса.

    Args:
        root:       Путь к директории с обучающими изображениями.
        transform:  Кастомный transform. Если None — используется
                    стандартный build_train_transform().
        recursive:  Искать ли изображения рекурсивно во вложенных папках.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        transform: Optional[Callable] = None,
        recursive: bool = True,
    ) -> None:
        super().__init__()

        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Директория не найдена: {self.root}")

        self.transform = transform if transform is not None else build_train_transform()
        self.image_paths = self._collect_images(self.root, recursive=recursive)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"В директории '{self.root}' не найдено ни одного изображения. "
                f"Поддерживаемые расширения: {sorted(_IMAGE_EXTENSIONS)}"
            )

    @staticmethod
    def _collect_images(root: Path, *, recursive: bool) -> list[Path]:
        """Собирает пути ко всем изображениям в директории."""
        pattern = "**/*" if recursive else "*"
        paths = [
            p
            for p in root.glob(pattern)
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        ]
        return sorted(paths)

    # Dataset API

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Возвращает один тензор изображения формы (C, H, W).

        Примечание: датасет только для train, поэтому маски / метки
        аномалий не возвращаются
        """
        image_path = self.image_paths[idx]

        image = Image.open(image_path).convert("RGB")
        return self.transform(image)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root='{self.root}', "
            f"n_images={len(self)}, "
            f"transform={self.transform}"
            f")"
        )


# Фабричная функция для создания DataLoader

def make_train_dataloader(
    root: Union[str, os.PathLike],
    batch_size: int = 32,
    num_workers: int = 1,
    transform: Optional[Callable] = None,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Создаёт DataLoader для обучающего датасета PatchCore.

    Shuffle=False: порядок итерации не влияет на качество,
    зато обеспечивает детерминированность между запусками.

    Args:
        root:        Путь к папке с обучающими изображениями.
        batch_size:  Количество изображений на батч.
        num_workers: Число параллельных процессов загрузки.
        transform:   Кастомный transform или None (стандартный).
        pin_memory:  Ускоряет перенос данных на GPU через pinned memory.

    Returns:
        Готовый к использованию DataLoader.
    """
    dataset = PatchCoreDataset(root=root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # PatchCore не требует перемешивания при fit
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

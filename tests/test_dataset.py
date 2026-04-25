"""
tests/test_dataset.py — тесты для patchcore/dataset.py

Покрываем:
  • build_train_transform: корректность pipeline, форма выхода, нормализация
  • PatchCoreDataset:      сбор изображений, __len__, __getitem__, обработка ошибок
  • make_train_dataloader: создание DataLoader, batch-форма
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from patchcore.dataset import (
    PatchCoreDataset,
    _CROP_SIZE,
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    _RESIZE_SIZE,
    build_train_transform,
    make_train_dataloader,
)


# -----------------------------------------------------------------------------
# build_train_transform
# -----------------------------------------------------------------------------

class TestBuildTrainTransform:
    """Тесты функции build_train_transform()."""

    def test_output_shape(self):
        """Трансформ должен вернуть тензор (3, 224, 224)."""
        transform = build_train_transform()
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        result = transform(img)
        assert result.shape == (3, _CROP_SIZE, _CROP_SIZE)

    def test_output_dtype(self):
        """Выход должен быть float32."""
        transform = build_train_transform()
        img = Image.fromarray(np.ones((256, 256, 3), dtype=np.uint8) * 128)
        result = transform(img)
        assert result.dtype == torch.float32

    def test_custom_sizes(self):
        """Кастомные resize/crop должны работать корректно."""
        transform = build_train_transform(resize=128, crop=96)
        img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
        result = transform(img)
        assert result.shape == (3, 96, 96)

    def test_normalization_range(self):
        """
        После нормализации значения должны быть примерно в [-3, 3].
        Белое изображение (255) - высокое значение, чёрное (0) - низкое.
        """
        transform = build_train_transform()
        white = Image.fromarray(np.full((300, 300, 3), 255, dtype=np.uint8))
        black = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
        white_t = transform(white)
        black_t = transform(black)
        # Нормализованное белое > нормализованное чёрное
        assert white_t.mean() > black_t.mean()

    def test_imagenet_normalization_applied(self):
        """
        Серое изображение (128/255 ≈ 0.502) нормализуется как (0.502 - mean) / std.
        Проверяем канал 0 с mean=0.485, std=0.229.
        """
        transform = build_train_transform()
        # Однородное серое изображение 256×256
        grey = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
        result = transform(grey)
        # Ожидаемое значение: (128/255 - 0.485) / 0.229 ≈ 0.075
        expected_ch0 = (128 / 255 - _IMAGENET_MEAN[0]) / _IMAGENET_STD[0]
        assert abs(float(result[0].mean()) - expected_ch0) < 0.02


# -----------------------------------------------------------------------------
# PatchCoreDataset
# -----------------------------------------------------------------------------

class TestPatchCoreDataset:
    """Тесты класса PatchCoreDataset."""

    def test_len(self, train_image_dir: Path):
        """Датасет должен найти все 5 изображений."""
        ds = PatchCoreDataset(root=train_image_dir)
        assert len(ds) == 5

    def test_getitem_shape(self, train_image_dir: Path):
        """Каждый элемент должен быть тензором (3, 224, 224)."""
        ds = PatchCoreDataset(root=train_image_dir)
        sample = ds[0]
        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (3, _CROP_SIZE, _CROP_SIZE)

    def test_getitem_dtype(self, train_image_dir: Path):
        """Тип данных должен быть float32."""
        ds = PatchCoreDataset(root=train_image_dir)
        assert ds[0].dtype == torch.float32

    def test_recursive_collection(self, nested_image_dir: Path):
        """Рекурсивный режим должен найти изображения в подпапках (6 = 3*2)."""
        ds = PatchCoreDataset(root=nested_image_dir, recursive=True)
        assert len(ds) == 6

    def test_non_recursive_collection(self, nested_image_dir: Path):
        """
        Нерекурсивный режим не находит изображения в подпапках.
        Все изображения лежат во вложенных директориях, поэтому
        датасет бросает RuntimeError — корректное поведение конструктора.
        """
        with pytest.raises(RuntimeError, match="не найдено ни одного изображения"):
            PatchCoreDataset(root=nested_image_dir, recursive=False)

    def test_nonexistent_dir_raises(self, tmp_path: Path):
        """Несуществующая директория должна вызывать ValueError."""
        with pytest.raises(ValueError, match="не найдена"):
            PatchCoreDataset(root=tmp_path / "does_not_exist")

    def test_empty_dir_raises(self, empty_dir: Path):
        """Пустая директория должна вызывать RuntimeError."""
        with pytest.raises(RuntimeError, match="не найдено ни одного изображения"):
            PatchCoreDataset(root=empty_dir)

    def test_non_image_files_ignored(self, non_image_dir: Path):
        """Файлы без расширений изображений должны игнорироваться - RuntimeError."""
        with pytest.raises(RuntimeError):
            PatchCoreDataset(root=non_image_dir)

    def test_custom_transform(self, train_image_dir: Path):
        """Кастомный трансформ должен применяться вместо стандартного."""
        from torchvision import transforms
        custom = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(48),
            transforms.ToTensor(),
        ])
        ds = PatchCoreDataset(root=train_image_dir, transform=custom)
        assert ds[0].shape == (3, 48, 48)

    def test_deterministic_order(self, train_image_dir: Path):
        """Два экземпляра датасета должны возвращать одинаковый первый элемент."""
        ds1 = PatchCoreDataset(root=train_image_dir)
        ds2 = PatchCoreDataset(root=train_image_dir)
        assert torch.allclose(ds1[0], ds2[0])

    def test_repr(self, train_image_dir: Path):
        """__repr__ не должен бросать исключений и содержать ключевую информацию."""
        ds = PatchCoreDataset(root=train_image_dir)
        r = repr(ds)
        assert "PatchCoreDataset" in r
        assert "n_images=5" in r

    def test_rgba_image_converted_to_rgb(self, tmp_path: Path):
        """RGBA-изображения должны конвертироваться в RGB без ошибок."""
        img_dir = tmp_path / "rgba"
        img_dir.mkdir()
        arr = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        Image.fromarray(arr, mode="RGBA").save(img_dir / "rgba.png")
        ds = PatchCoreDataset(root=img_dir)
        sample = ds[0]
        assert sample.shape[0] == 3  # RGB, не RGBA


# -----------------------------------------------------------------------------
# make_train_dataloader
# -----------------------------------------------------------------------------

class TestMakeTrainDataloader:
    """Тесты фабричной функции make_train_dataloader()."""

    def test_batch_shape(self, train_image_dir: Path):
        """Первый батч DataLoader должен иметь форму (batch_size, 3, 224, 224)."""
        loader = make_train_dataloader(
            root=train_image_dir,
            batch_size=3,
            num_workers=0,
        )
        batch = next(iter(loader))
        # Датасет содержит 5 изображений, batch_size=3 - первый батч = 3
        assert batch.shape == (3, 3, _CROP_SIZE, _CROP_SIZE)

    def test_total_samples(self, train_image_dir: Path):
        """DataLoader должен итерировать все изображения без потерь."""
        loader = make_train_dataloader(
            root=train_image_dir,
            batch_size=2,
            num_workers=0,
        )
        total = sum(batch.shape[0] for batch in loader)
        assert total == 5  # 5 изображений в датасете

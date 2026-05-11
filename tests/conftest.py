"""
conftest.py — общие фикстуры для всех тестов patchcore.

Фикстуры делятся на три группы:
  1. Тензорные примитивы — синтетические данные без реальных изображений
  2. Файловая система    — временные директории с .png-файлами для датасета
  3. Предобученные объекты (scope="session") — FeatureExtractor загружается
     один раз на всю сессию, чтобы не качать веса при каждом тесте
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# -----------------------------------------------------------------------------
# Константы для синтетических данных
# -----------------------------------------------------------------------------

BATCH_SIZE = 2          # маленький батч — тесты должны быть быстрыми
IMG_SIZE   = 224        # стандартный входной размер PatchCore
CHANNELS   = 3
FEAT_DIM   = 1024       # размерность патч-вектора после FeatureExtractor
SPATIAL_H  = 28         # H карты признаков для 224×224 входа
SPATIAL_W  = 28         # W карты признаков для 224×224 входа
N_PATCHES  = BATCH_SIZE * SPATIAL_H * SPATIAL_W  # 2 * 784 = 1568


# -----------------------------------------------------------------------------
# Группа 1: Тензорные примитивы
# -----------------------------------------------------------------------------

@pytest.fixture
def random_images() -> torch.Tensor:
    """
    Батч случайных нормализованных изображений (B, 3, 224, 224).
    Значения подобраны близко к ImageNet-нормализованным (μ≈0, σ≈1).
    """
    torch.manual_seed(42)
    return torch.randn(BATCH_SIZE, CHANNELS, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def random_patch_features() -> torch.Tensor:
    """Матрица патч-признаков (N_patches, FEAT_DIM) — выход FeatureExtractor."""
    torch.manual_seed(0)
    return torch.randn(N_PATCHES, FEAT_DIM)


@pytest.fixture
def small_feature_matrix() -> torch.Tensor:
    """
    Маленькая матрица (50, 16) для быстрого тестирования CoresetSampler
    и NearestNeighborIndex без накладных расходов на большие данные.
    """
    torch.manual_seed(7)
    return torch.randn(50, 16)


@pytest.fixture
def memory_bank_for_index() -> torch.Tensor:
    """
    Банк памяти (100, 32) для тестирования NearestNeighborIndex.
    Достаточно большой для значимого kNN-поиска.
    """
    torch.manual_seed(13)
    return torch.randn(100, 32)


# -----------------------------------------------------------------------------
# Группа 2: Файловая система (PatchCoreDataset)
# -----------------------------------------------------------------------------

def _make_png(path: Path, size: tuple[int, int] = (256, 256)) -> None:
    """Сохраняет случайное RGB-изображение в PNG-файл."""
    arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


@pytest.fixture
def train_image_dir(tmp_path: Path) -> Path:
    """
    Временная директория с 5 PNG-изображениями 256×256.
    Структура MVTec-подобная: tmp/train/good/*.png
    """
    good_dir = tmp_path / "train" / "good"
    good_dir.mkdir(parents=True)
    for i in range(5):
        _make_png(good_dir / f"img_{i:03d}.png")
    return good_dir


@pytest.fixture
def nested_image_dir(tmp_path: Path) -> Path:
    """
    Директория с изображениями в нескольких подпапках — тест рекурсивного сбора.
    """
    for sub in ["a", "b", "c"]:
        subdir = tmp_path / sub
        subdir.mkdir()
        for i in range(2):
            _make_png(subdir / f"{sub}_{i}.png")
    return tmp_path


@pytest.fixture
def empty_dir(tmp_path: Path) -> Path:
    """Пустая директория без изображений — должна вызывать RuntimeError."""
    d = tmp_path / "empty"
    d.mkdir()
    return d


@pytest.fixture
def non_image_dir(tmp_path: Path) -> Path:
    """Директория только с не-изображениями — должна вызывать RuntimeError."""
    d = tmp_path / "non_images"
    d.mkdir()
    (d / "file.txt").write_text("hello")
    (d / "data.json").write_text("{}")
    return d


# -----------------------------------------------------------------------------
# Группа 3: Метрики — синтетические GT-данные
# -----------------------------------------------------------------------------

@pytest.fixture
def perfect_detector_data():
    """
    Данные идеального детектора:
    аномальные изображения имеют score=1.0, нормальные — score=0.0.
    Image AUROC должен быть равен 1.0.
    """
    n_normal, n_anomaly = 10, 10
    gt_labels     = np.array([0] * n_normal + [1] * n_anomaly, dtype=np.int32)
    image_scores  = np.array([0.0] * n_normal + [1.0] * n_anomaly, dtype=np.float32)
    return gt_labels, image_scores


@pytest.fixture
def random_detector_data():
    """
    Данные случайного детектора.
    Image AUROC должен быть около 0.5.
    """
    rng = np.random.default_rng(42)
    n = 100
    gt_labels    = rng.integers(0, 2, size=n).astype(np.int32)
    # Гарантируем хотя бы один класс каждого типа
    gt_labels[0] = 0
    gt_labels[1] = 1
    image_scores = rng.random(size=n).astype(np.float32)
    return gt_labels, image_scores


@pytest.fixture
def pixel_metric_data():
    """
    Синтетические карты аномальности и GT-маски для pixel AUROC и PRO.
    4 изображения: 2 нормальных (маска = 0) + 2 аномальных (маска с дефектом).
    """
    H, W = 32, 32
    n_total, n_anomaly = 4, 2

    gt_labels = np.array([0, 0, 1, 1], dtype=np.int32)

    gt_masks = np.zeros((n_total, H, W), dtype=np.uint8)
    # На аномальных изображениях дефект в верхнем левом квадранте
    gt_masks[2, :H//2, :W//2] = 1
    gt_masks[3, :H//2, :W//2] = 1

    # Карты аномальности: аномальные имеют высокие значения там, где дефект
    anomaly_maps = np.zeros((n_total, H, W), dtype=np.float32)
    anomaly_maps[2, :H//2, :W//2] = 0.9
    anomaly_maps[3, :H//2, :W//2] = 0.8
    anomaly_maps[3, H//2:, W//2:] = 0.05  # небольшой шум

    # Image scores: max по карте
    image_scores = anomaly_maps.max(axis=(1, 2))

    return gt_labels, image_scores, anomaly_maps, gt_masks

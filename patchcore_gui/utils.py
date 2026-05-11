"""
Вспомогательные функции: списки изображений, подготовка кадра 224×224 под карту аномалий,
отрисовка тепловой карты и overlay в QPixmap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap


# Расширения и размеры совпадают с patchcore.dataset (resize 256 - center crop 224).
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)
_RESIZE: int = 256
_CROP: int = 224


def list_image_paths(folder: str) -> list[str]:
    """
    Возвращает отсортированные пути к изображениям в папке (не рекурсивно).

    Args:
        folder: Корневая папка «конвейера».

    Returns:
        Список абсолютных путей в виде строк.
    """
    root = Path(folder)
    if not root.is_dir():
        return []
    paths: list[Path] = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS:
            paths.append(p)
    return [str(p.resolve()) for p in sorted(paths, key=lambda x: x.name.lower())]


def load_display_rgb_224(image_path: str) -> np.ndarray:
    """
    Загружает RGB-кадр 224×224 в том же геометрическом пространстве, что и anomaly_map.

    Pipeline соответствует build_train_transform без нормализации: Resize(256), CenterCrop(224).

    Args:
        image_path: Путь к файлу изображения.

    Returns:
        Массив формы (224, 224, 3), dtype uint8.
    """
    pil = Image.open(image_path).convert("RGB")
    pil = pil.resize((_RESIZE, _RESIZE), Image.Resampling.BILINEAR)
    w, h = pil.size
    left = (w - _CROP) // 2
    top = (h - _CROP) // 2
    pil = pil.crop((left, top, left + _CROP, top + _CROP))
    return np.asarray(pil, dtype=np.uint8)


def anomaly_map_to_bgr_heatmap(
    anomaly_map: np.ndarray,
    score_min: float | None = None,
    score_max: float | None = None,
) -> np.ndarray:
    """
    Преобразует карту сырых L2-расстояний в цветную тепловую карту (BGR, uint8) через OpenCV JET.

    Нормализация выполняется по глобальным границам score_min/score_max из модели,
    что обеспечивает сравнимость карт между изображениями:
    нормальные кадры (низкие значения) -> синий, аномальные -> красный.

    Если границы не переданы или некорректны - используется локальный min/max карты
    (fallback: карты визуально красивы, но не сравнимы между собой).

    Args:
        anomaly_map: Двумерный массив сырых L2-расстояний (H, W).
        score_min:   Нижняя граница шкалы (из model.score_min).
        score_max:   Верхняя граница шкалы (из model.score_max).

    Returns:
        Массив (H, W, 3) BGR uint8.
    """
    import cv2

    m = np.asarray(anomaly_map, dtype=np.float32)
    if score_min is not None and score_max is not None and score_max > score_min:
        m = (m - float(score_min)) / float(score_max - score_min)
    else:
        # Fallback: нормализация по локальному min/max.
        # Защита от сплошного красного когда сырые значения >>1 и clip(0,1) даёт 1.0.
        lo, hi = float(m.min()), float(m.max())
        if hi > lo:
            m = (m - lo) / (hi - lo)
        else:
            m = np.zeros_like(m)
    m = np.clip(m, 0.0, 1.0)
    gray = (m * 255.0).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return heat_bgr


def blend_rgb_with_heat_bgr(
    rgb_uint8: np.ndarray,
    heat_bgr_uint8: np.ndarray,
    alpha: float = 0.45,
    intensity_map: np.ndarray | None = None,
) -> np.ndarray:
    """
    Смешивает RGB-оригинал с тепловой картой (BGR) в RGB uint8.

    Args:
        rgb_uint8: (H, W, 3) RGB.
        heat_bgr_uint8: (H, W, 3) BGR.
        alpha: Вес тепловой карты (0 — только оригинал, 1 — только heat).

    Returns:
        RGB uint8 (H, W, 3).
    """
    import cv2

    a = float(np.clip(alpha, 0.0, 1.0))
    heat_rgb = cv2.cvtColor(heat_bgr_uint8, cv2.COLOR_BGR2RGB)
    base = rgb_uint8.astype(np.float32)
    over = heat_rgb.astype(np.float32)
    if intensity_map is None:
        out = (1.0 - a) * base + a * over
    else:
        # Снижаем вклад overlay на "холодных" участках, чтобы нормальные кадры
        # оставались визуально ближе к оригиналу.
        w = np.asarray(intensity_map, dtype=np.float32)
        w = np.clip(w, 0.0, 1.0)
        a_map = (a * w)[..., None]
        out = (1.0 - a_map) * base + a_map * over
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def numpy_rgb_to_qpixmap(rgb: np.ndarray) -> QPixmap:
    """Конвертирует RGB uint8 (H, W, 3) в QPixmap."""
    arr = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, ch = arr.shape
    if ch != 3:
        raise ValueError("Ожидается RGB с 3 каналами.")
    bytes_per_line = ch * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


def scaled_pixmap(
    source: QPixmap,
    target_width: int,
    target_height: int,
) -> QPixmap:
    """
    Масштабирует pixmap с сохранением пропорций, вписывая в заданный прямоугольник.

    Args:
        source: Исходный pixmap.
        target_width: Максимальная ширина области отображения.
        target_height: Максимальная высота области отображения.

    Returns:
        Отмасштабированный QPixmap.
    """
    if source.isNull():
        return source
    return source.scaled(
        target_width,
        target_height,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )

"""Тип записи истории инференса для галереи."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class InferenceHistoryEntry:
    """Один кадр конвейера: сырые данные для отображения и пересчёта вердикта."""

    path: str
    raw_score: float
    rgb: np.ndarray
    anomaly_map: np.ndarray
    elapsed_ms: float

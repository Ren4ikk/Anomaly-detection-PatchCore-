"""
Фоновые потоки для загрузки весов и инференса PatchCore без блокировки GUI.
"""

from __future__ import annotations

import queue
import time
from typing import Optional

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from patchcore.dataset import build_train_transform
from patchcore.patchcore import PatchCore


def normalize_score(raw_score: float, score_min: float, score_max: float) -> float:
    """Линейно нормирует скор в [0, 1] по диапазону модели."""
    denom = score_max - score_min
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((raw_score - score_min) / denom, 0.0, 1.0))


class ModelLoadWorker(QThread):
    """
    Однократная загрузка чекпоинта в объекте PatchCore (в отдельном потоке).

    После успешной загрузки объект модели можно передавать в InferenceWorker
    только если дальнейшее использование будет в том же потоке — поэтому здесь
    мы только проверяем, что файл читается, и эмитим метаданные.

    Альтернатива: создавать PatchCore только внутри InferenceWorker (выбрано там).
    Этот воркер оставлен для явной валидации пути к .pt без блокировки UI.
    """

    load_ok = pyqtSignal(float, float, float)  # score_min, score_max, threshold_raw
    load_failed = pyqtSignal(str)

    def __init__(self, model_path: str, device: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._device = device

    def run(self) -> None:
        try:
            m = PatchCore(device=self._device)
            m.load(self._model_path)
            self.load_ok.emit(m.score_min, m.score_max, m.threshold)
        except Exception as exc:  # noqa: BLE001 — показать пользователю любую ошибку IO/weights
            self.load_failed.emit(str(exc))


class InferenceWorker(QThread):
    """
    Очередь инференса: модель создаётся и используется только в этом QThread.

    В главный поток уходят сырое значение скора, нормализованный скор [0,1],
    карта аномалий и время; ошибки — отдельным сигналом.
    """

    model_ready = pyqtSignal(float, float, float)  # score_min, score_max, threshold_raw
    inference_done = pyqtSignal(str, float, float, object, float)
    # path, raw_score, norm_score [0,1], anomaly_map (numpy), elapsed_ms
    inference_failed = pyqtSignal(str, str)  # path, message

    def __init__(self, model_path: str, device: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._device = device
        self._tasks: queue.Queue[Optional[str]] = queue.Queue()

    def enqueue_path(self, image_path: str) -> None:
        """Поставить изображение в очередь на обработку."""
        self._tasks.put(image_path)

    def request_stop(self) -> None:
        """Сигнал остановки: после завершения текущей задачи поток завершится."""
        self._tasks.put(None)

    def run(self) -> None:
        try:
            model = PatchCore(device=self._device)
            model.load(self._model_path)
            transform = build_train_transform()
            self.model_ready.emit(model.score_min, model.score_max, model.threshold)

            while True:
                path = self._tasks.get()
                if path is None:
                    break
                try:
                    t0 = time.perf_counter()
                    pil = Image.open(path).convert("RGB")
                    tensor = transform(pil).unsqueeze(0)
                    result = model.predict_single(tensor)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    norm = normalize_score(result.image_score, model.score_min, model.score_max)
                    self.inference_done.emit(
                        path,
                        float(result.image_score),
                        norm,
                        result.anomaly_map,
                        float(elapsed_ms),
                    )
                except Exception as exc:  # noqa: BLE001
                    self.inference_failed.emit(path, str(exc))
        except Exception as exc:  # noqa: BLE001 — сбой load / нет файла
            self.inference_failed.emit(self._model_path, str(exc))


class TrainingWorker(QThread):
    """
    Полный цикл обучения PatchCore в фоновом потоке: fit → compute_score_range → save.

    Папка ``train_image_dir`` должна содержать только изображения нормального класса
    (эталоны без дефектов) — по ним строится банк памяти и статистика порога.
    """

    training_started = pyqtSignal()
    # threshold, score_min, score_max (сырые значения с модели после compute_score_range)
    training_success = pyqtSignal(float, float, float)
    training_finished = pyqtSignal()
    training_failed = pyqtSignal(str)

    def __init__(
        self,
        train_image_dir: str,
        save_path: str,
        device: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._train_image_dir = train_image_dir
        self._save_path = save_path
        self._device = device

    def run(self) -> None:
        self.training_started.emit()
        try:
            model = PatchCore(device=self._device)
            model.fit(self._train_image_dir)
            model.compute_score_range(self._train_image_dir)
            model.save(self._save_path)
            thr = float(model.threshold)
            smin = float(model.score_min)
            smax = float(model.score_max)
            self.training_success.emit(thr, smin, smax)
        except Exception as exc:  # noqa: BLE001
            self.training_failed.emit(str(exc))
        else:
            self.training_finished.emit()


def select_device(preference: str = "auto") -> str:
    """
    Возвращает строку устройства для PatchCore.

    Args:
        preference: 'cuda', 'cpu' или 'auto' (CUDA если доступна).
    """
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

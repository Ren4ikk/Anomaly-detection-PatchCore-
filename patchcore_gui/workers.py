"""
Фоновые потоки для загрузки банка памяти и инференса PatchCore без блокировки GUI.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from patchcore.dataset import build_train_transform
from patchcore.metrics import Metrics
from patchcore.patchcore import PatchCore
from patchcore_gui.settings_dialog import TrainingSettings


def normalize_score(raw_score: float, score_min: float, score_max: float) -> float:
    """Линейно нормирует скор в [0, 1] по диапазону банка памяти."""
    denom = score_max - score_min
    if denom <= 1e-12:
        return 0.0
    return float(np.clip((raw_score - score_min) / denom, 0.0, 1.0))


class MetaLoadWorker(QThread):
    """
    Асинхронно читает метаданные из .pt файла банка памяти (score_min, score_max, threshold)
    БЕЗ создания FeatureExtractor и загрузки весов backbone.

    Используется в _choose_model и _try_restore_model_metadata вместо синхронного
    torch.load в главном потоке, что устраняет фриз GUI при чтении крупных файлов.

    Signals:
        meta_ready(score_min, score_max, threshold_raw, state_dict):
            Метаданные успешно прочитаны.
        meta_failed(message):
            Произошла ошибка чтения файла.
    """

    meta_ready = pyqtSignal(float, float, float, dict)
    meta_failed = pyqtSignal(str)

    def __init__(self, model_path: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._model_path = model_path

    def run(self) -> None:
        try:
            state = torch.load(self._model_path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("Ожидался словарь состояния PatchCore.")
            score_min = float(state.get("score_min", 0.0))
            score_max = float(state.get("score_max", 1.0))
            threshold = float(state.get("threshold", 0.5))
            self.meta_ready.emit(score_min, score_max, threshold, state)
        except Exception as exc:  # noqa: BLE001
            self.meta_failed.emit(str(exc))


class ModelLoadWorker(QThread):
    """
    Однократная загрузка чекпоинта в объекте PatchCore (в отдельном потоке).
    Оставлен для явной валидации пути к .pt без блокировки UI.
    """

    load_ok = pyqtSignal(float, float, float)  # score_min, score_max, threshold_raw
    load_failed = pyqtSignal(str)

    def __init__(self, model_path: str, device: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._model_path = model_path
        self._device = device

    def run(self) -> None:
        try:
            # FIX: читаем state один раз, передаём параметры в PatchCore,
            # затем вызываем load() — который тоже читает state, но это
            # единственный публичный способ инициализации nn_index.
            # Двойное чтение здесь оправдано: первый read — для параметров
            # конструктора (backbone_name, layers, patch_size), второй —
            # внутри load() для восстановления memory_bank.
            # В ModelLoadWorker это некритично — он используется только
            # для валидации файла, не для инференса.
            state = torch.load(self._model_path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("Ожидался словарь состояния PatchCore.")
            m = PatchCore(
                device=self._device,
                backbone_name=str(state.get("backbone_name", "wide_resnet50_2")),
                layers=tuple(state.get("layers", ("layer2", "layer3"))),
                patch_size=int(state.get("patch_size", 3)),
            )
            m.load(self._model_path)
            self.load_ok.emit(m.score_min, m.score_max, m.threshold)
        except Exception as exc:  # noqa: BLE001
            self.load_failed.emit(str(exc))


class InferenceWorker(QThread):
    """
    Очередь инференса: банк памяти создаётся и используется только в этом QThread.

    В главный поток уходят сырое значение скора, нормализованный скор [0,1],
    карта аномалий и время; ошибки — отдельным сигналом.

    Жизненный цикл:
      1. start()        — поток запускается, загружает backbone + корсет
      2. model_ready    — сигнал: метаданные готовы, можно ставить задачи
      3. enqueue_path() — добавить путь в очередь на обработку
      4. request_stop() — вставить sentinel None, поток завершится чисто
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
        # Флаг: backbone + корсет уже загружены и готовы к инференсу.
        # Устанавливается из главного потока в _on_model_ready.
        self.is_model_loaded: bool = False

    def enqueue_path(self, image_path: str) -> None:
        """Поставить изображение в очередь на обработку."""
        self._tasks.put(image_path)

    def request_stop(self) -> None:
        """Сигнал остановки: sentinel None завершает цикл обработки."""
        self._tasks.put(None)

    def run(self) -> None:
        try:
            # FIX: читаем state ОДИН РАЗ и передаём уже готовый state в load_from_state(),
            # чтобы избежать двойного torch.load (был: torch.load здесь + torch.load внутри
            # model.load()) и двойного создания FeatureExtractor (был: PatchCore.__init__
            # создавал backbone, затем model.load() пересоздавал его снова).
            state = torch.load(self._model_path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                raise ValueError("Ожидался словарь состояния PatchCore.")

            model = PatchCore(
                device=self._device,
                backbone_name=str(state.get("backbone_name", "wide_resnet50_2")),
                layers=tuple(state.get("layers", ("layer2", "layer3"))),
                patch_size=int(state.get("patch_size", 3)),
            )
            # FIX: используем load_from_state() вместо load() — передаём уже
            # прочитанный state dict, исключая повторное чтение файла с диска.
            model.load_from_state(state)

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
    Полный цикл формирования эталонного банка памяти PatchCore в фоновом потоке:
    fit → compute_score_range → save.

    Папка ``train_image_dir`` должна содержать только изображения нормального класса
    (эталоны без дефектов) — по ним строится банк памяти и статистика порога.
    """

    training_started = pyqtSignal()
    # threshold, score_min, score_max, metrics_dict
    training_success = pyqtSignal(float, float, float, dict)
    training_finished = pyqtSignal()
    training_failed = pyqtSignal(str)
    # Некритичное предупреждение: банк сформирован, но что-то пошло не так с метриками/масками
    training_warning = pyqtSignal(str)
    # Пользователь нажал «Отмена» — банк не сохранён
    training_cancelled = pyqtSignal()

    def __init__(
        self,
        train_image_dir: str,
        save_path: str,
        device: str,
        settings: TrainingSettings,
        metrics_val_dir: str | None = None,
        metrics_mask_dir: str | None = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._train_image_dir = train_image_dir
        self._save_path = save_path
        self._device = device
        self._settings = settings
        self._metrics_val_dir = metrics_val_dir
        self._metrics_mask_dir = metrics_mask_dir
        self._cancel_requested: bool = False

    def request_cancel(self) -> None:
        """Безопасная отмена: поднимает флаг, поток завершится на ближайшей точке проверки."""
        self._cancel_requested = True

    def _should_stop(self) -> bool:
        return self._cancel_requested

    def run(self) -> None:
        self.training_started.emit()
        try:
            if self._settings.device == "auto":
                torch_device = select_device()
            elif self._settings.device in {"cpu", "cuda"}:
                torch_device = self._settings.device
            else:
                torch_device = self._device

            model = PatchCore(
                device=torch_device,
                coreset_ratio=self._settings.coreset_ratio,
                n_reweight_nn=self._settings.n_reweight_nn,
                gaussian_sigma=self._settings.gaussian_sigma,
                backbone_name=self._settings.backbone_name,
                layers=self._settings.layers,
                patch_size=self._settings.patch_size,
                use_gpu_faiss=self._settings.use_gpu_faiss,
            )
            model.fit(self._train_image_dir, should_stop=self._should_stop)

            if self._should_stop():
                raise InterruptedError("Формирование банка отменено пользователем.")

            model.compute_score_range(self._train_image_dir, should_stop=self._should_stop)

            if self._should_stop():
                raise InterruptedError("Формирование банка отменено пользователем.")

            if self._settings.threshold_mode == "f1_optimal":
                self._apply_f1_threshold(model=model)

            # Вычисляем метрики если указана validation директория
            metrics_dict: dict = {}
            if self._metrics_val_dir:
                try:
                    metrics_dict = self._compute_metrics(model)
                    model.save_metrics(metrics_dict)
                except Exception as exc:  # noqa: BLE001
                    msg = str(exc)
                    print(f"[TrainingWorker] Не удалось вычислить метрики: {msg}")
                    self.training_warning.emit(
                        f"Эталонный банк памяти сформирован успешно, однако вычислить "
                        f"метрики качества не удалось:\n\n{msg}\n\n"
                        f"Проверьте структуру папки Validation (должны быть подпапки "
                        f"'good' и папки с дефектами) и при необходимости — папку GT-масок."
                    )

            model.save(self._save_path)
            thr = float(model.threshold)
            smin = float(model.score_min)
            smax = float(model.score_max)
            self.training_success.emit(thr, smin, smax, metrics_dict)
        except InterruptedError:
            print("[TrainingWorker] Формирование банка отменено пользователем.")
            self.training_cancelled.emit()
        except Exception as exc:  # noqa: BLE001
            self.training_failed.emit(str(exc))
        else:
            self.training_finished.emit()

    def _apply_f1_threshold(self, model: PatchCore) -> None:
        if not self._settings.validation_dir:
            raise ValueError("Не выбрана папка Validation для F1-оптимального порога.")

        val_images, val_labels, val_masks = self._load_validation_data(
            self._settings.validation_dir,
            self._settings.gt_mask_dir if self._settings.gt_mask_dir else None,
        )
        if len(val_images) == 0:
            raise ValueError("Validation папка не содержит изображений.")

        image_scores: list[float] = []
        anomaly_maps: list[np.ndarray] = []

        for i in range(0, len(val_images), model.batch_size):
            batch = torch.stack(val_images[i : i + model.batch_size])
            batch_results = model.predict(batch)
            for r in batch_results:
                image_scores.append(float(r.image_score))
                anomaly_maps.append(np.asarray(r.anomaly_map, dtype=np.float32))

        y_true_image = np.asarray(val_labels, dtype=np.int32)
        y_scores_image = np.asarray(image_scores, dtype=np.float32)

        if self._settings.threshold_objective == "pixel_f1":
            if not any(m is not None for m in val_masks):
                raise ValueError("Для Pixel-level F1 нужны GT-маски в выбранной папке.")
            gt_masks = []
            pred_maps = []
            for idx, pred in enumerate(anomaly_maps):
                mask = val_masks[idx]
                if mask is None:
                    gt_masks.append(np.zeros(pred.shape, dtype=np.uint8))
                else:
                    gt_masks.append(mask.astype(np.uint8))
                pred_maps.append(pred)

            y_true_pixel = np.stack(gt_masks).astype(np.int32).reshape(-1)
            y_scores_pixel = np.stack(pred_maps).astype(np.float32).reshape(-1)
            f1_thr, _ = Metrics.compute_f1_optimal_threshold(y_true_pixel, y_scores_pixel)
        else:
            f1_thr, _ = Metrics.compute_f1_optimal_threshold(y_true_image, y_scores_image)

        model.threshold = float(f1_thr)

    def _compute_metrics(self, model: "PatchCore") -> dict:
        """
        Прогоняет validation-данные через банк памяти и вычисляет метрики.
        Image AUROC всегда. Pixel AUROC и PRO — только если есть GT-маски.
        """
        from patchcore.metrics import Metrics

        val_images, val_labels, val_masks = self._load_validation_data(
            self._metrics_val_dir,
            self._metrics_mask_dir,
        )
        if len(val_images) == 0:
            raise ValueError("Validation папка для метрик не содержит изображений.")

        image_scores: list[float] = []
        anomaly_maps_list: list[np.ndarray] = []

        print(f"[TrainingWorker] Вычисление метрик ({len(val_images)} изображений)...")
        for i in range(0, len(val_images), model.batch_size):
            batch = torch.stack(val_images[i : i + model.batch_size])
            batch_results = model.predict(batch)
            for r in batch_results:
                image_scores.append(float(r.image_score))
                anomaly_maps_list.append(np.asarray(r.anomaly_map, dtype=np.float32))

        y_true = np.asarray(val_labels, dtype=np.int32)
        y_scores = np.asarray(image_scores, dtype=np.float32)

        gt_masks_arr = None
        maps_arr = None
        has_masks = any(m is not None for m in val_masks)

        if self._metrics_mask_dir and not has_masks:
            raise ValueError(
                f"Папка GT-масок указана ({self._metrics_mask_dir}), но ни одна маска "
                f"не найдена. Ожидаемая структура: <папка_масок>/<категория>/<имя_файла>.*  "
                f"(например: masks/crack/001.png). Pixel AUROC и PRO Score не будут вычислены."
            )

        if has_masks:
            gt_list = []
            map_list = []
            for idx, pred in enumerate(anomaly_maps_list):
                mask = val_masks[idx]
                gt_list.append(mask if mask is not None else np.zeros(pred.shape, dtype=np.uint8))
                map_list.append(pred)
            gt_masks_arr = np.stack(gt_list)
            maps_arr = np.stack(map_list)

        results = Metrics().compute(
            image_scores=y_scores,
            gt_labels=y_true,
            anomaly_maps=maps_arr,
            gt_masks=gt_masks_arr,
        )

        metrics_dict: dict = {
            "image_auroc": float(results.image_auroc),
            "image_fpr": results.image_fpr.tolist(),
            "image_tpr": results.image_tpr.tolist(),
        }

        if has_masks and results.pixel_auroc > 0:
            metrics_dict["pixel_auroc"] = float(results.pixel_auroc)
            metrics_dict["pixel_fpr"] = results.pixel_fpr.tolist()
            metrics_dict["pixel_tpr"] = results.pixel_tpr.tolist()
            metrics_dict["pro_score"] = float(results.pro_score)

            try:
                from patchcore.metrics import _compute_pro
                anomaly_idx = y_true == 1
                if anomaly_idx.sum() > 0 and gt_masks_arr is not None:
                    _, pro_fprs, pro_vals = _compute_pro(
                        maps_arr[anomaly_idx],
                        gt_masks_arr[anomaly_idx],
                        num_thresh=100,
                    )
                    metrics_dict["pro_fpr"] = pro_fprs.tolist()
                    metrics_dict["pro_tpr"] = pro_vals.tolist()
            except Exception as _exc:
                print(f"[TrainingWorker] PRO curve: {_exc}")

        print(f"[TrainingWorker] Image AUROC: {results.image_auroc:.4f}")
        if has_masks:
            print(f"[TrainingWorker] Pixel AUROC: {results.pixel_auroc:.4f}")
            print(f"[TrainingWorker] PRO Score  : {results.pro_score:.4f}")

        return metrics_dict

    @staticmethod
    def _load_validation_data(
        validation_dir: str,
        gt_mask_dir: str | None,
    ) -> tuple[list[torch.Tensor], list[int], list[np.ndarray | None]]:
        transform = build_train_transform()
        val_path = Path(validation_dir)
        if not val_path.is_dir():
            raise ValueError(f"Validation директория не найдена: {validation_dir}")

        category_dirs = sorted([p for p in val_path.iterdir() if p.is_dir()])
        good_dir = val_path / "good"
        if not good_dir.is_dir():
            raise ValueError("Validation директория должна содержать подпапку 'good'.")
        if len(category_dirs) < 2:
            raise ValueError(
                "Validation директория должна содержать 'good' и хотя бы одну папку дефектов."
            )

        image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        images: list[torch.Tensor] = []
        labels: list[int] = []
        masks: list[np.ndarray | None] = []

        mask_root = Path(gt_mask_dir) if gt_mask_dir else None
        if mask_root is not None and not mask_root.is_dir():
            raise ValueError(f"Директория GT масок не найдена: {gt_mask_dir}")

        for category_dir in category_dirs:
            category = category_dir.name
            label = 0 if category == "good" else 1
            for img_path in sorted(category_dir.iterdir()):
                if not img_path.is_file() or img_path.suffix.lower() not in image_ext:
                    continue
                image = Image.open(img_path).convert("RGB")
                images.append(transform(image))
                labels.append(label)

                if label == 0 or mask_root is None:
                    masks.append(None)
                    continue

                candidates = sorted(mask_root.glob(f"{category}/{img_path.stem}*"))
                if not candidates:
                    masks.append(None)
                    continue
                mask_img = Image.open(candidates[0]).convert("L")
                mask_img = mask_img.resize((224, 224), Image.NEAREST)
                masks.append((np.array(mask_img) > 0).astype(np.uint8))

        return images, labels, masks


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

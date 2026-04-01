"""
Скрипт запуска PatchCore.

Использование:

  # Обучение и оценка на MVTec AD (категория bottle):
  python run.py --train_dir ./data/bottle/train/good
                --test_dir  ./data/bottle/test
                --mask_dir  ./data/bottle/ground_truth
                --save_path ./models/bottle.pt

  # Только инференс (модель уже обучена):
  python run.py --load_path ./models/bottle.pt
                --test_dir  ./data/bottle/test
                --mask_dir  ./data/bottle/ground_truth

  # Минимальный запуск (без метрик):
  python run.py --train_dir ./data/bottle/train/good
                --test_dir  ./data/bottle/test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# ─── проверка GPU ─────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU: {name}")
        return torch.device("cuda")
    print("[Device] GPU недоступен — используется CPU.")
    return torch.device("cpu")


# ─── загрузка тестового датасета с масками ────────────────────────────────────

def _load_test_data(
    test_dir: str,
    mask_dir: str | None,
    transform,
) -> tuple[list[torch.Tensor], list[int], list[np.ndarray | None], list[str]]:
    """
    Загружает тестовые изображения из структуры MVTec AD:

      test/
        good/      ← нормальные (label=0, mask=None)
        defect_A/  ← аномальные (label=1, mask из mask_dir)
        defect_B/
        ...

    Returns:
        images:  список тензоров (3, 224, 224)
        labels:  0 = норма, 1 = аномалия
        masks:   GT-маски (H, W) uint8 или None для нормальных
        names:   имена файлов для логирования
    """
    test_path = Path(test_dir)
    mask_path = Path(mask_dir) if mask_dir else None

    images, labels, masks, names = [], [], [], []

    _IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

    for category_dir in sorted(test_path.iterdir()):
        if not category_dir.is_dir():
            continue
        is_normal = category_dir.name == "good"

        for img_file in sorted(category_dir.iterdir()):
            if img_file.suffix.lower() not in _IMAGE_EXT:
                continue

            # Изображение
            image = Image.open(img_file).convert("RGB")
            images.append(transform(image))
            labels.append(0 if is_normal else 1)
            names.append(f"{category_dir.name}/{img_file.name}")

            # Маска
            if is_normal or mask_path is None:
                masks.append(None)
            else:
                mask_file = mask_path / category_dir.name / img_file.stem
                # MVTec хранит маски как *_mask.png
                candidates = list(mask_path.glob(
                    f"{category_dir.name}/{img_file.stem}*"
                ))
                if candidates:
                    mask_img = Image.open(candidates[0]).convert("L")
                    mask_img = mask_img.resize((224, 224), Image.NEAREST)
                    mask_arr = (np.array(mask_img) > 0).astype(np.uint8)
                    masks.append(mask_arr)
                else:
                    masks.append(None)

    return images, labels, masks, names


# ─── основной пайплайн ────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    # Импорты здесь — чтобы ошибки импорта были понятны
    # from patchcore.dataset import build_train_transform
    from patchcore import PatchCore, build_train_transform, Metrics
    # from patchcore.metrics import Metrics

    device = _get_device()
    transform = build_train_transform()

    # ── Инициализация модели ──────────────────────────────────────────────────
    model = PatchCore(
        device=device,
        coreset_ratio=args.coreset_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_gpu_faiss=False,   # faiss-cpu — FAISS на CPU
    )

    # ── fit() или load() ─────────────────────────────────────────────────────
    if args.load_path:
        model.load(args.load_path)
    elif args.train_dir:
        t0 = time.time()
        model.fit(args.train_dir)
        print(f"[fit] Время: {time.time() - t0:.1f}с\n")

        # Вычисляем диапазон скоров по train-данным для корректной визуализации
        model.compute_score_range(args.train_dir)

        if args.save_path:
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            model.save(args.save_path)
    else:
        print("Ошибка: укажите --train_dir или --load_path")
        sys.exit(1)

    # ── predict() ────────────────────────────────────────────────────────────
    if not args.test_dir:
        print("--test_dir не указан, инференс пропущен.")
        return

    print(f"\n[Predict] Загрузка тестовых данных из: {args.test_dir}")
    images, labels, masks, names = _load_test_data(
        args.test_dir, args.mask_dir, transform
    )

    if len(images) == 0:
        print("Тестовые изображения не найдены.")
        return

    print(f"[Predict] Найдено {len(images)} изображений "
          f"({sum(labels)} аномальных, {len(labels) - sum(labels)} нормальных)\n")

    # Прогоняем батчами
    image_scores: list[float] = []
    anomaly_maps: list[np.ndarray] = []

    t0 = time.time()
    for i in range(0, len(images), args.batch_size):
        batch_imgs = torch.stack(images[i : i + args.batch_size])
        results = model.predict(batch_imgs)
        for r in results:
            image_scores.append(r.image_score)
            anomaly_maps.append(r.anomaly_map)

        processed = min(i + args.batch_size, len(images))
        print(f"  Обработано: {processed}/{len(images)}", end="\r")

    print(f"\n[Predict] Время инференса: {time.time() - t0:.1f}с")

    # ── Метрики ───────────────────────────────────────────────────────────────
    gt_labels = np.array(labels, dtype=np.int32)
    scores_arr = np.array(image_scores, dtype=np.float32)

    # Проверяем что есть оба класса (нужно для AUROC)
    if len(np.unique(gt_labels)) < 2:
        print("\n[Metrics] Нет обоих классов — AUROC не вычисляется.")
        _print_scores(names, image_scores, labels)
        return

    metrics = Metrics()

    # Pixel-level метрики — только если есть маски
    has_masks = any(m is not None for m in masks)
    if has_masks and args.mask_dir:
        # Формируем массивы только для изображений с масками
        valid_idx = [i for i, m in enumerate(masks) if m is not None]

        # Для нормальных изображений создаём нулевые маски
        gt_masks_list = []
        maps_list = []
        for i in range(len(images)):
            if masks[i] is not None:
                gt_masks_list.append(masks[i])
            else:
                gt_masks_list.append(np.zeros(anomaly_maps[i].shape, dtype=np.uint8))
            maps_list.append(anomaly_maps[i])

        gt_masks_arr = np.stack(gt_masks_list)   # (N, H, W)
        maps_arr = np.stack(maps_list)            # (N, H, W)

        results_metrics = metrics.compute(
            image_scores=scores_arr,
            gt_labels=gt_labels,
            anomaly_maps=maps_arr,
            gt_masks=gt_masks_arr,
        )
    else:
        results_metrics = metrics.compute(
            image_scores=scores_arr,
            gt_labels=gt_labels,
        )

    # ── Вывод результатов ─────────────────────────────────────────────────────
    print("\n" + "=" * 45)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 45)
    print(results_metrics)
    print("=" * 45)

    if args.verbose:
        _print_scores(names, image_scores, labels)


def _print_scores(
    names: list[str],
    scores: list[float],
    labels: list[int],
) -> None:
    """Выводит скоры по каждому изображению."""
    print("\nПодробные скоры:")
    print(f"  {'Файл':<45} {'Скор':>8}  {'GT'}")
    print("  " + "-" * 62)
    for name, score, label in zip(names, scores, labels):
        gt_str = "ANOMALY" if label == 1 else "normal "
        print(f"  {name:<45} {score:>8.4f}  {gt_str}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PatchCore — обнаружение аномалий",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Пути
    parser.add_argument(
        "--train_dir", type=str, default=None,
        help="Директория с нормальными train-изображениями"
    )
    parser.add_argument(
        "--test_dir", type=str, default=None,
        help="Директория с тестовыми изображениями (структура MVTec)"
    )
    parser.add_argument(
        "--mask_dir", type=str, default=None,
        help="Директория с GT-масками (структура MVTec ground_truth)"
    )
    parser.add_argument(
        "--save_path", type=str, default=None,
        help="Путь для сохранения обученной модели (.pt)"
    )
    parser.add_argument(
        "--load_path", type=str, default=None,
        help="Путь к сохранённой модели для загрузки (.pt)"
    )

    # Гиперпараметры
    parser.add_argument(
        "--coreset_ratio", type=float, default=0.1,
        help="Доля патчей в косете (0.1 = PatchCore-10%%)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Размер батча"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Число процессов DataLoader"
    )

    # Прочее
    parser.add_argument(
        "--verbose", action="store_true",
        help="Выводить скоры по каждому изображению"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())

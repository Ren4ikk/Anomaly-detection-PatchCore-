"""
Инференс PatchCore на одном изображении.

Сохраняет:
  - тепловую карту аномальности (heatmap)
  - наложение тепловой карты на оригинал (overlay)
  - side-by-side сравнение (original | heatmap | overlay)

Использование:
  python infer.py --model ./models/capsule.pt --image ./test_image.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # без GUI — работает на любой машине
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from dataset import build_train_transform
from patchcore import PatchCore


# ─────────────────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"[Device] GPU: {name}")
        return torch.device("cuda")
    print("[Device] GPU недоступен — используется CPU.")
    return torch.device("cpu")


def _save_results(
    original: np.ndarray,
    anomaly_map: np.ndarray,
    image_score: float,
    output_dir: Path,
    image_name: str,
    threshold: float,
) -> None:
    """
    Сохраняет три файла:
      {name}_heatmap.png  — тепловая карта
      {name}_overlay.png  — наложение на оригинал
      {name}_comparison.png — все три рядом
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_name).stem

    # Нормализуем карту в [0,1] только для визуализации
    # (в patchcore.py карта хранится в сырых L2-расстояниях для корректных метрик)
    map_min, map_max = anomaly_map.min(), anomaly_map.max()
    if map_max > map_min:
        anomaly_map_vis = (anomaly_map - map_min) / (map_max - map_min)
    else:
        anomaly_map_vis = anomaly_map.copy()

    # Цветовая карта: синий (норма) → красный (аномалия)
    colormap = cm.jet
    heatmap_rgba = colormap(anomaly_map_vis)       # (H, W, 4) float [0,1]
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Overlay: смешиваем оригинал и тепловую карту
    alpha = 0.5
    overlay = (
        (1 - alpha) * original.astype(np.float32) +
        alpha * heatmap_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    # ── Сохраняем heatmap ────────────────────────────────────────────────
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(anomaly_map_vis, cmap="jet", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Anomaly Score: {image_score:.4f}", fontsize=12)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Сохраняем overlay ────────────────────────────────────────────────
    overlay_path = output_dir / f"{stem}_overlay.png"
    Image.fromarray(overlay).save(overlay_path)

    # ── Сохраняем side-by-side comparison ────────────────────────────────
    comparison_path = output_dir / f"{stem}_comparison.png"
    is_anomaly = image_score >= threshold
    status = f"ANOMALY ({image_score:.4f})" if is_anomaly else f"NORMAL ({image_score:.4f})"
    status_color = "red" if is_anomaly else "green"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Оригинал", fontsize=13)
    axes[0].axis("off")

    im = axes[1].imshow(anomaly_map_vis, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Тепловая карта", fontsize=13)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay)
    axes[2].set_title("Наложение", fontsize=13)
    axes[2].axis("off")

    fig.suptitle(
        status,
        fontsize=15,
        fontweight="bold",
        color=status_color,
    )
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Тепловая карта : {heatmap_path}")
    print(f"  Наложение      : {overlay_path}")
    print(f"  Сравнение      : {comparison_path}")


# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:

    device = _get_device()

    # ── Загрузка модели ───────────────────────────────────────────────────
    print(f"\n[Infer] Загрузка модели: {args.model}")
    model = PatchCore(device=device)
    model.load(args.model)

    # ── Загрузка изображения ──────────────────────────────────────────────
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Ошибка: файл не найден — {image_path}")
        return

    print(f"[Infer] Изображение: {image_path}")

    transform = build_train_transform()
    pil_image = Image.open(image_path).convert("RGB")

    # Оригинал для визуализации (ресайзим до 224 для совпадения с тепловой картой)
    original_resized = np.array(pil_image.resize((224, 224), Image.BILINEAR))

    # Препроцессинг для модели
    tensor = transform(pil_image).unsqueeze(0)  # (1, 3, 224, 224)

    # ── Инференс ──────────────────────────────────────────────────────────
    print("[Infer] Запуск инференса...")
    result = model.predict_single(tensor)

    print(f"\n{'='*40}")
    print(f"  Image Score : {result.image_score:.6f}")
    print(f"  Порог       : {args.threshold}")
    verdict = "АНОМАЛИЯ" if result.image_score >= args.threshold else "НОРМА"
    verdict_sign = "🔴" if result.image_score >= args.threshold else "🟢"
    print(f"  Вердикт     : {verdict_sign} {verdict}")
    print(f"{'='*40}\n")

    # ── Сохранение результатов ────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    print(f"[Infer] Сохранение результатов в: {output_dir}")

    _save_results(
        original=original_resized,
        anomaly_map=result.anomaly_map,
        image_score=result.image_score,
        output_dir=output_dir,
        image_name=image_path.name,
        threshold=args.threshold,
    )

    print("\n[Infer] Готово.")


# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PatchCore — инференс на одном изображении",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Путь к сохранённой модели (.pt)"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Путь к тестовому изображению"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./results",
        help="Директория для сохранения результатов"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Порог для вынесения вердикта НОРМА/АНОМАЛИЯ"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())

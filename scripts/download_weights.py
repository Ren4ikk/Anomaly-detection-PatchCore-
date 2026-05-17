"""
scripts/download_weights.py
───────────────────────────
Скрипт для скачивания весов всех backbone-ов, используемых в приложении.
Запускается один раз во время сборки EXE на CI.

Сохраняет .pth файлы в ./bundled_weights/ (относительно корня репозитория).
Эта папка затем упаковывается в дистрибутив через PyInstaller --add-data.

Список backbone-ов должен совпадать с вариантами в SettingsDialog
(patchcore_gui/settings_dialog.py → _backbone_combo.addItems(...)).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import torch
import torchvision.models as tv_models

# ---------------------------------------------------------------------------
# Все backbone-ы, доступные в настройках приложения.
# Ключ  — имя модели (передаётся в tv_models.__dict__[name]).
# Значение — объект Weights с методом .url для скачивания.
# ---------------------------------------------------------------------------
# torchvision хранит «дефолтные» веса через атрибут DEFAULT у каждого
# *_Weights enum-а. Нам нужно только скачать файл — создание полной модели
# не требуется (экономит RAM на CI).
# ---------------------------------------------------------------------------

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

BACKBONE_WEIGHTS: dict[str, object] = {
    "wide_resnet50_2":    Wide_ResNet50_2_Weights.DEFAULT,
    "wide_resnet101_2":   Wide_ResNet101_2_Weights.DEFAULT,
    "resnet18":           ResNet18_Weights.DEFAULT,
    "resnet34":           ResNet34_Weights.DEFAULT,
    "resnet50":           ResNet50_Weights.DEFAULT,
    "resnet101":          ResNet101_Weights.DEFAULT,
    "resnext50_32x4d":    ResNeXt50_32X4D_Weights.DEFAULT,
    "resnext101_32x8d":   ResNeXt101_32X8D_Weights.DEFAULT,
}

# Папка назначения — ./bundled_weights/ рядом со скриптом (т.е. в корне репо)
DEST_DIR = Path(__file__).resolve().parent.parent / "bundled_weights"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Целевая папка: {DEST_DIR}")
    print()

    # torch.hub скачивает в TORCH_HOME/hub/checkpoints/
    # Читаем фактический путь кеша чтобы найти файлы после скачивания.
    hub_dir = Path(torch.hub.get_dir()) / "checkpoints"

    total = len(BACKBONE_WEIGHTS)
    for idx, (name, weights_enum) in enumerate(BACKBONE_WEIGHTS.items(), start=1):
        print(f"[{idx}/{total}] {name} ...", flush=True)

        # Получаем URL из метаданных weights enum-а
        url: str = weights_enum.url  # type: ignore[attr-defined]
        filename = url.split("/")[-1]          # например: wide_resnet50_2-95faca4d.pth
        dest_file = DEST_DIR / filename

        if dest_file.exists():
            print(f"  ✓ уже есть: {dest_file.name}  ({dest_file.stat().st_size // 1024 // 1024} МБ)")
            continue

        # torch.hub.load_state_dict_from_url скачивает в hub_dir и возвращает state_dict.
        # progress=True — показывает прогресс в логе CI.
        print(f"  ↓ скачиваю: {url}", flush=True)
        torch.hub.load_state_dict_from_url(
            url,
            model_dir=str(hub_dir),
            map_location="cpu",
            progress=True,
            check_hash=True,
        )

        # Копируем скачанный файл в bundled_weights/
        cached_file = hub_dir / filename
        if not cached_file.exists():
            # Некоторые версии torchvision используют другое имя кеша — ищем по частичному совпадению
            candidates = list(hub_dir.glob(f"{filename.split('-')[0]}*"))
            if not candidates:
                print(f"  ✗ ОШИБКА: файл не найден в {hub_dir}", file=sys.stderr)
                sys.exit(1)
            cached_file = candidates[0]

        shutil.copy2(cached_file, dest_file)
        size_mb = dest_file.stat().st_size // 1024 // 1024
        print(f"  ✓ сохранён: {dest_file.name}  ({size_mb} МБ)")

    print()
    print(f"Готово. Скачано / проверено {total} файлов весов в {DEST_DIR}")


if __name__ == "__main__":
    main()

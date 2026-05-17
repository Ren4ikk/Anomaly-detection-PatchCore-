"""
scripts/download_weights.py
───────────────────────────
Скрипт для скачивания весов всех backbone-ов, используемых в приложении.
Запускается один раз во время сборки EXE на CI (см. build-exe.yml).

Сохраняет .pth файлы в <repo_root>/bundled_weights/.
Эта папка упаковывается в дистрибутив через PyInstaller --add-data.

Список backbone-ов совпадает с вариантами в SettingsDialog
(patchcore_gui/settings_dialog.py → _backbone_combo.addItems(...)).

Запуск вручную (для локальной офлайн-сборки):
    python scripts/download_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

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

# ---------------------------------------------------------------------------
# Все backbone-ы, доступные в настройках приложения.
# Ключ   — отображаемое имя (для диагностики).
# Значение — WeightsEnum.DEFAULT с атрибутом .url.
# ---------------------------------------------------------------------------
BACKBONE_WEIGHTS = {
    "wide_resnet50_2":   Wide_ResNet50_2_Weights.DEFAULT,
    "wide_resnet101_2":  Wide_ResNet101_2_Weights.DEFAULT,
    "resnet18":          ResNet18_Weights.DEFAULT,
    "resnet34":          ResNet34_Weights.DEFAULT,
    "resnet50":          ResNet50_Weights.DEFAULT,
    "resnet101":         ResNet101_Weights.DEFAULT,
    "resnext50_32x4d":   ResNeXt50_32X4D_Weights.DEFAULT,
    "resnext101_32x8d":  ResNeXt101_32X8D_Weights.DEFAULT,
}

# FIX: скрипт лежит в scripts/, корень репо — на один уровень выше (.parent),
# а не на два (.parent.parent).
#   scripts/download_weights.py
#     └─ Path(__file__).resolve().parent  →  <repo>/scripts/
#          └─ .parent                     →  <repo>/          ← корень репо
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR  = REPO_ROOT / "bundled_weights"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Целевая папка: {DEST_DIR}")
    print()

    total = len(BACKBONE_WEIGHTS)
    for idx, (name, weights_enum) in enumerate(BACKBONE_WEIGHTS.items(), start=1):
        print(f"[{idx}/{total}] {name} ...", flush=True)

        # Имя файла берём из URL torchvision
        # Пример URL:  https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
        # Имя файла:   wide_resnet50_2-95faca4d.pth
        url: str = weights_enum.url  # type: ignore[attr-defined]
        filename = url.split("/")[-1]
        dest_file = DEST_DIR / filename

        if dest_file.is_file():
            size_mb = dest_file.stat().st_size // 1024 // 1024
            print(f"  ✓ уже есть: {dest_file.name}  ({size_mb} МБ)")
            continue

        print(f"  ↓ скачиваю: {url}", flush=True)

        # FIX: передаём DEST_DIR напрямую как model_dir.
        # load_state_dict_from_url кладёт файл прямо в model_dir —
        # никакого копирования не нужно, файл сразу оказывается в bundled_weights/.
        torch.hub.load_state_dict_from_url(
            url,
            model_dir=str(DEST_DIR),
            map_location="cpu",
            progress=True,
            check_hash=True,
        )

        # Проверяем что файл действительно появился
        if not dest_file.is_file():
            # Параноидальная проверка: в редких случаях torchvision
            # может добавить суффикс версии к имени файла
            candidates = sorted(DEST_DIR.glob(f"{filename.split('-')[0]}*.pth"))
            if not candidates:
                print(
                    f"  ✗ ОШИБКА: файл не найден в {DEST_DIR} после скачивания.",
                    file=sys.stderr,
                )
                sys.exit(1)
            actual_file = candidates[0]
            print(f"  ✓ сохранён под именем: {actual_file.name}")
        else:
            size_mb = dest_file.stat().st_size // 1024 // 1024
            print(f"  ✓ сохранён: {dest_file.name}  ({size_mb} МБ)")

    print()
    print(f"Готово. Скачано / проверено {total} файлов весов в {DEST_DIR}")
    print()
    # Итоговый список для верификации в логе CI
    print("Содержимое bundled_weights/:")
    for f in sorted(DEST_DIR.glob("*.pth")):
        print(f"  {f.name}  ({f.stat().st_size // 1024 // 1024} МБ)")


if __name__ == "__main__":
    main()

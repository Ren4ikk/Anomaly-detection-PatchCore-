"""
scripts/download_weights.py
---------------------------
Downloads weights for all backbone models used in the application.
Run once at build time before PyInstaller packaging (see build-exe.yml).

Saves .pth files to <repo_root>/bundled_weights/.
That directory is then packed into the distribution via --add-data.

The backbone list must match the options in SettingsDialog
(patchcore_gui/settings_dialog.py -> _backbone_combo.addItems(...)).

Manual run (for local offline build):
    python scripts/download_weights.py
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# FIX: Force UTF-8 output on Windows (default console uses CP1252 which
# cannot encode many Unicode characters and raises UnicodeEncodeError).
# Must be done before any print() calls.
# ---------------------------------------------------------------------------
import sys
import io

if sys.stdout.encoding and sys.stdout.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.upper() not in ("UTF-8", "UTF8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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
# All backbone models available in the application settings.
# Key   - model name (passed to tv_models.__dict__[name]).
# Value - WeightsEnum.DEFAULT with a .url attribute.
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

# Script is at scripts/download_weights.py:
#   Path(__file__).resolve()         -> <repo>/scripts/download_weights.py
#   .parent                          -> <repo>/scripts/
#   .parent.parent                   -> <repo>/              <- repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR  = REPO_ROOT / "bundled_weights"


def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Destination directory: {DEST_DIR}")
    print()

    total = len(BACKBONE_WEIGHTS)
    for idx, (name, weights_enum) in enumerate(BACKBONE_WEIGHTS.items(), start=1):
        print(f"[{idx}/{total}] {name} ...", flush=True)

        # Derive filename from the torchvision URL.
        # Example URL:  https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
        # Filename:     wide_resnet50_2-95faca4d.pth
        url: str = weights_enum.url  # type: ignore[attr-defined]
        filename = url.split("/")[-1]
        dest_file = DEST_DIR / filename

        if dest_file.is_file():
            size_mb = dest_file.stat().st_size // 1024 // 1024
            print(f"  [skip] already exists: {dest_file.name}  ({size_mb} MB)")
            continue

        print(f"  [download] {url}", flush=True)

        # Pass DEST_DIR as model_dir so the file lands directly there —
        # no extra copy step needed.
        torch.hub.load_state_dict_from_url(
            url,
            model_dir=str(DEST_DIR),
            map_location="cpu",
            progress=True,
            check_hash=True,
        )

        # Verify the file appeared (paranoia check for version mismatches).
        if not dest_file.is_file():
            stem = filename.split("-")[0]
            candidates = sorted(DEST_DIR.glob(f"{stem}*.pth"))
            if not candidates:
                print(
                    f"  [ERROR] file not found in {DEST_DIR} after download.",
                    file=sys.stderr,
                )
                sys.exit(1)
            actual_file = candidates[0]
            print(f"  [ok] saved as: {actual_file.name}")
        else:
            size_mb = dest_file.stat().st_size // 1024 // 1024
            print(f"  [ok] saved: {dest_file.name}  ({size_mb} MB)")

    print()
    print(f"Done. Downloaded / verified {total} weight files in {DEST_DIR}")
    print()
    print("Contents of bundled_weights/:")
    for f in sorted(DEST_DIR.glob("*.pth")):
        print(f"  {f.name}  ({f.stat().st_size // 1024 // 1024} MB)")


if __name__ == "__main__":
    main()

"""
Точка входа: десктопное приложение визуального контроля на базе PatchCore.

Запуск из корня репозитория::

    python main.py

Опционально зафиксировать устройство (по умолчанию — auto: CUDA при наличии)::

    set PATCHCORE_GUI_DEVICE=cpu
    python main.py
"""

from __future__ import annotations

import os
import sys

from PyQt6.QtWidgets import QApplication

from patchcore_gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("PatchCore QC")
    app.setOrganizationName("patchcore-realization")

    device_pref = os.environ.get("PATCHCORE_GUI_DEVICE", "auto").strip().lower()
    if device_pref not in ("auto", "cpu", "cuda"):
        device_pref = "auto"

    window = MainWindow(device_preference=device_pref)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()

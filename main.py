from __future__ import annotations

import os
import sys
import multiprocessing

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
    # ВАЖНО: Эта строчка предотвращает бесконечное открытие новых окон 
    # в скомпилированном .exe файле на Windows при использовании PyTorch/multiprocessing!
    # Должна идти самой первой инструкцией внутри блока __main__.
    multiprocessing.freeze_support()
    
    try:
        main()
    except Exception as e:
        import traceback
        print("КРИТИЧЕСКАЯ ОШИБКА ПРИ ЗАПУСКЕ:")
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")

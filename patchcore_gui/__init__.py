"""PyQt6-интерфейс для визуального контроля PatchCore (без изменений ML-ядра)."""

from __future__ import annotations

__all__ = ["MainWindow"]


def __getattr__(name: str):  # PEP 562: отложенный импорт (тяжёлые зависимости только по запросу)
    if name == "MainWindow":
        from patchcore_gui.main_window import MainWindow

        return MainWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

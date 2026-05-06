import sys
import platform
import os

from PySide6.QtWidgets import QApplication

from app.window import MainWindow


def _detect_compute_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "GPU:使用(CUDA)"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "GPU:使用(MPS)"
    except Exception:
        pass

    return "CPU:使用 / GPU:未使用"


def main() -> int:
    app = QApplication(sys.argv)
    os_info = platform.platform()
    compute_status = _detect_compute_device()
    os.environ["ANALYZER_COMPUTE_DEVICE"] = compute_status

    screen = app.primaryScreen()
    geometry = screen.availableGeometry() if screen is not None else None
    print(f"OS: {os_info}")
    print(f"計算デバイス: {compute_status}")
    if geometry is not None:
        print(f"表示領域: {geometry.width()}x{geometry.height()}")
    else:
        print("表示領域: 取得できませんでした")

    window = MainWindow(os_name=platform.system(), compute_status=compute_status)
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

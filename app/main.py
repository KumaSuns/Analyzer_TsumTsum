import sys
import platform
import os
from pathlib import Path


def _configure_qt_environment() -> None:
    """Fix Qt not finding PySide6 plugins (e.g. Microsoft Store python.exe → 'No QtMultimedia backends found')."""
    try:
        import PySide6  # noqa: F401  # ensures DLL dirs on Windows before Qt loads

        plugin_dir = Path(PySide6.__file__).resolve().parent / "plugins"
        if plugin_dir.is_dir():
            p = os.fspath(plugin_dir)
            prev = os.environ.get("QT_PLUGIN_PATH", "")
            if not prev:
                os.environ["QT_PLUGIN_PATH"] = p
            elif p not in prev.split(os.pathsep):
                os.environ["QT_PLUGIN_PATH"] = p + os.pathsep + prev
    except Exception:
        pass

    if sys.platform == "win32":
        # ffmpeg バックエンドの方が多くのコーデックに対応（WMF は「Unsupported media type」になりやすい）
        # うまくいかない場合のみ: set QT_MEDIA_BACKEND=windows
        os.environ.setdefault("QT_MEDIA_BACKEND", "ffmpeg")


_configure_qt_environment()

from PySide6.QtWidgets import QApplication

from app.window import MainWindow


def _detect_compute_device() -> str:
    try:
        from app.services.scene_cnn import describe_torch_device, pick_torch_device, torch_available

        if not torch_available():
            return "CPU:使用 / PyTorch 未導入"
        dev = pick_torch_device()
        if dev.type == "cuda":
            return f"GPU:使用(CUDA) {describe_torch_device(dev)}"
        if dev.type == "mps":
            return f"GPU:使用(MPS) {describe_torch_device(dev)}"
        return f"CPU:使用 ({describe_torch_device(dev)})"
    except Exception:
        pass

    return "CPU:使用 / GPU:未使用"


def main() -> int:
    # Intel Mac の AMD GPU では PyTorch MPS が Metal エラーになるため CPU 固定
    if platform.machine() != "arm64":
        os.environ.setdefault("ANALYZER_TORCH_DEVICE", "cpu")

    app = QApplication(sys.argv)
    os_info = platform.platform()
    compute_status = _detect_compute_device()
    os.environ["ANALYZER_COMPUTE_DEVICE"] = compute_status

    try:
        from app.services.scene_cnn import verify_torch_install

        torch_err = verify_torch_install()
        if torch_err:
            print(f"警告: {torch_err}")
    except Exception:
        pass

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

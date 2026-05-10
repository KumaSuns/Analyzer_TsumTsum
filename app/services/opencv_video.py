"""Video decode via OpenCV (bypasses broken Qt Multimedia on some Windows setups)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None  # type: ignore

import numpy as np
from PySide6.QtGui import QImage

# 巨大ファイルの一時コピーは避ける（バイト）
_MAX_TEMP_COPY_BYTES = int(1.8 * 1024 * 1024 * 1024)


def is_opencv_video_available() -> bool:
    return cv2 is not None


def _win_long_path_prefix(path: str) -> str:
    """Help OpenCV / Win32 APIs that mishandle some Unicode paths."""
    if sys.platform != "win32":
        return path
    ap = os.path.normpath(os.path.abspath(path))
    if ap.startswith("\\\\?\\") or ap.startswith("\\\\"):
        return path
    return "\\\\?\\" + ap


def _path_candidates(path: str) -> list[str]:
    raw = path.strip()
    seen: set[str] = set()
    out: list[str] = []

    def add(p: str) -> None:
        if p and p not in seen:
            seen.add(p)
            out.append(p)

    add(raw)
    try:
        rp = str(Path(raw).expanduser().resolve(strict=False))
        add(rp)
    except Exception:
        pass
    try:
        add(os.path.abspath(raw))
    except Exception:
        pass
    if "\\" in raw:
        add(raw.replace("\\", "/"))
    if sys.platform == "win32":
        try:
            add(_win_long_path_prefix(raw))
        except Exception:
            pass
    return out


def _capture_backends() -> list[int]:
    if cv2 is None:
        return []
    order: list[int] = []
    for name in ("CAP_FFMPEG", "CAP_MSMF", "CAP_ANY"):
        v = getattr(cv2, name, None)
        if isinstance(v, int) and v not in order:
            order.append(v)
    if 0 not in order:
        order.append(0)
    return order


def _try_capture(path: str) -> Optional["cv2.VideoCapture"]:
    if cv2 is None:
        return None
    for backend in _capture_backends():
        try:
            cap = cv2.VideoCapture(path, backend)
        except Exception:
            continue
        if not cap.isOpened():
            cap.release()
            continue
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            continue
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        except Exception:
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, 0.0)
            except Exception:
                pass
        return cap
    return None


class OpenCvVideoSource:
    def __init__(self) -> None:
        self._cap: Optional["cv2.VideoCapture"] = None
        self._temp_copy_path: Optional[str] = None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._temp_copy_path:
            try:
                os.unlink(self._temp_copy_path)
            except OSError:
                pass
            self._temp_copy_path = None

    def open(self, path: str) -> bool:
        self.release()
        if cv2 is None:
            return False

        for candidate in _path_candidates(path):
            cap = _try_capture(candidate)
            if cap is not None:
                self._cap = cap
                return True

        src = Path(path).expanduser()
        try:
            src = src.resolve(strict=False)
        except Exception:
            pass
        if src.is_file():
            try:
                size = src.stat().st_size
            except OSError:
                size = -1
            if 0 < size <= _MAX_TEMP_COPY_BYTES:
                suffix = src.suffix or ".mp4"
                fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="analyzer_video_")
                os.close(fd)
                try:
                    shutil.copy2(str(src), tmp)
                    cap = _try_capture(tmp)
                    if cap is not None:
                        self._cap = cap
                        self._temp_copy_path = tmp
                        return True
                except OSError:
                    pass
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

        return False

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def fps(self) -> float:
        if not self.is_open:
            return 30.0
        v = float(self._cap.get(cv2.CAP_PROP_FPS))
        return v if v > 1e-3 else 30.0

    def duration_ms(self) -> int:
        if not self.is_open:
            return 0
        fps = self.fps()
        fc = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fc <= 0 or fps <= 1e-3:
            return 0
        return int((fc / fps) * 1000.0)

    def seek_ms(self, ms: int) -> None:
        if not self.is_open:
            return
        self._cap.set(cv2.CAP_PROP_POS_MSEC, float(max(0, ms)))

    def read_qimage(self) -> Optional[QImage]:
        if not self.is_open:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

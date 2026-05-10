"""Video decode via imageio (FFmpeg). Helps when OpenCV fails (e.g. cloud paths, codecs)."""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtGui import QImage

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None  # type: ignore


def is_imageio_video_available() -> bool:
    return imageio is not None


class ImageIoVideoSource:
    def __init__(self) -> None:
        self._reader = None
        self._fps = 30.0
        self._duration_ms = 0
        self._nframes: Optional[int] = None
        self._seek_ms_cursor = 0

    def release(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None

    @property
    def is_open(self) -> bool:
        return self._reader is not None

    def open(self, path: str) -> bool:
        self.release()
        if imageio is None:
            return False
        try:
            r = imageio.get_reader(path, "ffmpeg")
        except Exception:
            return False
        try:
            meta = r.get_meta_data()
            fps = meta.get("fps", None)
            self._fps = float(fps) if fps and float(fps) > 1e-3 else 30.0
            dur = meta.get("duration", None)
            if dur is not None and float(dur) > 0:
                self._duration_ms = int(float(dur) * 1000)
            else:
                self._duration_ms = 0
            # count_frames() はクラウド上の動画で極端に遅いことがあるため使わない
            self._nframes = None
            fr0 = r.get_data(0)
            if fr0 is None or getattr(fr0, "size", 0) == 0:
                r.close()
                return False
            self._reader = r
            self._seek_ms_cursor = 0
            return True
        except Exception:
            try:
                r.close()
            except Exception:
                pass
            return False

    def fps(self) -> float:
        return self._fps

    def duration_ms(self) -> int:
        return self._duration_ms

    def seek_ms(self, ms: int) -> None:
        self._seek_ms_cursor = max(0, ms)

    def read_qimage(self) -> Optional[QImage]:
        if not self.is_open:
            return None
        idx = int((self._seek_ms_cursor / 1000.0) * self._fps)
        if self._nframes is not None:
            idx = max(0, min(idx, self._nframes - 1))
        else:
            idx = max(0, idx)
        try:
            arr = self._reader.get_data(idx)
        except Exception:
            return None
        if arr is None or arr.size == 0:
            return None
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        rgb = np.ascontiguousarray(arr[:, :, :3])
        h, w, ch = rgb.shape
        bpl = ch * w
        return QImage(rgb.data, w, h, bpl, QImage.Format.Format_RGB888).copy()

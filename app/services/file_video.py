"""Try OpenCV first, then imageio+FFmpeg (better for Google Drive / iPhone HEVC paths)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

from app.services.imageio_video import ImageIoVideoSource, is_imageio_video_available
from app.services.opencv_video import (
    OpenCvVideoSource,
    _MAX_TEMP_COPY_BYTES,
    _path_candidates,
    is_opencv_video_available,
)

_STREAM_CHUNK = 4 * 1024 * 1024


def is_file_video_available() -> bool:
    return is_opencv_video_available() or is_imageio_video_available()


def _needs_local_temp_copy(path: str) -> bool:
    """Google Drive / 非ASCII パスは FFmpeg・OpenCV が直接開けないことが多い。"""
    p = str(path)
    if "マイドライブ" in p or "My Drive" in p:
        return True
    try:
        p.encode("ascii")
    except UnicodeEncodeError:
        return True
    return False


def _subprocess_flags() -> int:
    if sys.platform == "win32":
        try:
            return int(subprocess.CREATE_NO_WINDOW)
        except AttributeError:
            return 0
    return 0


def _ffmpeg_input_path(src_path: str) -> str:
    """Windows で Unicode・長いパスを ffmpeg に渡すときの安定化。"""
    ap = os.path.normpath(os.path.abspath(os.path.expanduser(src_path)))
    if sys.platform == "win32":
        if ap.startswith("\\\\?\\") or ap.startswith("\\\\"):
            return ap
        try:
            if os.path.exists(ap):
                return "\\\\?\\" + ap
        except OSError:
            pass
    return ap


def _ffmpeg_copy_to_temp(src_path: str, owned: list[str], errors: list[str]) -> Optional[str]:
    """imageio 同梱の ffmpeg でストリーム再マルチプレクス（Python 読みが失敗するドライブ向け）。"""
    try:
        import imageio_ffmpeg
    except ImportError:
        errors.append("imageio_ffmpeg がありません")
        return None
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not exe or not os.path.isfile(exe):
        errors.append("ffmpeg 実行ファイルが見つかりません")
        return None
    abspath = _ffmpeg_input_path(src_path)
    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="analyzer_ff_")
    os.close(fd)
    try:
        os.unlink(tmp)
    except OSError:
        pass
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        abspath,
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        tmp,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,
            creationflags=_subprocess_flags(),
        )
    except subprocess.TimeoutExpired:
        errors.append("ffmpeg 実体化: タイムアウト（900s）")
        return None
    except OSError as exc:
        errors.append(f"ffmpeg 起動失敗: {exc}")
        return None
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        errors.append(f"ffmpeg 終了コード {proc.returncode}: {err[:500]}")
        try:
            if os.path.isfile(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return None
    try:
        if not os.path.isfile(tmp) or os.path.getsize(tmp) <= 0:
            errors.append("ffmpeg 出力が空です")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return None
    except OSError as exc:
        errors.append(f"ffmpeg 出力確認失敗: {exc}")
        return None
    owned.append(tmp)
    return tmp


def _ffmpeg_transcode_to_temp(src_path: str, owned: list[str], errors: list[str]) -> Optional[str]:
    """iPhone HEVC・moov 位置などでデコードできない場合に H.264 へ一度通す。"""
    try:
        import imageio_ffmpeg
    except ImportError:
        errors.append("imageio_ffmpeg がありません（トランスコード不可）")
        return None
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    if not exe or not os.path.isfile(exe):
        errors.append("ffmpeg 実行ファイルが見つかりません（トランスコード不可）")
        return None
    in_path = _ffmpeg_input_path(src_path)
    fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="analyzer_tx_")
    os.close(fd)
    try:
        os.unlink(tmp)
    except OSError:
        pass
    cmd = [
        exe,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        in_path,
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        tmp,
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,
            creationflags=_subprocess_flags(),
        )
    except subprocess.TimeoutExpired:
        errors.append("ffmpeg トランスコード: タイムアウト（120 分）")
        return None
    except OSError as exc:
        errors.append(f"ffmpeg トランスコード起動失敗: {exc}")
        return None
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        errors.append(f"ffmpeg トランスコード終了 {proc.returncode}: {err[:800]}")
        try:
            if os.path.isfile(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return None
    try:
        if not os.path.isfile(tmp) or os.path.getsize(tmp) <= 0:
            errors.append("ffmpeg トランスコード出力が空です")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return None
    except OSError as exc:
        errors.append(f"ffmpeg トランスコード出力確認失敗: {exc}")
        return None
    owned.append(tmp)
    return tmp


def _materialize_video_to_temp(path: str, owned: list[str], errors: list[str]) -> Optional[str]:
    """順次読みで TEMP に実体化。is_file() が偽でも exists かつ open できれば試す。"""
    src = Path(path).expanduser()
    try:
        src = src.resolve(strict=False)
    except Exception:
        src = Path(path)

    if src.is_dir():
        errors.append("パスがフォルダです")
        return None
    if not src.exists():
        errors.append("パスが存在しません（同期・オフライン・パス誤りを確認）")
        return None

    suffix = src.suffix or ".mp4"
    fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="analyzer_vid_")
    os.close(fd)

    try:
        size = -1
        try:
            size = src.stat().st_size
        except OSError as exc:
            errors.append(f"stat 失敗: {exc}")
            size = -1

        if size > 0 and size <= _MAX_TEMP_COPY_BYTES:
            try:
                shutil.copy2(str(src), tmp)
                if os.path.getsize(tmp) > 0:
                    owned.append(tmp)
                    return tmp
            except OSError as exc:
                errors.append(f"copy2 失敗: {exc}")
            try:
                with open(src, "rb") as f_in, open(tmp, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
                if os.path.getsize(tmp) > 0:
                    owned.append(tmp)
                    return tmp
            except OSError as exc:
                errors.append(f"copyfileobj 失敗: {exc}")

        try:
            os.unlink(tmp)
        except OSError:
            pass
        fd, tmp = tempfile.mkstemp(suffix=suffix, prefix="analyzer_vid_")
        os.close(fd)
        total = 0
        try:
            with open(src, "rb") as f_in, open(tmp, "wb") as f_out:
                while True:
                    chunk = f_in.read(_STREAM_CHUNK)
                    if not chunk:
                        break
                    f_out.write(chunk)
                    total += len(chunk)
                    if total > _MAX_TEMP_COPY_BYTES:
                        raise OSError("video exceeds size limit")
        except OSError as exc:
            errors.append(f"ストリーム読み込み失敗: {exc}")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return None
        if total <= 0:
            errors.append("ストリーム読み込み: 0 バイト（クラウド未取得の可能性）")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return None
        owned.append(tmp)
        return tmp
    except OSError as exc:
        errors.append(f"実体化例外: {exc}")
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None


class FileVideoSource:
    """Duck-compatible with OpenCvVideoSource for MainWindow."""

    def __init__(self) -> None:
        self._impl: Optional[Union[OpenCvVideoSource, ImageIoVideoSource]] = None
        self._owned_temp_paths: list[str] = []
        self.last_open_error: str = ""

    def release(self) -> None:
        if self._impl is not None:
            self._impl.release()
            self._impl = None
        for p in self._owned_temp_paths:
            try:
                os.unlink(p)
            except OSError:
                pass
        self._owned_temp_paths.clear()

    @property
    def is_open(self) -> bool:
        return self._impl is not None and self._impl.is_open

    def _try_opencv(self, primary: str, errors: list[str]) -> bool:
        if not is_opencv_video_available():
            return False
        oc = OpenCvVideoSource()
        if oc.open(primary):
            self._impl = oc
            return True
        oc.release()
        errors.append(f"OpenCV が開けません: {primary}")
        return False

    def _try_imageio(self, primary: str, errors: list[str]) -> bool:
        if not is_imageio_video_available():
            return False
        for cand in _path_candidates(primary):
            ij = ImageIoVideoSource()
            if ij.open(cand):
                self._impl = ij
                return True
            ij.release()
        errors.append(f"imageio+FFmpeg が開けません: {primary}")
        return False

    def open(self, path: str) -> bool:
        self.release()
        self.last_open_error = ""
        errors: list[str] = []
        primary = path

        if _needs_local_temp_copy(path):
            local = _materialize_video_to_temp(path, self._owned_temp_paths, errors)
            if local:
                primary = local
            else:
                ff_tmp = _ffmpeg_copy_to_temp(path, self._owned_temp_paths, errors)
                if ff_tmp:
                    primary = ff_tmp

        used_local_copy = primary != path
        if used_local_copy:
            opened = self._try_imageio(primary, errors) or self._try_opencv(primary, errors)
        else:
            opened = self._try_opencv(primary, errors) or self._try_imageio(primary, errors)
        if opened:
            self.last_open_error = ""
            return True

        transcode_sources: list[str] = []
        seen: set[str] = set()
        for s in (primary, path):
            if not s or s in seen:
                continue
            try:
                if os.path.isfile(s):
                    seen.add(s)
                    transcode_sources.append(s)
            except OSError:
                pass
        for src in transcode_sources:
            tx = _ffmpeg_transcode_to_temp(src, self._owned_temp_paths, errors)
            if tx and (
                self._try_imageio(tx, errors) or self._try_opencv(tx, errors)
            ):
                self.last_open_error = ""
                return True

        if primary == path and is_imageio_video_available():
            src = Path(path).expanduser()
            try:
                src = src.resolve(strict=False)
            except Exception:
                src = Path(path)
            if src.exists() and not src.is_dir():
                tmp = _materialize_video_to_temp(str(src), self._owned_temp_paths, errors)
                if tmp and (self._try_imageio(tmp, errors) or self._try_opencv(tmp, errors)):
                    self.last_open_error = ""
                    return True
                ff_tmp = _ffmpeg_copy_to_temp(str(src), self._owned_temp_paths, errors)
                if ff_tmp and (self._try_imageio(ff_tmp, errors) or self._try_opencv(ff_tmp, errors)):
                    self.last_open_error = ""
                    return True

        self.last_open_error = "\n".join(errors) if errors else "不明な理由で開けませんでした"
        return False

    def fps(self) -> float:
        return self._impl.fps() if self._impl else 30.0

    def duration_ms(self) -> int:
        return self._impl.duration_ms() if self._impl else 0

    def seek_ms(self, ms: int) -> None:
        if self._impl is not None:
            self._impl.seek_ms(ms)

    def read_qimage(self):
        if self._impl is None:
            return None
        return self._impl.read_qimage()

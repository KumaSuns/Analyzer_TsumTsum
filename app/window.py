from __future__ import annotations

import hashlib
import os
import platform
import json
import colorsys
import threading
import queue
import time
from datetime import datetime
from typing import Callable, Optional
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, Qt, QUrl, QSettings, Signal, QTimer
from PySide6.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap, QTextCharFormat, QTextCursor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QProgressBar,
    QStackedWidget,
    QPushButton,
    QRubberBand,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.analyzer import VideoAnalyzer
from app.services.image_save import save_training_png
from app.services.scene_model import (
    SceneCentroidModel,
    _NONE_VETO_ABSOLUTE_MAX,
    _NONE_VETO_EXEMPLAR_MAX,
    _NONE_VETO_SESSION_MAX,
    feature_matches_none_exemplars,
    image_to_feature,
)
from app.services.scene_dataset_advice import (
    analyze_scene_dataset,
    format_advice_report_text,
    summarize_skill_activation_images,
)
from app.services.scene_dataset_dedup import (
    SAVE_SKIP_DUPLICATE_PREFIX,
    deduplicate_scene_dataset,
    find_duplicate_scene_image,
    register_saved_scene_image,
)
from app.services.scene_dataset_review import SceneDatasetReviewDialog
from app.services.trainer import SimpleTrainer
from app.services.tsum_registry import TsumRegistry
from app.services.skill_classifier import (
    SKILL_IMAGE_CATEGORY_ACTIVATION,
    SkillClassifierPool,
    list_skill_categories,
    list_skill_tsum_dirs,
    resolve_tsum_dir,
    skill_category_label,
)
from app.services.use_tsum_classifier import UseTsumClassifier
from app.services.coin_gain_reader import (
    _MAX_HUD_ACCEPTABLE_ERR,
    _is_slot_decode_dbg,
    bonus_coin_hud_rect,
    coin_gain_confirmed_combined,
    consensus_coin_gain,
    opencv_available,
    plausible_coin_value,
    read_coin_gain_crop,
)
from app.services.coin_digit_cnn import coin_digit_cnn_available
from app.services.coin_digit_dataset import (
    count_saved_crops,
    save_labeled_crop,
    save_labeled_score_crop,
)
from app.services.result_digit_cnn import result_digit_cnn_available
from app.services.result_digit_dataset import (
    count_saved_crops as count_saved_result_crops,
    save_labeled_result_gain_crop,
)
from app.services.result_gain_reader import (
    RESULT_GAIN_FIELDS,
    RESULT_GAIN_FIELD_KEYS,
    consensus_result_gain,
    read_result_gain_crop,
    result_gain_confirmed,
)
from app.services.score_gain_reader import (
    consensus_score_gain,
    plausible_score_value,
    read_score_gain_crop,
    score_gain_confirmed,
)

_RESULT_GAIN_CROP_KEYS = frozenset(RESULT_GAIN_FIELD_KEYS)
from app.services.remaining_time_reader import read_remaining_seconds
from app.services.file_video import FileVideoSource, is_file_video_available

# 解析テスト: item〜result を「シーンに入った瞬間」で確認モーダルする対象にできる（none 除く）
_ANALYSIS_SCENE_CONFIRM_LABELS = tuple(c for c in SimpleTrainer.CLASSES if c != "none")
# 初回起動時に確認モーダルを ON にするシーン
_SCENE_CONFIRM_DEFAULT_ON = frozenset({"ready", "go", "fever", "timeup", "bonus", "coin", "result"})
_APP_VERSION = "Version_002_2026_06_12"
# いまは timeup 精度改善に集中するため解析中の fever 検知を止める（True で復帰）
_ANALYSIS_FEVER_ENABLED = False
# IN_GAME: 弱い fever は連続2サンプル（1サンプルだけの誤検知を出さない）
_FEVER_WEAK_STREAK_REQUIRED = 2
# go 画面を逃したとき、プレイ画面(none)が続いたら IN_GAME に入る（短すぎると go 未検知になる）
_FEVER_INGAME_ENTRY_NONE_STREAK = 12
# CNN: 2位以内で go のスコアがこれ未満なら go 候補（score = 1 - softmax）
_CNN_GO_SCORE_MAX = 0.42
# ready/go フェーズ専用（やや緩め: score=1-softmax、0.90 未満なら候補）
_CNN_PREGAME_SCORE_MAX = 0.90
# 1位との score 差がこれ以下なら pregame 候補（go は none との僅差誤検知が少ない）
_CNN_PREGAME_MARGIN_MAX = 0.30
# ready は go より none との僅差が出やすいので margin を広げる
_CNN_READY_MARGIN_MAX = 0.45
# ready/go: top1 かつ score がこれ未満（val は 0.01 前後。実動画用に weak も併用）
_CNN_READY_DETECT_MAX = 0.65
_CNN_GO_DETECT_MAX = 0.65
_CNN_PREGAME_WEAK_MAX = 0.78
# top1 以外: 1位との score 差がこれ以下なら weak 候補
_PREGAME_READY_STREAK_WEAK = 1
_PREGAME_GO_STREAK_WEAK = 1
# ready/go 候補の間に none が挟まってもストリークを維持
_PREGAME_HIT_NONE_GAP = 3
_CNN_GO_MARGIN_MAX = 0.45
# item 確定後の ready 探索（猶予は streak 用。WAIT_GO へ進むのは CNN ready のみ）
_WAIT_READY_GRACE_SAMPLES = 60
# ready: train/none 参照・セッション veto で none 画面の誤 ready を抑える（go は CNN のまま）
_PREGAME_READY_SUPPRESSION_ENABLED = True
_SESSION_READY_NONE_VETO_MAX = 30
_CNN_READY_NONE_MARGIN = 0.08
_READY_STRONG_AFTER_REJECT_MAX = 0.42
# item 選択画面の crop キー（use_tsum は使用アイテムに含めない）
_ITEM_SELECT_KEYS = ("score", "coin", "exp", "time", "bomb", "five_to_four", "combo")
# 時間スキップ専用（解析フロー・シーン判定では使わない）
_REQUIRED_TIMER_SKIP_CROP_KEY = "remaining_time"
# 動画ツール下段ボタン action（解析トグル）
_ANALYSIS_BUTTON_ACTION = "analyze"
_TIME_EFFICIENCY_BUTTON_ACTION = "時間効率"
# 時間効率: go 確定位置から動画をこの分だけ進めて timeup 付近へ（解析は継続）
_TIME_EFFICIENCY_GO_SKIP_DELAY_MS = 60_000
# 1 game = item 確定〜result 確定。ready/go/timeup/coin/result は各 game に 1 回のみ
_ANALYSIS_TOGGLE_ACTIONS = frozenset({_ANALYSIS_BUTTON_ACTION, _TIME_EFFICIENCY_BUTTON_ACTION})
# CNN: fever 候補（score = 1 - softmax、小さいほど fever らしい）
_CNN_FEVER_SCORE_MAX = 0.62
_CNN_FEVER_STRONG_SCORE_MAX = 0.38
# IN_GAME 確定: top1=fever かつ score がこれ未満（0.34 は取りこぼし多。0.45 で val 43/43・none FP 4）
_CNN_FEVER_DETECT_MAX = 0.45
_CNN_FEVER_RANKED_TOP = 8
_SESSION_FEVER_NONE_VETO_MAX = 30
# IN_GAME 確定: top1=timeup かつ score がこれ未満（低いほど自信。誤検知抑制のためやや厳しめ）
_CNN_TIMEUP_DETECT_MAX = 0.30
# top1=timeup でも none がこの差以内なら timeup 候補を出さない
_CNN_TIMEUP_NONE_MARGIN = 0.03
_SESSION_TIMEUP_NONE_VETO_MAX = 30
_TIMEUP_REJECT_COOLDOWN_SAMPLES = 24
# WAIT_BONUS 確定: top1=bonus かつ score がこれ未満（val 16/16、max≈0.508）
_CNN_BONUS_DETECT_MAX = 0.52
_SESSION_BONUS_NONE_VETO_MAX = 30
_BONUS_REJECT_COOLDOWN_SAMPLES = 24
# WAIT_COIN 確定: top1=coin かつ score がこれ未満（学習枚数が少ないときはやや緩め）
_CNN_COIN_DETECT_MAX = 0.55
_SESSION_COIN_NONE_VETO_MAX = 30
_COIN_REJECT_COOLDOWN_SAMPLES = 24
# WAIT_RESULT 確定: top1=result かつ score がこれ未満（val 26/26、max≈0.0125）
_CNN_RESULT_DETECT_MAX = 0.10
_SESSION_RESULT_NONE_VETO_MAX = 30
_RESULT_REJECT_COOLDOWN_SAMPLES = 24
# CNN fever: 連続 N 回で確定 → M サンプルは fever 維持（点滅防止）
_FEVER_CNN_CONFIRM_STREAK = 2
_FEVER_LATCH_HOLD_SAMPLES = 8
_FEVER_LATCH_RELEASE_STREAK = 4
# IN_GAME 中の判定間隔（通常より密に）
_INGAME_SAMPLE_EVERY_MAX = 5
# go 直後: 緩い fever 候補だけ抑える
_FEVER_IN_GAME_WARMUP_SAMPLES = 12
_FEVER_IN_GAME_WARMUP_SAMPLES_CNN = 8
# fever 確定途中に none が 1 回挟まってもストリーク維持
_FEVER_NONE_GAP_ALLOW = 1
# モーダルで fever 誤りとしたあと、しばらく fever 確定しない
_FEVER_REJECT_COOLDOWN_SAMPLES = 24
# 同じフィーバー中に fever 回数を二重カウントしない間隔（表示用）
_FEVER_COUNT_GAP_MS = 12_000
_CNN_TIMEUP_SCORE_MAX = 0.58
_CNN_BONUS_SCORE_MAX = 0.58
_CNN_COIN_SCORE_MAX = 0.58
_CNN_RESULT_SCORE_MAX = 0.58
_TIMEUP_STOP_CONFIRM_REQUIRED = 2
# IN_GAME へ遷移直後だけ go を救済（go 取りこぼし対策）
_INGAME_GO_GRACE_SAMPLES = 18

_SCENE_CONFIRM_PREVIEW_MAX_W = 560
_SCENE_CONFIRM_PREVIEW_MAX_H = 315
_COIN_GAIN_CROP_PREVIEW_MAX_W = 480
_COIN_GAIN_CROP_PREVIEW_MAX_H = 200
# 動画ツール左列モード（内部 ID は 1〜6 のまま）
_VIDEO_TOOL_MODE_LABELS: tuple[str, ...] = (
    "トリム",
    "シーン",
    "コイン",
    "結果",
    "—",
    "—",
)
_VIDEO_TOOL_MODE_TRIM = 1
_VIDEO_TOOL_MODE_SCENE = 2
_VIDEO_TOOL_MODE_COIN = 3
_VIDEO_TOOL_MODE_RESULT = 4


def _video_tool_mode_label(mode_id: int) -> str:
    if 1 <= mode_id <= len(_VIDEO_TOOL_MODE_LABELS):
        return _VIDEO_TOOL_MODE_LABELS[mode_id - 1]
    return str(mode_id)


def _preview_pixmap_for_scene_modal(frame_image, max_w: int, max_h: int) -> QPixmap:
    if frame_image is None or frame_image.isNull():
        pm = QPixmap(min(320, max_w), min(180, max_h))
        pm.fill(QColor(60, 60, 60))
        return pm
    pm = QPixmap.fromImage(frame_image)
    if pm.isNull():
        pm = QPixmap(min(320, max_w), min(180, max_h))
        pm.fill(QColor(60, 60, 60))
        return pm
    return pm.scaled(
        max_w,
        max_h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


def _qt_multimedia_environment_report() -> str:
    import sys

    lines = [f"Python: {sys.executable}"]
    try:
        import PySide6

        root = os.path.dirname(PySide6.__file__)
        lines.append(f"PySide6: {PySide6.__version__}")
        lines.append(f"PySide6 dir: {root}")
        mm = os.path.join(root, "plugins", "multimedia")
        lines.append(f"plugins/multimedia exists: {os.path.isdir(mm)}")
    except Exception as exc:
        lines.append(f"PySide6: (取得できません: {exc})")
    lines.append(f"QT_PLUGIN_PATH: {os.environ.get('QT_PLUGIN_PATH', '(未設定)')}")
    lines.append(f"QT_MEDIA_BACKEND: {os.environ.get('QT_MEDIA_BACKEND', '(未設定)')}")
    return "\n".join(lines)


def _is_streaming_drive_path(path: str) -> bool:
    p = path or ""
    return "マイドライブ" in p or "My Drive" in p


def _media_error_extra_hints(error_string: str, video_path: str = "") -> str:
    low = (error_string or "").lower()
    if "unsupported media type" not in low:
        return ""
    backend = os.environ.get("QT_MEDIA_BACKEND", "(未設定)")
    bl = backend.lower() if backend != "(未設定)" else ""
    other = "windows" if bl == "ffmpeg" else "ffmpeg"
    lines = [
        "\n--- よくある対処 ---\n",
        f"現在のバックエンド: {backend}\n",
    ]
    if _is_streaming_drive_path(video_path):
        lines.append(
            "マイドライブ上の iPhone 動画は、Qt ではほぼ開けません。"
            "エクスプローラで「オフラインで使用可能」にするか、C: などローカルにコピーしてから開いてください。\n"
        )
    if bl == "windows":
        lines.append(
            "WMF(windows) は形式によっては未対応になります。**まず ffmpeg を試してください**（下の1行目）。\n"
        )
    if bl == "ffmpeg":
        lines.append(
            f'別バックエンド(windows)を試す: $env:QT_MEDIA_BACKEND = "{other}"; python -m app.main\n'
        )
    elif bl == "windows":
        lines.append(
            f'別バックエンド(ffmpeg)を試す: $env:QT_MEDIA_BACKEND = "{other}"; python -m app.main\n'
        )
    else:
        lines.append(f'バックエンドを試す: $env:QT_MEDIA_BACKEND = "{other}"; python -m app.main\n')
    lines.append(
        "既定（main.py の ffmpeg）に戻す: Remove-Item Env:QT_MEDIA_BACKEND -ErrorAction SilentlyContinue; python -m app.main\n"
    )
    lines.append(
        "それでもダメなら、H.264 映像 + AAC 音声の MP4 に再エンコード（HandBrake、ffmpeg の libx264/aac）。\n"
    )
    return "".join(lines)


def _show_copyable_warning(parent: Optional[QWidget], title: str, summary: str, details: str) -> None:
    QApplication.clipboard().setText(details)
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(summary)
    box.setInformativeText(
        "詳細はメッセージボックスの「詳細を表示」で開くか、クリップボードにコピー済みの内容をメモ帳に貼り付け（Ctrl+V）してください。"
    )
    box.setDetailedText(details)
    box.setStandardButtons(QMessageBox.StandardButton.Ok)
    box.exec()


def _is_alive_qobject(obj) -> bool:
    if obj is None:
        return False
    try:
        obj.objectName()
        return True
    except RuntimeError:
        return False


def _item_select_keys_from(targets: list[str]) -> list[str]:
    """アイテム選択画面の7種（Score〜Combo）。use_tsum は除外。"""
    picked = {k for k in targets if k in _ITEM_SELECT_KEYS}
    return [k for k in _ITEM_SELECT_KEYS if k in picked]


def _format_duration_min_sec(elapsed_sec: float) -> str:
    """経過秒を 02分05秒 形式にする。"""
    total = max(0, int(round(elapsed_sec)))
    minutes, seconds = divmod(total, 60)
    return f"{minutes:02d}分{seconds:02d}秒"


class VideoCropOverlay(QWidget):
    """動画の上にトリミング枠を描画（QVideoWidget のネイティブ面の上に出す）。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._frame_rect = QRect()
        self.hide()

    def set_frame_rect(self, rect: QRect) -> None:
        self._frame_rect = rect
        self.setVisible(not rect.isNull())
        self.update()

    def frame_rect(self) -> QRect:
        return self._frame_rect

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self._frame_rect.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#00E5FF"), 3)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 229, 255, 55))
        painter.drawRect(self._frame_rect)


class AspectFitVideoContainer(QWidget):
    clicked = Signal()
    cropSelected = Signal(float, float, float, float)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._aspect_ratios = [
            9 / 19.5,   # iPhone portrait
            9 / 20,     # Android portrait
            3 / 4,      # iPad portrait
            19.5 / 9,   # iPhone landscape
            20 / 9,     # Android landscape
            4 / 3,      # iPad landscape
        ]
        self.video_widget = QVideoWidget(self)
        self.video_widget.setStyleSheet("background: #000;")
        self.frame_label = QLabel(self)
        self.frame_label.setStyleSheet("background: #000;")
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.hide()
        self._opencv_display = False
        self._crop_overlay = VideoCropOverlay(self)
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rubber_band.hide()
        self._rubber_band.setStyleSheet("border: 2px solid #00E5FF; background: rgba(0, 229, 255, 40);")
        self._drag_origin = QPoint()
        self._crop_dragging = False
        self._crop_enabled = False
        self._selected_rect = QRect()
        self._normalized_rect: Optional[tuple[float, float, float, float]] = None
        self._source_width = 0
        self._source_height = 0

    def set_crop_enabled(self, enabled: bool) -> None:
        self._crop_enabled = enabled
        for surface in (self.video_widget, self.frame_label):
            surface.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)
        if not enabled:
            self._crop_dragging = False
            self._rubber_band.hide()
        self._sync_crop_overlay()

    def set_source_size(self, width: int, height: int) -> None:
        self._source_width = max(0, width)
        self._source_height = max(0, height)
        self._sync_crop_overlay()

    def clear_crop_overlay(self) -> None:
        self._normalized_rect = None
        self._selected_rect = QRect()
        self._crop_overlay.set_frame_rect(QRect())
        self._rubber_band.hide()

    def set_opencv_display(self, on: bool) -> None:
        self._opencv_display = bool(on)
        self.video_widget.setVisible(not self._opencv_display)
        self.frame_label.setVisible(self._opencv_display)

    def _video_surface_rect(self) -> QRect:
        return self.frame_label.geometry() if self._opencv_display else self.video_widget.geometry()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        rect = self.contentsRect()
        if rect.width() <= 0 or rect.height() <= 0:
            return

        best_w, best_h = 0, 0
        best_area = -1
        for ratio in self._aspect_ratios:
            width = min(rect.width(), int(rect.height() * ratio))
            height = int(width / ratio)
            if height > rect.height():
                height = rect.height()
                width = int(height * ratio)
            area = width * height
            if area > best_area:
                best_area = area
                best_w, best_h = width, height

        x = rect.x() + (rect.width() - best_w) // 2
        y = rect.y() + (rect.height() - best_h) // 2
        r = QRect(x, y, best_w, best_h)
        self.video_widget.setGeometry(r)
        self.frame_label.setGeometry(r)
        self._sync_crop_overlay()

    def _local_rect_in_content(self, rect: QRect, content_rect: QRect) -> QRect:
        return QRect(
            rect.x() - content_rect.x(),
            rect.y() - content_rect.y(),
            rect.width(),
            rect.height(),
        )

    def _set_drag_overlay(self, rect: QRect) -> None:
        content_rect = self._content_rect()
        if content_rect.width() <= 0 or content_rect.height() <= 0 or rect.isNull():
            return
        clipped = rect.intersected(content_rect)
        self._crop_overlay.setGeometry(content_rect)
        self._crop_overlay.set_frame_rect(self._local_rect_in_content(clipped, content_rect))
        self._crop_overlay.raise_()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._crop_enabled and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            if self._content_rect().contains(pos):
                self._drag_origin = pos
                self._crop_dragging = True
                self._set_drag_overlay(QRect(self._drag_origin, self._drag_origin))
            super().mousePressEvent(event)
            return
        elif event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._crop_enabled and self._crop_dragging:
            current = event.position().toPoint()
            self._set_drag_overlay(QRect(self._drag_origin, current).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if (
            self._crop_enabled
            and self._crop_dragging
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._crop_dragging = False
            content_rect = self._content_rect()
            frame_rect = self._crop_overlay.frame_rect()
            if (
                frame_rect.width() > 2
                and frame_rect.height() > 2
                and content_rect.width() > 0
                and content_rect.height() > 0
            ):
                clipped = QRect(
                    content_rect.x() + frame_rect.x(),
                    content_rect.y() + frame_rect.y(),
                    frame_rect.width(),
                    frame_rect.height(),
                )
                self._selected_rect = clipped
                nx = frame_rect.x() / content_rect.width()
                ny = frame_rect.y() / content_rect.height()
                nw = frame_rect.width() / content_rect.width()
                nh = frame_rect.height() / content_rect.height()
                self._normalized_rect = (nx, ny, nw, nh)
                self.cropSelected.emit(nx, ny, nw, nh)
                self._sync_crop_overlay()
        super().mouseReleaseEvent(event)

    def set_selected_normalized_rect(self, nx: float, ny: float, nw: float, nh: float) -> None:
        self._normalized_rect = (float(nx), float(ny), float(nw), float(nh))
        self._sync_crop_overlay()

    def _sync_crop_overlay(self) -> None:
        if self._normalized_rect is None:
            self._crop_overlay.set_frame_rect(QRect())
            return
        content_rect = self._content_rect()
        if content_rect.width() <= 0 or content_rect.height() <= 0:
            return
        nx, ny, nw, nh = self._normalized_rect
        lx = int(nx * content_rect.width())
        ly = int(ny * content_rect.height())
        lw = max(1, int(nw * content_rect.width()))
        lh = max(1, int(nh * content_rect.height()))
        self._selected_rect = QRect(content_rect.x() + lx, content_rect.y() + ly, lw, lh)
        self._crop_overlay.setGeometry(content_rect)
        self._crop_overlay.set_frame_rect(QRect(lx, ly, lw, lh))
        self._crop_overlay.raise_()

    def selected_pixmap(self) -> QPixmap:
        if self._selected_rect.isNull():
            return QPixmap()
        surface = self.frame_label if self._opencv_display else self.video_widget
        video_rect = surface.geometry()
        local_rect = QRect(
            self._selected_rect.x() - video_rect.x(),
            self._selected_rect.y() - video_rect.y(),
            self._selected_rect.width(),
            self._selected_rect.height(),
        )
        return surface.grab(local_rect)

    def _content_rect(self) -> QRect:
        video_rect = self._video_surface_rect()
        if video_rect.width() <= 0 or video_rect.height() <= 0:
            return video_rect
        if self._source_width <= 0 or self._source_height <= 0:
            # Source frame size is unknown yet: disallow crop region.
            return QRect()

        src_ratio = self._source_width / self._source_height
        dst_ratio = video_rect.width() / video_rect.height()
        if src_ratio > dst_ratio:
            w = video_rect.width()
            h = int(w / src_ratio)
            x = video_rect.x()
            y = video_rect.y() + (video_rect.height() - h) // 2
        else:
            h = video_rect.height()
            w = int(h * src_ratio)
            x = video_rect.x() + (video_rect.width() - w) // 2
            y = video_rect.y()
        return QRect(x, y, w, h)


_TRAIN_ACTION_META: dict[str, dict[str, str]] = {
    "coin_digit": {
        "title": "コイン桁 CNN だけ保存",
        "subtitle": "獲得コイン・獲得スコアの数字読取 → coin_digit.pt",
        "tooltip": "シーン CNN を再学習せず coin_digit.pt のみ作成（獲得コイン DL 用）",
        "progress": "コイン桁 CNN 学習・保存中…",
        "bar_color": "#FFB300",
    },
    "result_digit": {
        "title": "結果桁 CNN だけ保存",
        "subtitle": "最終獲得コイン（result_coin_gain）→ result_digit.pt",
        "tooltip": "シーン CNN を再学習せず result_digit.pt のみ作成（最終獲得コイン DL 用）",
        "progress": "結果コイン CNN 学習・保存中…",
        "bar_color": "#42A5F5",
    },
    "skill": {
        "title": "スキル・使用ツムだけ再学習",
        "subtitle": "スキル発動判定・使用ツム判定（シーン CNN は触らない）",
        "tooltip": "シーンを触らず use_tsum / skill モデルだけ更新",
        "progress": "スキル・使用ツム 再学習中…",
        "bar_color": "#66BB6A",
    },
}


class MainWindow(QMainWindow):
    def __init__(self, os_name: str = "", compute_status: str = "") -> None:
        super().__init__()
        self.setWindowTitle("ツムツム解析アプリ")
        self.resize(960, 640)
        self.project_root = Path(__file__).resolve().parents[1]
        self.os_name = os_name or platform.system()
        self.compute_status = compute_status or "CPU:使用 / GPU:未使用"
        self.last_video_path = ""
        self.settings = QSettings("Analyzer_TsumTsum", "Analyzer_TsumTsum")
        self.estimated_fps = 30.0
        self.media_duration_ms = 0
        self.video_analyzer = VideoAnalyzer(
            sample_every_frames=15,
            model_root=self.project_root / "app/models/main_model",
            use_tsum_models_root=self.project_root / "app/models/use_tsum",
            images_root=self.project_root / "app/assets/images",
        )
        self.analysis_running = False
        self._analysis_profile = _ANALYSIS_BUTTON_ACTION
        self._analysis_toggle_buttons = []
        self.analysis_warmup_until_ms = 0
        self.analysis_frame_seq = 0
        self._analysis_log_scroll_counter = 0
        self._analysis_confirm_edge_prev = ""
        self._analysis_confirm_edge_key = ""
        self._analysis_last_logged_scene = ""
        self._analysis_progress_ui_last_ms = 0
        self._playback_ui_last_ms = 0
        self._cv_display_tick = 0
        self._video_frame_size: Optional[tuple[int, int]] = None
        self.flow_phase = "WAIT_ITEM"
        self.flow_game_index = 1
        self._fever_raw_streak = 0
        self._fever_none_gap_used = 0
        self._fever_reject_cooldown = 0
        self._fever_veto_frame_indices: set[int] = set()
        self._analysis_fever_none_veto: list[list[float]] = []
        self._analysis_timeup_none_veto: list[list[float]] = []
        self._timeup_veto_frame_indices: set[int] = set()
        self._timeup_reject_cooldown = 0
        self._analysis_bonus_none_veto: list[list[float]] = []
        self._bonus_veto_frame_indices: set[int] = set()
        self._bonus_reject_cooldown = 0
        self._analysis_coin_none_veto: list[list[float]] = []
        self._coin_veto_frame_indices: set[int] = set()
        self._coin_reject_cooldown = 0
        self._analysis_result_none_veto: list[list[float]] = []
        self._result_veto_frame_indices: set[int] = set()
        self._result_reject_cooldown = 0
        self._analysis_ready_none_veto: list[list[float]] = []
        self._ready_veto_frame_indices: set[int] = set()
        self._pregame_ready_reject_cooldown = 0
        self._in_game_fever_warmup = 0
        self._last_raw_scene_in_game = ""
        self._fever_count = 0
        self._fever_last_count_ms = -1
        self._fever_episode_latched = False
        self._fever_episode_hold = 0
        self._fever_confirm_streak = 0
        self._fever_clear_streak = 0
        self._ingame_go_grace = 0
        self._timeup_stop_seen = 0
        self._skill_count = 0
        self._skill_episode_active = False
        self._skill_off_streak = 0
        self._skill_raw_streak = 0
        self.locked_item_targets: list[str] = []
        self.locked_use_tsum = "-"
        self.locked_item_fixed = False
        self._ingame_sample_saved: Optional[int] = None
        self._pregame_ready_streak = 0
        self._pregame_go_streak = 0
        self._pregame_ready_none_gap = 0
        self._pregame_go_none_gap = 0
        self._pregame_ready_pending = False
        self._pregame_go_pending = False
        self._pregame_ready_latched = False
        self._pregame_go_latched = False
        self._game_active = False
        self._ready_confirmed_this_game = False
        self._go_confirmed_this_game = False
        self._timeup_confirmed_this_game = False
        self._coin_confirmed_this_game = False
        self._coin_cnn_detected_this_game = False
        self._coin_post_detect_ocr_left = 0
        self._coin_result_ocr_left = 0
        self._coin_modal_acknowledged = False
        self._coin_flow_latched = False
        self._result_confirmed_this_game = False
        self._go_clock_anchor_ms: Optional[int] = None
        self._go_to_timeup_sec: Optional[float] = None
        self._go_to_result_sec: Optional[float] = None
        self._timeup_position_ms: int = 0
        self._ready_suppressed_this_game = False
        self._wait_ready_grace = 0
        self._item_scan_targets: set[str] = set()
        self._item_scene_active = False
        self._item_use_tsum_pending = "-"
        self._item_confirmed_this_game = False
        self._item_scan_suppressed = False
        self.pending_crop_rect = None
        self.current_video_container: Optional[AspectFitVideoContainer] = None
        self.current_video_frame_image = None
        self._video_frame_counter = 0
        self.crop_playback_lock = False
        self.crop_positions_for_analysis = {}
        self.crop_targets = [
            ("Score", "score"),
            ("Coin", "coin"),
            ("Exp", "exp"),
            ("time", "time"),
            ("Bomb", "bomb"),
            ("5>4", "five_to_four"),
            ("Combo", "combo"),
            ("残り時間", "remaining_time"),
            ("UseTsum", "use_tsum"),
            ("獲得コイン", "coin_gain"),
            ("獲得スコア", "score_gain"),
            ("最終獲得スコア", "result_score_gain"),
            ("スコアボーナス", "score_bonus_gain"),
            ("獲得経験値", "result_exp_gain"),
            ("最終獲得コイン", "result_coin_gain"),
        ]
        self.crop_target_display = {key: display for display, key in self.crop_targets}
        self._coin_gain_best: Optional[int] = None
        self._hud_coin_reads: list[tuple[int, float, str]] = []
        self._gain_coin_reads: list[tuple[int, float, str]] = []
        self._coin_gain_last_debug = ""
        self._coin_gain_last_capture_frame: Optional[QImage] = None
        self._score_gain_best: Optional[int] = None
        self._score_gain_reads: list[tuple[int, float, str]] = []
        self._score_gain_last_debug = ""
        self._score_gain_last_capture_frame: Optional[QImage] = None
        self._result_gain_best: dict[str, Optional[int]] = {
            k: None for k in RESULT_GAIN_FIELD_KEYS
        }
        self._result_gain_reads: dict[str, list[tuple[int, float, str]]] = {
            k: [] for k in RESULT_GAIN_FIELD_KEYS
        }
        self._result_gain_last_debug: dict[str, str] = {k: "" for k in RESULT_GAIN_FIELD_KEYS}
        self._result_gain_last_capture_frame: Optional[QImage] = None
        self._video_tool_result_fields: dict[str, dict] = {}
        self._last_completed_round: Optional[dict] = None
        self.trainer = SimpleTrainer(
            images_root=self.project_root / "app/assets/images",
            model_root=self.project_root / "app/models/main_model",
        )
        self.tsum_registry = TsumRegistry(self.project_root / "app/models/use_tsum")
        self.use_tsum_classifier = UseTsumClassifier(
            models_root=self.project_root / "app/models/use_tsum",
            registry_path=self.project_root / "app/models/use_tsum/registry.json",
        )
        self.skill_classifier_pool = SkillClassifierPool(
            models_root=self.project_root / "app/models/skill",
        )
        self.train_timer = QTimer(self)
        self.train_timer.setInterval(500)
        self.train_timer.timeout.connect(self._on_train_timer_tick)
        self.train_poll_timer = QTimer(self)
        self.train_poll_timer.setInterval(100)
        self.train_poll_timer.timeout.connect(self._poll_train_messages)
        self.train_message_queue: queue.Queue[str] = queue.Queue()
        self.train_thread: Optional[threading.Thread] = None
        self.train_busy = False
        self._train_busy_task = ""
        self._scene_refit_busy = False
        self.train_started_at = 0.0
        self._trainer_scene_dirty = False
        self._last_fever_blocked_log: tuple[str, bool] | None = None
        self._ingame_entered_logged = False
        self._ingame_entry_none_streak = 0
        self.use_opencv_for_video = False
        self._last_file_video_open_error = ""
        self.cv_source: Optional[FileVideoSource] = FileVideoSource() if is_file_video_available() else None
        self._cv_position_ms = 0
        self._cv_playing = False
        self._cv_playback_rate = 1.0
        self.cv_play_timer = QTimer(self)
        self.cv_play_timer.timeout.connect(self._cv_timer_tick)
        self._time_efficiency_skip_timer = QTimer(self)
        self._time_efficiency_skip_timer.setSingleShot(True)
        self._time_efficiency_skip_timer.timeout.connect(
            self._on_time_efficiency_go_skip_timeout
        )
        self._setup_menu()
        self._setup_status_bar()
        self._setup_central_widget()

    def _setup_menu(self) -> None:
        file_menu = self.menuBar().addMenu("ファイル")

        exit_action = QAction("終了", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        help_menu = self.menuBar().addMenu("ヘルプ")
        about_action = QAction("このアプリについて", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_status_bar(self) -> None:
        self.statusBar().hide()

    def _setup_central_widget(self) -> None:
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(0)

        header = QFrame(central)
        header.setObjectName("headerSection")
        header.setStyleSheet("QFrame#headerSection { border: none; background: #FFF; }")
        header.setFixedHeight(50)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(8)
        button_height = 36
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.feature_buttons = []
        button_labels = [
            "テスト・処理",
            "動画ツール",
            "学習",
            "button4",
            "button5",
            "button6",
            "button7",
            "button8",
        ]
        for index in range(8):
            button = QPushButton(button_labels[index], header)
            button.setFixedSize(100, button_height)
            button.setCheckable(True)
            if index == 0:
                button.setChecked(True)
            header_layout.addWidget(button)
            self.button_group.addButton(button, index + 1)
            self.feature_buttons.append(button)
        self.button_group.idClicked.connect(self._on_feature_changed)
        close_button = QPushButton("✖", header)
        close_button.setFixedSize(button_height, button_height)
        close_button.clicked.connect(self.close)
        header_layout.addWidget(close_button)

        main = QFrame(central)
        main.setObjectName("mainSection")
        main.setStyleSheet("QFrame#mainSection { border: none; background: #FFF; }")
        main_layout = QHBoxLayout(main)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left = QFrame(main)
        left.setObjectName("leftSection")
        left.setStyleSheet("QFrame#leftSection { border: none; background: #FFF; }")
        left.setFixedWidth(300)
        self.left_layout = QVBoxLayout(left)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self._left_section_frame = left

        center = QFrame(main)
        center.setObjectName("centerSection")
        center.setStyleSheet("QFrame#centerSection { border: none; background: #FFF; }")
        center.setFixedWidth(480)
        self.center_layout = QVBoxLayout(center)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_layout.setSpacing(0)

        right = QFrame(main)
        right.setObjectName("rightSection")
        right.setStyleSheet("QFrame#rightSection { border: none; background: #FFF; }")
        right.setFixedWidth(520)
        self.right_layout = QVBoxLayout(right)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self._right_section_frame = right

        main_layout.addWidget(left, 0)
        main_layout.addWidget(center, 0)
        main_layout.addWidget(right, 0)

        footer = QFrame(central)
        footer.setObjectName("footerSection")
        footer.setStyleSheet("QFrame#footerSection { border: none; background: #FFF; }")
        footer.setFixedHeight(50)
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 16, 0)
        version_label = QLabel(
            f"{_APP_VERSION}  {self.os_name}  {self.compute_status}",
            footer,
        )
        version_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(version_label, 1)

        layout.addWidget(header)
        layout.addWidget(main, 1)
        layout.addWidget(footer)
        self.setCentralWidget(central)
        self._render_feature_ui(1)

    def _on_feature_changed(self, feature_id: int) -> None:
        if feature_id != 2:
            self._hide_trim_overlay_on_video()
        self._render_feature_ui(feature_id)

    def _on_video_tool_quick_mode_clicked(self, mode_id: int) -> None:
        """動画ツール: トリム/シーン/コイン/結果（内部 ID 1〜4）、5〜6=未割当。"""
        if self.button_group.checkedId() != 2:
            return
        stack = getattr(self, "_video_tool_right_stack", None)
        if stack is None or not _is_alive_qobject(stack):
            return
        if mode_id == 1:
            stack.setCurrentIndex(0)
            stack.setVisible(True)
            if self.pending_crop_rect is not None:
                self._prepare_trim_overlay_on_video()
            elif hasattr(self, "crop_target_buttons"):
                self._on_crop_target_selection_changed()
        elif mode_id == 2:
            stack.setCurrentIndex(1)
            stack.setVisible(True)
            self._hide_trim_overlay_on_video()
        elif mode_id == 3:
            stack.setCurrentIndex(2)
            stack.setVisible(True)
            self._hide_trim_overlay_on_video()
            self._refresh_video_tool_coin_digit_preview(reset_read=True)
        elif mode_id == 4:
            stack.setCurrentIndex(3)
            stack.setVisible(True)
            self._hide_trim_overlay_on_video()
            self._refresh_video_tool_result_digit_preview(reset_read=True)
        else:
            stack.setVisible(False)
            self._hide_trim_overlay_on_video()

    def _is_video_tool_coin_digit_mode(self) -> bool:
        if self.button_group.checkedId() != 2:
            return False
        group = getattr(self, "_video_tool_mode_group", None)
        if group is None or not _is_alive_qobject(group):
            return False
        return group.checkedId() == 3

    def _is_video_tool_result_digit_mode(self) -> bool:
        if self.button_group.checkedId() != 2:
            return False
        group = getattr(self, "_video_tool_mode_group", None)
        if group is None or not _is_alive_qobject(group):
            return False
        return group.checkedId() == 4

    def _refresh_video_tool_digit_preview_if_active(self, *, reset_read: bool = False) -> None:
        if self._is_video_tool_coin_digit_mode():
            self._refresh_video_tool_coin_digit_preview(reset_read=reset_read)
        elif self._is_video_tool_result_digit_mode():
            self._refresh_video_tool_result_digit_preview(reset_read=reset_read)

    def _video_tool_digit_frame_image(self) -> Optional[QImage]:
        if (
            self._is_video_tool_coin_digit_mode() or self._is_video_tool_result_digit_mode()
        ) and self._is_player_paused():
            ms = self._playback_position_ms()
            img = self._read_frame_at_ms(ms)
            if img is not None and not img.isNull():
                self.current_video_frame_image = img
                return img
        image = self.current_video_frame_image
        if image is not None and not image.isNull():
            return image
        return None

    def _video_tool_coin_digit_frame_image(self) -> Optional[QImage]:
        return self._video_tool_digit_frame_image()

    def _refresh_video_tool_result_field_preview(
        self,
        field_key: str,
        frame_image=None,
        positions: Optional[dict] = None,
        *,
        reset_read: bool = False,
    ) -> None:
        widgets = self._video_tool_result_fields.get(field_key)
        if not widgets:
            return
        preview = widgets.get("preview")
        read_label = widgets.get("read_label")
        if preview is None or not _is_alive_qobject(preview):
            return
        spec = next((f for f in RESULT_GAIN_FIELDS if f.key == field_key), None)
        if reset_read and read_label is not None and _is_alive_qobject(read_label):
            read_label.setText(f"{spec.label if spec else field_key}読取: 未実行")
        if frame_image is None:
            frame_image = self._video_tool_digit_frame_image()
        if positions is None:
            positions = self._load_crop_positions()
        if frame_image is None or not self._crop_rect_defined(positions, field_key):
            preview.setText(
                f"{field_key} 範囲未設定"
                if not self._crop_rect_defined(positions, field_key)
                else "フレームなし"
            )
            preview.setPixmap(QPixmap())
            return
        crop = self._crop_frame_roi_from_positions(frame_image, positions, field_key)
        if crop is None or crop.isNull():
            preview.setText("切り抜き失敗")
            preview.setPixmap(QPixmap())
            return
        pix = QPixmap.fromImage(crop)
        if pix.isNull():
            preview.setText("プレビューなし")
            preview.setPixmap(QPixmap())
        else:
            preview.setText("")
            preview.setPixmap(
                pix.scaled(
                    preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _refresh_video_tool_result_digit_preview(self, *, reset_read: bool = False) -> None:
        count_label = getattr(self, "_video_tool_result_count_label", None)
        assets = self.project_root / "app/assets/images"
        train_n, val_n = count_saved_result_crops(assets)
        if count_label is not None and _is_alive_qobject(count_label):
            count_label.setText(f"保存済み: train={train_n} val={val_n}")
        frame_image = self._video_tool_digit_frame_image()
        positions = self._load_crop_positions()
        for field_key in RESULT_GAIN_FIELD_KEYS:
            self._refresh_video_tool_result_field_preview(
                field_key,
                frame_image,
                positions,
                reset_read=reset_read,
            )

    def _on_video_tool_result_digit_read_clicked(self) -> None:
        status = getattr(self, "_video_tool_result_status", None)
        if self.button_group.checkedId() != 2:
            return
        if not self._is_player_paused():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("一時停止してから読取してください。")
            return
        frame_image = self._video_tool_digit_frame_image()
        if frame_image is None:
            if status is not None and _is_alive_qobject(status):
                status.setText("フレームがありません。")
            return
        if not opencv_available() or not result_digit_cnn_available():
            if status is not None and _is_alive_qobject(status):
                status.setText("opencv または result_digit 未準備です。")
            return
        positions = self._load_crop_positions()
        any_ok = False
        for spec in RESULT_GAIN_FIELDS:
            widgets = self._video_tool_result_fields.get(spec.key, {})
            read_label = widgets.get("read_label")
            value_spin = widgets.get("value_spin")
            preview = widgets.get("preview")
            if not self._crop_rect_defined(positions, spec.key):
                if read_label is not None and _is_alive_qobject(read_label):
                    read_label.setText(f"{spec.label}読取: 範囲未設定")
                continue
            crop = self._crop_frame_roi_from_positions(frame_image, positions, spec.key)
            if crop is None or crop.isNull():
                if read_label is not None and _is_alive_qobject(read_label):
                    read_label.setText(f"{spec.label}読取: 切り抜き失敗")
                continue
            if preview is not None and _is_alive_qobject(preview):
                pix = QPixmap.fromImage(crop)
                if pix.isNull():
                    preview.setText("プレビューなし")
                    preview.setPixmap(QPixmap())
                else:
                    preview.setText("")
                    preview.setPixmap(
                        pix.scaled(
                            preview.size(),
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
            val, _err, dbg = read_result_gain_crop(crop, field_key=spec.key)
            if val is not None and spec.plausible(val):
                any_ok = True
                if read_label is not None and _is_alive_qobject(read_label):
                    read_label.setText(f"{spec.label}読取: {int(val):,}")
                if value_spin is not None and _is_alive_qobject(value_spin):
                    value_spin.blockSignals(True)
                    value_spin.setValue(int(val))
                    value_spin.blockSignals(False)
            elif read_label is not None and _is_alive_qobject(read_label):
                read_label.setText(f"{spec.label}読取: 失敗 ({dbg})")
        if status is not None and _is_alive_qobject(status):
            if any_ok:
                status.setStyleSheet("color: #555;")
                status.setText("読取完了。見た目と照合してください。")
            else:
                status.setStyleSheet("color: #C62828;")
                status.setText(
                    "読取失敗。正解値を手入力→「学習用に保存」→"
                    "学習タブ「結果桁 CNN だけ保存」で再学習してください。"
                )

    def _on_video_tool_result_field_save_clicked(self, field_key: str) -> None:
        status = getattr(self, "_video_tool_result_status", None)
        widgets = self._video_tool_result_fields.get(field_key, {})
        value_spin = widgets.get("value_spin")
        spec = next((f for f in RESULT_GAIN_FIELDS if f.key == field_key), None)
        if (
            self.button_group.checkedId() != 2
            or value_spin is None
            or not _is_alive_qobject(value_spin)
            or spec is None
        ):
            return
        if not self._is_player_paused():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("一時停止してから保存してください。")
            return
        frame_image = self._video_tool_digit_frame_image()
        if frame_image is None:
            if status is not None and _is_alive_qobject(status):
                status.setText("フレームがありません。")
            return
        positions = self._load_crop_positions()
        if not self._crop_rect_defined(positions, field_key):
            if status is not None and _is_alive_qobject(status):
                status.setText(f"{spec.label}: 範囲未設定")
            return
        crop = self._crop_frame_roi_from_positions(frame_image, positions, field_key)
        if crop is None or crop.isNull():
            if status is not None and _is_alive_qobject(status):
                status.setText(f"{spec.label}: 切り抜き失敗")
            return
        label = int(value_spin.value())
        if not spec.plausible(label):
            if status is not None and _is_alive_qobject(status):
                status.setText(f"{spec.label}: 正解値が範囲外です。")
            return
        frame_index = int(
            (max(self._playback_position_ms(), 0) / 1000.0) * float(self.estimated_fps or 30.0)
        )
        ok, msg = save_labeled_result_gain_crop(
            crop,
            label,
            field_key,
            assets_root=self.project_root / "app/assets/images",
            frame_index=frame_index,
        )
        if status is not None and _is_alive_qobject(status):
            if ok:
                status.setStyleSheet("color: #2E7D32;")
                status.setText(f"{spec.label} 保存 → {msg}")
            else:
                status.setStyleSheet("color: #C62828;")
                status.setText(f"保存失敗: {msg}")
        if ok:
            self._train_log(f"リザルト桁学習用保存({spec.label}): {msg}")
        self._refresh_video_tool_result_digit_preview()

    def _refresh_video_tool_coin_digit_preview(self, *, reset_read: bool = False) -> None:
        """切り抜きプレビューと保存枚数のみ更新（CNN 読取は行わない）。"""
        preview = getattr(self, "_video_tool_coin_preview_label", None)
        read_label = getattr(self, "_video_tool_coin_read_label", None)
        count_label = getattr(self, "_video_tool_coin_count_label", None)
        if preview is None or not _is_alive_qobject(preview):
            return

        assets = self.project_root / "app/assets/images"
        train_n, val_n = count_saved_crops(assets)
        if count_label is not None and _is_alive_qobject(count_label):
            count_label.setText(f"保存済み: train={train_n} val={val_n}")

        if reset_read and read_label is not None and _is_alive_qobject(read_label):
            read_label.setText("コイン読取: 未実行")

        frame_image = self._video_tool_coin_digit_frame_image()
        if frame_image is None:
            preview.setText("フレームなし")
            preview.setPixmap(QPixmap())
            if reset_read and read_label is not None and _is_alive_qobject(read_label):
                read_label.setText("読取: 動画を止めてフレームを表示してください")
            return

        positions = self._load_crop_positions()
        if not self._crop_rect_defined(positions, "coin_gain"):
            preview.setText("coin_gain 範囲未設定")
            preview.setPixmap(QPixmap())
            if reset_read and read_label is not None and _is_alive_qobject(read_label):
                read_label.setText("読取: 「トリム」で獲得コイン範囲を保存してください")
            return

        crop = self._crop_frame_roi(frame_image, "coin_gain")
        if crop is None or crop.isNull():
            preview.setText("切り抜き失敗")
            preview.setPixmap(QPixmap())
            if reset_read and read_label is not None and _is_alive_qobject(read_label):
                read_label.setText("読取: 切り抜きできませんでした")
            return

        pix = QPixmap.fromImage(crop)
        if pix.isNull():
            preview.setText("プレビューなし")
            preview.setPixmap(QPixmap())
        else:
            preview.setText("")
            preview.setPixmap(
                pix.scaled(
                    preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._refresh_video_tool_score_digit_preview(
            frame_image, positions, reset_read=reset_read
        )

    def _refresh_video_tool_score_digit_preview(
        self,
        frame_image=None,
        positions: Optional[dict] = None,
        *,
        reset_read: bool = False,
    ) -> None:
        preview = getattr(self, "_video_tool_score_preview_label", None)
        read_label = getattr(self, "_video_tool_score_read_label", None)
        if preview is None or not _is_alive_qobject(preview):
            return
        if reset_read and read_label is not None and _is_alive_qobject(read_label):
            read_label.setText("スコア読取: 未実行")
        if frame_image is None:
            frame_image = self._video_tool_coin_digit_frame_image()
        if positions is None:
            positions = self._load_crop_positions()
        if frame_image is None or not self._crop_rect_defined(positions, "score_gain"):
            preview.setText(
                "score_gain 範囲未設定"
                if not self._crop_rect_defined(positions, "score_gain")
                else "フレームなし"
            )
            preview.setPixmap(QPixmap())
            if reset_read and read_label is not None and _is_alive_qobject(read_label):
                if not self._crop_rect_defined(positions, "score_gain"):
                    read_label.setText(
                        "スコア読取: 「トリム」で獲得スコア範囲を保存してください"
                    )
            return
        crop = self._crop_frame_roi_from_positions(frame_image, positions, "score_gain")
        if crop is None or crop.isNull():
            preview.setText("切り抜き失敗")
            preview.setPixmap(QPixmap())
            return
        pix = QPixmap.fromImage(crop)
        if pix.isNull():
            preview.setText("プレビューなし")
            preview.setPixmap(QPixmap())
        else:
            preview.setText("")
            preview.setPixmap(
                pix.scaled(
                    preview.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def _on_video_tool_coin_digit_read_clicked(self) -> None:
        """動画ツール3: 現在フレームの coin_gain を CNN で読取。"""
        status = getattr(self, "_video_tool_coin_status", None)
        read_label = getattr(self, "_video_tool_coin_read_label", None)
        value_spin = getattr(self, "_video_tool_coin_value_spin", None)
        if self.button_group.checkedId() != 2:
            return
        if not self._is_player_paused():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("一時停止してから読取してください。")
            return

        frame_image = self._video_tool_coin_digit_frame_image()
        if frame_image is None:
            if read_label is not None and _is_alive_qobject(read_label):
                read_label.setText("読取: フレームがありません")
            return

        crop = self._crop_frame_roi(frame_image, "coin_gain")
        if crop is None or crop.isNull():
            if read_label is not None and _is_alive_qobject(read_label):
                read_label.setText("読取: coin_gain 範囲未設定か切り抜き失敗")
            return

        if not opencv_available():
            if read_label is not None and _is_alive_qobject(read_label):
                read_label.setText("読取: opencv 未導入")
            return
        if not coin_digit_cnn_available():
            if read_label is not None and _is_alive_qobject(read_label):
                read_label.setText("読取: coin_digit 未学習（学習タブでモデル保存）")
            return

        crop_val, _err, dbg = read_coin_gain_crop(crop)
        coin_ok = crop_val is not None and plausible_coin_value(crop_val)
        if coin_ok:
            if read_label is not None and _is_alive_qobject(read_label):
                read_label.setText(f"コイン読取: {int(crop_val):,}")
            if value_spin is not None and _is_alive_qobject(value_spin):
                value_spin.blockSignals(True)
                value_spin.setValue(int(crop_val))
                value_spin.blockSignals(False)
        else:
            if read_label is not None and _is_alive_qobject(read_label):
                read_label.setText(f"コイン読取: 読み取れませんでした ({dbg})")

        score_read_label = getattr(self, "_video_tool_score_read_label", None)
        positions = self._load_crop_positions()
        score_ok = False
        if self._crop_rect_defined(positions, "score_gain"):
            score_crop = self._crop_frame_roi_from_positions(
                frame_image, positions, "score_gain"
            )
            if score_crop is not None and not score_crop.isNull():
                score_val, _s_err, s_dbg = read_score_gain_crop(score_crop)
                score_ok = score_val is not None and plausible_score_value(score_val)
                if score_read_label is not None and _is_alive_qobject(score_read_label):
                    if score_ok:
                        score_read_label.setText(f"スコア読取: {int(score_val):,}")
                        score_spin = getattr(self, "_video_tool_score_value_spin", None)
                        if score_spin is not None and _is_alive_qobject(score_spin):
                            score_spin.blockSignals(True)
                            score_spin.setValue(int(score_val))
                            score_spin.blockSignals(False)
                    else:
                        score_read_label.setText(
                            f"スコア読取: 読み取れませんでした ({s_dbg})"
                        )
        elif score_read_label is not None and _is_alive_qobject(score_read_label):
            score_read_label.setText(
                "スコア読取: 「トリム」で獲得スコア範囲を保存してください"
            )

        if status is not None and _is_alive_qobject(status):
            if coin_ok or score_ok:
                status.setStyleSheet("color: #555;")
                status.setText("読取完了。見た目と照合してください。")
            else:
                status.setStyleSheet("color: #C62828;")
                status.setText("読取失敗。正解値を手入力して保存できます。")

    def _on_video_tool_coin_digit_save_clicked(self) -> None:
        """動画ツール3: 獲得コイン切り抜きを正解ラベル付きで学習用保存。"""
        status = getattr(self, "_video_tool_coin_status", None)
        value_spin = getattr(self, "_video_tool_coin_value_spin", None)
        if self.button_group.checkedId() != 2 or value_spin is None or not _is_alive_qobject(value_spin):
            return
        if not self._is_player_paused():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("一時停止してから保存してください。")
            return

        frame_image = self._video_tool_coin_digit_frame_image()
        if frame_image is None:
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("フレームがありません。動画を止めてから保存してください。")
            return

        crop = self._crop_frame_roi(frame_image, "coin_gain")
        if crop is None or crop.isNull():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("coin_gain 範囲が未設定か切り抜きに失敗しました。")
            return

        label = int(value_spin.value())
        if not plausible_coin_value(label):
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("正解値は 1〜9999999（1〜7桁）の範囲で入力してください。")
            return

        frame_index = int(
            (max(self._playback_position_ms(), 0) / 1000.0) * float(self.estimated_fps or 30.0)
        )
        ok, msg = save_labeled_crop(
            crop,
            label,
            assets_root=self.project_root / "app/assets/images",
            frame_index=frame_index,
        )
        if status is not None and _is_alive_qobject(status):
            if ok:
                status.setStyleSheet("color: #2E7D32;")
                status.setText(f"保存しました → {msg}")
            else:
                status.setStyleSheet("color: #C62828;")
                status.setText(f"保存失敗: {msg}")
        if ok:
            self._train_log(f"コイン桁学習用保存: {msg}")
            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(f"コイン桁学習用保存: {msg}")
        else:
            self._train_log(f"コイン桁学習用保存失敗: {msg}")
        self._refresh_video_tool_coin_digit_preview()

    def _on_video_tool_score_digit_save_clicked(self) -> None:
        """動画ツール3: 獲得スコア切り抜きを正解ラベル付きで学習用保存。"""
        status = getattr(self, "_video_tool_coin_status", None)
        value_spin = getattr(self, "_video_tool_score_value_spin", None)
        if self.button_group.checkedId() != 2 or value_spin is None or not _is_alive_qobject(value_spin):
            return
        if not self._is_player_paused():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("一時停止してから保存してください。")
            return

        frame_image = self._video_tool_coin_digit_frame_image()
        if frame_image is None:
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("フレームがありません。動画を止めてから保存してください。")
            return

        positions = self._load_crop_positions()
        if not self._crop_rect_defined(positions, "score_gain"):
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("score_gain 範囲が未設定です。「トリム」で設定してください。")
            return

        crop = self._crop_frame_roi_from_positions(frame_image, positions, "score_gain")
        if crop is None or crop.isNull():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("score_gain 切り抜きに失敗しました。")
            return

        label = int(value_spin.value())
        if not plausible_score_value(label):
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("正解値は 1〜9999999999（1〜10桁）の範囲で入力してください。")
            return

        frame_index = int(
            (max(self._playback_position_ms(), 0) / 1000.0) * float(self.estimated_fps or 30.0)
        )
        ok, msg = save_labeled_score_crop(
            crop,
            label,
            assets_root=self.project_root / "app/assets/images",
            frame_index=frame_index,
        )
        if status is not None and _is_alive_qobject(status):
            if ok:
                status.setStyleSheet("color: #2E7D32;")
                status.setText(f"スコア保存しました → {msg}")
            else:
                status.setStyleSheet("color: #C62828;")
                status.setText(f"スコア保存失敗: {msg}")
        if ok:
            self._train_log(f"スコア桁学習用保存: {msg}")
            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(f"スコア桁学習用保存: {msg}")
        else:
            self._train_log(f"スコア桁学習用保存失敗: {msg}")
        self._refresh_video_tool_coin_digit_preview()

    def _on_video_tool_scene_save_clicked(self) -> None:
        """右列・画像保存: 停止中フレームを PNG で書き出す（train/val は自動）。"""
        status = getattr(self, "_video_tool_scene_status", None)
        combo = getattr(self, "_video_tool_scene_class_combo", None)
        if self.button_group.checkedId() != 2 or combo is None or not _is_alive_qobject(combo):
            return
        image = self.current_video_frame_image
        if image is None or image.isNull():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("フレームがありません。動画を一時停止または停止してから保存してください。")
            return
        choice = combo.currentText()
        if not choice:
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("クラスを選択してください。")
            return
        ok, msg = self._save_frame_to_scene_class(image, choice)
        if msg.startswith(SAVE_SKIP_DUPLICATE_PREFIX):
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #555;")
                status.setText(
                    f"同一画像のため保存スキップ → {msg[len(SAVE_SKIP_DUPLICATE_PREFIX):]}"
                )
                self._train_log(f"画像保存スキップ（同一）: {msg[len(SAVE_SKIP_DUPLICATE_PREFIX):]}")
            return
        if ok:
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #2E7D32;")
                parts = msg.split("/", 2)
                if len(parts) == 3:
                    split, cls, fname = parts[0], parts[1], parts[2]
                    status.setText(f"保存しました → {split} / {cls} / {fname}")
                    self._train_log(f"画像保存: {split}/{cls} -> {fname}")
                else:
                    status.setText(f"保存しました → {msg}")
                    self._train_log(f"画像保存: {msg}")
        else:
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText(f"書き込み失敗: {msg}")

    def _on_video_tool_skill_save_clicked(self) -> None:
        """動画ツール2: スキル発動画像を skills/<dir>/<種類>/ へ保存。"""
        status = getattr(self, "_video_tool_scene_status", None)
        save_btn = getattr(self, "_video_tool_skill_save_btn", None)
        tsum_combo = getattr(self, "_video_tool_skill_tsum_combo", None)
        cat_combo = getattr(self, "_video_tool_skill_cat_combo", None)
        if self.button_group.checkedId() != 2 or tsum_combo is None or cat_combo is None:
            return
        if not _is_alive_qobject(tsum_combo) or not _is_alive_qobject(cat_combo):
            return
        image = self.current_video_frame_image
        if image is None or image.isNull():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("フレームがありません。動画を一時停止または停止してから保存してください。")
            return
        tsum_dir = tsum_combo.currentData()
        category = cat_combo.currentData()
        if not tsum_dir or not category:
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("ツムと種類を選択してください。")
            return

        if save_btn is not None and _is_alive_qobject(save_btn):
            save_btn.setEnabled(False)
        if status is not None and _is_alive_qobject(status):
            status.setStyleSheet("color: #555;")
            status.setText("スキル保存中…")
        QApplication.processEvents()

        try:
            ok, msg = self._save_frame_to_skill(image, str(tsum_dir), str(category))
        finally:
            if save_btn is not None and _is_alive_qobject(save_btn):
                save_btn.setEnabled(True)

        if status is None or not _is_alive_qobject(status):
            return
        if ok:
            status.setStyleSheet("color: #2E7D32;")
            status.setText(f"スキル保存 → {msg}")
        else:
            status.setStyleSheet("color: #C62828;")
            status.setText(f"スキル保存失敗: {msg}")

    def _choose_scene_train_val_split(self, cls: str) -> str:
        base = self.project_root / "app/assets/images"
        train_dir = base / "train" / cls
        val_dir = base / "val" / cls
        train_count = self._count_image_files(train_dir)
        val_count = self._count_image_files(val_dir)
        total = train_count + val_count
        expected_val_after = int((total + 1) * 0.2)
        if val_count < expected_val_after:
            return "val"
        return "train"

    def _skills_images_root(self) -> Path:
        return self.project_root / "app/assets/images/skills"

    def _default_skill_tsum_dir(self) -> str:
        if self.locked_item_fixed and self.locked_use_tsum not in {"-", ""}:
            resolved = self._resolve_tsum_dir(self.locked_use_tsum)
            if resolved:
                return resolved
        if hasattr(self, "item_tsum_combo") and _is_alive_qobject(self.item_tsum_combo):
            selected = self.item_tsum_combo.currentText()
            if selected and selected != "auto":
                return self._resolve_tsum_dir(selected)
        return ""

    def _save_frame_to_skill(self, image, tsum_dir: str, category: str) -> tuple[bool, str]:
        if image is None or image.isNull():
            return False, "画像がありません"
        if not tsum_dir or not category:
            return False, "ツムまたは種類が未選択です"
        out_dir = self._skills_images_root() / tsum_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_index = int((max(self._playback_position_ms(), 0) / 1000.0) * float(self.estimated_fps or 30.0))
        out_path = out_dir / f"skill_{category}_{timestamp}_f{frame_index}.png"
        if save_training_png(image, out_path):
            return True, f"skills/{tsum_dir}/{category}/{out_path.name}"
        return False, str(out_path)

    def _create_skill_tsum_category_combos(self, default_tsum_dir: str) -> tuple[QComboBox, QComboBox]:
        tsum_combo = QComboBox()
        tsum_ids = list_skill_tsum_dirs(self._skills_images_root())
        if not tsum_ids:
            tsum_ids = self.tsum_registry.list_tsum_ids()
        for tid in tsum_ids:
            display = self.use_tsum_classifier._display_map.get(tid, tid)
            tsum_combo.addItem(f"{display} ({tid})", tid)
        if default_tsum_dir:
            idx = tsum_combo.findData(default_tsum_dir)
            if idx >= 0:
                tsum_combo.setCurrentIndex(idx)

        cat_combo = QComboBox()

        def refill_categories() -> None:
            cat_combo.clear()
            tid = tsum_combo.currentData()
            if not tid:
                return
            for cat in list_skill_categories(self._skills_images_root(), str(tid)):
                cat_combo.addItem(skill_category_label(cat), cat)
            act_idx = cat_combo.findData(SKILL_IMAGE_CATEGORY_ACTIVATION)
            if act_idx >= 0:
                cat_combo.setCurrentIndex(act_idx)

        tsum_combo.currentIndexChanged.connect(lambda _i: refill_categories())
        refill_categories()
        return tsum_combo, cat_combo

    def _build_skill_save_group(
        self,
        parent_layout: QVBoxLayout,
        *,
        show_fever_hint: bool,
        default_tsum_dir: str,
    ) -> tuple[QCheckBox, QComboBox, QComboBox]:
        """シーン確認ダイアログ用: スキル発動画面の任意保存 UI。"""
        group = QGroupBox("スキル発動画面（任意）")
        g_layout = QVBoxLayout(group)
        hint = (
            "フィーバー表示とスキル発動が重なることがあります。"
            "このフレームをスキル学習用に保存する場合は種類を選んでください。"
        )
        if show_fever_hint:
            hint = "※ フィーバーと重なっている場合、下で発動時などを選んで保存できます。\n" + hint
        hint_lbl = QLabel(hint)
        hint_lbl.setWordWrap(True)
        g_layout.addWidget(hint_lbl)

        save_cb = QCheckBox("スキル発動画像も保存する")
        save_cb.setChecked(False)
        g_layout.addWidget(save_cb)

        tsum_combo, cat_combo = self._create_skill_tsum_category_combos(default_tsum_dir)
        tsum_row = QHBoxLayout()
        tsum_row.addWidget(QLabel("ツム (dir)"))
        tsum_row.addWidget(tsum_combo, 1)
        g_layout.addLayout(tsum_row)
        cat_row = QHBoxLayout()
        cat_row.addWidget(QLabel("種類"))
        cat_row.addWidget(cat_combo, 1)
        g_layout.addLayout(cat_row)

        save_cb.toggled.connect(tsum_combo.setEnabled)
        save_cb.toggled.connect(cat_combo.setEnabled)
        tsum_combo.setEnabled(False)
        cat_combo.setEnabled(False)

        parent_layout.addWidget(group)
        return save_cb, tsum_combo, cat_combo

    def _try_skill_save_from_dialog(
        self,
        frame_image,
        save_cb: QCheckBox,
        tsum_combo: QComboBox,
        cat_combo: QComboBox,
    ) -> None:
        if not save_cb.isChecked():
            return
        tsum_dir = tsum_combo.currentData()
        category = cat_combo.currentData()
        if not tsum_dir or not category:
            return
        ok, msg = self._save_frame_to_skill(frame_image, str(tsum_dir), str(category))
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(("スキル画像保存: " if ok else "スキル画像保存失敗: ") + msg)
        if ok:
            self._train_log(f"スキル画像保存: {msg}")

    def _save_frame_to_scene_class(self, image, choice: str) -> tuple[bool, str]:
        """現在位置のフレームを train/val 振り分けで PNG 保存（動画ツールの画像保存と同じ規則）。"""
        if image is None or image.isNull():
            return False, "画像がありません"
        images_root = self.project_root / "app/assets/images"
        existing = find_duplicate_scene_image(images_root, image)
        if existing is not None:
            try:
                rel = existing.relative_to(images_root).as_posix()
            except ValueError:
                rel = existing.as_posix()
            return True, f"{SAVE_SKIP_DUPLICATE_PREFIX}{rel}"
        split = self._choose_scene_train_val_split(choice)
        out_dir = images_root / split / choice
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_index = int((max(self._playback_position_ms(), 0) / 1000.0) * float(self.estimated_fps or 30.0))
        out_path = out_dir / f"scene_{choice}_{timestamp}_f{frame_index}.png"
        if save_training_png(image, out_path):
            register_saved_scene_image(out_path)
            return True, f"{split}/{choice}/{out_path.name}"
        return False, str(out_path)

    def _scene_class_dataset_count(self, cls: str) -> int:
        base = self.project_root / "app/assets/images"
        return self._count_image_files(base / "train" / cls) + self._count_image_files(
            base / "val" / cls
        )

    def _maybe_save_scene_class_on_confirm_yes(self, frame_image, scene_key: str) -> None:
        """確認モーダル「はい」時、当該クラスの train/val 画像が0枚なら1枚保存。"""
        if scene_key not in SimpleTrainer.CLASSES or scene_key == "none":
            return
        if frame_image is None or frame_image.isNull():
            return
        if self._scene_class_dataset_count(scene_key) > 0:
            return
        save_ok, msg = self._save_frame_to_scene_class(frame_image, scene_key)
        if not hasattr(self, "log_view") or not _is_alive_qobject(self.log_view):
            return
        if msg.startswith(SAVE_SKIP_DUPLICATE_PREFIX):
            return
        if save_ok:
            self.log_view.append(
                f"初回サンプル保存: {msg}（{scene_key}）。"
                "学習タブで学習→保存すると反映されます。"
            )
            self._train_log(f"確認モーダル初回保存: {msg}")
            self._refit_scene_model_after_correction(scene_key, frame_image)
        else:
            self.log_view.append(f"初回サンプル保存失敗: {msg}")

    def _append_none_veto_from_frame(self, frame_image) -> None:
        """修正保存した none 画面を centroid 誤検知リストへ（CNN 解析中は使わない）。"""
        if frame_image is None or frame_image.isNull():
            return
        feat = image_to_feature(frame_image)
        if not feat:
            return
        model = self.video_analyzer.scene_classifier.model
        model.none_veto_exemplars.insert(0, feat)
        del model.none_veto_exemplars[_NONE_VETO_EXEMPLAR_MAX:]

    def _append_session_fever_none_veto(self, frame_image) -> None:
        """今回の解析で fever 誤検知として none にしたフレームだけ記録。"""
        if frame_image is None or frame_image.isNull():
            return
        feat = image_to_feature(frame_image)
        if not feat:
            return
        self._analysis_fever_none_veto.insert(0, feat)
        del self._analysis_fever_none_veto[_SESSION_FEVER_NONE_VETO_MAX:]

    @staticmethod
    def _fever_score_from_ranked(ranked: list) -> float | None:
        for cls, score in ranked:
            if cls == "fever":
                return float(score)
        return None

    def _fever_cnn_top1_detected(self, ranked: list) -> bool:
        """CNN top1 が fever かつ val 学習データと同程度の自信度。"""
        return bool(ranked) and ranked[0][0] == "fever" and ranked[0][1] < _CNN_FEVER_DETECT_MAX

    def _fever_strong_cnn_signal(
        self, ranked: list, *, fever_strong_hit: bool, fever_top1: bool
    ) -> bool:
        return self._fever_cnn_top1_detected(ranked)

    def _fever_clear_cnn_signal(
        self, ranked: list, *, fever_strong_hit: bool, fever_top1: bool
    ) -> bool:
        return self._fever_cnn_top1_detected(ranked)

    def _fever_cnn_confirm_need(
        self, ranked: list, *, fever_strong_hit: bool, fever_top1: bool
    ) -> int:
        return 1 if self._fever_cnn_top1_detected(ranked) else _FEVER_CNN_CONFIRM_STREAK

    def _fever_cnn_gates_ok(
        self, ranked: list, *, fever_strong_hit: bool, fever_top1: bool
    ) -> bool:
        """誤検知拒否直後でも、はっきりした fever は通す。"""
        if self._fever_reject_cooldown <= 0:
            return True
        return self._fever_clear_cnn_signal(
            ranked, fever_strong_hit=fever_strong_hit, fever_top1=fever_top1
        )

    def _fever_blocked_by_none_veto(
        self,
        frame_image,
        scene_feature=None,
        *,
        ranked: list | None = None,
        fever_strong_hit: bool = False,
        fever_top1: bool = False,
    ) -> bool:
        fever_score = self._fever_score_from_ranked(ranked or [])
        if fever_score is not None and fever_score <= 0.12:
            return False
        if self._fever_clear_cnn_signal(
            ranked or [], fever_strong_hit=fever_strong_hit, fever_top1=fever_top1
        ):
            return False
        if self.video_analyzer.scene_classifier.uses_cnn():
            exemplars = self._analysis_fever_none_veto
            max_dist = _NONE_VETO_SESSION_MAX
        else:
            exemplars = self.video_analyzer.scene_classifier.model.none_veto_exemplars
            max_dist = _NONE_VETO_ABSOLUTE_MAX
        feat = scene_feature
        if not feat and frame_image is not None and not frame_image.isNull():
            feat = image_to_feature(frame_image)
        if not feat:
            return False
        return feature_matches_none_exemplars(feat, exemplars, max_dist)

    def _fever_frame_vetoed(self, frame_index: int) -> bool:
        indices = getattr(self, "_fever_veto_frame_indices", None)
        if not indices:
            return False
        return int(frame_index) in indices

    def _record_fever_none_rejection(self, frame_image, frame_index: int) -> None:
        self._reset_fever_latch()
        self._fever_raw_streak = 0
        self._fever_none_gap_used = 0
        self._append_session_fever_none_veto(frame_image)
        self._fever_veto_frame_indices.add(int(frame_index))
        self._fever_reject_cooldown = _FEVER_REJECT_COOLDOWN_SAMPLES

    def _reject_fever_detection(self, frame_image=None, frame_index: int = -1) -> None:
        """モーダルで fever を拒否した直後: ラッチ解除・再検知抑止。"""
        self._reset_fever_latch()
        self._fever_raw_streak = 0
        self._fever_none_gap_used = 0
        self._fever_reject_cooldown = _FEVER_REJECT_COOLDOWN_SAMPLES
        if frame_index >= 0:
            self._fever_veto_frame_indices.add(int(frame_index))
        if frame_image is not None and not frame_image.isNull():
            self._append_session_fever_none_veto(frame_image)

    @staticmethod
    def _timeup_score_from_ranked(ranked: list) -> float | None:
        for cls, score in ranked:
            if cls == "timeup":
                return float(score)
        return None

    def _timeup_cnn_top1_detected(self, ranked: list) -> bool:
        return bool(ranked) and ranked[0][0] == "timeup" and ranked[0][1] < _CNN_TIMEUP_DETECT_MAX

    def _timeup_cnn_detected(self, ranked: list) -> bool:
        """IN_GAME: top1=timeup または top-k 内の僅差 timeup 候補。"""
        if not ranked:
            return False
        if self._timeup_cnn_top1_detected(ranked):
            return True
        top_score = float(ranked[0][1])
        for i, (cls, score) in enumerate(ranked[:4]):
            if cls != "timeup":
                continue
            sc = float(score)
            if sc >= _CNN_TIMEUP_SCORE_MAX:
                continue
            if i == 0:
                return True
            if sc - top_score <= 0.14:
                return True
        return False

    @staticmethod
    def _timeup_cnn_ambiguous_with_none(ranked: list) -> bool:
        """CNN で timeup 1 位でも none が僅差なら timeup 確定しない。"""
        if not ranked:
            return False
        if ranked[0][0] == "none":
            return True
        timeup_score: float | None = None
        none_score: float | None = None
        for cls, score in ranked:
            if cls == "timeup":
                timeup_score = float(score) if timeup_score is None else min(timeup_score, float(score))
            elif cls == "none":
                none_score = float(score) if none_score is None else min(none_score, float(score))
        if timeup_score is None:
            return False
        if none_score is None:
            return False
        if ranked[0][0] != "timeup":
            return False
        # score が十分小さい（強い timeup）ときは僅差判定で潰さない。
        if timeup_score <= 0.12:
            return False
        return none_score <= timeup_score + _CNN_TIMEUP_NONE_MARGIN

    def _append_session_timeup_none_veto(self, frame_image) -> None:
        if frame_image is None or frame_image.isNull():
            return
        feat = image_to_feature(frame_image)
        if not feat:
            return
        self._analysis_timeup_none_veto.insert(0, feat)
        del self._analysis_timeup_none_veto[_SESSION_TIMEUP_NONE_VETO_MAX:]

    def _timeup_frame_vetoed(self, frame_index: int) -> bool:
        return int(frame_index) in self._timeup_veto_frame_indices

    def _timeup_blocked_by_session_veto(
        self, frame_image, scene_feature=None, ranked: list | None = None
    ) -> bool:
        timeup_score = self._timeup_score_from_ranked(ranked or [])
        # 強い timeup 候補は veto しない（本物まで潰れるのを防ぐ）。
        if timeup_score is not None and timeup_score <= 0.12:
            return False
        if ranked and self.video_analyzer.scene_classifier.uses_cnn():
            if self._timeup_cnn_ambiguous_with_none(ranked):
                return True
        feat = scene_feature
        if not feat and frame_image is not None and not frame_image.isNull():
            feat = image_to_feature(frame_image)
        if not feat:
            return False
        if self._analysis_timeup_none_veto and feature_matches_none_exemplars(
            feat, self._analysis_timeup_none_veto, _NONE_VETO_SESSION_MAX
        ):
            return True
        centroid = self.video_analyzer.scene_classifier.model
        return bool(centroid.none_veto_exemplars) and centroid.none_veto_blocks_timeup(
            feat, timeup_score=timeup_score
        )

    def _reset_per_game_scene_flags(self) -> None:
        self._item_confirmed_this_game = False
        self._ready_confirmed_this_game = False
        self._go_confirmed_this_game = False
        self._timeup_confirmed_this_game = False
        self._coin_confirmed_this_game = False
        self._coin_cnn_detected_this_game = False
        self._coin_post_detect_ocr_left = 0
        self._coin_result_ocr_left = 0
        self._coin_modal_acknowledged = False
        self._coin_flow_latched = False
        self._coin_reject_cooldown = 0
        self._analysis_coin_none_veto = []
        self._coin_veto_frame_indices = set()
        self._result_confirmed_this_game = False
        self._ready_suppressed_this_game = False
        self._analysis_ready_none_veto = []
        self._ready_veto_frame_indices = set()
        self._pregame_ready_reject_cooldown = 0
        self._pregame_ready_pending = False
        self._pregame_go_pending = False
        self._pregame_ready_streak = 0
        self._pregame_go_streak = 0
        self._pregame_ready_none_gap = 0
        self._pregame_go_none_gap = 0
        self._pregame_ready_latched = False
        self._pregame_go_latched = False
        self._wait_ready_grace = 0
        self._analysis_confirm_edge_key = ""
        self._clear_analysis_confirm_edge_prefix("coin")

    def _refresh_round_timing_label(self) -> None:
        label = getattr(self, "counter_round_timing_label", None)
        if label is None or not _is_alive_qobject(label):
            return
        if (
            self._go_clock_anchor_ms is None
            and self._go_to_timeup_sec is None
            and self._go_to_result_sec is None
        ):
            label.setText("go→timeup -- / go→result --")
            return
        tu = (
            _format_duration_min_sec(self._go_to_timeup_sec)
            if self._go_to_timeup_sec is not None
            else "--"
        )
        rs = (
            _format_duration_min_sec(self._go_to_result_sec)
            if self._go_to_result_sec is not None
            else "--"
        )
        label.setText(f"go→timeup {tu} / go→result {rs}")

    def _elapsed_sec_since_go(self, position_ms: int) -> Optional[float]:
        if self._go_clock_anchor_ms is None:
            return None
        delta_ms = max(0, int(position_ms)) - int(self._go_clock_anchor_ms)
        return delta_ms / 1000.0

    def _start_go_round_clock(self, position_ms: int) -> None:
        self._go_clock_anchor_ms = max(0, int(position_ms))
        self._go_to_timeup_sec = None
        self._go_to_result_sec = None
        self._refresh_round_timing_label()
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(
                f"G{self.flow_game_index} ラウンド計測開始: go "
                f"{_format_duration_min_sec(self._go_clock_anchor_ms / 1000.0)}"
            )

    def _log_go_elapsed(self, event: str, position_ms: int, elapsed_sec: float) -> None:
        if not hasattr(self, "log_view") or not _is_alive_qobject(self.log_view):
            return
        anchor_s = self._go_clock_anchor_ms / 1000.0 if self._go_clock_anchor_ms is not None else 0.0
        pos_s = max(0, int(position_ms)) / 1000.0
        self.log_view.append(
            f"G{self.flow_game_index} go→{event}: {_format_duration_min_sec(elapsed_sec)} "
            f"(go {_format_duration_min_sec(anchor_s)} → {event} {_format_duration_min_sec(pos_s)})"
        )

    def _begin_new_game_from_item(self) -> None:
        self._reset_per_game_scene_flags()
        self._game_active = True
        self._clear_time_efficiency_go_skip_state()

    def _scene_once_confirmed_this_game(self, scene_key: str) -> bool:
        if scene_key == "ready":
            return self._ready_confirmed_this_game
        if scene_key == "go":
            return self._go_confirmed_this_game
        if scene_key == "timeup":
            return self._timeup_confirmed_this_game
        if scene_key == "coin":
            return self._coin_confirmed_this_game
        if scene_key == "result":
            return self._result_confirmed_this_game
        return False

    def _can_detect_scene_once_per_game(self, scene_key: str, flow_phase: str) -> bool:
        if scene_key in ("ready", "go", "timeup", "coin", "result") and not self._game_active:
            return False
        if self._scene_once_confirmed_this_game(scene_key):
            return False
        if scene_key == "ready":
            return flow_phase in ("WAIT_READY", "WAIT_GO")
        if scene_key == "go":
            return flow_phase == "WAIT_GO" and self._ready_confirmed_this_game
        if scene_key == "timeup":
            return flow_phase == "IN_GAME"
        if scene_key == "coin":
            if self._coin_confirmed_this_game or self._coin_cnn_detected_this_game:
                return False
            return flow_phase in ("WAIT_BONUS", "WAIT_COIN")
        if scene_key == "result":
            return flow_phase == "WAIT_RESULT"
        return True

    def _confirm_timeup_detection(self, position_ms: int = 0) -> None:
        if self._timeup_confirmed_this_game:
            return
        self._timeup_confirmed_this_game = True
        self._timeup_position_ms = max(0, int(position_ms))
        elapsed = self._elapsed_sec_since_go(position_ms)
        if elapsed is not None:
            self._go_to_timeup_sec = elapsed
            self._log_go_elapsed("timeup", position_ms, elapsed)
            self._refresh_round_timing_label()
        if self._analysis_bonus_enabled():
            self.flow_phase = "WAIT_BONUS"
        else:
            self.flow_phase = "WAIT_COIN"
            self._reset_coin_gain_capture()
            self._reset_coin_hunt_state()
            self._coin_flow_latched = True
        self._reset_fever_latch()
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    def _record_timeup_none_rejection(self, frame_image, frame_index: int) -> None:
        self._timeup_reject_cooldown = _TIMEUP_REJECT_COOLDOWN_SAMPLES
        self._append_session_timeup_none_veto(frame_image)
        self._timeup_veto_frame_indices.add(int(frame_index))
        self._clear_analysis_confirm_edge_prefix("timeup")

    def _reject_timeup_detection(self, frame_image=None, frame_index: int = -1) -> None:
        self.flow_phase = "IN_GAME"
        self._timeup_reject_cooldown = _TIMEUP_REJECT_COOLDOWN_SAMPLES
        if frame_index >= 0:
            self._timeup_veto_frame_indices.add(int(frame_index))
        if frame_image is not None and not frame_image.isNull():
            self._append_session_timeup_none_veto(frame_image)
        self._clear_analysis_confirm_edge_prefix("timeup")
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    @staticmethod
    def _bonus_score_from_ranked(ranked: list) -> float | None:
        for cls, score in ranked:
            if cls == "bonus":
                return float(score)
        return None

    def _bonus_cnn_top1_detected(self, ranked: list) -> bool:
        return bool(ranked) and ranked[0][0] == "bonus" and ranked[0][1] < _CNN_BONUS_DETECT_MAX

    def _bonus_cnn_detected(self, ranked: list) -> bool:
        """WAIT_BONUS: top1=bonus または top-k 内の僅差 bonus 候補。"""
        if not ranked:
            return False
        if self._bonus_cnn_top1_detected(ranked):
            return True
        top_score = float(ranked[0][1])
        for i, (cls, score) in enumerate(ranked[:4]):
            if cls != "bonus":
                continue
            sc = float(score)
            if sc >= _CNN_BONUS_SCORE_MAX:
                continue
            if i == 0:
                return True
            if sc - top_score <= 0.14:
                return True
        return False

    def _append_session_bonus_none_veto(self, frame_image) -> None:
        if frame_image is None or frame_image.isNull():
            return
        feat = image_to_feature(frame_image)
        if not feat:
            return
        self._analysis_bonus_none_veto.insert(0, feat)
        del self._analysis_bonus_none_veto[_SESSION_BONUS_NONE_VETO_MAX:]

    def _bonus_frame_vetoed(self, frame_index: int) -> bool:
        return int(frame_index) in self._bonus_veto_frame_indices

    def _bonus_blocked_by_session_veto(
        self, frame_image, scene_feature=None, ranked: list | None = None
    ) -> bool:
        if ranked and self._bonus_cnn_detected(ranked):
            return False
        if not self._analysis_bonus_none_veto:
            return False
        feat = scene_feature
        if not feat and frame_image is not None and not frame_image.isNull():
            feat = image_to_feature(frame_image)
        if not feat:
            return False
        return feature_matches_none_exemplars(
            feat, self._analysis_bonus_none_veto, _NONE_VETO_SESSION_MAX
        )

    def _reset_coin_gain_capture(self) -> None:
        self._coin_gain_best = None
        self._hud_coin_reads = []
        self._gain_coin_reads = []
        self._coin_gain_last_debug = ""
        self._coin_gain_last_capture_frame = None
        self._refresh_coin_gain_label()
        self._reset_score_gain_capture()
        self._reset_result_gain_capture()

    def _reset_score_gain_capture(self) -> None:
        self._score_gain_best = None
        self._score_gain_reads = []
        self._score_gain_last_debug = ""
        self._score_gain_last_capture_frame = None
        self._refresh_score_gain_label()

    def _reset_result_gain_capture(self) -> None:
        self._result_gain_best = {k: None for k in RESULT_GAIN_FIELD_KEYS}
        self._result_gain_reads = {k: [] for k in RESULT_GAIN_FIELD_KEYS}
        self._result_gain_last_debug = {k: "" for k in RESULT_GAIN_FIELD_KEYS}
        self._result_gain_last_capture_frame = None
        self._refresh_result_gain_labels()

    @staticmethod
    def _unreliable_slot_coin_read(dbg: str) -> bool:
        return _is_slot_decode_dbg(dbg) and "n=4" not in dbg and "head_tail" not in dbg

    def _recompute_coin_gain_best(self) -> None:
        hud_pick = consensus_coin_gain(
            [(v, e) for v, e, _d in self._hud_coin_reads]
        )
        gain_pick = consensus_coin_gain(
            [(v, e) for v, e, _d in self._gain_coin_reads]
        )
        if hud_pick is not None and (
            gain_pick is None
            or len(self._hud_coin_reads) >= len(self._gain_coin_reads)
        ):
            self._coin_gain_best = hud_pick
        else:
            self._coin_gain_best = gain_pick

    def _reset_coin_hunt_state(self) -> None:
        """timeup/bonus 後の coin 探索開始時に、誤検知 veto で検知不能になるのを防ぐ。"""
        self._analysis_coin_none_veto = []
        self._coin_veto_frame_indices = set()
        self._coin_reject_cooldown = 0
        self._coin_flow_latched = False
        self._coin_modal_acknowledged = False
        self._clear_analysis_confirm_edge_prefix("coin")

    def _snapshot_completed_round(self) -> None:
        self._last_completed_round = {
            "use_tsum": self.locked_use_tsum,
            "item_targets": list(self.locked_item_targets),
            "coin_gain": self._coin_gain_best,
            "score_gain": self._score_gain_best,
            "result_score": self._result_gain_best.get("result_score_gain"),
            "score_bonus": self._result_gain_best.get("score_bonus_gain"),
            "result_exp": self._result_gain_best.get("result_exp_gain"),
            "result_coin": self._result_gain_best.get("result_coin_gain"),
            "fever_count": self._fever_count,
            "skill_count": self._skill_count,
        }

    def _refresh_coin_gain_label(self) -> None:
        label = getattr(self, "counter_coin_gain_label", None)
        if label is None or not _is_alive_qobject(label):
            return
        coin_gain = self._coin_gain_best
        if coin_gain is None and self._last_completed_round:
            coin_gain = self._last_completed_round.get("coin_gain")
        if coin_gain is not None:
            label.setText(f"獲得コイン: {coin_gain:,}")
        else:
            label.setText("獲得コイン: --")

    def _refresh_score_gain_label(self) -> None:
        label = getattr(self, "counter_score_gain_label", None)
        if label is None or not _is_alive_qobject(label):
            return
        score_gain = self._score_gain_best
        if score_gain is None and self._last_completed_round:
            score_gain = self._last_completed_round.get("score_gain")
        if score_gain is not None:
            label.setText(f"獲得スコア: {score_gain:,}")
        else:
            label.setText("獲得スコア: --")

    def _refresh_result_gain_labels(self) -> None:
        mapping = (
            ("counter_result_score_label", "result_score_gain", "result_score", "最終スコア"),
            ("counter_score_bonus_label", "score_bonus_gain", "score_bonus", "スコアボーナス"),
            ("counter_result_exp_label", "result_exp_gain", "result_exp", "獲得EXP"),
            ("counter_result_coin_label", "result_coin_gain", "result_coin", "最終コイン"),
        )
        for attr, state_key, snap_key, title in mapping:
            label = getattr(self, attr, None)
            if label is None or not _is_alive_qobject(label):
                continue
            value = self._result_gain_best.get(state_key)
            if value is None and self._last_completed_round:
                value = self._last_completed_round.get(snap_key)
            if value is not None:
                label.setText(f"{title}: {value:,}")
            else:
                label.setText(f"{title}: --")

    def _refresh_fever_skill_counter_labels(self) -> None:
        fever = self._fever_count
        skill = self._skill_count
        if (
            self._last_completed_round
            and not self.locked_item_fixed
            and not self._game_active
        ):
            fever = self._last_completed_round.get("fever_count", fever)
            skill = self._last_completed_round.get("skill_count", skill)
        if hasattr(self, "counter_fever_count_label") and _is_alive_qobject(
            self.counter_fever_count_label
        ):
            self.counter_fever_count_label.setText(f"fever回数: {fever}")
        if hasattr(self, "counter_skill_count_label") and _is_alive_qobject(
            self.counter_skill_count_label
        ):
            self.counter_skill_count_label.setText(f"スキル回数: {skill}")

    def _crop_frame_roi(self, frame_image, key: str) -> Optional[QImage]:
        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        return self._crop_frame_roi_from_positions(frame_image, positions, key)

    def _coin_gain_crop_preview_pixmap(
        self,
        frame_image,
        max_w: int = _COIN_GAIN_CROP_PREVIEW_MAX_W,
        max_h: int = _COIN_GAIN_CROP_PREVIEW_MAX_H,
    ) -> QPixmap:
        crop = self._crop_frame_roi(frame_image, "coin_gain")
        if crop is None or crop.isNull():
            pm = QPixmap(min(320, max_w), min(120, max_h))
            pm.fill(QColor(60, 60, 60))
            return pm
        pm = QPixmap.fromImage(crop)
        if pm.isNull():
            pm = QPixmap(min(320, max_w), min(120, max_h))
            pm.fill(QColor(60, 60, 60))
            return pm
        return pm.scaled(
            max_w,
            max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _ask_coin_gain_training_value(
        self, initial: Optional[int] = None
    ) -> Optional[int]:
        dlg = QDialog(self)
        dlg.setWindowTitle("獲得コインの正解値")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel("coin_gain 切り抜きの正解コイン数を入力してください。"))
        spin = QSpinBox()
        spin.setRange(1, 9_999_999)
        if initial is not None and plausible_coin_value(initial):
            spin.setValue(int(initial))
        layout.addWidget(spin)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        value = int(spin.value())
        return value if plausible_coin_value(value) else None

    def _show_coin_gain_crop_confirm_dialog(
        self, frame_image, coin_value: Optional[int], score_value: Optional[int] = None
    ) -> None:
        if frame_image is None or frame_image.isNull():
            return
        was_cv = getattr(self, "use_opencv_for_video", False)
        was_playing = self._is_player_playing()
        if was_playing:
            if was_cv:
                self._cv_pause()
            elif hasattr(self, "player") and _is_alive_qobject(self.player):
                self.player.pause()
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("コイン・スコア確定")
            dlg.setModal(True)
            layout = QVBoxLayout(dlg)
            crop = self._crop_frame_roi(frame_image, "coin_gain")
            crop_val: Optional[int] = None
            if crop is not None and not crop.isNull():
                read_val, _crop_err, _crop_dbg = read_coin_gain_crop(crop)
                if read_val is not None and plausible_coin_value(read_val):
                    crop_val = int(read_val)
            if crop_val is not None:
                info = QLabel(f"コイン読取: {crop_val:,} (coin確定)")
            else:
                info = QLabel("コイン読取: 読み取れませんでした (coin確定)")
            info.setWordWrap(True)
            layout.addWidget(info)
            positions = self.crop_positions_for_analysis or self._load_crop_positions()
            if self._crop_rect_defined(positions, "score_gain"):
                score_crop = self._crop_frame_roi(frame_image, "score_gain")
                score_crop_val: Optional[int] = None
                if score_crop is not None and not score_crop.isNull():
                    score_read, _s_err, _s_dbg = read_score_gain_crop(score_crop)
                    if score_read is not None and plausible_score_value(score_read):
                        score_crop_val = int(score_read)
                if score_crop_val is not None:
                    score_info = QLabel(f"スコア読取: {score_crop_val:,} (coin確定)")
                elif score_value is not None and plausible_score_value(score_value):
                    score_info = QLabel(f"スコア読取: {score_value:,} (coin確定)")
                else:
                    score_info = QLabel("スコア読取: 読み取れませんでした (coin確定)")
                score_info.setWordWrap(True)
                layout.addWidget(score_info)
            value_row = QHBoxLayout()
            value_row.addWidget(QLabel("正解値（学習用）"))
            value_spin = QSpinBox()
            value_spin.setRange(1, 9_999_999)
            value_spin.setSingleStep(1)
            spin_initial = crop_val
            if spin_initial is None and coin_value is not None and plausible_coin_value(coin_value):
                spin_initial = int(coin_value)
            if spin_initial is not None:
                value_spin.setValue(spin_initial)
            value_row.addWidget(value_spin, 1)
            layout.addLayout(value_row)
            train_n, val_n = count_saved_crops(self.project_root / "app/assets/images")
            save_hint = QLabel(
                f"学習用 crop 保存先: app/assets/images/coin_digits/ "
                f"(現在 train={train_n} val={val_n})"
            )
            save_hint.setWordWrap(True)
            save_hint.setStyleSheet("color: #444;")
            layout.addWidget(save_hint)
            crop_title = QLabel("coin_gain 切り抜き（読取範囲）")
            crop_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(crop_title)
            crop_img = QLabel()
            crop_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            crop_img.setPixmap(self._coin_gain_crop_preview_pixmap(frame_image))
            layout.addWidget(crop_img)

            def _save_coin_digit_training_sample() -> None:
                if crop is None or crop.isNull():
                    return
                label = int(value_spin.value())
                frame_index = 0
                if hasattr(self, "player") and _is_alive_qobject(self.player):
                    frame_index = int(
                        (max(self._playback_position_ms(), 0) / 1000.0)
                        * float(self.estimated_fps or 30.0)
                    )
                ok, msg = save_labeled_crop(
                    crop,
                    label,
                    assets_root=self.project_root / "app/assets/images",
                    frame_index=frame_index,
                )
                self._train_log(
                    f"コイン桁学習用保存: {msg}" if ok else f"コイン桁学習用保存失敗: {msg}"
                )
                if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                    suffix = " → 学習タブで「コイン桁 CNN だけ保存」"
                    self.log_view.append(
                        (f"コイン桁学習用保存: {msg}{suffix}" if ok else f"保存失敗: {msg}")
                    )
                if ok and plausible_coin_value(label):
                    nonlocal coin_value
                    coin_value = label
                    read_part = f"読取: {crop_val:,} → " if crop_val is not None else ""
                    info.setText(
                        f"{read_part}正解値 {coin_value:,} を学習用に保存しました"
                    )
                    tn, vn = count_saved_crops(self.project_root / "app/assets/images")
                    save_hint.setText(
                        f"学習用 crop 保存先: app/assets/images/coin_digits/ "
                        f"(現在 train={tn} val={vn})"
                    )

            save_train_btn = QPushButton("学習用に保存")
            save_train_btn.clicked.connect(_save_coin_digit_training_sample)
            ok_btn = QPushButton("OK")
            ok_btn.setDefault(True)
            ok_btn.clicked.connect(dlg.accept)
            row = QHBoxLayout()
            row.addWidget(save_train_btn)
            row.addStretch(1)
            row.addWidget(ok_btn)
            layout.addLayout(row)
            dlg.exec()
            if crop is not None and not crop.isNull() and plausible_coin_value(value_spin.value()):
                self._coin_gain_best = int(value_spin.value())
                self._refresh_coin_gain_label()
        finally:
            if was_playing and self.analysis_running:
                if was_cv:
                    self._cv_play()
                elif hasattr(self, "player") and _is_alive_qobject(self.player):
                    self.player.play()

    @staticmethod
    def _crop_rect_defined(positions: dict, key: str) -> bool:
        rect = positions.get(key) if isinstance(positions, dict) else None
        if not isinstance(rect, list) or len(rect) != 4:
            return False
        try:
            _nx, _ny, nw, nh = (float(rect[i]) for i in range(4))
            return nw > 0 and nh > 0
        except (TypeError, ValueError):
            return False

    def _crop_frame_roi_from_positions(
        self, frame_image, positions: dict, key: str
    ) -> Optional[QImage]:
        if not self._crop_rect_defined(positions, key):
            return None
        rect = positions.get(key) if isinstance(positions, dict) else None
        if not isinstance(rect, list) or len(rect) != 4:
            return None
        try:
            nx, ny, nw, nh = (
                float(rect[0]),
                float(rect[1]),
                float(rect[2]),
                float(rect[3]),
            )
        except Exception:
            return None
        cropped = UseTsumClassifier._crop_by_normalized_rect(frame_image, (nx, ny, nw, nh))
        if cropped.isNull():
            return None
        return cropped

    def _expanded_normalized_rect(
        self, rect: list, *, pad_x: float, pad_y: float, grow_w: float, grow_h: float, min_h: float = 0.0
    ) -> tuple[float, float, float, float]:
        nx, ny, nw, nh = (float(rect[i]) for i in range(4))
        nh = max(nh, min_h)
        left = max(0.0, nx - nw * pad_x)
        top = max(0.0, ny - nh * pad_y)
        width = min(1.0 - left, nw * grow_w)
        height = min(1.0 - top, nh * grow_h)
        return (left, top, width, height)

    def _coin_gain_rois_for_capture(self, frame_image, scene: str = "") -> list:
        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        scene_key = (self._coin_effective_capture_scene(scene) or "").strip().lower()
        rois: list = []
        if scene_key == "bonus":
            bonus_rect = bonus_coin_hud_rect(positions)
            bonus_roi = UseTsumClassifier._crop_by_normalized_rect(
                frame_image, bonus_rect
            )
            if bonus_roi is not None and not bonus_roi.isNull():
                rois.append(bonus_roi)
        coin_rect = positions.get("coin") if isinstance(positions, dict) else None
        if isinstance(coin_rect, list) and len(coin_rect) == 4:
            try:
                nx, ny, nw, nh = (float(coin_rect[i]) for i in range(4))
                nh = max(nh, 0.055)
                tight = UseTsumClassifier._crop_by_normalized_rect(
                    frame_image, (nx, ny, nw, nh)
                )
                if tight is not None and not tight.isNull():
                    rois.append(tight)
                hud_rect = (
                    nx,
                    max(0.0, ny - nh * 0.35),
                    min(1.0 - nx, nw * 1.08),
                    min(1.0 - max(0.0, ny - nh * 0.35), nh * 1.55),
                )
                coin_hud = UseTsumClassifier._crop_by_normalized_rect(
                    frame_image, hud_rect
                )
                if coin_hud is not None and not coin_hud.isNull():
                    rois.append(coin_hud)
            except (TypeError, ValueError):
                pass
        gain_rois: list = []
        primary = self._crop_frame_roi_from_positions(frame_image, positions, "coin_gain")
        if primary is not None and not primary.isNull():
            gain_rois.append(primary)
        gain_rect = positions.get("coin_gain") if isinstance(positions, dict) else None
        if isinstance(gain_rect, list) and len(gain_rect) == 4:
            try:
                expanded_rect = self._expanded_normalized_rect(
                    gain_rect, pad_x=0.04, pad_y=0.35, grow_w=1.08, grow_h=1.7, min_h=0.048
                )
                expanded = UseTsumClassifier._crop_by_normalized_rect(
                    frame_image, expanded_rect
                )
                if expanded is not None and not expanded.isNull():
                    gain_rois.append(expanded)
            except (TypeError, ValueError):
                pass
        if scene_key == "bonus":
            return rois[:1] if rois else rois
        if scene_key == "coin":
            if gain_rois:
                return gain_rois[:1]
            return rois[:1] if rois else rois
        if scene_key == "result":
            if gain_rois:
                return gain_rois[:1]
            return rois[:1] if rois else rois
        return rois + gain_rois

    def _coin_roi_read_jobs(
        self, rois: list, scene: str
    ) -> list[tuple[QImage, str]]:
        scene_key = (scene or "").strip().lower()
        jobs: list[tuple[QImage, str]] = []
        n_hud = min(2, len(rois))
        for idx, roi in enumerate(rois):
            if scene_key == "bonus":
                source = "hud"
            elif scene_key == "coin":
                source = "gain" if idx < max(0, len(rois) - n_hud) else "hud"
            else:
                source = "hud" if idx < n_hud else "gain"
            jobs.append((roi, source))
        return jobs

    def _coin_effective_capture_scene(self, scene: str) -> str:
        return scene

    def _coin_capture_scene_allowed(self, scene: str) -> bool:
        return (scene or "").strip().lower() == "coin"

    def _update_coin_gain_capture(self, frame_image, scene: str = "") -> None:
        if frame_image is None or frame_image.isNull():
            return
        scene = self._coin_effective_capture_scene(scene)
        if not self._coin_capture_scene_allowed(scene):
            return
        self._update_score_gain_capture(frame_image)
        if not opencv_available():
            self._coin_gain_last_debug = "opencv未導入"
            self._refresh_coin_gain_label()
            return
        if not coin_digit_cnn_available():
            self._coin_gain_last_debug = "coin_digit未学習（学習タブでモデル保存）"
            self._refresh_coin_gain_label()
            return
        rois = self._coin_gain_rois_for_capture(frame_image, scene=scene)
        if not rois:
            self._coin_gain_last_debug = "範囲未設定"
            self._refresh_coin_gain_label()
            return
        frame_reads: list[tuple[int, float, str, str]] = []
        for roi, source in self._coin_roi_read_jobs(rois, scene):
            value, err, dbg = read_coin_gain_crop(roi)
            if scene == "coin":
                source = "hud"
            if value is None or not plausible_coin_value(value):
                self._coin_gain_last_debug = dbg
                continue
            frame_reads.append((value, err, dbg, source))
        if not frame_reads:
            self._refresh_coin_gain_label()
            return
        value, err, dbg, source = min(frame_reads, key=lambda item: item[1])
        if source == "gain" and self._unreliable_slot_coin_read(dbg):
            self._coin_gain_last_debug = f"skip_gain {value} ({dbg})"
            self._refresh_coin_gain_label()
            self._maybe_finalize_coin_confirmation()
            return
        self._coin_gain_last_debug = dbg
        self._coin_gain_last_capture_frame = frame_image.copy()
        if source == "hud":
            self._hud_coin_reads.append((value, err, dbg))
        else:
            self._gain_coin_reads.append((value, err, dbg))
        self._recompute_coin_gain_best()
        self._refresh_coin_gain_label()
        self._maybe_finalize_coin_confirmation()

    def _recompute_score_gain_best(self) -> None:
        pick = consensus_score_gain(
            [(v, e) for v, e, _d in self._score_gain_reads]
        )
        if pick is not None:
            self._score_gain_best = pick

    def _update_score_gain_capture(self, frame_image) -> None:
        if frame_image is None or frame_image.isNull():
            return
        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        if not self._crop_rect_defined(positions, "score_gain"):
            return
        if not opencv_available():
            self._score_gain_last_debug = "opencv未導入"
            self._refresh_score_gain_label()
            return
        if not coin_digit_cnn_available():
            self._score_gain_last_debug = "coin_digit未学習（学習タブでモデル保存）"
            self._refresh_score_gain_label()
            return
        crop = self._crop_frame_roi_from_positions(frame_image, positions, "score_gain")
        if crop is None or crop.isNull():
            self._score_gain_last_debug = "範囲未設定"
            self._refresh_score_gain_label()
            return
        value, err, dbg = read_score_gain_crop(crop)
        if value is None or not plausible_score_value(value):
            self._score_gain_last_debug = dbg
            self._refresh_score_gain_label()
            return
        self._score_gain_last_debug = dbg
        self._score_gain_last_capture_frame = frame_image.copy()
        self._score_gain_reads.append((value, err, dbg))
        self._recompute_score_gain_best()
        self._refresh_score_gain_label()

    def _log_coin_gain_capture(self, note: str = "") -> None:
        if not hasattr(self, "log_view") or not _is_alive_qobject(self.log_view):
            return
        suffix = f" ({note})" if note else ""
        if self._coin_gain_best is not None:
            self.log_view.append(
                f"獲得コイン: {self._coin_gain_best:,}{suffix} [{self._coin_gain_last_debug}]"
            )
        else:
            self.log_view.append(
                f"獲得コイン: 読み取れませんでした ({self._coin_gain_last_debug}){suffix}"
            )

    def _log_score_gain_capture(self, note: str = "") -> None:
        if not hasattr(self, "log_view") or not _is_alive_qobject(self.log_view):
            return
        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        if not self._crop_rect_defined(positions, "score_gain"):
            return
        suffix = f" ({note})" if note else ""
        if self._score_gain_best is not None:
            self.log_view.append(
                f"獲得スコア: {self._score_gain_best:,}{suffix} [{self._score_gain_last_debug}]"
            )
        else:
            self.log_view.append(
                f"獲得スコア: 読み取れませんでした ({self._score_gain_last_debug}){suffix}"
            )

    def _confirm_bonus_detection(self) -> None:
        self.flow_phase = "WAIT_COIN"
        self._reset_coin_gain_capture()
        self._reset_coin_hunt_state()
        self._coin_flow_latched = True
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    @staticmethod
    def _coin_score_from_ranked(ranked: list) -> float | None:
        for cls, score in ranked:
            if cls == "coin":
                return float(score)
        return None

    def _coin_cnn_top1_detected(self, ranked: list) -> bool:
        return bool(ranked) and ranked[0][0] == "coin" and ranked[0][1] < _CNN_COIN_DETECT_MAX

    def _coin_cnn_detected(self, ranked: list) -> bool:
        """WAIT_COIN: top1=coin または top-k 内の僅差 coin 候補。"""
        if not ranked:
            return False
        if self._coin_cnn_top1_detected(ranked):
            return True
        top_score = float(ranked[0][1])
        for i, (cls, score) in enumerate(ranked[:4]):
            if cls != "coin":
                continue
            sc = float(score)
            if sc >= _CNN_COIN_SCORE_MAX:
                continue
            if i == 0:
                return True
            if sc - top_score <= 0.14:
                return True
        return False

    def _coin_scene_detected(
        self,
        *,
        raw_scene: str,
        coin_hit: bool,
        use_scene_cnn: bool,
        ranked: list,
        frame_index: int,
        scene_feature,
        frame_image,
    ) -> bool:
        if use_scene_cnn:
            cooldown_ok = (
                self.flow_phase == "WAIT_COIN" or self._coin_reject_cooldown <= 0
            )
            return (
                self._coin_cnn_detected(ranked)
                and cooldown_ok
                and not (
                    (frame_index >= 0 and self._coin_frame_vetoed(frame_index))
                    or self._coin_blocked_by_session_veto(
                        frame_image, scene_feature, ranked=ranked
                    )
                )
            )
        return raw_scene == "coin" or coin_hit

    def _append_session_coin_none_veto(self, frame_image) -> None:
        if frame_image is None or frame_image.isNull():
            return
        feat = image_to_feature(frame_image)
        if not feat:
            return
        self._analysis_coin_none_veto.insert(0, feat)
        del self._analysis_coin_none_veto[_SESSION_COIN_NONE_VETO_MAX:]

    def _coin_frame_vetoed(self, frame_index: int) -> bool:
        return int(frame_index) in self._coin_veto_frame_indices

    def _coin_blocked_by_session_veto(
        self, frame_image, scene_feature=None, ranked: list | None = None
    ) -> bool:
        if self.flow_phase == "WAIT_COIN":
            return False
        if ranked and self._coin_cnn_detected(ranked):
            return False
        if not self._analysis_coin_none_veto:
            return False
        feat = scene_feature
        if not feat and frame_image is not None and not frame_image.isNull():
            feat = image_to_feature(frame_image)
        if not feat:
            return False
        return feature_matches_none_exemplars(
            feat, self._analysis_coin_none_veto, _NONE_VETO_SESSION_MAX
        )

    def _read_coin_gain_hud_crop(self, frame_image) -> tuple[Optional[int], float, str]:
        crop = self._crop_frame_roi(frame_image, "coin_gain")
        if crop is None or crop.isNull():
            return None, 1e9, "cropなし"
        return read_coin_gain_crop(crop)

    def _read_score_gain_crop(self, frame_image) -> tuple[Optional[int], float, str]:
        crop = self._crop_frame_roi(frame_image, "score_gain")
        if crop is None or crop.isNull():
            return None, 1e9, "cropなし"
        return read_score_gain_crop(crop)

    def _finalize_score_gain_at_coin(self, frame_image) -> None:
        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        if not self._crop_rect_defined(positions, "score_gain"):
            return
        if score_gain_confirmed(self._score_gain_reads):
            if frame_image is not None and not frame_image.isNull():
                crop_val, crop_err, crop_dbg = self._read_score_gain_crop(frame_image)
                if crop_val is not None and plausible_score_value(crop_val):
                    if crop_err <= _MAX_HUD_ACCEPTABLE_ERR:
                        self._score_gain_best = crop_val
                        self._score_gain_last_debug = crop_dbg
        self._log_score_gain_capture("coin確定")
        self._refresh_score_gain_label()

    def _update_result_gain_capture(self, frame_image) -> None:
        if frame_image is None or frame_image.isNull():
            return
        if self.flow_phase != "WAIT_RESULT" or self._result_confirmed_this_game:
            return
        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        if not opencv_available() or not result_digit_cnn_available():
            return
        any_read = False
        for spec in RESULT_GAIN_FIELDS:
            if not self._crop_rect_defined(positions, spec.key):
                continue
            crop = self._crop_frame_roi_from_positions(frame_image, positions, spec.key)
            if crop is None or crop.isNull():
                continue
            value, err, dbg = read_result_gain_crop(crop, field_key=spec.key)
            if value is None or not spec.plausible(value):
                self._result_gain_last_debug[spec.key] = dbg
                continue
            any_read = True
            self._result_gain_last_debug[spec.key] = dbg
            self._result_gain_reads[spec.key].append((value, err, dbg))
            pick = consensus_result_gain(
                [(v, e) for v, e, _d in self._result_gain_reads[spec.key]],
                field_key=spec.key,
            )
            if pick is not None:
                self._result_gain_best[spec.key] = pick
        if any_read:
            self._result_gain_last_capture_frame = frame_image.copy()
        self._refresh_result_gain_labels()

    def _finalize_result_gain_capture(self, frame_image) -> None:
        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        for spec in RESULT_GAIN_FIELDS:
            if not self._crop_rect_defined(positions, spec.key):
                continue
            if not result_gain_confirmed(
                self._result_gain_reads[spec.key], field_key=spec.key
            ):
                continue
            if frame_image is not None and not frame_image.isNull():
                crop = self._crop_frame_roi_from_positions(
                    frame_image, positions, spec.key
                )
                if crop is not None and not crop.isNull():
                    crop_val, crop_err, crop_dbg = read_result_gain_crop(
                        crop, field_key=spec.key
                    )
                    if crop_val is not None and spec.plausible(crop_val):
                        if crop_err <= _MAX_HUD_ACCEPTABLE_ERR:
                            self._result_gain_best[spec.key] = crop_val
                            self._result_gain_last_debug[spec.key] = crop_dbg
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            parts = []
            for spec in RESULT_GAIN_FIELDS:
                val = self._result_gain_best.get(spec.key)
                dbg = self._result_gain_last_debug.get(spec.key, "")
                if val is not None:
                    parts.append(f"{spec.label}={val:,}")
                elif self._crop_rect_defined(positions, spec.key):
                    parts.append(f"{spec.label}=--({dbg})")
            if parts:
                self.log_view.append("リザルト確定: " + " | ".join(parts))
        self._refresh_result_gain_labels()

    def _finalize_coin_confirmation(self) -> None:
        if self._coin_confirmed_this_game:
            return
        if not plausible_coin_value(self._coin_gain_best):
            return
        if not coin_gain_confirmed_combined(self._hud_coin_reads, self._gain_coin_reads):
            return
        frame = self._coin_gain_last_capture_frame
        if frame is not None and not frame.isNull():
            crop_val, crop_err, crop_dbg = self._read_coin_gain_hud_crop(frame)
            if crop_val is not None and plausible_coin_value(crop_val):
                if crop_err <= _MAX_HUD_ACCEPTABLE_ERR or "dl n=4" in crop_dbg or "n=4" in crop_dbg:
                    self._coin_gain_best = crop_val
                    self._coin_gain_last_debug = crop_dbg
        self._finalize_score_gain_at_coin(frame)
        self._coin_confirmed_this_game = True
        self._log_coin_gain_capture("coin確定")
        if self._analysis_scene_confirm_enabled("coin"):
            self._show_coin_gain_crop_confirm_dialog(
                self._coin_gain_last_capture_frame,
                self._coin_gain_best,
                self._score_gain_best,
            )
        self.flow_phase = "WAIT_RESULT"
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    def _maybe_finalize_coin_confirmation(self, frame_image=None) -> None:
        if self._coin_confirmed_this_game:
            return
        if (
            self._analysis_scene_confirm_enabled("coin")
            and not self._coin_modal_acknowledged
        ):
            return
        if not self._coin_flow_latched:
            return
        if not coin_gain_confirmed_combined(self._hud_coin_reads, self._gain_coin_reads):
            return
        self._finalize_coin_confirmation()

    def _acknowledge_coin_scene(self, frame_image=None) -> None:
        """コイン画面の確認（モーダル「はい」）。数値が読めたら WAIT_RESULT へ進む。"""
        self._coin_modal_acknowledged = True
        self._coin_flow_latched = True
        if frame_image is not None and not frame_image.isNull():
            self._update_coin_gain_capture(frame_image, scene="coin")
        self._maybe_finalize_coin_confirmation()

    def _mark_coin_cnn_detected(self) -> None:
        """coin シーン CNN 検知は1ゲーム1回。"""
        self._coin_cnn_detected_this_game = True
        self._coin_flow_latched = True
        self._coin_post_detect_ocr_left = 6

    def _coin_ocr_scene_this_step(self, flow_raw: str) -> str | None:
        """coin 検知後: WAIT_COIN 中に数フレーム OCR。"""
        if self._coin_confirmed_this_game or not self._coin_cnn_detected_this_game:
            return None
        if self._coin_post_detect_ocr_left <= 0:
            return None
        if self.flow_phase != "WAIT_COIN":
            return None
        self._coin_post_detect_ocr_left -= 1
        return "coin"

    def _confirm_coin_detection(self, frame_image=None, scene: str = "") -> None:
        if self._coin_confirmed_this_game:
            return
        if frame_image is not None and not frame_image.isNull():
            self._update_coin_gain_capture(frame_image, scene=scene)
        self._maybe_finalize_coin_confirmation()

    def _record_coin_none_rejection(self, frame_image, frame_index: int) -> None:
        self._coin_reject_cooldown = _COIN_REJECT_COOLDOWN_SAMPLES
        self._append_session_coin_none_veto(frame_image)
        self._coin_veto_frame_indices.add(int(frame_index))
        self._clear_analysis_confirm_edge_prefix("coin")

    def _reject_coin_detection(self, frame_image=None, frame_index: int = -1) -> None:
        self._coin_flow_latched = False
        self.flow_phase = "WAIT_COIN"
        self._coin_reject_cooldown = _COIN_REJECT_COOLDOWN_SAMPLES
        if frame_index >= 0:
            self._coin_veto_frame_indices.add(int(frame_index))
        if frame_image is not None and not frame_image.isNull():
            self._append_session_coin_none_veto(frame_image)
        self._clear_analysis_confirm_edge_prefix("coin")
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    def _record_bonus_none_rejection(self, frame_image, frame_index: int) -> None:
        self._bonus_reject_cooldown = _BONUS_REJECT_COOLDOWN_SAMPLES
        self._append_session_bonus_none_veto(frame_image)
        self._bonus_veto_frame_indices.add(int(frame_index))
        self._clear_analysis_confirm_edge_prefix("bonus")

    def _reject_bonus_detection(self, frame_image=None, frame_index: int = -1) -> None:
        self.flow_phase = "WAIT_BONUS"
        self._bonus_reject_cooldown = _BONUS_REJECT_COOLDOWN_SAMPLES
        if frame_index >= 0:
            self._bonus_veto_frame_indices.add(int(frame_index))
        if frame_image is not None and not frame_image.isNull():
            self._append_session_bonus_none_veto(frame_image)
        self._clear_analysis_confirm_edge_prefix("bonus")
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    @staticmethod
    def _result_score_from_ranked(ranked: list) -> float | None:
        for cls, score in ranked:
            if cls == "result":
                return float(score)
        return None

    def _result_cnn_top1_detected(self, ranked: list) -> bool:
        return bool(ranked) and ranked[0][0] == "result" and ranked[0][1] < _CNN_RESULT_DETECT_MAX

    def _result_cnn_detected(self, ranked: list) -> bool:
        """WAIT_RESULT: top1=result または score が僅差の result 候補。"""
        if not ranked:
            return False
        if self._result_cnn_top1_detected(ranked):
            return True
        top_score = float(ranked[0][1])
        for i, (cls, score) in enumerate(ranked[:4]):
            if cls != "result":
                continue
            sc = float(score)
            if sc >= _CNN_RESULT_DETECT_MAX:
                continue
            if i == 0:
                return True
            if sc - top_score <= 0.14:
                return True
        return False

    def _append_session_result_none_veto(self, frame_image) -> None:
        if frame_image is None or frame_image.isNull():
            return
        feat = image_to_feature(frame_image)
        if not feat:
            return
        self._analysis_result_none_veto.insert(0, feat)
        del self._analysis_result_none_veto[_SESSION_RESULT_NONE_VETO_MAX:]

    def _result_frame_vetoed(self, frame_index: int) -> bool:
        return int(frame_index) in self._result_veto_frame_indices

    def _result_blocked_by_session_veto(
        self, frame_image, scene_feature=None, ranked: list | None = None
    ) -> bool:
        if ranked and self._result_cnn_detected(ranked):
            return False
        if not self._analysis_result_none_veto:
            return False
        feat = scene_feature
        if not feat and frame_image is not None and not frame_image.isNull():
            feat = image_to_feature(frame_image)
        if not feat:
            return False
        return feature_matches_none_exemplars(
            feat, self._analysis_result_none_veto, _NONE_VETO_SESSION_MAX
        )

    def _advance_after_result_confirmed(self) -> None:
        self.flow_game_index += 1
        self._game_active = False
        self.flow_phase = "WAIT_ITEM"
        self.locked_item_targets = []
        self.locked_use_tsum = "-"
        self.locked_item_fixed = False
        self._reset_per_game_scene_flags()
        self._item_scan_targets = set()
        self._item_scene_active = False
        self._item_use_tsum_pending = "-"
        self._item_scan_suppressed = False
        self._reset_coin_gain_capture()
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )
        self._refresh_item_counter_labels()
        self._refresh_fever_skill_counter_labels()

    def _confirm_result_detection(
        self, position_ms: int = 0, frame_image=None
    ) -> None:
        if self._result_confirmed_this_game:
            return
        elapsed = self._elapsed_sec_since_go(position_ms)
        if elapsed is not None:
            self._go_to_result_sec = elapsed
            self._log_go_elapsed("result", position_ms, elapsed)
            self._refresh_round_timing_label()
        frame = frame_image
        if frame is None or frame.isNull():
            frame = self._result_gain_last_capture_frame
        self._finalize_result_gain_capture(frame)
        self._snapshot_completed_round()
        self._result_confirmed_this_game = True
        self._advance_after_result_confirmed()

    def _record_result_none_rejection(self, frame_image, frame_index: int) -> None:
        self._result_reject_cooldown = _RESULT_REJECT_COOLDOWN_SAMPLES
        self._append_session_result_none_veto(frame_image)
        self._result_veto_frame_indices.add(int(frame_index))
        self._clear_analysis_confirm_edge_prefix("result")

    def _reject_result_detection(self, frame_image=None, frame_index: int = -1) -> None:
        self.flow_phase = "WAIT_RESULT"
        self._result_reject_cooldown = _RESULT_REJECT_COOLDOWN_SAMPLES
        if frame_index >= 0:
            self._result_veto_frame_indices.add(int(frame_index))
        if frame_image is not None and not frame_image.isNull():
            self._append_session_result_none_veto(frame_image)
        self._clear_analysis_confirm_edge_prefix("result")
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    def _refit_scene_model_after_correction(self, choice: str, frame_image=None) -> None:
        """解析中の修正保存後、モデルを更新（CNN は学習タブで保存が必要）。"""
        if choice not in SimpleTrainer.CLASSES:
            return
        if choice == "none" and frame_image is not None and _PREGAME_READY_SUPPRESSION_ENABLED:
            self._append_none_veto_from_frame(frame_image)
            images_root = self.project_root / "app/assets/images"
            n_none = self.video_analyzer.scene_classifier.centroid.rebuild_none_veto_exemplars(
                images_root
            )
            self._fever_reject_cooldown = _FEVER_REJECT_COOLDOWN_SAMPLES
            if n_none and hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(
                    f"none参照を更新しました（{n_none}枚）。同じ見た目の ready 誤検知を抑制します。"
                )
        if self.video_analyzer.scene_classifier.uses_cnn():
            self._trainer_scene_dirty = True
            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(
                    f"修正画像を保存しました（{choice}）。"
                    " 学習タブで学習→保存すると反映されます。"
                )
            return
        if self._scene_refit_busy:
            return
        images_root = self.project_root / "app/assets/images"
        self._scene_refit_busy = True
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(
                "モデル再計算を開始しました（数十秒かかることがあります）。"
                "動画は再開します。"
            )

        def work() -> None:
            err: Optional[str] = None
            centroids: Optional[dict] = None
            calib: Optional[dict] = None
            try:
                self.trainer.scene_model.fit_from_dataset(images_root)
                centroids = dict(self.trainer.scene_model.centroids)
                calib = dict(self.trainer.scene_model.fever_calib)
            except Exception as exc:
                err = str(exc)

            def finish() -> None:
                self._scene_refit_busy = False
                if not hasattr(self, "log_view") or not _is_alive_qobject(self.log_view):
                    return
                if err is not None:
                    self.log_view.append(f"モデル再計算失敗: {err}")
                    return
                if centroids:
                    self.video_analyzer.apply_scene_centroids(centroids, calib)
                    self._trainer_scene_dirty = True
                    n = len(self.trainer.scene_model.none_veto_exemplars)
                    self.log_view.append(
                        "モデル再計算が完了し、解析に反映しました。"
                        f"（none参照画像: {n}枚）"
                        "ファイルへ書き込むには学習タブの「モデル保存」。"
                    )

            QTimer.singleShot(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _release_counter_and_log_widgets(self) -> None:
        for name in (
            "log_view",
            "counter_frame",
            "counter_use_tsum_label",
            "counter_use_item_label",
            "counter_fever_count_label",
            "counter_skill_count_label",
            "counter_coin_gain_label",
            "counter_score_gain_label",
            "counter_result_score_label",
            "counter_score_bonus_label",
            "counter_result_exp_label",
            "counter_result_coin_label",
            "counter_round_timing_label",
            "counter_analysis_state_label",
            "counter_progress_label",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def _dispose_media_player(self) -> None:
        """タブ切替で古い QMediaPlayer / QAudioOutput が子として残らないようにする。"""
        pl = getattr(self, "player", None)
        if pl is not None and _is_alive_qobject(pl):
            pl.blockSignals(True)
            try:
                pl.stop()
            except RuntimeError:
                pass
            pl.deleteLater()
            delattr(self, "player")
        ao = getattr(self, "audio_output", None)
        if ao is not None and _is_alive_qobject(ao):
            ao.blockSignals(True)
            ao.deleteLater()
            delattr(self, "audio_output")

    def _render_feature_ui(self, feature_id: int) -> None:
        if hasattr(self, "detail_log_check") and _is_alive_qobject(getattr(self, "detail_log_check", None)):
            self._save_analysis_settings_ui()
        if hasattr(self, "cv_play_timer"):
            self.cv_play_timer.stop()
            self._cv_playing = False
        if getattr(self, "cv_source", None) is not None:
            self.cv_source.release()
        g = getattr(self, "_video_tool_mode_group", None)
        if g is not None:
            for b in list(g.buttons()):
                g.removeButton(b)
            g.deleteLater()
            delattr(self, "_video_tool_mode_group")
        pl = getattr(self, "player", None)
        if pl is not None and _is_alive_qobject(pl):
            pl.blockSignals(True)
            try:
                pl.stop()
            except RuntimeError:
                pass
        self._clear_layout(self.left_layout)
        self._clear_layout(self.center_layout)
        self.current_video_container = None
        self._clear_layout(self.right_layout)
        self._dispose_media_player()
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        self.left_layout.setSpacing(0)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        self.center_layout.setSpacing(0)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)
        self._right_section_frame.setVisible(True)
        if feature_id == 2:
            self._release_counter_and_log_widgets()

        # 処理テスト(1)・動画ツール(2)・学習(3) 共通: left 300 + center 480 + right 520（列幅の例外なし）
        self._left_section_frame.setFixedWidth(300)
        self._right_section_frame.setFixedWidth(520)

        if feature_id == 1:
            self.left_layout.setContentsMargins(2, 2, 2, 2)
            self.left_layout.setSpacing(2)
            self.left_layout.addWidget(QLabel("解析設定"))
            compact_h = 22

            sample_row = QFrame()
            sample_row_layout = QHBoxLayout(sample_row)
            sample_row_layout.setContentsMargins(0, 0, 0, 0)
            sample_row_layout.setSpacing(2)
            self.sample_frame_spin = QSpinBox()
            self.sample_frame_spin.setRange(1, 60)
            self.sample_frame_spin.setValue(15)
            self.sample_frame_spin.setFixedHeight(compact_h)
            self.sample_frame_spin.valueChanged.connect(self._on_sample_frame_changed)
            sample_row_layout.addWidget(QLabel("間隔"))
            sample_row_layout.addWidget(self.sample_frame_spin, 1)
            self.left_layout.addWidget(sample_row)
            interval_hint = QLabel("間隔＝判定の間引きのみ（再生速度は変わりません）。重いときは20〜30。")
            interval_hint.setWordWrap(True)
            interval_hint.setStyleSheet("color: #555; font-size: 11px;")
            self.left_layout.addWidget(interval_hint)

            self.analysis_light_check = QCheckBox("軽量解析（スキル検知オフ・UI更新抑制）")
            self.analysis_light_check.setChecked(False)
            self.left_layout.addWidget(self.analysis_light_check)

            model_row = QFrame()
            model_row_layout = QHBoxLayout(model_row)
            model_row_layout.setContentsMargins(0, 0, 0, 0)
            model_row_layout.setSpacing(2)
            self.model_version_combo = QComboBox()
            self.model_version_combo.addItems(["version_1", "version_2", "version_3"])
            self.model_version_combo.setFixedHeight(compact_h)
            model_row_layout.addWidget(QLabel("モデル"))
            model_row_layout.addWidget(self.model_version_combo, 1)
            self.left_layout.addWidget(model_row)

            tsum_row = QFrame()
            tsum_row_layout = QHBoxLayout(tsum_row)
            tsum_row_layout.setContentsMargins(0, 0, 0, 0)
            tsum_row_layout.setSpacing(2)
            self.item_tsum_combo = QComboBox()
            tsum_ids = self.tsum_registry.list_tsum_ids()
            self.item_tsum_combo.addItem("auto")
            self.item_tsum_combo.addItems(tsum_ids)
            self.item_tsum_combo.setFixedHeight(compact_h)
            tsum_row_layout.addWidget(QLabel("ツム"))
            tsum_row_layout.addWidget(self.item_tsum_combo, 1)
            self.left_layout.addWidget(tsum_row)

            self.detail_log_check = QCheckBox("詳細ログ")
            self.detail_log_check.setChecked(False)
            self.left_layout.addWidget(self.detail_log_check)

            self.start_from_zero_check = QCheckBox("先頭から")
            self.start_from_zero_check.setChecked(True)
            self.left_layout.addWidget(self.start_from_zero_check)

            self.analysis_stop_on_timeup_check = QCheckBox("timeupで解析を自動停止")
            self.analysis_stop_on_timeup_check.setChecked(False)
            self.analysis_stop_on_timeup_check.setToolTip(
                "オンにすると timeup 確定後に解析を止めます。"
                "オフ（推奨）なら bonus/result まで続けます。"
            )
            self.left_layout.addWidget(self.analysis_stop_on_timeup_check)

            self.left_layout.addWidget(QLabel("シーン判定の確認（入った瞬間・モーダル）"))
            confirm_wrap = QWidget()
            confirm_layout = QVBoxLayout(confirm_wrap)
            confirm_layout.setContentsMargins(0, 0, 0, 0)
            confirm_layout.setSpacing(1)
            self.analysis_scene_confirm_checks = {}
            for scene_key in _ANALYSIS_SCENE_CONFIRM_LABELS:
                cb = QCheckBox(scene_key)
                cb.setChecked(scene_key in _SCENE_CONFIRM_DEFAULT_ON)
                if scene_key == "fever" and not _ANALYSIS_FEVER_ENABLED:
                    cb.setChecked(False)
                    cb.setEnabled(False)
                    cb.setToolTip("現在 fever 検知は停止中（timeup 作業中）")
                self.analysis_scene_confirm_checks[scene_key] = cb
                confirm_layout.addWidget(cb)
            self.left_layout.addWidget(confirm_wrap)

            skill_confirm_hint = QLabel(
                "スキル発動はシーン一覧に含まれません（別判定）。"
                "IN_GAME 中に検知され、ログへ skill=ツム名 と出ます。"
            )
            skill_confirm_hint.setWordWrap(True)
            skill_confirm_hint.setStyleSheet("color: #555; font-size: 11px;")
            self.left_layout.addWidget(skill_confirm_hint)
            self.analysis_skill_confirm_check = QCheckBox("スキル発動（検知時モーダル）")
            self.analysis_skill_confirm_check.setChecked(False)
            self.left_layout.addWidget(self.analysis_skill_confirm_check)

            self._bind_analysis_settings_persistence()

            self.left_layout.addStretch(1)
        else:
            if feature_id == 2:
                self.right_layout.setContentsMargins(2, 2, 2, 2)
                self.right_layout.setSpacing(2)
                trim_page = QWidget()
                self._video_tool_right_inner = trim_page
                inner_layout = QVBoxLayout(trim_page)
                inner_layout.setContentsMargins(0, 0, 0, 0)
                inner_layout.setSpacing(2)
                inner_layout.addWidget(QLabel("トリミング"))
                inner_layout.addWidget(QLabel("対象(複数選択)"))
                self.crop_target_buttons = {}
                target_grid = QFrame()
                target_grid_layout = QGridLayout(target_grid)
                target_grid_layout.setContentsMargins(0, 0, 0, 0)
                target_grid_layout.setHorizontalSpacing(4)
                target_grid_layout.setVerticalSpacing(4)
                cols_per_row = 8
                for index, (display, key) in enumerate(self.crop_targets):
                    btn = QPushButton(display)
                    btn.setCheckable(True)
                    btn.setFixedHeight(24)
                    btn.setMinimumWidth(btn.fontMetrics().horizontalAdvance(display) + 14)
                    btn.setToolTip(display)
                    btn.setStyleSheet(
                        "QPushButton { border:1px solid #888; background:#F5F5F5; padding:0 6px; }"
                        "QPushButton:checked { background:#FFD54F; border:1px solid #C9A227; }"
                    )
                    btn.toggled.connect(self._on_crop_target_selection_changed)
                    self.crop_target_buttons[key] = btn
                    target_grid_layout.addWidget(
                        btn, index // cols_per_row, index % cols_per_row
                    )
                inner_layout.addWidget(target_grid)
                # Default selection to avoid "saved but no target selected" confusion.
                if "score" in self.crop_target_buttons:
                    self.crop_target_buttons["score"].setChecked(True)

                self.crop_rect_label = QLabel("範囲: 未選択")
                inner_layout.addWidget(self.crop_rect_label)

                self.crop_start_button = QPushButton("トリミング開始")
                self.crop_start_button.setCheckable(True)
                self.crop_start_button.toggled.connect(self._on_toggle_crop_mode)
                inner_layout.addWidget(self.crop_start_button)

                self.crop_status_label = QLabel("状態: 停止")
                inner_layout.addWidget(self.crop_status_label)

                adjust_row1 = QFrame()
                adjust_row1_layout = QHBoxLayout(adjust_row1)
                adjust_row1_layout.setContentsMargins(0, 0, 0, 0)
                adjust_row1_layout.setSpacing(2)
                adjust_row1_layout.addWidget(QLabel("x"))
                self.crop_x_spin = QDoubleSpinBox()
                self.crop_x_spin.setRange(0.0, 1.0)
                self.crop_x_spin.setSingleStep(0.005)
                self.crop_x_spin.setDecimals(4)
                self.crop_x_spin.setFixedHeight(22)
                self.crop_x_spin.valueChanged.connect(self._on_crop_spin_changed)
                adjust_row1_layout.addWidget(self.crop_x_spin, 1)
                adjust_row1_layout.addWidget(QLabel("y"))
                self.crop_y_spin = QDoubleSpinBox()
                self.crop_y_spin.setRange(0.0, 1.0)
                self.crop_y_spin.setSingleStep(0.005)
                self.crop_y_spin.setDecimals(4)
                self.crop_y_spin.setFixedHeight(22)
                self.crop_y_spin.valueChanged.connect(self._on_crop_spin_changed)
                adjust_row1_layout.addWidget(self.crop_y_spin, 1)
                inner_layout.addWidget(adjust_row1)

                adjust_row2 = QFrame()
                adjust_row2_layout = QHBoxLayout(adjust_row2)
                adjust_row2_layout.setContentsMargins(0, 0, 0, 0)
                adjust_row2_layout.setSpacing(2)
                adjust_row2_layout.addWidget(QLabel("w"))
                self.crop_w_spin = QDoubleSpinBox()
                self.crop_w_spin.setRange(0.0, 1.0)
                self.crop_w_spin.setSingleStep(0.005)
                self.crop_w_spin.setDecimals(4)
                self.crop_w_spin.setFixedHeight(22)
                self.crop_w_spin.valueChanged.connect(self._on_crop_spin_changed)
                adjust_row2_layout.addWidget(self.crop_w_spin, 1)
                adjust_row2_layout.addWidget(QLabel("h"))
                self.crop_h_spin = QDoubleSpinBox()
                self.crop_h_spin.setRange(0.0, 1.0)
                self.crop_h_spin.setSingleStep(0.005)
                self.crop_h_spin.setDecimals(4)
                self.crop_h_spin.setFixedHeight(22)
                self.crop_h_spin.valueChanged.connect(self._on_crop_spin_changed)
                adjust_row2_layout.addWidget(self.crop_h_spin, 1)
                inner_layout.addWidget(adjust_row2)

                save_crop_button = QPushButton("位置を保存")
                save_crop_button.clicked.connect(self._on_save_crop_clicked)
                inner_layout.addWidget(save_crop_button)
                save_dataset_button = QPushButton("対象画像保存(自動train/val)")
                save_dataset_button.clicked.connect(self._on_save_target_images_clicked)
                inner_layout.addWidget(save_dataset_button)
                check_crop_button = QPushButton("停止画で切り抜き確認")
                check_crop_button.clicked.connect(self._on_check_crop_preview_clicked)
                inner_layout.addWidget(check_crop_button)
                inner_layout.addWidget(QLabel("動画上をドラッグで指定"))

                self.crop_preview_label = QLabel("プレビューなし")
                self.crop_preview_label.setFixedSize(180, 100)
                self.crop_preview_label.setStyleSheet("border:1px solid #888; background:#111; color:#DDD;")
                self.crop_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                inner_layout.addWidget(self.crop_preview_label)
                self.crop_check_label = QLabel("確認プレビューなし")
                self.crop_check_label.setFixedSize(280, 180)
                self.crop_check_label.setStyleSheet("border:1px solid #888; background:#111; color:#DDD;")
                self.crop_check_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                inner_layout.addWidget(self.crop_check_label)
                inner_layout.addStretch(1)

                scene_page = QWidget()
                sv = QVBoxLayout(scene_page)
                sv.setContentsMargins(0, 0, 0, 0)
                sv.setSpacing(10)
                scene_title = QLabel("画像の保存")
                scene_title.setStyleSheet("font-weight: bold; font-size: 14px;")
                sv.addWidget(scene_title)
                scene_hint = QLabel(
                    "一時停止または停止中の画面1枚を PNG で書き出します。\n"
                    "保存先: app/assets/images の train または val / クラス名（振り分けは自動・目安 val 約20%）。"
                )
                scene_hint.setWordWrap(True)
                scene_hint.setStyleSheet("color: #444;")
                sv.addWidget(scene_hint)

                class_row = QFrame()
                class_row_layout = QHBoxLayout(class_row)
                class_row_layout.setContentsMargins(0, 0, 0, 0)
                class_row_layout.setSpacing(8)
                class_row_layout.addWidget(QLabel("クラス"))
                self._video_tool_scene_class_combo = QComboBox()
                self._video_tool_scene_class_combo.addItems(list(SimpleTrainer.CLASSES))
                self._video_tool_scene_class_combo.setMinimumWidth(220)
                class_row_layout.addWidget(self._video_tool_scene_class_combo, 1)
                sv.addWidget(class_row)

                self._video_tool_scene_save_btn = QPushButton("画像を保存")
                self._video_tool_scene_save_btn.setToolTip("選択クラスへ保存（train/val は自動）")
                self._video_tool_scene_save_btn.setAutoDefault(True)
                self._video_tool_scene_save_btn.clicked.connect(self._on_video_tool_scene_save_clicked)
                sv.addWidget(self._video_tool_scene_save_btn)

                skill_title = QLabel("スキル発動画像")
                skill_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 8px;")
                sv.addWidget(skill_title)
                skill_hint = QLabel(
                    "保存先: app/assets/images/skills/<dir名>/<種類>/（例: namine/activation）。\n"
                    "フィーバーと重なっている画面も、こちらで発動時として保存できます。"
                )
                skill_hint.setWordWrap(True)
                skill_hint.setStyleSheet("color: #444;")
                sv.addWidget(skill_hint)
                skill_tsum_row = QFrame()
                skill_tsum_row_layout = QHBoxLayout(skill_tsum_row)
                skill_tsum_row_layout.setContentsMargins(0, 0, 0, 0)
                skill_tsum_row_layout.setSpacing(8)
                skill_tsum_row_layout.addWidget(QLabel("ツム"))
                self._video_tool_skill_tsum_combo, self._video_tool_skill_cat_combo = (
                    self._create_skill_tsum_category_combos(self._default_skill_tsum_dir())
                )
                self._video_tool_skill_tsum_combo.setMinimumWidth(180)
                skill_tsum_row_layout.addWidget(self._video_tool_skill_tsum_combo, 1)
                sv.addWidget(skill_tsum_row)
                skill_cat_row = QFrame()
                skill_cat_row_layout = QHBoxLayout(skill_cat_row)
                skill_cat_row_layout.setContentsMargins(0, 0, 0, 0)
                skill_cat_row_layout.setSpacing(8)
                skill_cat_row_layout.addWidget(QLabel("種類"))
                self._video_tool_skill_cat_combo.setMinimumWidth(180)
                skill_cat_row_layout.addWidget(self._video_tool_skill_cat_combo, 1)
                sv.addWidget(skill_cat_row)
                self._video_tool_skill_save_btn = QPushButton("スキル発動画像を保存")
                self._video_tool_skill_save_btn.setToolTip(
                    "skills/<dir>/activation/ などへ保存（学習タブのモデル保存で反映）"
                )
                self._video_tool_skill_save_btn.clicked.connect(self._on_video_tool_skill_save_clicked)
                sv.addWidget(self._video_tool_skill_save_btn)

                self._video_tool_scene_status = QLabel(
                    "左の「シーン」で表示。クラスを選んで「画像を保存」。"
                    "スキルはツム・種類を選んで「スキル発動画像を保存」。"
                )
                self._video_tool_scene_status.setWordWrap(True)
                self._video_tool_scene_status.setStyleSheet("color: #555;")
                sv.addWidget(self._video_tool_scene_status)
                sv.addStretch(1)

                coin_page = QWidget()
                cv = QVBoxLayout(coin_page)
                cv.setContentsMargins(0, 0, 0, 0)
                cv.setSpacing(8)
                coin_title = QLabel("獲得コイン・スコア")
                coin_title.setStyleSheet("font-weight: bold; font-size: 14px;")
                cv.addWidget(coin_title)
                coin_hint = QLabel(
                    "スライダー・コマ送りで coin 画面へ移動し、「読取」で CNN 結果を確認します。\n"
                    "範囲は「トリム」で coin_gain / score_gain をそれぞれ設定。\n"
                    "読取が間違うときは正解値を直して「学習用に保存」"
                    "（coin / score とも app/assets/images/coin_digits/）。\n"
                    "十分貯まったら学習タブの「コイン桁 CNN だけ保存」。"
                    "（coin 画面専用・result とは別モデル）"
                )
                coin_hint.setWordWrap(True)
                coin_hint.setStyleSheet("color: #444;")
                cv.addWidget(coin_hint)

                self._video_tool_coin_read_label = QLabel("コイン読取: 未実行")
                self._video_tool_coin_read_label.setWordWrap(True)
                cv.addWidget(self._video_tool_coin_read_label)
                self._video_tool_score_read_label = QLabel("スコア読取: 未実行")
                self._video_tool_score_read_label.setWordWrap(True)
                cv.addWidget(self._video_tool_score_read_label)

                self._video_tool_coin_read_btn = QPushButton("読取")
                self._video_tool_coin_read_btn.setToolTip(
                    "現在フレームの coin_gain / score_gain を CNN で読取（手動実行）"
                )
                self._video_tool_coin_read_btn.clicked.connect(self._on_video_tool_coin_digit_read_clicked)
                cv.addWidget(self._video_tool_coin_read_btn)

                value_row = QFrame()
                value_row_layout = QHBoxLayout(value_row)
                value_row_layout.setContentsMargins(0, 0, 0, 0)
                value_row_layout.setSpacing(8)
                value_row_layout.addWidget(QLabel("コイン正解値"))
                self._video_tool_coin_value_spin = QSpinBox()
                self._video_tool_coin_value_spin.setRange(1, 9_999_999)
                self._video_tool_coin_value_spin.setMinimumWidth(120)
                value_row_layout.addWidget(self._video_tool_coin_value_spin, 1)
                cv.addWidget(value_row)

                self._video_tool_coin_save_btn = QPushButton("コイン学習用に保存")
                self._video_tool_coin_save_btn.setToolTip(
                    "coin_gain 切り抜きを正解ラベル付きで coin_digits/train|val へ保存"
                )
                self._video_tool_coin_save_btn.clicked.connect(self._on_video_tool_coin_digit_save_clicked)
                cv.addWidget(self._video_tool_coin_save_btn)

                score_value_row = QFrame()
                score_value_row_layout = QHBoxLayout(score_value_row)
                score_value_row_layout.setContentsMargins(0, 0, 0, 0)
                score_value_row_layout.setSpacing(8)
                score_value_row_layout.addWidget(QLabel("スコア正解値"))
                self._video_tool_score_value_spin = QSpinBox()
                self._video_tool_score_value_spin.setRange(1, 2_147_483_647)
                self._video_tool_score_value_spin.setMinimumWidth(120)
                score_value_row_layout.addWidget(self._video_tool_score_value_spin, 1)
                cv.addWidget(score_value_row)

                self._video_tool_score_save_btn = QPushButton("スコア学習用に保存")
                self._video_tool_score_save_btn.setToolTip(
                    "score_gain 切り抜きを正解ラベル付きで coin_digits/train|val へ保存"
                )
                self._video_tool_score_save_btn.clicked.connect(
                    self._on_video_tool_score_digit_save_clicked
                )
                cv.addWidget(self._video_tool_score_save_btn)

                self._video_tool_coin_count_label = QLabel("保存済み: train=0 val=0")
                self._video_tool_coin_count_label.setStyleSheet("color: #555;")
                cv.addWidget(self._video_tool_coin_count_label)

                coin_preview_title = QLabel("coin_gain プレビュー")
                coin_preview_title.setStyleSheet("color: #555;")
                cv.addWidget(coin_preview_title)
                self._video_tool_coin_preview_label = QLabel("プレビューなし")
                self._video_tool_coin_preview_label.setFixedSize(300, 100)
                self._video_tool_coin_preview_label.setStyleSheet(
                    "border:1px solid #888; background:#111; color:#DDD;"
                )
                self._video_tool_coin_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cv.addWidget(self._video_tool_coin_preview_label)

                score_preview_title = QLabel("score_gain プレビュー")
                score_preview_title.setStyleSheet("color: #555;")
                cv.addWidget(score_preview_title)
                self._video_tool_score_preview_label = QLabel("プレビューなし")
                self._video_tool_score_preview_label.setFixedSize(300, 100)
                self._video_tool_score_preview_label.setStyleSheet(
                    "border:1px solid #888; background:#111; color:#DDD;"
                )
                self._video_tool_score_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cv.addWidget(self._video_tool_score_preview_label)

                self._video_tool_coin_status = QLabel(
                    "左の「コイン」で表示。範囲は「トリム」で設定。"
                )
                self._video_tool_coin_status.setWordWrap(True)
                self._video_tool_coin_status.setStyleSheet("color: #555;")
                cv.addWidget(self._video_tool_coin_status)
                cv.addStretch(1)

                result_page = QWidget()
                rv = QVBoxLayout(result_page)
                rv.setContentsMargins(0, 0, 0, 0)
                rv.setSpacing(6)
                result_title = QLabel("リザルト4項目")
                result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
                rv.addWidget(result_title)
                result_hint = QLabel(
                    "まず最終獲得コインから。result 画面で切り抜きを「トリム」で設定し、"
                    "「読取」で確認。間違いは正解値を直して「学習用に保存」。"
                    "保存先: app/assets/images/result_digits/。"
                    "学習タブの「結果桁 CNN だけ保存」。解析時は result確定で記録。"
                )
                result_hint.setWordWrap(True)
                result_hint.setStyleSheet("color: #444;")
                rv.addWidget(result_hint)
                self._video_tool_result_read_btn = QPushButton("読取")
                self._video_tool_result_read_btn.clicked.connect(
                    self._on_video_tool_result_digit_read_clicked
                )
                rv.addWidget(self._video_tool_result_read_btn)
                self._video_tool_result_count_label = QLabel("保存済み: train=0 val=0")
                self._video_tool_result_count_label.setStyleSheet("color: #555;")
                rv.addWidget(self._video_tool_result_count_label)
                self._video_tool_result_fields = {}
                for spec in RESULT_GAIN_FIELDS:
                    read_lbl = QLabel(f"{spec.label}読取: 未実行")
                    read_lbl.setWordWrap(True)
                    rv.addWidget(read_lbl)
                    row = QHBoxLayout()
                    row_w = QWidget()
                    row_w.setLayout(row)
                    spin = QSpinBox()
                    spin.setRange(
                        1,
                        9_999_999 if spec.max_digits <= 7 else 2_147_483_647,
                    )
                    row.addWidget(QLabel("正解値"))
                    row.addWidget(spin, 1)
                    save_btn = QPushButton("学習用に保存")
                    save_btn.clicked.connect(
                        lambda _c=False, k=spec.key: self._on_video_tool_result_field_save_clicked(
                            k
                        )
                    )
                    row.addWidget(save_btn)
                    rv.addWidget(row_w)
                    preview = QLabel("プレビューなし")
                    preview.setFixedSize(280, 72)
                    preview.setStyleSheet(
                        "border:1px solid #888; background:#111; color:#DDD;"
                    )
                    preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    rv.addWidget(preview)
                    self._video_tool_result_fields[spec.key] = {
                        "read_label": read_lbl,
                        "value_spin": spin,
                        "preview": preview,
                        "save_btn": save_btn,
                    }
                self._video_tool_result_status = QLabel(
                    "左の「結果」で表示。4項目の範囲は「トリム」で設定。"
                )
                self._video_tool_result_status.setWordWrap(True)
                self._video_tool_result_status.setStyleSheet("color: #555;")
                rv.addWidget(self._video_tool_result_status)
                rv.addStretch(1)

                self._video_tool_right_stack = QStackedWidget()
                self._video_tool_right_stack.addWidget(trim_page)
                self._video_tool_right_stack.addWidget(scene_page)
                self._video_tool_right_stack.addWidget(coin_page)
                self._video_tool_right_stack.addWidget(result_page)
                self.right_layout.addWidget(self._video_tool_right_stack)
                self._on_crop_target_selection_changed()
            elif feature_id not in (3,):
                self.left_layout.addWidget(QLabel(f"left - button{feature_id}"))
        if feature_id == 2:
            video_tool_btn_style = (
                "QPushButton {"
                "  border: 1px solid #888;"
                "  border-radius: 4px;"
                "  background: #F5F5F5;"
                "  padding: 2px 4px;"
                "}"
                "QPushButton:hover { background: #EAEAEA; }"
                "QPushButton:pressed { background: #DCDCDC; }"
                "QPushButton:checked { background: #FFD54F; border: 1px solid #C9A227; }"
            )
            video_tool_btn_row = QFrame()
            video_tool_btn_row.setStyleSheet("background: transparent;")
            # 縦ボタン列が左列いっぱいに伸びないよう幅を抑える。
            video_tool_btn_row.setMaximumWidth(108)
            btn_row_layout = QVBoxLayout(video_tool_btn_row)
            btn_row_layout.setContentsMargins(4, 4, 4, 4)
            btn_row_layout.setSpacing(4)
            self._video_tool_mode_group = QButtonGroup(self)
            self._video_tool_mode_group.setExclusive(True)
            for i in range(6):
                mode_id = i + 1
                label = _VIDEO_TOOL_MODE_LABELS[i]
                btn = QPushButton(label)
                btn.setCheckable(True)
                btn.setFixedHeight(28)
                btn.setStyleSheet(video_tool_btn_style)
                if mode_id == _VIDEO_TOOL_MODE_TRIM:
                    btn.setToolTip("右列: 切り抜き範囲の指定・位置保存")
                elif mode_id == _VIDEO_TOOL_MODE_SCENE:
                    btn.setToolTip("右列: シーン画像・スキル発動画像の保存")
                elif mode_id == _VIDEO_TOOL_MODE_COIN:
                    btn.setToolTip("右列: coin画面の獲得コイン・獲得スコア（読取・学習用保存）")
                elif mode_id == _VIDEO_TOOL_MODE_RESULT:
                    btn.setToolTip("右列: result画面の4項目（読取・学習用保存）")
                else:
                    btn.setEnabled(False)
                    btn.setToolTip("未割当")
                btn_row_layout.addWidget(btn)
                self._video_tool_mode_group.addButton(btn, mode_id)
            self._video_tool_mode_group.idClicked.connect(self._on_video_tool_quick_mode_clicked)
            first_mode = self._video_tool_mode_group.button(1)
            if first_mode is not None:
                first_mode.setChecked(True)
            stack = getattr(self, "_video_tool_right_stack", None)
            if stack is not None:
                stack.setCurrentIndex(0)
                stack.setVisible(True)
            btn_row_layout.addStretch(1)
            self.left_layout.addWidget(video_tool_btn_row)
            self.left_layout.addStretch(1)
        else:
            self.log_view = QTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setAcceptRichText(True)
            doc = self.log_view.document()
            if doc is not None and hasattr(doc, "setMaximumBlockCount"):
                doc.setMaximumBlockCount(12000)
            self.counter_frame = QFrame()
            self.counter_frame.setFixedHeight(178)
            self.counter_frame.setStyleSheet("border: none; background: transparent;")
            counter_layout = QVBoxLayout(self.counter_frame)
            counter_layout.setContentsMargins(0, 0, 0, 0)
            counter_layout.setSpacing(2)
            self.counter_use_tsum_label = QLabel("使用ツム: --")
            self.counter_use_item_label = QLabel("使用アイテム: --")
            self.counter_fever_count_label = QLabel("fever回数: 0")
            self.counter_skill_count_label = QLabel("スキル回数: 0")
            self.counter_coin_gain_label = QLabel("獲得コイン: --")
            self.counter_score_gain_label = QLabel("獲得スコア: --")
            self.counter_result_score_label = QLabel("最終スコア: --")
            self.counter_score_bonus_label = QLabel("スコアボーナス: --")
            self.counter_result_exp_label = QLabel("獲得EXP: --")
            self.counter_result_coin_label = QLabel("最終コイン: --")
            self.counter_round_timing_label = QLabel("go→timeup -- / go→result --")
            self.counter_analysis_state_label = QLabel("解析状態: 停止")
            self.counter_progress_label = QLabel("進行: --")
            counter_layout.addWidget(self.counter_use_tsum_label)
            counter_layout.addWidget(self.counter_use_item_label)
            counter_layout.addWidget(self.counter_fever_count_label)
            counter_layout.addWidget(self.counter_skill_count_label)
            counter_layout.addWidget(self.counter_coin_gain_label)
            counter_layout.addWidget(self.counter_score_gain_label)
            counter_layout.addWidget(self.counter_result_score_label)
            counter_layout.addWidget(self.counter_score_bonus_label)
            counter_layout.addWidget(self.counter_result_exp_label)
            counter_layout.addWidget(self.counter_result_coin_label)
            counter_layout.addWidget(self.counter_round_timing_label)
            counter_layout.addWidget(self.counter_analysis_state_label)
            counter_layout.addWidget(self.counter_progress_label)
            self.right_layout.addWidget(self.counter_frame)
            self.right_layout.addWidget(self.log_view, 1)
        if feature_id in (1, 2):
            self._analysis_toggle_buttons = []
            self.player = QMediaPlayer(self)
            self.audio_output = QAudioOutput(self)
            self.player.setAudioOutput(self.audio_output)
            self.is_muted = self.settings.value("is_muted", True, type=bool)
            self.audio_output.setMuted(self.is_muted)
            self.player.positionChanged.connect(self._on_player_position_changed)
            self.player.durationChanged.connect(self._on_player_duration_changed)
            self.player.playbackStateChanged.connect(self._on_playback_state_changed)
            self.player.mediaStatusChanged.connect(self._on_media_status_changed)
            self.player.setLoops(1)
            self.player.errorOccurred.connect(self._on_player_error)

            top_box = QFrame()
            top_box.setStyleSheet("border: none; background: #222222;")
            top_layout = QVBoxLayout(top_box)
            top_layout.setContentsMargins(0, 0, 0, 0)
            top_layout.setSpacing(0)

            top_extra_frame = QFrame()
            top_extra_frame.setStyleSheet("border: none; background: #FFF;")
            top_extra_frame.setFixedHeight(30)
            top_extra_layout = QHBoxLayout(top_extra_frame)
            top_extra_layout.setContentsMargins(8, 0, 8, 0)
            top_extra_layout.setSpacing(8)
            open_video_button = QPushButton("動画を開く", top_extra_frame)
            open_video_button.setFixedHeight(26)
            open_video_button.setStyleSheet(
                "QPushButton {"
                "  border: 1px solid #888;"
                "  border-radius: 4px;"
                "  background: #F5F5F5;"
                "  padding: 0 10px;"
                "}"
                "QPushButton:hover { background: #EAEAEA; }"
                "QPushButton:pressed { background: #DCDCDC; }"
            )
            open_video_button.clicked.connect(self._open_video_file)
            top_extra_layout.addWidget(open_video_button)
            self.video_file_label = QLabel("未選択", top_extra_frame)
            self.video_file_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.video_file_label.setStyleSheet("color: #333;")
            top_extra_layout.addWidget(self.video_file_label, 1)
            self.mute_checkbox = QCheckBox("消音", top_extra_frame)
            self.mute_checkbox.setChecked(self.is_muted)
            self.mute_checkbox.toggled.connect(self._on_mute_toggled)
            top_extra_layout.addWidget(self.mute_checkbox)

            top_layout.addWidget(top_extra_frame)

            video_container = AspectFitVideoContainer(top_box)
            self.current_video_container = video_container
            self.use_opencv_for_video = False
            # Windows に加え macOS でも Qt Multimedia がファイルを出さない環境があるため、
            # FileVideoSource が使えるときはファイル動画モード（OpenCV / imageio）に寄せる。
            if self.cv_source is not None and platform.system() in ("Windows", "Darwin"):
                self.use_opencv_for_video = True
            if not self.use_opencv_for_video:
                self.player.setVideoOutput(video_container.video_widget)
                sink = video_container.video_widget.videoSink()
                if sink is not None:
                    sink.videoFrameChanged.connect(self._on_video_frame_changed)
                elif self.cv_source is not None:
                    self.use_opencv_for_video = True
            if self.use_opencv_for_video:
                video_container.set_opencv_display(True)
                self.mute_checkbox.setEnabled(False)
                self.mute_checkbox.setToolTip("ファイル動画モードでは音声がありません")
            else:
                video_container.set_opencv_display(False)
                self.mute_checkbox.setEnabled(True)
                self.mute_checkbox.setToolTip("")
                if video_container.video_widget.videoSink() is None and self.cv_source is None:
                    diag = (
                        "QVideoSink を作成できませんでした。\n"
                        "ファイル動画モードを使うには pip install（opencv-python-headless, imageio, imageio-ffmpeg）を実行してください。\n\n"
                        + _qt_multimedia_environment_report()
                    )
                    QTimer.singleShot(
                        0,
                        lambda w=self, d=diag: _show_copyable_warning(
                            w,
                            "動画機能が使えません",
                            "Qt でもファイル動画デコーダでも初期化できませんでした。",
                            d,
                        ),
                    )
            video_container.clicked.connect(self._on_play_pause_clicked)
            video_container.cropSelected.connect(self._on_crop_selected)
            video_container.set_crop_enabled(False)
            top_layout.addWidget(video_container, 1)

            bottom_box = QFrame()
            bottom_box.setStyleSheet("border: none; background: #FFF;")
            screen = QApplication.primaryScreen()
            if screen is not None:
                bottom_box.setFixedHeight(screen.availableGeometry().height() // 5)
            bottom_layout = QVBoxLayout(bottom_box)
            bottom_layout.setContentsMargins(0, 0, 0, 0)
            bottom_layout.setSpacing(0)

            for index in range(4):
                section = QFrame()
                section.setStyleSheet("border: none; background: #FFF;")
                section_layout = QHBoxLayout(section)
                section_layout.setContentsMargins(0, 0, 0, 0)
                section_layout.setSpacing(0)
                if index == 0:
                    self.time_label = QLabel("00:00 / 00:00", section)
                    self.time_label.setFixedWidth(120)
                    self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    section_layout.addWidget(self.time_label)

                    self.seek_slider = QSlider(Qt.Orientation.Horizontal, section)
                    self.seek_slider.setMinimum(0)
                    self.seek_slider.setMaximum(0)
                    self.seek_slider.setFixedHeight(24)
                    self.seek_slider.sliderMoved.connect(self._on_slider_moved)
                    section_layout.addWidget(self.seek_slider, 1)

                    self.frame_label = QLabel("0 / 0", section)
                    self.frame_label.setFixedWidth(90)
                    self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    section_layout.addWidget(self.frame_label)
                elif index == 1:
                    control_button_style = (
                        "QPushButton {"
                        "  border: 1px solid #888;"
                        "  border-radius: 4px;"
                        "  background: #F5F5F5;"
                        "  padding: 2px 4px;"
                        "}"
                        "QPushButton:hover { background: #EAEAEA; }"
                        "QPushButton:pressed { background: #DCDCDC; }"
                        "QPushButton:disabled { color: #9A9A9A; background: #F2F2F2; border-color: #CCC; }"
                    )
                    section_layout.setContentsMargins(4, 0, 4, 0)
                    section_layout.setSpacing(4)
                    self.step_huge_left_button = QPushButton("-30", section)
                    self.step_huge_left_button.setStyleSheet(control_button_style)
                    self.step_huge_left_button.setFixedSize(46, 24)
                    self.step_huge_left_button.clicked.connect(lambda: self._step_backward(30))
                    section_layout.addWidget(self.step_huge_left_button)

                    self.step_large_left_button = QPushButton("-10", section)
                    self.step_large_left_button.setStyleSheet(control_button_style)
                    self.step_large_left_button.setFixedSize(46, 24)
                    self.step_large_left_button.clicked.connect(lambda: self._step_backward(10))
                    section_layout.addWidget(self.step_large_left_button)

                    self.step_small_left_button = QPushButton("-1", section)
                    self.step_small_left_button.setStyleSheet(control_button_style)
                    self.step_small_left_button.setFixedSize(40, 24)
                    self.step_small_left_button.clicked.connect(lambda: self._step_backward(1))
                    section_layout.addWidget(self.step_small_left_button)

                    self.play_pause_button = QPushButton("再生", section)
                    self.play_pause_button.setStyleSheet(control_button_style)
                    self.play_pause_button.setFixedSize(52, 24)
                    self.play_pause_button.clicked.connect(self._on_play_pause_clicked)
                    section_layout.addWidget(self.play_pause_button)

                    self.stop_button = QPushButton("停止", section)
                    self.stop_button.setStyleSheet(control_button_style)
                    self.stop_button.setFixedSize(52, 24)
                    self.stop_button.clicked.connect(self._on_stop_clicked)
                    section_layout.addWidget(self.stop_button)

                    self.step_small_right_button = QPushButton("+1", section)
                    self.step_small_right_button.setStyleSheet(control_button_style)
                    self.step_small_right_button.setFixedSize(40, 24)
                    self.step_small_right_button.clicked.connect(lambda: self._step_forward(1))
                    section_layout.addWidget(self.step_small_right_button)

                    self.step_large_right_button = QPushButton("+10", section)
                    self.step_large_right_button.setStyleSheet(control_button_style)
                    self.step_large_right_button.setFixedSize(46, 24)
                    self.step_large_right_button.clicked.connect(lambda: self._step_forward(10))
                    section_layout.addWidget(self.step_large_right_button)

                    self.step_huge_right_button = QPushButton("+30", section)
                    self.step_huge_right_button.setStyleSheet(control_button_style)
                    self.step_huge_right_button.setFixedSize(46, 24)
                    self.step_huge_right_button.clicked.connect(lambda: self._step_forward(30))
                    section_layout.addWidget(self.step_huge_right_button)
                elif index == 2:
                    speed_button_style = (
                        "QPushButton {"
                        "  border: 1px solid #888;"
                        "  border-radius: 4px;"
                        "  background: #F5F5F5;"
                        "  padding: 2px 6px;"
                        "}"
                        "QPushButton:hover { background: #EAEAEA; }"
                        "QPushButton:pressed { background: #DCDCDC; }"
                    )
                    section_layout.setContentsMargins(4, 0, 4, 0)
                    section_layout.setSpacing(4)
                    for label, rate in [("1/4", 0.25), ("1/2", 0.5), ("1", 1.0), ("2", 2.0), ("4", 4.0)]:
                        button = QPushButton(label, section)
                        button.setFixedSize(52, 24)
                        button.setStyleSheet(speed_button_style)
                        button.clicked.connect(lambda _checked=False, r=rate: self._set_playback_rate(r))
                        section_layout.addWidget(button)
                else:
                    section_layout.setContentsMargins(4, 0, 4, 0)
                    section_layout.setSpacing(4)
                    analyze_row_button_style = (
                        "QPushButton {"
                        "  border: 1px solid #888;"
                        "  border-radius: 4px;"
                        "  background: #F5F5F5;"
                        "  padding: 2px 8px;"
                        "}"
                        "QPushButton:hover { background: #EAEAEA; }"
                        "QPushButton:pressed { background: #DCDCDC; }"
                        "QPushButton:disabled { color: #9A9A9A; background: #F2F2F2; border-color: #CCC; }"
                    )
                    if feature_id == 1:
                        row_buttons = [
                            ("時間スキップ", "timer_skip"),
                            ("解析", _ANALYSIS_BUTTON_ACTION),
                            ("時間効率", _TIME_EFFICIENCY_BUTTON_ACTION),
                            ("仮4", None),
                        ]
                    else:
                        row_buttons = [
                            ("時間スキップ", "timer_skip"),
                            ("仮2", None),
                            ("時間効率", _TIME_EFFICIENCY_BUTTON_ACTION),
                            ("仮4", None),
                        ]
                    for label, action in row_buttons:
                        button = QPushButton(label, section)
                        button.setStyleSheet(analyze_row_button_style)
                        if action == _ANALYSIS_BUTTON_ACTION:
                            button.setFixedSize(88, 24)
                        elif action == _TIME_EFFICIENCY_BUTTON_ACTION:
                            button.setFixedSize(80, 24)
                        elif action == "timer_skip":
                            button.setFixedSize(80, 24)
                        else:
                            button.setFixedSize(64, 24)
                        if action == "timer_skip":
                            button.setToolTip(
                                "現在フレームの残り時間（最大3桁）を読み取り、"
                                "動画内で同じ残り時間のフレームへジャンプします。"
                                "事前に動画ツールで「残り時間」トリミングを保存してください（必須）。"
                                "item 行の time とは別です。"
                            )
                            button.clicked.connect(self._on_temp1_timer_skip_clicked)
                        elif action in _ANALYSIS_TOGGLE_ACTIONS:
                            if action == _ANALYSIS_BUTTON_ACTION:
                                self.analyze_button = button
                            tip = "シーン解析を開始/停止します（解析ボタンと同じ）。"
                            if action == _TIME_EFFICIENCY_BUTTON_ACTION:
                                tip += (
                                    " fever/bonus は検知しません。"
                                    " go 確定位置から動画60秒先へ自動ジャンプします。"
                                )
                            button.setToolTip(tip)
                            self._register_analysis_toggle_button(button, action)
                            button.clicked.connect(
                                lambda _checked=False, a=action: self._on_analyze_clicked(a)
                            )
                        section_layout.addWidget(button)
                bottom_layout.addWidget(section, 1)

            self.center_layout.addWidget(top_box, 1)
            self.center_layout.addWidget(bottom_box, 1)
        elif feature_id == 3:
            self.left_layout.setContentsMargins(4, 4, 4, 4)
            self.left_layout.setSpacing(4)
            self.left_layout.addWidget(QLabel("学習データ"))
            self.left_layout.addWidget(QLabel("train: app/assets/images/train"))
            self.left_layout.addWidget(QLabel("val: app/assets/images/val"))
            self.left_layout.addWidget(QLabel("保存先: app/models/main_model/version_1"))
            self.left_layout.addStretch(1)

            train_page_content = QWidget()
            train_page_layout = QVBoxLayout(train_page_content)
            train_page_layout.setContentsMargins(8, 6, 8, 6)
            train_page_layout.setSpacing(4)

            data_check_button = QPushButton("データ確認")
            data_check_button.setFixedHeight(28)
            data_check_button.clicked.connect(self._on_train_data_check_clicked)
            self.train_shortage_button = QPushButton("不足画像を確認")
            self.train_shortage_button.setFixedHeight(28)
            self.train_shortage_button.setToolTip(
                "各シーンクラスの train/val 枚数を見て、"
                "足りない画像の種類と追加目安を右の学習ログに表示します。"
            )
            self.train_shortage_button.clicked.connect(self._on_train_shortage_check_clicked)
            data_row = QHBoxLayout()
            data_row.setSpacing(4)
            data_row.addWidget(data_check_button, 1)
            data_row.addWidget(self.train_shortage_button, 1)
            data_row_w = QWidget()
            data_row_w.setLayout(data_row)
            train_page_layout.addWidget(data_row_w)

            self.train_data_advice_status = QLabel("データ状況: 未確認")
            self.train_data_advice_status.setWordWrap(True)
            self.train_data_advice_status.setStyleSheet(
                "color: #333; font-size: 11px; padding: 4px 6px;"
                "background: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 4px;"
            )
            train_page_layout.addWidget(self.train_data_advice_status)

            dedup_preview_button = QPushButton("重複確認")
            dedup_preview_button.setFixedHeight(28)
            dedup_preview_button.setToolTip(
                "train/val 全体でバイト完全一致の重複を検出します（削除はしません）。"
            )
            dedup_preview_button.clicked.connect(self._on_train_dedup_preview_clicked)
            dedup_button = QPushButton("重複削除")
            dedup_button.setFixedHeight(28)
            dedup_button.setToolTip(
                "train/val の完全同一画像を1枚にまとめます。"
                "残す優先: ファイル名とクラス一致 → train → 先に保存した枚。"
            )
            dedup_button.clicked.connect(self._on_train_dedup_clicked)
            self.train_dedup_button = dedup_button
            dataset_review_button = QPushButton("画像レビュー")
            dataset_review_button.setFixedHeight(28)
            dataset_review_button.setToolTip(
                "train/val のシーン画像を1枚ずつ表示し、不要な画像を削除します。"
                " ← → で移動、Del で削除。CNN 読込時は top1 も表示。"
            )
            dataset_review_button.clicked.connect(self._on_train_dataset_review_clicked)
            tool_row = QHBoxLayout()
            tool_row.setSpacing(4)
            tool_row.addWidget(dedup_preview_button, 1)
            tool_row.addWidget(dedup_button, 1)
            tool_row.addWidget(dataset_review_button, 1)
            tool_row_w = QWidget()
            tool_row_w.setLayout(tool_row)
            train_page_layout.addWidget(tool_row_w)

            train_start_button = QPushButton("学習開始")
            train_start_button.setFixedHeight(28)
            train_start_button.clicked.connect(self._on_train_start_clicked)
            self.train_start_button = train_start_button
            train_stop_button = QPushButton("学習停止")
            train_stop_button.setFixedHeight(28)
            train_stop_button.clicked.connect(self._on_train_stop_clicked)
            self.train_stop_button = train_stop_button
            train_save_button = QPushButton("モデル保存")
            train_save_button.setFixedHeight(28)
            train_save_button.clicked.connect(self._on_train_save_clicked)
            self.train_save_button = train_save_button
            train_row = QHBoxLayout()
            train_row.setSpacing(4)
            train_row.addWidget(train_start_button, 1)
            train_row.addWidget(train_stop_button, 1)
            train_row.addWidget(train_save_button, 1)
            train_row_w = QWidget()
            train_row_w.setLayout(train_row)
            train_page_layout.addWidget(train_row_w)

            status_row = QHBoxLayout()
            status_row.setSpacing(6)
            self.train_status_label = QLabel("状態: 待機中")
            status_row.addWidget(self.train_status_label, 1)
            self.train_progress_bar = QProgressBar()
            self.train_progress_bar.setFixedHeight(12)
            self.train_progress_bar.setRange(0, 100)
            self.train_progress_bar.setValue(0)
            self.train_progress_bar.setTextVisible(True)
            self.train_progress_bar.setFormat("")
            self.train_progress_bar.setVisible(False)
            status_row.addWidget(self.train_progress_bar, 2)
            status_row_w = QWidget()
            status_row_w.setLayout(status_row)
            train_page_layout.addWidget(status_row_w)

            use_tsum_row = QHBoxLayout()
            use_tsum_row.setSpacing(4)
            use_tsum_row.addWidget(QLabel("表示名"))
            self.use_tsum_name_input = QLineEdit()
            self.use_tsum_name_input.setPlaceholderText("ナミネ")
            self.use_tsum_name_input.setFixedHeight(26)
            use_tsum_row.addWidget(self.use_tsum_name_input, 1)
            use_tsum_row.addWidget(QLabel("dir"))
            self.use_tsum_dir_input = QLineEdit()
            self.use_tsum_dir_input.setPlaceholderText("namine")
            self.use_tsum_dir_input.setFixedHeight(26)
            use_tsum_row.addWidget(self.use_tsum_dir_input, 1)
            create_use_tsum_button = QPushButton("ツム追加")
            create_use_tsum_button.setFixedHeight(28)
            create_use_tsum_button.setToolTip(
                "使用ツムのモデル用フォルダと app/assets/images/use_tsums/<dir>/ を作成"
            )
            create_use_tsum_button.clicked.connect(self._on_create_use_tsum_clicked)
            use_tsum_row.addWidget(create_use_tsum_button)
            use_tsum_row_w = QWidget()
            use_tsum_row_w.setLayout(use_tsum_row)
            train_page_layout.addWidget(use_tsum_row_w)

            train_models_label = QLabel("個別モデル更新")
            train_models_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 2px;")
            train_page_layout.addWidget(train_models_label)
            self.train_coin_digit_save_button = self._make_train_action_button(
                "coin_digit", self._on_train_save_coin_digit_only_clicked
            )
            train_page_layout.addWidget(self.train_coin_digit_save_button)
            self.train_result_digit_save_button = self._make_train_action_button(
                "result_digit", self._on_train_save_result_digit_only_clicked
            )
            train_page_layout.addWidget(self.train_result_digit_save_button)
            self.train_skill_only_button = self._make_train_action_button(
                "skill", self._on_train_skill_only_clicked
            )
            train_page_layout.addWidget(self.train_skill_only_button)
            train_page_layout.addStretch(1)

            self.center_layout.addWidget(train_page_content, 1)
            self._set_train_ui_state()
            self._refresh_train_dataset_advice()

            self.right_layout.addWidget(QLabel("学習ログ"))
            self.log_view.append("学習ページを表示しました。")
        else:
            self.center_layout.addWidget(QLabel(f"center - button{feature_id}"))
            self.log_view.append(f"right - button{feature_id}")

    def _open_video_file(self) -> None:
        initial_path = self.settings.value("last_video_dir", str(Path.home()), type=str)
        if self.last_video_path:
            last_path = Path(self.last_video_path)
            if last_path.exists():
                initial_path = str(last_path.parent)

        # exec()+selectedFiles() は環境によって空になりやすいので getOpenFileName に統一する。
        filter_str = "動画 (*.mp4 *.mov *.m4v *.avi *.mkv *.webm);;すべて (*.*)"
        path, _ = QFileDialog.getOpenFileName(
            self,
            "動画ファイルを選択",
            initial_path,
            filter_str,
        )
        if not path:
            return
        self._load_video(path)

    def _show_video_open_failed(self, resolved: str, detail: str) -> None:
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(f"動画を開けませんでした: {resolved}")
            if detail.strip():
                self.log_view.append(detail.strip())
        drive_hint = ""
        if _is_streaming_drive_path(resolved):
            drive_hint = (
                "\n\nマイドライブ上のファイルは、開くときにバックグラウンドで順次ダウンロードします。"
                "初回は容量・回線状況により数十秒〜かかることがあります。"
                "「オフラインで使用可能」にするか、C: などローカルへコピーしてから開いてください。"
                "別アプリでファイルを開きっぱなしにしていると失敗することがあります。"
            )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("動画を開けません")
        box.setText(
            "動画の取り込みまたはデコードに失敗しました（順次読み・ffmpeg コピー・OpenCV・FFmpeg・H.264 トランスコードを試行済み）。"
            "ネットワーク・ディスク容量・他アプリのロックを確認してください。"
            + drive_hint
        )
        if detail.strip():
            box.setDetailedText(detail.strip())
            QApplication.clipboard().setText(detail.strip())
            box.setInformativeText("詳細は「詳細を表示」、またはクリップボードにコピー済みです。")
        box.exec()

    def _switch_to_qt_video_and_load(self, resolved: str) -> bool:
        """OpenCV / imageio 経路が失敗したとき、同一ファイルを Qt Multimedia で開き直す。"""
        if _is_streaming_drive_path(resolved):
            return False
        if self.cv_source is not None:
            self.cv_source.release()
        self.cv_play_timer.stop()
        self._cv_playing = False
        self.use_opencv_for_video = False
        vc = self.current_video_container
        if vc is None or not hasattr(self, "player") or not _is_alive_qobject(self.player):
            return False
        vc.set_opencv_display(False)
        if hasattr(self, "mute_checkbox") and _is_alive_qobject(self.mute_checkbox):
            self.mute_checkbox.setEnabled(True)
            self.mute_checkbox.setToolTip("")
        self.player.setVideoOutput(vc.video_widget)
        sink = vc.video_widget.videoSink()
        if sink is None:
            return False
        try:
            sink.videoFrameChanged.disconnect(self._on_video_frame_changed)
        except TypeError:
            pass
        sink.videoFrameChanged.connect(self._on_video_frame_changed)
        self.player.setSource(QUrl.fromLocalFile(resolved))
        self.player.pause()
        self._on_playback_state_changed(self.player.playbackState())
        return True

    def _load_video(self, path: str) -> None:
        resolved = str(Path(path).expanduser().resolve(strict=False))
        self.last_video_path = resolved
        self.settings.setValue("last_video_dir", str(Path(resolved).parent))
        if hasattr(self, "video_file_label") and _is_alive_qobject(self.video_file_label):
            self.video_file_label.setText(Path(resolved).name)
        self.analysis_running = False
        self.analysis_warmup_until_ms = 0
        self.analysis_frame_seq = 0
        self._reset_analysis_flow()
        self.video_analyzer.reset()
        if getattr(self, "use_opencv_for_video", False) and self.cv_source is not None:
            self.cv_play_timer.stop()
            self._cv_playing = False
            prog: Optional[QProgressDialog] = None
            if "マイドライブ" in resolved or "My Drive" in resolved:
                prog = QProgressDialog(
                    "マイドライブから動画を取り込んでいます。初回・大容量では時間がかかることがあります…",
                    None,
                    0,
                    0,
                    self,
                )
                prog.setWindowTitle("読み込み中")
                prog.setWindowModality(Qt.WindowModality.ApplicationModal)
                prog.setMinimumDuration(0)
                prog.show()
                QApplication.processEvents()
            try:
                opened_ok = self.cv_source.open(resolved)
            finally:
                if prog is not None:
                    prog.close()
            if not opened_ok:
                detail = getattr(self.cv_source, "last_open_error", "") or ""
                self._last_file_video_open_error = detail
                if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                    self.log_view.append(f"ファイル動画モードで開けませんでした: {resolved}")
                    if detail.strip():
                        self.log_view.append(detail.strip())
                if self._switch_to_qt_video_and_load(resolved):
                    if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                        self.log_view.append(
                            f"ファイル動画モードで開けなかったため、Qt で再試行しました: {resolved}"
                        )
                    return
                self._show_video_open_failed(resolved, detail)
                return
            self.estimated_fps = self.cv_source.fps()
            self.media_duration_ms = self.cv_source.duration_ms()
            if hasattr(self, "seek_slider"):
                self.seek_slider.setMaximum(max(self.media_duration_ms, 1))
            self._cv_position_ms = 0
            self.cv_source.seek_ms(0)
            img = self.cv_source.read_qimage()
            if img is None or img.isNull():
                self.cv_source.release()
                detail = (
                    getattr(self.cv_source, "last_open_error", "") or "最初のフレームを読み取れませんでした。"
                )
                self._last_file_video_open_error = detail
                if self._switch_to_qt_video_and_load(resolved):
                    if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                        self.log_view.append(
                            f"最初のフレーム取得に失敗したため、Qt で再試行しました: {resolved}"
                        )
                    return
                self._show_video_open_failed(resolved, detail)
                return
            self._video_frame_counter = 0
            self._cv_push_frame(img)
            QApplication.processEvents()
            self._refresh_play_button_text()
            self._update_step_buttons_enabled()
            QTimer.singleShot(0, lambda: self._cv_seek_and_show(self._cv_position_ms))
            return
        self.player.setSource(QUrl.fromLocalFile(resolved))
        self.player.pause()
        self._on_playback_state_changed(self.player.playbackState())

    def _playback_position_ms(self) -> int:
        if getattr(self, "use_opencv_for_video", False):
            return self._cv_position_ms
        return self.player.position()

    def _is_player_playing(self) -> bool:
        if getattr(self, "use_opencv_for_video", False):
            return self._cv_playing
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def _is_player_paused(self) -> bool:
        if getattr(self, "use_opencv_for_video", False):
            return not self._cv_playing
        return self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState

    def _refresh_play_button_text(self) -> None:
        if not hasattr(self, "play_pause_button") or not _is_alive_qobject(self.play_pause_button):
            return
        self.play_pause_button.setText("一時停止" if self._is_player_playing() else "再生")

    def _update_step_buttons_enabled(self) -> None:
        pause_only = self._is_player_paused()
        if hasattr(self, "step_small_left_button"):
            self.step_huge_left_button.setEnabled(pause_only)
            self.step_small_left_button.setEnabled(pause_only)
            self.step_large_left_button.setEnabled(pause_only)
            self.step_small_right_button.setEnabled(pause_only)
            self.step_large_right_button.setEnabled(pause_only)
            self.step_huge_right_button.setEnabled(pause_only)

    def _sync_cv_timer_interval(self) -> None:
        """再生速度は常に動画FPS基準。解析の「間隔」とは切り離す。"""
        fps = max(float(self.estimated_fps), 1.0)
        rate = max(float(getattr(self, "_cv_playback_rate", 1.0)), 0.05)
        self.cv_play_timer.setInterval(max(int(1000.0 / (fps * rate)), 1))

    def _analysis_sample_step(self) -> int:
        step = max(1, self.video_analyzer.sample_every_frames)
        if self._analysis_light_mode():
            step *= 2
        return step

    def _cv_push_frame(self, image: QImage, *, refresh_display: bool = True) -> None:
        if image is None or image.isNull():
            return
        self.current_video_frame_image = image
        if not refresh_display:
            return
        if self.current_video_container is not None:
            w, h = image.width(), image.height()
            if self._video_frame_size != (w, h):
                self._video_frame_size = (w, h)
                self.current_video_container.set_source_size(w, h)
            elif self.pending_crop_rect is not None:
                self._apply_pending_rect_to_video()
            lbl = self.current_video_container.frame_label
            geo = lbl.geometry()
            tw, th = geo.width(), geo.height()
            # レイアウト直後は geometry が 0 のことがあり、この分岐だと一生 pixmap が付かず真っ黒になる。
            if tw <= 0 or th <= 0:
                cr = self.current_video_container.contentsRect()
                tw = max(cr.width(), 1)
                th = max(cr.height(), 1)
            if tw <= 0 or th <= 0:
                tw = max(min(image.width(), 1280), 1)
                th = max(min(image.height(), 720), 1)
            scale_mode = (
                Qt.TransformationMode.FastTransformation
                if self.analysis_running
                else Qt.TransformationMode.SmoothTransformation
            )
            lbl.setPixmap(
                QPixmap.fromImage(image).scaled(
                    tw,
                    th,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    scale_mode,
                )
            )

    def _cv_seek_and_show(self, ms: int) -> None:
        if not getattr(self, "use_opencv_for_video", False) or self.cv_source is None or not self.cv_source.is_open:
            return
        cap = self.media_duration_ms
        self._cv_position_ms = max(0, min(ms, cap) if cap > 0 else max(0, ms))
        self.cv_source.seek_ms(self._cv_position_ms)
        img = self.cv_source.read_qimage()
        if img is not None and not img.isNull():
            self._cv_push_frame(img)
        self._update_playback_indicators(self._cv_position_ms)
        self._refresh_video_tool_digit_preview_if_active(reset_read=True)

    def _cv_timer_tick(self) -> None:
        if not getattr(self, "use_opencv_for_video", False) or not self._cv_playing:
            return
        if self.cv_source is None or not self.cv_source.is_open:
            return
        if self.crop_playback_lock:
            return
        analyze_now = False
        if self.analysis_running:
            self.analysis_frame_seq += 1
            step = self._analysis_sample_step()
            analyze_now = self.analysis_frame_seq % step == 0
            img = self.cv_source.read_next_qimage()
            if img is not None and not img.isNull():
                self._cv_position_ms = self.cv_source.current_position_ms()
        else:
            interval = self.cv_play_timer.interval()
            delta = max(1, int(interval * self._cv_playback_rate))
            next_ms = self._cv_position_ms + delta
            if self.media_duration_ms > 0 and next_ms >= self.media_duration_ms:
                self._cv_reached_end()
                return
            self._cv_position_ms = next_ms
            self.cv_source.seek_ms(self._cv_position_ms)
            img = self.cv_source.read_qimage()
        if img is None or img.isNull():
            self._cv_reached_end()
            return
        self._cv_push_frame(img, refresh_display=True)
        self._on_player_position_changed(self._cv_position_ms)
        if analyze_now:
            self._run_analysis_step(self._cv_position_ms, img)
            QApplication.processEvents()

    def _cv_play(self) -> None:
        if not getattr(self, "use_opencv_for_video", False) or self.cv_source is None or not self.cv_source.is_open:
            return
        if self.crop_playback_lock:
            return
        self._cv_playing = True
        self._sync_cv_timer_interval()
        self.cv_play_timer.start()
        self._refresh_play_button_text()

    def _cv_pause(self) -> None:
        self._cv_playing = False
        self.cv_play_timer.stop()
        self._refresh_play_button_text()

    def _cv_stop(self) -> None:
        self._cv_pause()
        self._cv_seek_and_show(0)

    def _cv_reached_end(self) -> None:
        self._cv_pause()
        if self.media_duration_ms > 0:
            self._cv_position_ms = self.media_duration_ms
        self._update_playback_indicators(self._cv_position_ms)
        self._on_end_of_media_cleanup()

    def _after_analysis_coin_phase(self) -> None:
        if self._coin_confirmed_this_game:
            return
        if (
            self.flow_phase == "WAIT_COIN"
            and coin_gain_confirmed_combined(
                self._hud_coin_reads, self._gain_coin_reads
            )
        ):
            self._finalize_coin_confirmation()
        if (
            hasattr(self, "log_view")
            and _is_alive_qobject(self.log_view)
            and self.flow_phase == "WAIT_COIN"
            and not self._coin_confirmed_this_game
        ):
            self.log_view.append(
                "獲得コイン未確定: "
                f"hud={len(self._hud_coin_reads)} gain={len(self._gain_coin_reads)} "
                f"[{self._coin_gain_last_debug}]"
            )

    def _on_end_of_media_cleanup(self) -> None:
        if self.analysis_running:
            self.analysis_running = False
            self.analysis_warmup_until_ms = 0
            self.analysis_frame_seq = 0
            if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
                self.counter_analysis_state_label.setText("解析状態: 完了")
            self._sync_analysis_control_buttons(running=False)
            if hasattr(self, "log_view"):
                self.log_view.append("解析完了: 動画終端に到達しました。")
            if (
                self.flow_phase in ("WAIT_COIN", "WAIT_BONUS")
                and not self._coin_confirmed_this_game
            ):
                self._after_analysis_coin_phase()
        self._update_step_buttons_enabled()

    def _on_player_error(self, error, error_string: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        msg = (error_string or "").strip() or "動画を読み込めませんでした（コーデックまたは形式の問題の可能性があります）。"
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(f"メディアエラー: {msg}")
        video_path = getattr(self, "last_video_path", "") or ""
        hints = _media_error_extra_hints(msg, video_path)
        file_detail = getattr(self, "_last_file_video_open_error", "") or ""
        details = f"{msg}{hints}"
        if file_detail.strip():
            details += f"\n\n--- ファイル動画モード（先に試行）---\n{file_detail.strip()}"
        details += f"\n{_qt_multimedia_environment_report()}"
        _show_copyable_warning(self, "動画を開けません", msg, details)

    def _on_mute_toggled(self, checked: bool) -> None:
        self.audio_output.setMuted(checked)
        self.settings.setValue("is_muted", checked)

    def _on_player_duration_changed(self, duration_ms: int) -> None:
        self.media_duration_ms = max(duration_ms, 0)
        if hasattr(self, "seek_slider"):
            self.seek_slider.setMaximum(self.media_duration_ms)
        self._update_playback_indicators(self._playback_position_ms())

    def _on_player_position_changed(self, position_ms: int) -> None:
        self._update_playback_indicators(position_ms)
        if self._is_video_tool_coin_digit_mode() and self._is_player_paused():
            self._refresh_video_tool_coin_digit_preview(reset_read=True)
        if hasattr(self, "counter_progress_label") and _is_alive_qobject(self.counter_progress_label):
            if self.analysis_running:
                if position_ms - self._analysis_progress_ui_last_ms < 500:
                    return
                self._analysis_progress_ui_last_ms = position_ms
            frame_index = int((max(position_ms, 0) / 1000.0) * self.estimated_fps)
            self.counter_progress_label.setText(f"進行: {position_ms/1000.0:.2f}s / f{frame_index}")

    def _on_slider_moved(self, position_ms: int) -> None:
        if getattr(self, "use_opencv_for_video", False):
            self._cv_seek_and_show(position_ms)
            return
        self.player.setPosition(position_ms)

    def _on_play_pause_clicked(self) -> None:
        crop_checked = False
        if hasattr(self, "crop_start_button") and _is_alive_qobject(self.crop_start_button):
            crop_checked = self.crop_start_button.isChecked()
        if self.crop_playback_lock or crop_checked:
            # Never toggle playback while cropping.
            return
        if self._is_player_playing():
            if getattr(self, "use_opencv_for_video", False):
                self._cv_pause()
            else:
                self.player.pause()
        else:
            if getattr(self, "use_opencv_for_video", False):
                self._cv_play()
            else:
                self.player.play()

    def _on_stop_clicked(self) -> None:
        if getattr(self, "use_opencv_for_video", False):
            self._cv_stop()
        else:
            self.player.stop()
        self.analysis_running = False
        self.analysis_warmup_until_ms = 0
        self.analysis_frame_seq = 0
        self._reset_analysis_flow()
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
            self.counter_analysis_state_label.setText("解析状態: 停止")
        self._sync_analysis_control_buttons(running=False)

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if self.crop_playback_lock:
            if getattr(self, "use_opencv_for_video", False):
                if self._cv_playing:
                    self._cv_pause()
            elif state == QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
            self._refresh_play_button_text()
            return
        if not hasattr(self, "play_pause_button") or not _is_alive_qobject(self.play_pause_button):
            return
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self.play_pause_button.setText("一時停止")
        else:
            self.play_pause_button.setText("再生")
            if state == QMediaPlayer.PlaybackState.StoppedState:
                self.analysis_running = False
                self.analysis_warmup_until_ms = 0
                self.analysis_frame_seq = 0
                if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
                    self.counter_analysis_state_label.setText("解析状態: 停止")
                self._sync_analysis_control_buttons(running=False)

    def _on_media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if status != QMediaPlayer.MediaStatus.EndOfMedia:
            return
        self._on_end_of_media_cleanup()

    def _step_forward(self, frames: int) -> None:
        if not self._is_player_paused():
            return
        step_ms = int((frames / self.estimated_fps) * 1000)
        pos = self._playback_position_ms() + step_ms
        if self.media_duration_ms > 0:
            pos = min(pos, self.media_duration_ms)
        target = max(pos, 0)
        if getattr(self, "use_opencv_for_video", False):
            self._cv_seek_and_show(target)
        else:
            self.player.setPosition(target)
            if self._is_video_tool_coin_digit_mode() or self._is_video_tool_result_digit_mode():
                QTimer.singleShot(
                    0, lambda: self._refresh_video_tool_digit_preview_if_active(reset_read=True)
                )

    def _step_backward(self, frames: int) -> None:
        if not self._is_player_paused():
            return
        step_ms = int((frames / self.estimated_fps) * 1000)
        target = max(self._playback_position_ms() - step_ms, 0)
        if getattr(self, "use_opencv_for_video", False):
            self._cv_seek_and_show(target)
        else:
            self.player.setPosition(target)
            if self._is_video_tool_coin_digit_mode() or self._is_video_tool_result_digit_mode():
                QTimer.singleShot(
                    0, lambda: self._refresh_video_tool_digit_preview_if_active(reset_read=True)
                )

    def _read_frame_at_ms(self, ms: int) -> Optional[QImage]:
        if getattr(self, "use_opencv_for_video", False):
            if self.cv_source is None or not self.cv_source.is_open:
                return None
            cap = self.media_duration_ms
            pos = max(0, min(ms, cap) if cap > 0 else max(0, ms))
            self.cv_source.seek_ms(pos)
            img = self.cv_source.read_qimage()
            if img is not None and not img.isNull():
                return img
            return None
        if self._is_player_playing():
            self.player.pause()
            self._refresh_play_button_text()
        self.player.setPosition(max(0, ms))
        for _ in range(10):
            QApplication.processEvents()
        img = self.current_video_frame_image
        if img is not None and not img.isNull():
            return img
        return None

    def _append_timer_skip_log(self, message: str) -> None:
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(message)

    def _run_timer_skip(
        self,
        *,
        require_paused: bool = True,
        show_progress: bool = True,
        log_prefix: str = "",
    ) -> bool:
        """残り時間を読み取り、動画内の同じ表示秒へジャンプする。"""
        prefix = f"{log_prefix}: " if log_prefix else ""
        if require_paused and self._is_player_playing():
            self._append_timer_skip_log(f"{prefix}スキップ: 一時停止してから実行してください。")
            return False
        if not opencv_available():
            self._append_timer_skip_log(f"{prefix}スキップ: opencv-python が必要です。")
            return False
        if self.media_duration_ms <= 0 and not getattr(self, "use_opencv_for_video", False):
            self._append_timer_skip_log(f"{prefix}スキップ: 動画を開いてください。")
            return False
        if getattr(self, "use_opencv_for_video", False):
            if self.cv_source is None or not self.cv_source.is_open:
                self._append_timer_skip_log(f"{prefix}スキップ: 動画を開いてください。")
                return False

        frame = self.current_video_frame_image
        if frame is None or frame.isNull():
            self._append_timer_skip_log(f"{prefix}スキップ: 表示中のフレームがありません。")
            return False

        positions = self.crop_positions_for_analysis or self._load_crop_positions()
        crop_key = _REQUIRED_TIMER_SKIP_CROP_KEY
        if not self._crop_rect_defined(positions, crop_key):
            self._append_timer_skip_log(
                f"{prefix}スキップ: トリミング「残り時間」の保存が必須です。"
                " 動画ツールで IN_GAME の秒数表示（最大3桁）に合わせて保存してください。"
                "（item の time ではありません）"
            )
            return False
        roi = self._crop_frame_roi_from_positions(frame, positions, crop_key)
        if roi is None or roi.isNull():
            self._append_timer_skip_log(
                f"{prefix}スキップ: トリミング「残り時間」から画像を切り出せませんでした。"
                " 範囲を保存し直してください。"
            )
            return False

        target_sec, dbg = read_remaining_seconds(roi)
        if target_sec is None:
            self._append_timer_skip_log(f"{prefix}スキップ: 残り時間を読み取れませんでした ({dbg})")
            return False
        if "n=1" in dbg and target_sec <= 9:
            self._append_timer_skip_log(
                f"{prefix}スキップ: 残り{target_sec}秒は信頼度が低いため中止しました（{dbg}）。"
                " 動画ツールで「残り時間」を秒数（最大3桁）がすべて入るよう合わせ直してください。"
            )
            return False

        start_ms = self._playback_position_ms()
        fps = max(float(self.estimated_fps or 30.0), 1.0)
        step_ms = max(200, int(1000.0 / fps))
        found_ms: Optional[int] = None

        prog = None
        if show_progress:
            prog = QProgressDialog("残り時間のフレームを検索中…", "キャンセル", 0, 0, self)
            prog.setWindowTitle("時間スキップ")
            prog.setWindowModality(Qt.WindowModality.WindowModal)
            prog.setMinimumDuration(0)
            prog.setValue(0)

        ms = start_ms
        while ms >= 0:
            if prog is not None and prog.wasCanceled():
                break
            if prog is not None:
                prog.setLabelText(f"検索中… {ms/1000:.1f}s / 残り{target_sec}秒")
            QApplication.processEvents()
            img = self._read_frame_at_ms(ms)
            if img is not None and not img.isNull():
                probe = self._crop_frame_roi_from_positions(img, positions, crop_key)
                if probe is not None:
                    sec, _ = read_remaining_seconds(probe)
                    if sec == target_sec:
                        found_ms = ms
                        break
            ms -= step_ms
            if prog is not None:
                prog.setValue(prog.value() + 1)

        if prog is not None:
            prog.close()

        if found_ms is None:
            self._append_timer_skip_log(
                f"{prefix}スキップ: 残り{target_sec}秒のフレームが見つかりませんでした（{dbg}）。"
                " トリミング位置を確認してください。"
            )
            if getattr(self, "use_opencv_for_video", False):
                self._cv_seek_and_show(start_ms)
            else:
                self.player.setPosition(start_ms)
            return False

        if getattr(self, "use_opencv_for_video", False):
            self._cv_seek_and_show(found_ms)
        else:
            self.player.setPosition(found_ms)
            for _ in range(6):
                QApplication.processEvents()

        self._append_timer_skip_log(
            f"{prefix}スキップ: 残り{target_sec}秒のフレームへジャンプ "
            f"({found_ms/1000:.2f}s, {crop_key}, {dbg})"
        )
        return True

    def _on_temp1_timer_skip_clicked(self) -> None:
        self._run_timer_skip()

    def _clear_time_efficiency_go_skip_state(self) -> None:
        self._time_efficiency_go_skip_pending = False
        self._time_efficiency_go_skip_done = False
        self._time_efficiency_go_skip_anchor_ms = 0
        timer = getattr(self, "_time_efficiency_skip_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()

    def _arm_time_efficiency_go_skip_timer(self, position_ms: int) -> None:
        self._clear_time_efficiency_go_skip_state()
        self._time_efficiency_go_skip_anchor_ms = max(0, int(position_ms))
        self._time_efficiency_go_skip_pending = True
        target_ms = self._time_efficiency_go_skip_target_ms()
        self._time_efficiency_skip_timer.start(_TIME_EFFICIENCY_GO_SKIP_DELAY_MS)
        self._append_timer_skip_log(
            f"時間効率: go 確定位置 {self._time_efficiency_go_skip_anchor_ms}ms →"
            f" 動画 {target_ms}ms へジャンプ予定（+{_TIME_EFFICIENCY_GO_SKIP_DELAY_MS // 1000}秒）"
        )

    def _time_efficiency_go_skip_target_ms(self) -> int:
        anchor = max(0, int(getattr(self, "_time_efficiency_go_skip_anchor_ms", 0)))
        target_ms = anchor + _TIME_EFFICIENCY_GO_SKIP_DELAY_MS
        if self.media_duration_ms > 0:
            target_ms = min(target_ms, max(0, self.media_duration_ms - 500))
        return max(0, target_ms)

    def _perform_time_efficiency_go_skip(self) -> None:
        if not self.analysis_running:
            return
        if self._analysis_profile != _TIME_EFFICIENCY_BUTTON_ACTION:
            return
        if self.flow_phase != "IN_GAME":
            self._append_timer_skip_log(
                f"時間効率: ジャンプ中止（フェーズ={self.flow_phase}）"
            )
            return
        target_ms = self._time_efficiency_go_skip_target_ms()
        current_ms = self._playback_position_ms()
        if current_ms >= target_ms - 800:
            self._append_timer_skip_log(
                f"時間効率: ジャンプ不要（現在 {current_ms}ms は目標 {target_ms}ms 付近）"
            )
            return
        self._append_timer_skip_log(
            f"時間効率: go+{_TIME_EFFICIENCY_GO_SKIP_DELAY_MS // 1000}秒へジャンプ"
            f" ({current_ms/1000:.2f}s → {target_ms/1000:.2f}s)"
        )
        if getattr(self, "use_opencv_for_video", False):
            self._cv_seek_and_show(target_ms)
        else:
            self.player.setPosition(target_ms)
            for _ in range(8):
                QApplication.processEvents()
        self.analysis_warmup_until_ms = target_ms + 500

    def _maybe_time_efficiency_go_skip(self, position_ms: int) -> None:
        if not getattr(self, "_time_efficiency_go_skip_pending", False):
            return
        if getattr(self, "_time_efficiency_go_skip_done", False):
            return
        if not self.analysis_running:
            return
        if self._analysis_profile != _TIME_EFFICIENCY_BUTTON_ACTION:
            return
        if self.flow_phase != "IN_GAME":
            return
        if position_ms < self._time_efficiency_go_skip_target_ms():
            return
        self._time_efficiency_go_skip_done = True
        self._time_efficiency_go_skip_pending = False
        timer = getattr(self, "_time_efficiency_skip_timer", None)
        if timer is not None and timer.isActive():
            timer.stop()
        self._perform_time_efficiency_go_skip()

    def _on_time_efficiency_go_skip_timeout(self) -> None:
        if getattr(self, "_time_efficiency_go_skip_done", False):
            return
        if not getattr(self, "_time_efficiency_go_skip_pending", False):
            return
        self._time_efficiency_go_skip_done = True
        self._time_efficiency_go_skip_pending = False
        self._perform_time_efficiency_go_skip()

    def _set_playback_rate(self, rate: float) -> None:
        self._cv_playback_rate = max(rate, 0.05)
        if not getattr(self, "use_opencv_for_video", False):
            self.player.setPlaybackRate(rate)
        elif self._cv_playing:
            self._sync_cv_timer_interval()

    def _register_analysis_toggle_button(self, button: QPushButton, action: str) -> None:
        self._analysis_toggle_buttons.append((button, action))
        self._sync_analysis_control_buttons(running=self.analysis_running)

    def _analysis_fever_enabled(self) -> bool:
        if self._analysis_profile == _TIME_EFFICIENCY_BUTTON_ACTION:
            return False
        return _ANALYSIS_FEVER_ENABLED

    def _analysis_bonus_enabled(self) -> bool:
        return self._analysis_profile != _TIME_EFFICIENCY_BUTTON_ACTION

    def _sync_analysis_control_buttons(self, *, running: bool) -> None:
        for btn, action in list(getattr(self, "_analysis_toggle_buttons", [])):
            if not _is_alive_qobject(btn):
                continue
            if action == _ANALYSIS_BUTTON_ACTION:
                btn.setText("解析停止" if running else "解析開始")
            elif action == _TIME_EFFICIENCY_BUTTON_ACTION:
                btn.setText("停止" if running else "時間効率")

    def _on_analyze_clicked(self, profile: str = _ANALYSIS_BUTTON_ACTION) -> None:
        if self.analysis_running:
            self.analysis_running = False
            self._clear_time_efficiency_go_skip_state()
            self._analysis_profile = _ANALYSIS_BUTTON_ACTION
            self.analysis_warmup_until_ms = 0
            self.analysis_frame_seq = 0
            self._reset_analysis_flow()
            if getattr(self, "use_opencv_for_video", False):
                self._cv_pause()
            else:
                self.player.pause()
            if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
                self.counter_analysis_state_label.setText("解析状態: 停止")
            self._sync_analysis_control_buttons(running=False)
            if hasattr(self, "log_view"):
                self.log_view.append("解析停止。")
            return
        if not self.last_video_path:
            if hasattr(self, "log_view"):
                self.log_view.append("動画未選択: 先に動画を開いてください。")
            return
        self.crop_positions_for_analysis = self._load_crop_positions()
        self.use_tsum_classifier.reload()
        self.skill_classifier_pool.reload()
        model_version = self._selected_analysis_model_version()
        self.video_analyzer.reload_model(version=model_version)
        if self._trainer_scene_dirty and self.trainer.scene_cnn.is_loaded():
            self.video_analyzer.reload_model(version=model_version)
            self._trainer_scene_dirty = False
        elif self._trainer_scene_dirty and self.trainer.scene_model.centroids:
            self.video_analyzer.apply_scene_centroids(
                self.trainer.scene_model.centroids,
                self.trainer.scene_model.fever_calib,
            )
        self.video_analyzer.reset()
        self._analysis_profile = profile
        images_root = self.project_root / "app/assets/images"
        if (
            _PREGAME_READY_SUPPRESSION_ENABLED
            and self.video_analyzer.scene_classifier.model is not None
        ):
            n_none = self.video_analyzer.scene_classifier.model.rebuild_none_veto_exemplars(
                images_root
            )
            if n_none and hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(
                    f"none誤検知参照: train/none から {n_none} 枚読込（ready 抑制に使用）"
                )
        self.analysis_running = True
        self._hide_trim_overlay_on_video()
        if self.current_video_container is not None:
            self.current_video_container.set_crop_enabled(False)
        if hasattr(self, "crop_start_button") and _is_alive_qobject(self.crop_start_button):
            self.crop_start_button.blockSignals(True)
            self.crop_start_button.setChecked(False)
            self.crop_start_button.blockSignals(False)
        self.analysis_frame_seq = 0
        self._analysis_log_scroll_counter = 0
        self._analysis_confirm_edge_prev = ""
        self._analysis_confirm_edge_key = ""
        self._analysis_last_logged_scene = ""
        self._analysis_progress_ui_last_ms = 0
        self._reset_analysis_flow()
        self._last_fever_blocked_log = None
        self._ingame_entered_logged = False
        self._log_analysis_model_diagnostics()
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
            self.counter_analysis_state_label.setText("解析状態: 実行中")
        self._sync_analysis_control_buttons(running=True)
        if hasattr(self, "start_from_zero_check") and self.start_from_zero_check.isChecked():
            if getattr(self, "use_opencv_for_video", False):
                self._cv_seek_and_show(0)
            else:
                self.player.setPosition(0)
        # Ignore unstable first frames right after start/seek.
        self.analysis_warmup_until_ms = max(self._playback_position_ms(), 0) + 500
        self._cv_display_tick = 0
        if getattr(self, "use_opencv_for_video", False):
            self._sync_cv_timer_interval()
            self._cv_play()
        else:
            self.player.play()
        if hasattr(self, "log_view"):
            self.log_view.clear()
            if hasattr(self, "counter_use_item_label"):
                self.counter_use_item_label.setText("使用アイテム: --")
            light = self._analysis_light_mode()
            self.log_view.append(
                f"解析開始: {self.video_analyzer.sample_every_frames}フレームごとに判定します。"
                f" 軽量={'ON' if light else 'OFF'} model={self.video_analyzer.active_model_version}"
            )
            self.log_view.append(
                "1game: item→result / ready・go・timeup・coin・result は各1回"
            )
            if self._analysis_profile == _TIME_EFFICIENCY_BUTTON_ACTION:
                self.log_view.append("時間効率モード: fever/bonus 検知 OFF（timeup 優先）")
                self.log_view.append(
                    f"時間効率: go 確定位置から動画 +{_TIME_EFFICIENCY_GO_SKIP_DELAY_MS // 1000}秒へ"
                    " 自動ジャンプ（解析継続）"
                )
            elif not self._analysis_fever_enabled():
                self.log_view.append(
                    "fever検知: OFF（timeup 精度改善作業中。復帰は _ANALYSIS_FEVER_ENABLED）"
                )
            self.log_view.append(
                f"scene_model: {self.video_analyzer.scene_model_path}"
            )
            meta = self.video_analyzer.active_model_meta()
            if meta.get("saved_at"):
                self.log_view.append(
                    f"モデル保存日時: {meta['saved_at']} "
                    f"(train={meta.get('train_total', '?')} val={meta.get('val_total', '?')})"
                )
            if meta.get("fever_val_total"):
                self.log_view.append(
                    "fever(val): "
                    f"strong={meta.get('fever_val_strong', '?')}/"
                    f"{meta.get('fever_val_total', '?')} "
                    f"any={meta.get('fever_val_any', '?')}/"
                    f"{meta.get('fever_val_total', '?')} "
                    f"誤(none/go) strong={meta.get('fever_false_none_go_strong', '?')} "
                    f"weak={meta.get('fever_false_none_go_weak', '?')}"
                )
            if self._trainer_scene_dirty and self.trainer.scene_model.centroids:
                self.log_view.append("※ 未保存の学習結果（メモリ）を解析に使用中。保存後は日時が更新されます。")
            self.log_view.append(
                "モデル状態: "
                f"loaded={self.video_analyzer.scene_model_loaded} "
                f"class_count={self.video_analyzer.scene_class_count}"
            )
            backend = getattr(self.video_analyzer, "scene_backend", "none")
            self.log_view.append(f"シーン分類: {backend}（cnn=深層学習 / centroid=従来）")
            if self.video_analyzer.scene_classifier.uses_cnn():
                self.log_view.append(
                    "クラス: " + ", ".join(self.video_analyzer.scene_classifier.cnn.classes)
                )
            elif self.video_analyzer.scene_model_loaded:
                labels = sorted(self.video_analyzer.scene_classifier.model.centroids.keys())
                self.log_view.append(
                    "シーン centroid のクラス（学習で train に画像があったもののみ）: "
                    + ", ".join(labels)
                )
            else:
                p = getattr(self.video_analyzer, "scene_model_path", None)
                if p is not None:
                    reason = SceneCentroidModel.describe_load_failure(p)
                    self.log_view.append(f"scene_model 読込失敗: {p} （{reason}）")
            self.log_view.append(f"トリミング読込: {list(self.crop_positions_for_analysis.keys())}")
            skill_dirs = self.skill_classifier_pool.known_dirs()
            if skill_dirs:
                self.log_view.append(f"スキルモデル読込: {', '.join(skill_dirs)}")
            else:
                self.log_view.append(
                    "スキルモデル: 未読込（skills/<dir名>/activation/ に発動時画像→学習タブで保存）"
                )
            if not self.video_analyzer.scene_model_loaded:
                ver = self.video_analyzer.active_model_version
                self.log_view.append(
                    f"警告: モデル {ver} が未読込です。"
                    "解析タブのモデル版を version_1 にするか、学習タブで学習→モデル保存してください。"
                )

    def _scene_model_runtime_fingerprint(self) -> str:
        """解析に実際に載っている centroid / fever 閾値の短い指紋。"""
        model = self.video_analyzer.scene_classifier.model
        if not model.centroids:
            return "empty"
        fever = model.centroids.get("fever", [])
        sample = [round(v, 5) for v in fever[:12]] if fever else []
        calib = model.fever_calib or {}
        blob = json.dumps({"f": sample, "c": calib}, sort_keys=True)
        return hashlib.md5(blob.encode()).hexdigest()[:10]

    def _log_analysis_model_diagnostics(self) -> None:
        if not hasattr(self, "log_view") or not _is_alive_qobject(self.log_view):
            return
        path = self.video_analyzer.scene_model_path
        self.log_view.append(f"モデル指紋(実行中): {self._scene_model_runtime_fingerprint()}")
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            self.log_view.append(f"モデルファイル更新: {mtime}")
        meta = self.video_analyzer.active_model_meta()
        if self._trainer_scene_dirty and self.trainer.last_fever_metrics:
            fm = self.trainer.last_fever_metrics
            self.log_view.append(
                "fever(学習直後・メモリ): "
                f"strong={fm.get('fever_val_strong', '?')}/{fm.get('fever_val_total', '?')} "
                f"any={fm.get('fever_val_any', '?')}/{fm.get('fever_val_total', '?')} "
                f"誤(none/go) strong={fm.get('false_none_go_strong', '?')} "
                f"weak={fm.get('false_none_go_weak', '?')}"
            )
        elif meta.get("fever_val_total"):
            self.log_view.append(
                "fever(保存済み): "
                f"strong={meta.get('fever_val_strong', '?')}/{meta.get('fever_val_total', '?')} "
                f"any={meta.get('fever_val_any', '?')}/{meta.get('fever_val_total', '?')} "
                f"誤(none/go) strong={meta.get('fever_false_none_go_strong', '?')} "
                f"weak={meta.get('fever_false_none_go_weak', '?')}"
            )
        calib = self.video_analyzer.scene_classifier.model.fever_calib
        self.log_view.append(
            "fever閾値: "
            f"top1_lead={calib.get('top1_lead')} "
            f"none_go_gap={calib.get('none_go_gap')} "
            f"weak_gap={calib.get('weak_none_go_gap')}"
        )
        n_veto = len(self.video_analyzer.scene_classifier.model.none_veto_exemplars)
        self.log_view.append(
            f"none誤検知抑制: train/none から新しい順 {n_veto} 枚を参照（保存した画面に近いと fever しません）"
        )

    def _reset_fever_latch(self) -> None:
        self._fever_episode_latched = False
        self._fever_episode_hold = 0
        self._fever_confirm_streak = 0
        self._fever_clear_streak = 0

    def _scene_for_confirm_dialog(
        self,
        scene_label: str,
        raw_scene: str,
        *,
        ranked: list,
        use_scene_cnn: bool,
        fever_hit: bool,
        go_hit: bool,
        ready_hit: bool,
        flow_phase: str,
        ready_confirmed: bool = False,
        ready_locked: bool = False,
        go_locked: bool = False,
    ) -> str:
        """フロー上 none でも、go/ready 候補なら確認ダイアログ用ラベルにする。"""
        if scene_label in _ANALYSIS_SCENE_CONFIRM_LABELS:
            if scene_label == "ready" and ready_locked:
                return "none"
            if scene_label == "go" and go_locked:
                return "none"
            if scene_label == "timeup" and self._timeup_confirmed_this_game:
                return "none"
            if scene_label == "result" and self._result_confirmed_this_game:
                return "none"
            if scene_label == "coin" and self._coin_confirmed_this_game:
                return "none"
            if scene_label == "fever" and not self._analysis_fever_enabled():
                return "none"
            if scene_label == "bonus" and not self._analysis_bonus_enabled():
                return "none"
            return scene_label
        # 生ラベルはフェーズに応じて制限（WAIT_ITEM中の bonus/raw などを防ぐ）
        if raw_scene == "item" and flow_phase == "WAIT_ITEM":
            if self._item_confirmed_this_game or self._item_scan_suppressed:
                return "none"
            return raw_scene
        if not ready_locked and self._can_detect_scene_once_per_game("ready", flow_phase):
            if raw_scene == "ready" and flow_phase in ("WAIT_READY", "WAIT_GO"):
                return raw_scene
            if flow_phase == "WAIT_READY" and ready_hit:
                return "ready"
        if not go_locked and self._can_detect_scene_once_per_game("go", flow_phase):
            if raw_scene == "go" and flow_phase == "WAIT_GO" and ready_confirmed:
                return raw_scene
            if flow_phase == "WAIT_GO" and ready_confirmed and go_hit:
                return "go"
        if raw_scene == "fever" and flow_phase == "IN_GAME" and self._analysis_fever_enabled():
            return raw_scene
        if self._can_detect_scene_once_per_game("timeup", flow_phase) and (
            raw_scene == "timeup"
            or (use_scene_cnn and self._timeup_cnn_detected(ranked))
        ):
            return "timeup"
        if (
            flow_phase == "WAIT_BONUS"
            and self._analysis_bonus_enabled()
            and (
                raw_scene == "bonus"
                or (use_scene_cnn and self._bonus_cnn_detected(ranked))
            )
        ):
            return "bonus"
        if self._can_detect_scene_once_per_game("coin", flow_phase) and (
            raw_scene == "coin"
            or (use_scene_cnn and self._coin_cnn_detected(ranked))
        ):
            return "coin"
        if self._can_detect_scene_once_per_game("result", flow_phase) and raw_scene == "result":
            return raw_scene
        if (
            flow_phase == "IN_GAME"
            and self._analysis_fever_enabled()
            and (fever_hit or raw_scene == "fever")
        ):
            return "fever"
        return scene_label

    def _update_cnn_fever_latch(
        self,
        *,
        fever_hit: bool,
        fever_strong_hit: bool,
        fever_top1: bool,
        ranked: list,
        gates_ok: bool,
    ) -> bool:
        """CNN: 2回で fever 確定し、取りこぼしサンプルがあってもしばらく fever を維持する。"""
        if not gates_ok:
            return False

        signal = fever_hit or fever_strong_hit
        if signal:
            self._fever_confirm_streak = min(self._fever_confirm_streak + 1, 30)
            self._fever_clear_streak = 0
        else:
            self._fever_confirm_streak = 0

        if not self._fever_episode_latched:
            need = self._fever_cnn_confirm_need(
                ranked, fever_strong_hit=fever_strong_hit, fever_top1=fever_top1
            )
            if self._fever_confirm_streak >= need:
                self._fever_episode_latched = True
                self._fever_episode_hold = _FEVER_LATCH_HOLD_SAMPLES
        else:
            if signal:
                self._fever_episode_hold = _FEVER_LATCH_HOLD_SAMPLES
                self._fever_clear_streak = 0
            else:
                self._fever_clear_streak += 1
                if self._fever_episode_hold > 0:
                    self._fever_episode_hold -= 1
                if (
                    self._fever_clear_streak >= _FEVER_LATCH_RELEASE_STREAK
                    and self._fever_episode_hold <= 0
                ):
                    self._reset_fever_latch()

        return self._fever_episode_latched

    def _enter_ingame_log(self, reason: str) -> None:
        self._last_raw_scene_in_game = ""
        self._fever_raw_streak = 0
        self._fever_none_gap_used = 0
        self._reset_fever_latch()
        use_cnn = self.video_analyzer.scene_classifier.uses_cnn()
        self._in_game_fever_warmup = (
            0
            if not self._analysis_fever_enabled()
            else (
                _FEVER_IN_GAME_WARMUP_SAMPLES_CNN
                if use_cnn
                else _FEVER_IN_GAME_WARMUP_SAMPLES
            )
        )
        self._ingame_entered_logged = True
        self._ingame_go_grace = _INGAME_GO_GRACE_SAMPLES
        self._restore_analysis_sampling()
        if use_cnn:
            self._boost_analysis_sampling_for_ingame()

    def _restore_analysis_sampling(self) -> None:
        if self._ingame_sample_saved is not None:
            self.video_analyzer.sample_every_frames = self._ingame_sample_saved
            self._ingame_sample_saved = None

    def _boost_analysis_sampling_for_ingame(self) -> None:
        if self._ingame_sample_saved is not None:
            return
        self._ingame_sample_saved = self.video_analyzer.sample_every_frames
        self.video_analyzer.sample_every_frames = max(
            1, min(_INGAME_SAMPLE_EVERY_MAX, self._ingame_sample_saved)
        )

    def _scene_likely_class(
        self,
        raw_scene: str,
        ranked: list,
        *,
        use_cnn: bool,
        target: str,
        cnn_max_score: float = _CNN_GO_SCORE_MAX,
        top_n: int = 4,
    ) -> bool:
        if raw_scene == target:
            return True
        if not ranked:
            return False
        for cls, score in ranked[:top_n]:
            if cls != target:
                continue
            if use_cnn:
                if score < cnn_max_score:
                    return True
            else:
                top_cls, top_score = ranked[0]
                if cls == top_cls:
                    return True
                if score <= top_score * 1.08:
                    return True
        return False

    @staticmethod
    def _pregame_scene_strong(raw_scene: str, ranked: list, target: str) -> bool:
        """ready/go フェーズ遷移: CNN の raw または top1 のみ（weak・margin 不可）。"""
        if raw_scene == target:
            return True
        return bool(ranked) and ranked[0][0] == target

    @staticmethod
    def _pregame_cnn_top1_detected(ranked: list, target: str, max_score: float) -> bool:
        return bool(ranked) and ranked[0][0] == target and ranked[0][1] < max_score

    def _pregame_cnn_hit_strength(
        self, raw_scene: str, ranked: list, *, target: str
    ) -> str:
        """CNN pregame: strong=top1/raw, weak=top-k+margin。"""
        detect_max = _CNN_READY_DETECT_MAX if target == "ready" else _CNN_GO_DETECT_MAX
        margin_max = _CNN_READY_MARGIN_MAX if target == "ready" else _CNN_GO_MARGIN_MAX
        if raw_scene == target:
            return "strong"
        if self._pregame_cnn_top1_detected(ranked, target, detect_max):
            return "strong"
        if not ranked:
            return "none"
        top_score = ranked[0][1]
        for cls, score in ranked[:6]:
            if cls != target:
                continue
            if score >= _CNN_PREGAME_WEAK_MAX:
                continue
            if ranked[0][0] == target:
                return "weak"
            if score - top_score <= margin_max:
                return "weak"
        return "none"

    def _pregame_class_hit(
        self,
        raw_scene: str,
        ranked: list,
        *,
        use_cnn: bool,
        target: str,
        margin_max: float = _CNN_PREGAME_MARGIN_MAX,
    ) -> bool:
        """WAIT_READY / WAIT_GO 用: ready・go だけ拾う。"""
        if use_cnn:
            return self._pregame_cnn_hit_strength(raw_scene, ranked, target=target) != "none"
        if raw_scene == target:
            return True
        if not ranked:
            return False
        if ranked[0][0] == target:
            return True
        return self._scene_likely_class(
            raw_scene, ranked, use_cnn=False, target=target, cnn_max_score=_CNN_PREGAME_SCORE_MAX
        )

    def _pregame_hit_strength(
        self,
        raw_scene: str,
        ranked: list,
        *,
        use_cnn: bool,
        target: str,
        margin_max: float = _CNN_PREGAME_MARGIN_MAX,
        frame_image=None,
        scene_feature=None,
        frame_index: int = -1,
    ) -> str:
        if use_cnn:
            strength = self._pregame_cnn_hit_strength(raw_scene, ranked, target=target)
            if (
                target == "ready"
                and strength != "none"
                and (
                    (frame_index >= 0 and self._ready_frame_vetoed(frame_index))
                    or self._ready_blocked_by_session_veto(
                        frame_image, scene_feature, ranked=ranked
                    )
                )
            ):
                return "none"
            return strength
        if raw_scene == target:
            return "strong"
        if ranked and ranked[0][0] == target:
            return "strong"
        if self._pregame_class_hit(
            raw_scene, ranked, use_cnn=False, target=target, margin_max=margin_max
        ):
            return "weak"
        return "none"

    def _bump_pregame_streak(self, hit: bool, *, streak_attr: str, gap_attr: str) -> int:
        streak = int(getattr(self, streak_attr, 0))
        gap = int(getattr(self, gap_attr, 0))
        if hit:
            streak = min(streak + 1, 30)
            gap = 0
        else:
            gap += 1
            if gap > _PREGAME_HIT_NONE_GAP:
                streak = 0
                gap = 0
        setattr(self, streak_attr, streak)
        setattr(self, gap_attr, gap)
        return streak

    def _analysis_scene_confirm_enabled(self, scene_key: str) -> bool:
        if scene_key == "fever" and not self._analysis_fever_enabled():
            return False
        if scene_key == "bonus" and not self._analysis_bonus_enabled():
            return False
        checks = getattr(self, "analysis_scene_confirm_checks", None)
        if not isinstance(checks, dict):
            return False
        cb = checks.get(scene_key)
        return cb is not None and _is_alive_qobject(cb) and cb.isChecked()

    def _refresh_flow_phase_label(self) -> None:
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(
            self.counter_analysis_state_label
        ):
            self.counter_analysis_state_label.setText(
                f"解析状態: G{self.flow_game_index} {self.flow_phase}"
            )

    def _confirm_item_detection(self) -> None:
        if self._item_confirmed_this_game:
            return
        self._item_confirmed_this_game = True
        self.locked_item_targets = _item_select_keys_from(list(self._item_scan_targets))
        if self._item_use_tsum_pending not in {"-", "unknown", ""}:
            self.locked_use_tsum = self._item_use_tsum_pending
        self.locked_item_fixed = True
        self._item_scene_active = False
        self._refresh_item_counter_labels()
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(f"G{self.flow_game_index} item 確定")

    def _reject_item_detection(self, frame_image=None, frame_index: int = -1) -> None:
        self._item_confirmed_this_game = False
        self._item_scan_suppressed = True
        self.locked_item_fixed = False
        self.locked_item_targets = []
        self.locked_use_tsum = "-"
        self._item_scan_targets = set()
        self._item_scene_active = False
        self._item_use_tsum_pending = "-"
        if self.flow_phase != "WAIT_ITEM":
            self.flow_phase = "WAIT_ITEM"
            self._game_active = False
            self._reset_per_game_scene_flags()
        self._clear_analysis_confirm_edge_prefix("item")
        self._refresh_item_counter_labels()
        self._refresh_flow_phase_label()

    def _confirm_pregame_ready(self, position_ms: int = 0, frame_index: int = -1) -> None:
        if self._ready_confirmed_this_game:
            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(
                    f"ready: 既に確定済みです（解析状態: G{self.flow_game_index} {self.flow_phase}）"
                )
            return
        if not self._game_active:
            self._begin_new_game_from_item()
        self._log_pregame_scene_detect("ready", position_ms, frame_index, " 確定")
        self.flow_phase = "WAIT_GO"
        self._ready_confirmed_this_game = True
        self._ready_suppressed_this_game = False
        self._pregame_ready_pending = False
        self._pregame_ready_latched = False
        self._pregame_go_pending = False
        self._pregame_go_latched = False
        self._pregame_ready_streak = 0
        self._pregame_ready_none_gap = 0
        self._pregame_go_streak = 0
        self._pregame_go_none_gap = 0
        self._ingame_entry_none_streak = 0
        self._refresh_flow_phase_label()
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append("フロー: WAIT_READY → WAIT_GO")
            self.log_view.append(
                "★ ready 確定。次は go 画面でモーダル「はい」→ IN_GAME へ進みます。"
            )

    def _append_session_ready_none_veto(self, frame_image) -> None:
        if frame_image is None or frame_image.isNull():
            return
        feat = image_to_feature(frame_image)
        if not feat:
            return
        self._analysis_ready_none_veto.insert(0, feat)
        del self._analysis_ready_none_veto[_SESSION_READY_NONE_VETO_MAX:]

    def _ready_frame_vetoed(self, frame_index: int) -> bool:
        return int(frame_index) in self._ready_veto_frame_indices

    @staticmethod
    def _ready_cnn_none_ranked_blocks(ranked: list | None) -> bool:
        if not _PREGAME_READY_SUPPRESSION_ENABLED or not ranked or ranked[0][0] != "ready":
            return False
        ready_sc = float(ranked[0][1])
        # top1=ready かつスコアが十分低い（自信あり）なら none 僅差では止めない
        if ready_sc < _CNN_READY_DETECT_MAX:
            return False
        if len(ranked) >= 2 and ranked[1][0] == "none":
            none_sc = float(ranked[1][1])
            if none_sc <= ready_sc + _CNN_READY_NONE_MARGIN:
                return True
        none_sc = next((float(s) for c, s in ranked if c == "none"), None)
        if none_sc is not None and none_sc <= ready_sc + _CNN_READY_NONE_MARGIN:
            return True
        return False

    def _ready_is_strong_cnn(self, ranked: list | None) -> bool:
        if not _PREGAME_READY_SUPPRESSION_ENABLED:
            return True
        if not ranked or ranked[0][0] != "ready":
            return False
        return float(ranked[0][1]) < _READY_STRONG_AFTER_REJECT_MAX

    def _effective_raw_scene_for_flow(
        self,
        raw_scene: str,
        frame_image,
        scene_feature,
        ranked: list | None,
        *,
        use_scene_cnn: bool,
    ) -> str:
        return raw_scene

    def _ready_blocked_by_session_veto(
        self, frame_image, scene_feature=None, ranked: list | None = None
    ) -> bool:
        if not _PREGAME_READY_SUPPRESSION_ENABLED:
            return False
        if ranked and ranked[0][0] == "ready":
            ready_sc = float(ranked[0][1])
            if ready_sc < _CNN_READY_DETECT_MAX:
                return False
            if ready_sc < _READY_STRONG_AFTER_REJECT_MAX:
                return False
        feat = scene_feature
        if not feat and frame_image is not None and not frame_image.isNull():
            feat = image_to_feature(frame_image)
        if not feat:
            return False
        # 解析中に「いいえ→none」した画面は CNN の自信度より優先
        if self._analysis_ready_none_veto and feature_matches_none_exemplars(
            feat, self._analysis_ready_none_veto, _NONE_VETO_SESSION_MAX
        ):
            return True
        if self._analysis_ready_none_veto and feature_matches_none_exemplars(
            feat, self._analysis_ready_none_veto, _NONE_VETO_ABSOLUTE_MAX
        ):
            return True
        ready_score = None
        if ranked:
            ready_score = next((float(s) for c, s in ranked if c == "ready"), None)
        centroid = self.video_analyzer.scene_classifier.model
        if centroid.none_veto_exemplars:
            if feature_matches_none_exemplars(
                feat, centroid.none_veto_exemplars, _NONE_VETO_SESSION_MAX
            ):
                return True
            if feature_matches_none_exemplars(
                feat, centroid.none_veto_exemplars, _NONE_VETO_ABSOLUTE_MAX
            ):
                return True
            if centroid.none_veto_blocks_ready(feat, ready_score=ready_score):
                return True
        if self._ready_cnn_none_ranked_blocks(ranked):
            return True
        return False

    def _record_ready_rejection(self, frame_image=None, frame_index: int = -1) -> None:
        if _PREGAME_READY_SUPPRESSION_ENABLED:
            if frame_index >= 0:
                self._ready_veto_frame_indices.add(int(frame_index))
            if frame_image is not None and not frame_image.isNull():
                self._append_session_ready_none_veto(frame_image)
            self._ready_suppressed_this_game = True
            self._pregame_ready_reject_cooldown = 45

    def _reject_pregame_ready(
        self, frame_image=None, frame_index: int = -1, *, record_veto: bool = False
    ) -> None:
        if record_veto:
            self._record_ready_rejection(frame_image, frame_index)
        self._clear_analysis_confirm_edge_prefix("ready")
        self.flow_phase = "WAIT_READY"
        self._ready_confirmed_this_game = False
        self._pregame_ready_pending = False
        self._pregame_ready_latched = False
        self._pregame_ready_streak = 0
        self._pregame_ready_none_gap = 0
        if self._analysis_last_logged_scene == "ready":
            self._analysis_last_logged_scene = "none"
        self._refresh_flow_phase_label()

    def _confirm_pregame_go(self, position_ms: int = 0, frame_index: int = -1) -> None:
        if not self._ready_confirmed_this_game:
            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(
                    "go: ready が未確定のため進めません。先に ready の「はい」を押してください。"
                )
            return
        if self._go_confirmed_this_game:
            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                self.log_view.append(
                    f"go: 既に確定済みです（解析状態: G{self.flow_game_index} {self.flow_phase}）"
                )
            return
        if not self._game_active:
            self._begin_new_game_from_item()
        self._log_pregame_scene_detect("go", position_ms, frame_index, " 確定")
        self._go_confirmed_this_game = True
        self._start_go_round_clock(position_ms)
        self._pregame_go_pending = False
        self._pregame_go_latched = False
        self._pregame_go_streak = 0
        self._pregame_go_none_gap = 0
        self.flow_phase = "IN_GAME"
        self._enter_ingame_log("go")
        self._refresh_flow_phase_label()
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append("フロー: WAIT_GO → IN_GAME")
            self.log_view.append(
                "★ go 確定。プレイ中（IN_GAME）→ timeup / coin / result を検知します。"
            )
        if (
            self.analysis_running
            and self._analysis_profile == _TIME_EFFICIENCY_BUTTON_ACTION
        ):
            self._arm_time_efficiency_go_skip_timer(position_ms)

    def _reject_pregame_go(self) -> None:
        self._clear_analysis_confirm_edge_prefix("go")
        self.flow_phase = "WAIT_GO" if self._ready_confirmed_this_game else "WAIT_READY"
        self._pregame_go_pending = False
        self._pregame_go_latched = False
        self._pregame_go_streak = 0
        self._pregame_go_none_gap = 0
        if self._analysis_last_logged_scene == "go":
            self._analysis_last_logged_scene = "none"
        self._refresh_flow_phase_label()

    def _selected_analysis_model_version(self) -> str:
        combo = getattr(self, "model_version_combo", None)
        if combo is not None and _is_alive_qobject(combo):
            version = combo.currentText().strip()
            if version:
                return version
        return "version_1"

    def _on_analysis_model_version_changed(self) -> None:
        self._save_analysis_settings_ui()
        version = self._selected_analysis_model_version()
        self.video_analyzer.reload_model(version=version)
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view) and not self.analysis_running:
            self.log_view.append(
                f"解析モデルを切替: {version} ({self.video_analyzer.scene_model_path})"
            )

    def _on_sample_frame_changed(self, value: int) -> None:
        self.video_analyzer.sample_every_frames = max(1, value)

    def _analysis_light_mode(self) -> bool:
        cb = getattr(self, "analysis_light_check", None)
        return cb is not None and _is_alive_qobject(cb) and cb.isChecked()

    def _analysis_stop_on_timeup_enabled(self) -> bool:
        cb = getattr(self, "analysis_stop_on_timeup_check", None)
        return cb is None or (not _is_alive_qobject(cb)) or cb.isChecked()

    def _load_analysis_settings_ui(self) -> None:
        """解析設定（左列）を QSettings から復元。"""
        s = self.settings

        def _set_spin(spin: QSpinBox, key: str, default: int) -> None:
            spin.blockSignals(True)
            spin.setValue(max(spin.minimum(), min(spin.maximum(), int(s.value(key, default, type=int)))))
            spin.blockSignals(False)

        def _set_check(cb: QCheckBox, key: str, default: bool) -> None:
            cb.blockSignals(True)
            cb.setChecked(bool(s.value(key, default, type=bool)))
            cb.blockSignals(False)

        if hasattr(self, "sample_frame_spin") and _is_alive_qobject(self.sample_frame_spin):
            _set_spin(self.sample_frame_spin, "analysis/sample_every_frames", 15)
            self.video_analyzer.sample_every_frames = self.sample_frame_spin.value()
        if hasattr(self, "model_version_combo") and _is_alive_qobject(self.model_version_combo):
            mv = str(s.value("analysis/model_version", "version_1", type=str))
            self.model_version_combo.blockSignals(True)
            idx = self.model_version_combo.findText(mv)
            if idx >= 0:
                self.model_version_combo.setCurrentIndex(idx)
            self.model_version_combo.blockSignals(False)
        if hasattr(self, "item_tsum_combo") and _is_alive_qobject(self.item_tsum_combo):
            tsum = str(s.value("analysis/item_tsum", "auto", type=str))
            self.item_tsum_combo.blockSignals(True)
            idx = self.item_tsum_combo.findText(tsum)
            if idx < 0:
                idx = self.item_tsum_combo.findData(tsum)
            if idx >= 0:
                self.item_tsum_combo.setCurrentIndex(idx)
            self.item_tsum_combo.blockSignals(False)
        if hasattr(self, "detail_log_check") and _is_alive_qobject(self.detail_log_check):
            _set_check(self.detail_log_check, "analysis/detail_log", False)
        if hasattr(self, "analysis_light_check") and _is_alive_qobject(self.analysis_light_check):
            _set_check(self.analysis_light_check, "analysis/light_mode", False)
        if hasattr(self, "start_from_zero_check") and _is_alive_qobject(self.start_from_zero_check):
            _set_check(self.start_from_zero_check, "analysis/start_from_zero", True)
        if hasattr(self, "analysis_stop_on_timeup_check") and _is_alive_qobject(
            self.analysis_stop_on_timeup_check
        ):
            _set_check(self.analysis_stop_on_timeup_check, "analysis/stop_on_timeup", False)
        checks = getattr(self, "analysis_scene_confirm_checks", None)
        if isinstance(checks, dict):
            for scene_key, cb in checks.items():
                if _is_alive_qobject(cb):
                    _set_check(
                        cb,
                        f"analysis/scene_confirm/{scene_key}",
                        scene_key in _SCENE_CONFIRM_DEFAULT_ON,
                    )
        if hasattr(self, "analysis_skill_confirm_check") and _is_alive_qobject(
            self.analysis_skill_confirm_check
        ):
            _set_check(self.analysis_skill_confirm_check, "analysis/skill_confirm", False)

    def _save_analysis_settings_ui(self) -> None:
        """解析設定を QSettings へ保存。"""
        s = self.settings
        if hasattr(self, "sample_frame_spin") and _is_alive_qobject(self.sample_frame_spin):
            s.setValue("analysis/sample_every_frames", self.sample_frame_spin.value())
        if hasattr(self, "model_version_combo") and _is_alive_qobject(self.model_version_combo):
            s.setValue("analysis/model_version", self.model_version_combo.currentText())
        if hasattr(self, "item_tsum_combo") and _is_alive_qobject(self.item_tsum_combo):
            s.setValue("analysis/item_tsum", self.item_tsum_combo.currentText())
        if hasattr(self, "detail_log_check") and _is_alive_qobject(self.detail_log_check):
            s.setValue("analysis/detail_log", self.detail_log_check.isChecked())
        if hasattr(self, "analysis_light_check") and _is_alive_qobject(self.analysis_light_check):
            s.setValue("analysis/light_mode", self.analysis_light_check.isChecked())
        if hasattr(self, "start_from_zero_check") and _is_alive_qobject(self.start_from_zero_check):
            s.setValue("analysis/start_from_zero", self.start_from_zero_check.isChecked())
        if hasattr(self, "analysis_stop_on_timeup_check") and _is_alive_qobject(
            self.analysis_stop_on_timeup_check
        ):
            s.setValue("analysis/stop_on_timeup", self.analysis_stop_on_timeup_check.isChecked())
        checks = getattr(self, "analysis_scene_confirm_checks", None)
        if isinstance(checks, dict):
            for scene_key, cb in checks.items():
                if _is_alive_qobject(cb):
                    s.setValue(f"analysis/scene_confirm/{scene_key}", cb.isChecked())
        if hasattr(self, "analysis_skill_confirm_check") and _is_alive_qobject(
            self.analysis_skill_confirm_check
        ):
            s.setValue("analysis/skill_confirm", self.analysis_skill_confirm_check.isChecked())
        s.sync()

    def _bind_analysis_settings_persistence(self) -> None:
        self._load_analysis_settings_ui()

        def _wire_save(signal, handler) -> None:
            signal.connect(handler)

        if hasattr(self, "sample_frame_spin") and _is_alive_qobject(self.sample_frame_spin):
            _wire_save(self.sample_frame_spin.valueChanged, lambda _v: self._save_analysis_settings_ui())
        if hasattr(self, "model_version_combo") and _is_alive_qobject(self.model_version_combo):
            _wire_save(
                self.model_version_combo.currentIndexChanged,
                lambda _i: self._on_analysis_model_version_changed(),
            )
        if hasattr(self, "item_tsum_combo") and _is_alive_qobject(self.item_tsum_combo):
            _wire_save(self.item_tsum_combo.currentIndexChanged, lambda _i: self._save_analysis_settings_ui())
        if hasattr(self, "detail_log_check") and _is_alive_qobject(self.detail_log_check):
            _wire_save(self.detail_log_check.toggled, lambda _c: self._save_analysis_settings_ui())
        if hasattr(self, "analysis_light_check") and _is_alive_qobject(self.analysis_light_check):
            _wire_save(self.analysis_light_check.toggled, lambda _c: self._save_analysis_settings_ui())
        if hasattr(self, "start_from_zero_check") and _is_alive_qobject(self.start_from_zero_check):
            _wire_save(self.start_from_zero_check.toggled, lambda _c: self._save_analysis_settings_ui())
        if hasattr(self, "analysis_stop_on_timeup_check") and _is_alive_qobject(
            self.analysis_stop_on_timeup_check
        ):
            _wire_save(
                self.analysis_stop_on_timeup_check.toggled,
                lambda _c: self._save_analysis_settings_ui(),
            )
        checks = getattr(self, "analysis_scene_confirm_checks", None)
        if isinstance(checks, dict):
            for cb in checks.values():
                if _is_alive_qobject(cb):
                    _wire_save(cb.toggled, lambda _c: self._save_analysis_settings_ui())
        if hasattr(self, "analysis_skill_confirm_check") and _is_alive_qobject(
            self.analysis_skill_confirm_check
        ):
            _wire_save(self.analysis_skill_confirm_check.toggled, lambda _c: self._save_analysis_settings_ui())

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.button_group.checkedId() == 1:
            self._save_analysis_settings_ui()
        super().closeEvent(event)

    def _should_show_trim_overlay(self) -> bool:
        """動画ツール・モード1で「トリミング開始」ON のときだけ枠を出す。"""
        if self.button_group.checkedId() != 2:
            return False
        stack = getattr(self, "_video_tool_right_stack", None)
        if stack is None or not _is_alive_qobject(stack):
            return False
        if not stack.isVisible() or stack.currentIndex() != 0:
            return False
        crop_btn = getattr(self, "crop_start_button", None)
        return (
            crop_btn is not None
            and _is_alive_qobject(crop_btn)
            and crop_btn.isChecked()
        )

    def _hide_trim_overlay_on_video(self) -> None:
        if self.current_video_container is not None:
            self.current_video_container.clear_crop_overlay()

    def _on_crop_selected(self, x: float, y: float, w: float, h: float) -> None:
        self.pending_crop_rect = [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
        self._update_crop_controls_from_pending()
        self._refresh_crop_preview()

    def _prepare_trim_overlay_on_video(self) -> None:
        """動画ツール・トリミング: 保存済み範囲を動画上に表示する。"""
        if not self._should_show_trim_overlay():
            self._hide_trim_overlay_on_video()
            return
        if self.current_video_container is None:
            return
        image = self.current_video_frame_image
        if image is not None and not image.isNull():
            w, h = image.width(), image.height()
            if w > 0 and h > 0:
                self.current_video_container.set_source_size(w, h)
        if self.pending_crop_rect is None and hasattr(self, "crop_target_buttons"):
            selected = [
                k
                for k, b in self.crop_target_buttons.items()
                if _is_alive_qobject(b) and b.isChecked()
            ]
            if selected:
                rect = self._load_crop_positions().get(selected[0])
                if isinstance(rect, list) and len(rect) == 4:
                    self.pending_crop_rect = [float(rect[i]) for i in range(4)]
                    self._update_crop_controls_from_pending()
        self._apply_pending_rect_to_video()

    def _on_crop_target_selection_changed(self) -> None:
        if not hasattr(self, "crop_target_buttons") or not hasattr(self, "crop_rect_label"):
            return
        if not _is_alive_qobject(getattr(self, "crop_rect_label", None)):
            return
        selected = [k for k, b in self.crop_target_buttons.items() if _is_alive_qobject(b) and b.isChecked()]
        if not selected:
            self.crop_rect_label.setText("範囲: 未選択 (対象未選択)")
            return
        data = self._load_crop_positions()
        first_key = selected[0]
        rect = data.get(first_key)
        if rect:
            self.pending_crop_rect = [float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])]
            self._update_crop_controls_from_pending()
            if self._should_show_trim_overlay():
                self._prepare_trim_overlay_on_video()
            self._refresh_crop_preview()
        else:
            self.crop_rect_label.setText("範囲: 未選択")
            self._hide_trim_overlay_on_video()

    def _on_save_crop_clicked(self) -> None:
        if self.pending_crop_rect is None:
            self._train_log("保存失敗: 先に動画上で範囲を指定してください。")
            return
        if self.current_video_frame_image is None or self.current_video_frame_image.isNull():
            self._train_log("保存失敗: 動画フレームが取得できていません。動画を再生/一時停止後に再試行してください。")
            return
        if not hasattr(self, "crop_target_buttons"):
            return
        selected_keys = [k for k, b in self.crop_target_buttons.items() if _is_alive_qobject(b) and b.isChecked()]
        if not selected_keys:
            self._train_log("保存失敗: 対象を1つ以上選択してください。")
            return
        data = self._load_crop_positions()
        data.pop("version", None)
        for key in selected_keys:
            data[key] = self.pending_crop_rect
        path = self.project_root / "app/models/main_model/crop_positions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self.crop_positions_for_analysis = dict(data)
        self._train_log(f"トリミング保存: {','.join(selected_keys)} -> {self.pending_crop_rect}")

    def _on_save_target_images_clicked(self) -> None:
        if self.current_video_frame_image is None or self.current_video_frame_image.isNull():
            self._train_log("保存失敗: 停止画フレームがありません。動画を一時停止して再試行してください。")
            return
        if not hasattr(self, "crop_target_buttons"):
            self._train_log("保存失敗: 対象ボタンが見つかりません。")
            return
        selected_keys = [k for k, b in self.crop_target_buttons.items() if _is_alive_qobject(b) and b.isChecked()]
        if not selected_keys:
            self._train_log("保存失敗: 対象を1つ以上選択してください。")
            return

        crop_data = self._load_crop_positions()
        image = self.current_video_frame_image
        width = image.width()
        height = image.height()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_index = (
            int((max(self._playback_position_ms(), 0) / 1000.0) * self.estimated_fps) if hasattr(self, "player") else 0
        )
        saved = 0

        for key in selected_keys:
            rect = crop_data.get(key)
            if not isinstance(rect, list) or len(rect) != 4:
                self._train_log(f"保存スキップ: {key} はトリミング範囲未保存です。")
                continue
            try:
                nx, ny, nw, nh = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
            except Exception:
                self._train_log(f"保存スキップ: {key} の範囲形式が不正です。")
                continue

            x = max(0, min(int(nx * width), width - 1))
            y = max(0, min(int(ny * height), height - 1))
            w = max(1, min(int(nw * width), width - x))
            h = max(1, min(int(nh * height), height - y))
            roi = image.copy(x, y, w, h)
            if roi.isNull():
                self._train_log(f"保存スキップ: {key} 切り抜き画像が空です。")
                continue

            if key in {"coin_gain", "score_gain", *_RESULT_GAIN_CROP_KEYS}:
                self._train_log(
                    "保存スキップ: 数値読取は「コイン」「結果」で学習用保存してください。"
                )
                continue

            split = self._choose_train_val_split(key)
            out_dir = self.project_root / "app/assets/images/item_icons" / split / key
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{key}_{timestamp}_f{frame_index}.png"
            if roi.save(str(out_path), "PNG"):
                saved += 1
                self._train_log(f"保存: {split}/{key} -> {out_path.name}")
            else:
                self._train_log(f"保存失敗: {split}/{key} -> {out_path.name}")

        if saved == 0:
            self._train_log("対象画像保存: 保存件数0")
            return
        self._train_log(f"対象画像保存完了: {saved}件")

    def _choose_train_val_split(self, target_key: str) -> str:
        train_dir = self.project_root / "app/assets/images/item_icons/train" / target_key
        val_dir = self.project_root / "app/assets/images/item_icons/val" / target_key
        train_count = self._count_image_files(train_dir)
        val_count = self._count_image_files(val_dir)
        total = train_count + val_count
        # Keep val roughly 20% by current per-target status.
        expected_val_after = int((total + 1) * 0.2)
        if val_count < expected_val_after:
            return "val"
        return "train"

    @staticmethod
    def _count_image_files(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        for file in path.iterdir():
            if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
                count += 1
        return count

    def _on_crop_spin_changed(self, _value: float) -> None:
        spin_names = ["crop_x_spin", "crop_y_spin", "crop_w_spin", "crop_h_spin"]
        if not all(hasattr(self, name) for name in spin_names):
            return
        if not all(_is_alive_qobject(getattr(self, name, None)) for name in spin_names):
            return
        self.pending_crop_rect = [
            round(self.crop_x_spin.value(), 4),
            round(self.crop_y_spin.value(), 4),
            round(self.crop_w_spin.value(), 4),
            round(self.crop_h_spin.value(), 4),
        ]
        if hasattr(self, "crop_rect_label") and _is_alive_qobject(self.crop_rect_label):
            self.crop_rect_label.setText(f"範囲: {self.pending_crop_rect}")
        self._prepare_trim_overlay_on_video()
        self._refresh_crop_preview()

    def _on_check_crop_preview_clicked(self) -> None:
        if not hasattr(self, "crop_check_label"):
            return
        image = self.current_video_frame_image
        if image is None or image.isNull():
            self._train_log("確認失敗: 停止画フレームがありません。動画を一時停止して再試行してください。")
            self.crop_check_label.setText("確認プレビューなし")
            self.crop_check_label.setPixmap(QPixmap())
            return

        data = self._load_crop_positions()
        selected_keys: list[str] = []
        if hasattr(self, "crop_target_buttons"):
            selected_keys = [k for k, b in self.crop_target_buttons.items() if _is_alive_qobject(b) and b.isChecked()]
        keys = [k for k in selected_keys if k in data]
        if not keys:
            keys = [k for k in data.keys()]
        if not keys:
            self._train_log("確認失敗: 保存済みトリミング範囲がありません。")
            self.crop_check_label.setText("確認プレビューなし")
            self.crop_check_label.setPixmap(QPixmap())
            return

        preview = QPixmap.fromImage(image.copy())
        painter = QPainter(preview)
        pen_colors = [
            QColor("#FFEB3B"),
            QColor("#00E5FF"),
            QColor("#FF8A80"),
            QColor("#69F0AE"),
            QColor("#B388FF"),
            QColor("#FFD180"),
        ]
        width = image.width()
        height = image.height()
        valid_count = 0
        stats_lines: list[str] = []
        detected, evaluations = self._evaluate_targets(image, data)
        for index, key in enumerate(keys):
            eval_item = evaluations.get(key)
            if eval_item is None:
                continue
            x, y, w, h = eval_item["rect_px"]
            pen = QPen(pen_colors[index % len(pen_colors)], 3)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)
            yellow_ratio = eval_item["yellow_ratio"]
            blue_ratio = eval_item["blue_ratio"]
            is_on = key in detected
            result_text = "ON" if is_on else "OFF"
            display_name = self.crop_target_display.get(key, key)
            painter.drawText(
                x + 4,
                max(14, y + 14),
                f"{display_name} Y:{yellow_ratio:.3f} B:{blue_ratio:.3f} {result_text}",
            )
            stats_lines.append(
                f"{display_name}: yellow={yellow_ratio:.3f} blue={blue_ratio:.3f} 判定={result_text}"
            )
            valid_count += 1
        painter.end()

        if valid_count == 0:
            self._train_log("確認失敗: 有効なトリミング範囲がありません。")
            self.crop_check_label.setText("確認プレビューなし")
            self.crop_check_label.setPixmap(QPixmap())
            return

        self.crop_check_label.setText("")
        self.crop_check_label.setPixmap(
            preview.scaled(
                self.crop_check_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._show_large_crop_check_preview(preview, stats_lines)
        self._train_log(f"停止画確認: {valid_count}件の切り抜き範囲を表示しました。")
        for line in stats_lines:
            self._train_log(f"  {line}")

    def _show_large_crop_check_preview(self, pixmap: QPixmap, stats_lines: list[str]) -> None:
        if pixmap.isNull():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("停止画トリミング確認")
        dialog.resize(1000, 760)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        info_text = "保存済み切り抜き範囲を停止画に重ねて表示"
        if stats_lines:
            info_text += "\n" + " / ".join(stats_lines)
        info = QLabel(info_text)
        info.setWordWrap(True)
        layout.addWidget(info)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_label.setStyleSheet("background:#111; border:1px solid #666;")
        image_label.setPixmap(
            pixmap.scaled(
                980,
                700,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        layout.addWidget(image_label, 1)
        dialog.exec()

    def _update_crop_controls_from_pending(self) -> None:
        if self.pending_crop_rect is None:
            return
        if hasattr(self, "crop_rect_label") and _is_alive_qobject(self.crop_rect_label):
            self.crop_rect_label.setText(f"範囲: {self.pending_crop_rect}")
        for spin_name, value in [
            ("crop_x_spin", self.pending_crop_rect[0]),
            ("crop_y_spin", self.pending_crop_rect[1]),
            ("crop_w_spin", self.pending_crop_rect[2]),
            ("crop_h_spin", self.pending_crop_rect[3]),
        ]:
            if hasattr(self, spin_name):
                spin = getattr(self, spin_name)
                if not _is_alive_qobject(spin):
                    continue
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

    def _on_toggle_crop_mode(self, enabled: bool) -> None:
        self.crop_playback_lock = enabled
        if enabled and hasattr(self, "player"):
            if self._is_player_playing():
                if getattr(self, "use_opencv_for_video", False):
                    self._cv_pause()
                else:
                    self.player.pause()
                self._train_log("トリミング開始: 再生中のため一時停止しました。")
            elif self.current_video_frame_image is None or self.current_video_frame_image.isNull():
                self._train_log("トリミング開始: フレーム取得待ちです。動画を少し動かすと選択できます。")
        if self.current_video_container is not None:
            self.current_video_container.set_crop_enabled(enabled)
        if enabled:
            self._prepare_trim_overlay_on_video()
        else:
            self._hide_trim_overlay_on_video()
        if hasattr(self, "crop_status_label"):
            self.crop_status_label.setText("状態: 選択中" if enabled else "状態: 停止")
        if hasattr(self, "crop_start_button"):
            self.crop_start_button.setText("トリミング終了" if enabled else "トリミング開始")

    def _apply_pending_rect_to_video(self) -> None:
        if self.pending_crop_rect is None or self.current_video_container is None:
            return
        if not self._should_show_trim_overlay():
            self._hide_trim_overlay_on_video()
            return
        image = self.current_video_frame_image
        if image is not None and not image.isNull():
            w, h = image.width(), image.height()
            if w > 0 and h > 0:
                self.current_video_container.set_source_size(w, h)
        self.current_video_container.set_selected_normalized_rect(
            self.pending_crop_rect[0],
            self.pending_crop_rect[1],
            self.pending_crop_rect[2],
            self.pending_crop_rect[3],
        )

    def _refresh_crop_preview(self) -> None:
        if not hasattr(self, "crop_preview_label") or not _is_alive_qobject(self.crop_preview_label):
            return
        if self.current_video_container is None or self.pending_crop_rect is None:
            self.crop_preview_label.setText("プレビューなし")
            self.crop_preview_label.setPixmap(QPixmap())
            return
        if self.current_video_frame_image is None:
            self.crop_preview_label.setText("プレビューなし")
            self.crop_preview_label.setPixmap(QPixmap())
            return

        image = self.current_video_frame_image
        x = int(self.pending_crop_rect[0] * image.width())
        y = int(self.pending_crop_rect[1] * image.height())
        w = int(self.pending_crop_rect[2] * image.width())
        h = int(self.pending_crop_rect[3] * image.height())

        x = max(0, min(x, max(0, image.width() - 1)))
        y = max(0, min(y, max(0, image.height() - 1)))
        w = max(1, min(w, image.width() - x))
        h = max(1, min(h, image.height() - y))

        pix = QPixmap.fromImage(image.copy(x, y, w, h))
        if pix.isNull():
            self.crop_preview_label.setText("プレビューなし")
            self.crop_preview_label.setPixmap(QPixmap())
            return
        self.crop_preview_label.setText("")
        self.crop_preview_label.setPixmap(
            pix.scaled(
                self.crop_preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _on_video_frame_changed(self, frame) -> None:
        self._video_frame_counter += 1
        if not self.analysis_running and self._video_frame_counter % 2 != 0:
            return
        try:
            image = frame.toImage()
        except Exception:
            return
        if image is None or image.isNull():
            return
        self.current_video_frame_image = image
        if self.current_video_container is not None:
            w, h = image.width(), image.height()
            if self._video_frame_size != (w, h):
                self._video_frame_size = (w, h)
                self.current_video_container.set_source_size(w, h)
            elif self.pending_crop_rect is not None:
                self._apply_pending_rect_to_video()
        if self.analysis_running:
            self.analysis_frame_seq += 1
            if self.analysis_frame_seq % self._analysis_sample_step() == 0:
                self._run_analysis_step(self._playback_position_ms(), image)
            return

    def _run_analysis_step(self, position_ms: int, frame_image) -> None:
        if position_ms < self.analysis_warmup_until_ms:
            return
        self._maybe_time_efficiency_go_skip(position_ms)
        # crop_positions は解析開始時に読み込み済み（毎フレームの disk I/O を避ける）
        selected_tsum = "auto"
        if hasattr(self, "item_tsum_combo") and _is_alive_qobject(self.item_tsum_combo):
            selected_tsum = self.item_tsum_combo.currentText()
        use_tsum_dir = ""
        if self.locked_item_fixed:
            use_tsum_dir = self._resolve_tsum_dir(self.locked_use_tsum)
        elif selected_tsum != "auto":
            use_tsum_dir = self._resolve_tsum_dir(selected_tsum)
        scene_feature = None
        if frame_image is not None and not frame_image.isNull():
            scene_feature = image_to_feature(frame_image)
        results = self.video_analyzer.process_frame(
            self.analysis_frame_seq,
            position_ms,
            selected_tsum,
            frame_image,
            use_tsum_dir=use_tsum_dir,
            scene_feature=scene_feature,
        )
        for result in results:
            raw_scene = result.scene_label
            fever_raw = raw_scene
            fever_hit = False
            fever_strong_hit = False
            skill_detected_now = False
            skill_gate_tsum_dir = ""
            use_scene_cnn = self.video_analyzer.scene_classifier.uses_cnn()
            scene_model = self.video_analyzer.scene_classifier.model
            ranked: list = []
            if frame_image is not None and not frame_image.isNull():
                ranked = self.video_analyzer.scene_classifier.ranked_distances(frame_image)
            elif not use_scene_cnn and scene_feature and len(scene_feature) > 0:
                ranked = scene_model.ranked_from_feature(scene_feature)
            fever_top1 = False
            if not self._analysis_fever_enabled():
                fever_hit = False
                fever_strong_hit = False
                fever_raw = raw_scene
            elif use_scene_cnn:
                fever_top1 = bool(ranked) and ranked[0][0] == "fever"
                fever_detected = self._fever_cnn_top1_detected(ranked)
                fever_hit = fever_detected
                fever_strong_hit = fever_detected
                if (
                    not fever_detected
                    and fever_top1
                    and scene_feature
                    and feature_matches_none_exemplars(
                        scene_feature,
                        self._analysis_fever_none_veto,
                        _NONE_VETO_SESSION_MAX,
                    )
                ):
                    fever_hit = False
                    fever_strong_hit = False
                fever_raw = "fever" if fever_hit else raw_scene
            elif ranked:
                fever_strong_hit = scene_model.fever_strong(ranked)
                fever_top1 = ranked[0][0] == "fever"
                if self.flow_phase == "IN_GAME":
                    fever_hit = scene_model.in_game_fever_candidate(ranked)
                else:
                    fever_hit = scene_model.in_game_fever_raw(ranked) == "fever"
                if (
                    self.flow_phase == "IN_GAME"
                    and scene_feature
                    and len(scene_feature) > 0
                    and scene_model.exemplar_blocks_fever_feature(scene_feature)
                    and not fever_top1
                    and not fever_strong_hit
                ):
                    fever_hit = False
                fever_raw = "fever" if fever_hit else "none"
            if self.flow_phase == "IN_GAME" and self.locked_item_fixed:
                skill_gate_tsum_dir = self._resolve_tsum_dir(self.locked_use_tsum)
                if skill_gate_tsum_dir:
                    skill_hit, _skill_gate_dist = self._predict_skill_hit(
                        frame_image, skill_gate_tsum_dir
                    )
                    skill_detected_now = skill_hit and not fever_hit
            flow_raw = self._effective_raw_scene_for_flow(
                raw_scene,
                frame_image,
                scene_feature,
                ranked,
                use_scene_cnn=use_scene_cnn,
            )
            go_hit_pre = self._scene_likely_class(
                flow_raw, ranked, use_cnn=use_scene_cnn, target="go"
            )
            ready_hit_pre = self._scene_likely_class(
                flow_raw, ranked, use_cnn=use_scene_cnn, target="ready", cnn_max_score=0.45
            )
            if not self._go_confirmed_this_game and self.flow_phase in (
                "WAIT_READY",
                "WAIT_GO",
                "IN_GAME",
            ):
                go_hit_pre = self._pregame_class_hit(
                    flow_raw,
                    ranked,
                    use_cnn=use_scene_cnn,
                    target="go",
                    margin_max=_CNN_GO_MARGIN_MAX,
                )
            if not self._ready_confirmed_this_game and self.flow_phase in ("WAIT_ITEM", "WAIT_READY"):
                ready_hit_pre = self._pregame_class_hit(
                    flow_raw,
                    ranked,
                    use_cnn=use_scene_cnn,
                    target="ready",
                    margin_max=_CNN_READY_MARGIN_MAX,
                )
            timeup_hit_pre = self._scene_likely_class(
                raw_scene,
                ranked,
                use_cnn=use_scene_cnn,
                target="timeup",
                cnn_max_score=_CNN_TIMEUP_SCORE_MAX,
            )
            bonus_hit_pre = self._scene_likely_class(
                raw_scene,
                ranked,
                use_cnn=use_scene_cnn,
                target="bonus",
                cnn_max_score=_CNN_BONUS_SCORE_MAX,
            )
            coin_hit_pre = self._scene_likely_class(
                raw_scene,
                ranked,
                use_cnn=use_scene_cnn,
                target="coin",
                cnn_max_score=_CNN_COIN_SCORE_MAX,
            )
            result_hit_pre = self._scene_likely_class(
                raw_scene,
                ranked,
                use_cnn=use_scene_cnn,
                target="result",
                cnn_max_score=_CNN_RESULT_SCORE_MAX,
            )
            evaluations: dict[str, dict] = {}
            item_debug = ""
            item_detected = False
            selected_targets: list[str] = []
            use_tsum_detected = "-"
            if self.flow_phase == "WAIT_ITEM":
                raw_item_scene = result.scene_label == "item"
                selected_targets, evaluations = [], {}
                if not raw_item_scene and self._item_scan_suppressed:
                    self._item_scan_suppressed = False
                if raw_item_scene and not self._item_scan_suppressed:
                    self._item_scene_active = True
                    selected_targets, evaluations = self._evaluate_targets(
                        frame_image, self.crop_positions_for_analysis
                    )
                    self._item_scan_targets.update(_item_select_keys_from(selected_targets))
                    self._refresh_item_counter_labels()
                    if _item_select_keys_from(selected_targets):
                        self._item_scene_active = True
                    if selected_tsum == "auto":
                        use_tsum_detected = self._detect_use_tsum(frame_image)
                    else:
                        use_tsum_detected = selected_tsum
                    if use_tsum_detected not in {"-", "unknown", ""}:
                        self._item_use_tsum_pending = use_tsum_detected
                elif self._item_scene_active and not raw_item_scene and not self._item_scan_suppressed:
                    selected_targets, evaluations = self._evaluate_targets(
                        frame_image, self.crop_positions_for_analysis
                    )
                    self._item_scan_targets.update(_item_select_keys_from(selected_targets))
                if _item_select_keys_from(list(self._item_scan_targets)):
                    self._item_scene_active = True
                leaving_item_scene = self._item_scene_active and not raw_item_scene
                if leaving_item_scene:
                    item_lock_ok = (
                        not self._analysis_scene_confirm_enabled("item")
                        or self._item_confirmed_this_game
                    )
                    if item_lock_ok:
                        if not self.locked_item_fixed:
                            self.locked_item_targets = _item_select_keys_from(
                                list(self._item_scan_targets)
                            )
                            if self._item_use_tsum_pending not in {"-", "unknown", ""}:
                                self.locked_use_tsum = self._item_use_tsum_pending
                            self.locked_item_fixed = True
                        self._item_scene_active = False
                        self._refresh_item_counter_labels()
                item_phase_completed = leaving_item_scene and (
                    not self._analysis_scene_confirm_enabled("item")
                    or self._item_confirmed_this_game
                )
                item_detected = bool(self._item_scan_targets) or self.locked_item_fixed
                scene_label = self._apply_scene_flow(
                    flow_raw,
                    item_phase_completed,
                    use_tsum_detected,
                    result.timestamp_ms,
                    ranked=ranked,
                    use_scene_cnn=use_scene_cnn,
                    frame_index=result.frame_index,
                    scene_feature=scene_feature,
                    frame_image=frame_image,
                )
                item_debug = self._format_item_debug(evaluations, selected_targets)
            else:
                # After item is established, never re-detect until next game.
                selected_targets = list(self.locked_item_targets)
                item_detected = self.locked_item_fixed
                use_tsum_detected = self.locked_use_tsum if self.locked_item_fixed else "-"
                scene_label = self._apply_scene_flow(
                    flow_raw,
                    False,
                    use_tsum_detected,
                    result.timestamp_ms,
                    fever_raw=fever_raw,
                    fever_hit=fever_hit,
                    fever_strong_hit=fever_strong_hit,
                    fever_top1=fever_top1,
                    fever_simple=use_scene_cnn,
                    skill_detected_now=skill_detected_now,
                    ranked=ranked,
                    use_scene_cnn=use_scene_cnn,
                    frame_index=result.frame_index,
                    scene_feature=scene_feature,
                    frame_image=frame_image,
                )
                item_debug = "item_lock:ON"
            result.scene_label = scene_label
            coin_ocr_scene = self._coin_ocr_scene_this_step(flow_raw)
            if (
                coin_ocr_scene
                and frame_image is not None
                and not frame_image.isNull()
            ):
                self._update_coin_gain_capture(frame_image, scene=coin_ocr_scene)
            if (
                self.flow_phase == "WAIT_RESULT"
                and not self._result_confirmed_this_game
                and frame_image is not None
                and not frame_image.isNull()
            ):
                self._update_result_gain_capture(frame_image)
            skill_tsum_dir = (
                self._resolve_tsum_dir(self.locked_use_tsum) if self.locked_item_fixed else use_tsum_dir
            )
            skill_new = False
            skill_log_dir = ""
            if (
                not self._analysis_light_mode()
                and self.locked_item_fixed
                and skill_tsum_dir
                and self.flow_phase == "IN_GAME"
            ):
                pending_dist = self._probe_skill_new_episode(frame_image, skill_tsum_dir)
                if pending_dist is not None:
                    use_skill_modal = False
                    skill_cb = getattr(self, "analysis_skill_confirm_check", None)
                    if skill_cb is not None and _is_alive_qobject(skill_cb) and skill_cb.isChecked():
                        use_skill_modal = True
                    if use_skill_modal:
                        if self._maybe_modal_skill_confirm(
                            skill_tsum_dir, frame_image, pending_dist
                        ):
                            self._commit_skill_episode(skill_tsum_dir, pending_dist)
                            skill_new = True
                            skill_log_dir = skill_tsum_dir
                        else:
                            self._reject_skill_episode()
                            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                                self.log_view.append(
                                    f"t={result.timestamp_ms / 1000.0:7.2f}s "
                                    f"frame={result.frame_index:6d} "
                                    f"skill誤検知: 取り消し ({skill_tsum_dir})"
                                )
                    else:
                        self._commit_skill_episode(skill_tsum_dir, pending_dist)
                        skill_new = True
                        skill_log_dir = skill_tsum_dir
            confirm_label = self._scene_for_confirm_dialog(
                scene_label,
                flow_raw,
                ranked=ranked,
                use_scene_cnn=use_scene_cnn,
                fever_hit=fever_hit,
                go_hit=go_hit_pre,
                ready_hit=ready_hit_pre,
                flow_phase=self.flow_phase,
                ready_confirmed=self._ready_confirmed_this_game,
                ready_locked=self._ready_confirmed_this_game,
                go_locked=self._go_confirmed_this_game,
            )
            if confirm_label == "fever" and (
                self._fever_frame_vetoed(result.frame_index)
                or self._fever_blocked_by_none_veto(
                    frame_image,
                    scene_feature,
                    ranked=ranked,
                    fever_strong_hit=fever_strong_hit,
                    fever_top1=fever_top1,
                )
            ):
                confirm_label = scene_label
            if confirm_label == "timeup" and (
                self._timeup_frame_vetoed(result.frame_index)
                or self._timeup_blocked_by_session_veto(
                    frame_image, scene_feature, ranked=ranked
                )
            ):
                confirm_label = "none"
            if confirm_label == "bonus" and (
                self._bonus_frame_vetoed(result.frame_index)
                or self._bonus_blocked_by_session_veto(
                    frame_image, scene_feature, ranked=ranked
                )
            ):
                confirm_label = "none"
            if confirm_label == "ready" and (
                self._ready_frame_vetoed(result.frame_index)
                or self._ready_blocked_by_session_veto(
                    frame_image, scene_feature, ranked=ranked
                )
            ):
                confirm_label = "none"
            if confirm_label == "result" and (
                self._result_frame_vetoed(result.frame_index)
                or self._result_blocked_by_session_veto(
                    frame_image, scene_feature, ranked=ranked
                )
            ):
                confirm_label = "none"
            self._maybe_modal_scene_confirm(
                confirm_label,
                frame_image,
                flow_scene=scene_label,
                ready_confirmed=self._ready_confirmed_this_game,
                frame_index=result.frame_index,
                position_ms=result.timestamp_ms,
            )
            if scene_label == "ready" and self._ready_confirmed_this_game:
                scene_label = "none"
            elif scene_label == "ready" and self.flow_phase not in ("WAIT_READY", "WAIT_GO"):
                scene_label = "none"
            elif scene_label == "go" and self._go_confirmed_this_game:
                scene_label = "none"
            elif scene_label == "go" and self.flow_phase not in ("WAIT_GO", "IN_GAME"):
                scene_label = "none"
            elif scene_label == "timeup" and self._timeup_confirmed_this_game:
                scene_label = "none"
            elif scene_label == "timeup" and self.flow_phase != "IN_GAME":
                scene_label = "none"
            elif scene_label == "result" and self._result_confirmed_this_game:
                scene_label = "none"
            elif scene_label == "result" and self.flow_phase != "WAIT_RESULT":
                scene_label = "none"
            elif scene_label == "coin" and self._coin_confirmed_this_game:
                scene_label = "none"
            elif scene_label == "coin" and self.flow_phase not in ("WAIT_BONUS", "WAIT_COIN"):
                scene_label = "none"
            elif scene_label == "fever" and self._fever_frame_vetoed(result.frame_index):
                scene_label = "none"
            elif (
                self._analysis_last_logged_scene == "fever"
                and scene_label != "fever"
            ):
                self._clear_analysis_confirm_edge_prefix("fever")
            elif scene_label == "timeup" and self._timeup_frame_vetoed(result.frame_index):
                scene_label = "none"
            elif scene_label == "bonus" and self.flow_phase not in (
                "WAIT_BONUS",
                "WAIT_COIN",
                "WAIT_RESULT",
            ):
                scene_label = "none"
            elif scene_label == "bonus" and self._bonus_frame_vetoed(result.frame_index):
                scene_label = "none"
            elif scene_label == "coin" and self._coin_frame_vetoed(result.frame_index):
                scene_label = "none"
            elif scene_label == "result" and self.flow_phase not in ("WAIT_RESULT", "WAIT_ITEM"):
                scene_label = "none"
            elif scene_label == "result" and self._result_frame_vetoed(result.frame_index):
                scene_label = "none"
            result.scene_label = scene_label
            if skill_new or scene_label != self._analysis_last_logged_scene:
                self._append_analysis_log(
                    result,
                    selected_targets,
                    use_tsum_detected,
                    item_detected,
                    item_debug,
                    skill_detected=skill_new,
                    skill_tsum_dir=skill_log_dir,
                )
                self._analysis_last_logged_scene = scene_label
            if (
                use_scene_cnn
                and self.flow_phase == "IN_GAME"
                and ranked
                and hasattr(self, "log_view")
                and _is_alive_qobject(self.log_view)
                and hasattr(self, "detail_log_check")
                and self.detail_log_check.isChecked()
            ):
                fs = self._fever_score_from_ranked(ranked)
                top = ranked[0][0] if ranked else "-"
                if top == "fever" or (fs is not None and fs < 0.50):
                    det = self._fever_cnn_top1_detected(ranked)
                    self.log_view.append(
                        f"  fever_cnn: top1={top} score={fs:.3f} "
                        f"detect={'YES' if det else 'NO'} (閾値<{_CNN_FEVER_DETECT_MAX})"
                    )
                if scene_label != "fever" and (
                    top == "fever" or (fs is not None and fs < _CNN_FEVER_DETECT_MAX)
                ):
                    vetoed = self._fever_blocked_by_none_veto(
                        frame_image,
                        scene_feature,
                        ranked=ranked,
                        fever_strong_hit=fever_strong_hit,
                        fever_top1=fever_top1,
                    )
                    self.log_view.append(
                        f"  fever_veto: blocked={vetoed} "
                        f"latch={self._fever_episode_latched} "
                        f"cooldown={self._fever_reject_cooldown}"
                    )
            if (
                use_scene_cnn
                and self.flow_phase in ("IN_GAME", "WAIT_BONUS")
                and ranked
                and hasattr(self, "log_view")
                and _is_alive_qobject(self.log_view)
                and hasattr(self, "detail_log_check")
                and self.detail_log_check.isChecked()
            ):
                ts = self._timeup_score_from_ranked(ranked)
                top = ranked[0][0] if ranked else "-"
                if top == "timeup" or (ts is not None and ts < 0.50):
                    det = self._timeup_cnn_detected(ranked)
                    self.log_view.append(
                        f"  timeup_cnn: top1={top} score={ts:.3f} "
                        f"detect={'YES' if det else 'NO'} "
                        f"(top1<{_CNN_TIMEUP_DETECT_MAX} or margin<{_CNN_TIMEUP_SCORE_MAX})"
                    )
                if scene_label != "timeup" and (
                    top == "timeup" or (ts is not None and ts < _CNN_TIMEUP_DETECT_MAX)
                ):
                    blocked = self._timeup_blocked_by_session_veto(
                        frame_image, scene_feature, ranked=ranked
                    )
                    ambiguous = self._timeup_cnn_ambiguous_with_none(ranked)
                    self.log_view.append(
                        f"  timeup_veto: blocked={blocked} ambiguous_none={ambiguous} "
                        f"cooldown={self._timeup_reject_cooldown}"
                    )
            if (
                use_scene_cnn
                and self.flow_phase == "WAIT_BONUS"
                and ranked
                and hasattr(self, "log_view")
                and _is_alive_qobject(self.log_view)
                and hasattr(self, "detail_log_check")
                and self.detail_log_check.isChecked()
            ):
                bs = self._bonus_score_from_ranked(ranked)
                top = ranked[0][0] if ranked else "-"
                if top == "bonus" or (bs is not None and bs < 0.55):
                    det = self._bonus_cnn_detected(ranked)
                    self.log_view.append(
                        f"  bonus_cnn: top1={top} score={bs:.3f} "
                        f"detect={'YES' if det else 'NO'} "
                        f"(top1<{_CNN_BONUS_DETECT_MAX} or margin<{_CNN_BONUS_SCORE_MAX})"
                    )
            if (
                use_scene_cnn
                and self.flow_phase == "WAIT_COIN"
                and not self._coin_confirmed_this_game
                and not self._coin_flow_latched
                and ranked
                and hasattr(self, "log_view")
                and _is_alive_qobject(self.log_view)
                and hasattr(self, "detail_log_check")
                and self.detail_log_check.isChecked()
            ):
                cs = self._coin_score_from_ranked(ranked)
                top = ranked[0][0] if ranked else "-"
                if top == "coin" or (cs is not None and cs < 0.58):
                    det = self._coin_cnn_detected(ranked)
                    self.log_view.append(
                        f"  coin_cnn: top1={top} score={cs:.3f} "
                        f"detect={'YES' if det else 'NO'} "
                        f"(top1<{_CNN_COIN_DETECT_MAX} or margin<{_CNN_COIN_SCORE_MAX})"
                    )
            if (
                use_scene_cnn
                and self.flow_phase in ("WAIT_READY", "WAIT_GO")
                and ranked
                and hasattr(self, "log_view")
                and _is_alive_qobject(self.log_view)
                and hasattr(self, "detail_log_check")
                and self.detail_log_check.isChecked()
            ):
                top = ranked[0][0] if ranked else "-"
                ready_s = next((s for c, s in ranked if c == "ready"), None)
                go_s = next((s for c, s in ranked if c == "go"), None)
                ready_txt = f"{ready_s:.3f}" if ready_s is not None else "-"
                go_txt = f"{go_s:.3f}" if go_s is not None else "-"
                rs = self._pregame_cnn_hit_strength(raw_scene, ranked, target="ready") if use_scene_cnn else "-"
                gs = self._pregame_cnn_hit_strength(raw_scene, ranked, target="go") if use_scene_cnn else "-"
                self.log_view.append(
                    f"  pregame_cnn: phase={self.flow_phase} top1={top} "
                    f"ready={ready_txt}({rs}) go={go_txt}({gs}) "
                    f"ready_pending={self._pregame_ready_pending} streak={self._pregame_ready_streak} "
                    f"go_pending={self._pregame_go_pending} go_streak={self._pregame_go_streak}"
                )
            if (
                use_scene_cnn
                and self.flow_phase == "WAIT_RESULT"
                and ranked
                and hasattr(self, "log_view")
                and _is_alive_qobject(self.log_view)
                and hasattr(self, "detail_log_check")
                and self.detail_log_check.isChecked()
            ):
                rs = self._result_score_from_ranked(ranked)
                top = ranked[0][0] if ranked else "-"
                if top == "result" or (rs is not None and rs < 0.20):
                    det = self._result_cnn_detected(ranked)
                    self.log_view.append(
                        f"  result_cnn: top1={top} score={rs:.3f} "
                        f"detect={'YES' if det else 'NO'} (閾値<{_CNN_RESULT_DETECT_MAX})"
                    )
            if (
                scene_label == "timeup"
                and self.flow_phase == "WAIT_BONUS"
                and self._analysis_stop_on_timeup_enabled()
            ):
                self._timeup_stop_seen += 1
                if self._timeup_stop_seen < _TIMEUP_STOP_CONFIRM_REQUIRED:
                    if hasattr(self, "log_view"):
                        self.log_view.append(
                            f"timeup候補 {self._timeup_stop_seen}/{_TIMEUP_STOP_CONFIRM_REQUIRED}: "
                            "誤検知防止のため継続します。"
                        )
                    continue
                self.analysis_running = False
                self.analysis_warmup_until_ms = 0
                self.analysis_frame_seq = 0
                if getattr(self, "use_opencv_for_video", False):
                    self._cv_pause()
                else:
                    self.player.pause()
                if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
                    self.counter_analysis_state_label.setText("解析状態: 完了(timeup)")
                self._sync_analysis_control_buttons(running=False)
                if hasattr(self, "log_view"):
                    self.log_view.append("解析完了: timeupを検知したため停止しました。")
                break
            else:
                self._timeup_stop_seen = 0

    def _reset_analysis_flow(self) -> None:
        self._last_completed_round = None
        self._clear_time_efficiency_go_skip_state()
        self.flow_phase = "WAIT_ITEM"
        self.flow_game_index = 1
        self._fever_raw_streak = 0
        self._fever_none_gap_used = 0
        self._fever_reject_cooldown = 0
        self._fever_veto_frame_indices = set()
        self._analysis_fever_none_veto = []
        self._analysis_timeup_none_veto = []
        self._timeup_veto_frame_indices = set()
        self._timeup_reject_cooldown = 0
        self._analysis_bonus_none_veto = []
        self._bonus_veto_frame_indices = set()
        self._bonus_reject_cooldown = 0
        self._analysis_coin_none_veto = []
        self._coin_veto_frame_indices = set()
        self._coin_reject_cooldown = 0
        self._analysis_result_none_veto = []
        self._result_veto_frame_indices = set()
        self._result_reject_cooldown = 0
        self._analysis_ready_none_veto = []
        self._ready_veto_frame_indices = set()
        self._pregame_ready_reject_cooldown = 0
        self._in_game_fever_warmup = 0
        self._last_raw_scene_in_game = ""
        self._fever_count = 0
        self._fever_last_count_ms = -1
        self._reset_fever_latch()
        self._ingame_go_grace = 0
        self._timeup_stop_seen = 0
        self._analysis_last_logged_scene = ""
        self._ingame_entered_logged = False
        self._ingame_entry_none_streak = 0
        self._pregame_ready_streak = 0
        self._pregame_go_streak = 0
        self._pregame_ready_none_gap = 0
        self._pregame_go_none_gap = 0
        self._pregame_ready_pending = False
        self._pregame_go_pending = False
        self._pregame_ready_latched = False
        self._pregame_go_latched = False
        self._game_active = False
        self._reset_per_game_scene_flags()
        self._go_clock_anchor_ms = None
        self._go_to_timeup_sec = None
        self._go_to_result_sec = None
        self._timeup_position_ms = 0
        self._refresh_round_timing_label()
        self._item_scan_targets = set()
        self._item_scene_active = False
        self._item_use_tsum_pending = "-"
        self._item_scan_suppressed = False
        self._reset_coin_gain_capture()
        self._skill_count = 0
        self._skill_episode_active = False
        self._skill_off_streak = 0
        self._skill_raw_streak = 0
        self.locked_item_targets = []
        self.locked_use_tsum = "-"
        self.locked_item_fixed = False
        self._item_scan_targets = set()
        self._item_scene_active = False
        self._item_use_tsum_pending = "-"
        self._restore_analysis_sampling()
        self._ingame_sample_saved = None
        self._refresh_item_counter_labels()
        self._refresh_coin_gain_label()
        self._refresh_fever_skill_counter_labels()

    def _register_fever_count(self, position_ms: int) -> None:
        """fever 確定のたびにカウンタ（同一フィーバーは間隔でまとめる）。"""
        now_ms = max(0, int(position_ms))
        if (
            self._fever_last_count_ms < 0
            or now_ms - self._fever_last_count_ms >= _FEVER_COUNT_GAP_MS
        ):
            self._fever_count += 1
            self._fever_last_count_ms = now_ms
            if hasattr(self, "counter_fever_count_label") and _is_alive_qobject(
                self.counter_fever_count_label
            ):
                self.counter_fever_count_label.setText(f"fever回数: {self._fever_count}")

    def _ingame_fever_allowed(
        self,
        frame_image,
        scene_feature,
        *,
        ranked: list | None = None,
        fever_strong_hit: bool = False,
        fever_top1: bool = False,
        frame_index: int = -1,
    ) -> bool:
        if frame_index >= 0 and self._fever_frame_vetoed(frame_index):
            return False
        return not self._fever_blocked_by_none_veto(
            frame_image,
            scene_feature,
            ranked=ranked,
            fever_strong_hit=fever_strong_hit,
            fever_top1=fever_top1,
        )

    def _on_flow_scene_confirmed(self, scene: str, position_ms: int) -> None:
        """フロー上のシーン確定時の副作用（fever 回数など）。"""
        if scene == "fever":
            if self._analysis_scene_confirm_enabled("fever"):
                return
            self._register_fever_count(position_ms)
        elif scene in ("timeup", "bonus", "result"):
            self._fever_last_count_ms = -1

    def _apply_scene_flow(
        self,
        raw_scene: str,
        item_phase_completed: bool,
        use_tsum_detected: str,
        position_ms: int,
        *,
        fever_raw: str | None = None,
        fever_hit: bool = False,
        fever_strong_hit: bool = False,
        fever_top1: bool = False,
        fever_simple: bool = False,
        skill_detected_now: bool = False,
        ranked: Optional[list] = None,
        use_scene_cnn: bool = False,
        frame_index: int = -1,
        scene_feature=None,
        frame_image=None,
    ) -> str:
        """Apply game-order constraints:
        item -> ready -> go -> fever/timeup -> bonus? -> coin -> result -> next game.
        (bonus は省略されることがある)
        """
        phase = self.flow_phase
        now_ms = max(0, int(position_ms))
        fever_signal = fever_raw if fever_raw is not None else raw_scene

        scene = "none"
        ranked = ranked or []
        go_hit = self._scene_likely_class(
            raw_scene, ranked, use_cnn=use_scene_cnn, target="go"
        )
        ready_hit = self._scene_likely_class(
            raw_scene, ranked, use_cnn=use_scene_cnn, target="ready", cnn_max_score=0.45
        )
        timeup_hit = self._scene_likely_class(
            raw_scene, ranked, use_cnn=use_scene_cnn, target="timeup", cnn_max_score=_CNN_TIMEUP_SCORE_MAX
        )
        bonus_hit = self._scene_likely_class(
            raw_scene, ranked, use_cnn=use_scene_cnn, target="bonus", cnn_max_score=_CNN_BONUS_SCORE_MAX
        )
        coin_hit = self._scene_likely_class(
            raw_scene, ranked, use_cnn=use_scene_cnn, target="coin", cnn_max_score=_CNN_COIN_SCORE_MAX
        )
        result_hit = self._scene_likely_class(
            raw_scene, ranked, use_cnn=use_scene_cnn, target="result", cnn_max_score=_CNN_RESULT_SCORE_MAX
        )
        timeup_signal = raw_scene == "timeup" or timeup_hit
        if phase == "WAIT_ITEM":
            if raw_scene == "item":
                scene = "item"
            item_confirm_ok = (
                not self._analysis_scene_confirm_enabled("item")
                or self._item_confirmed_this_game
            )
            advance_pregame = item_confirm_ok and (
                item_phase_completed
                or (self.locked_item_fixed and raw_scene in ("ready", "go"))
                or (self._item_scene_active and raw_scene in ("ready", "go"))
            )
            if advance_pregame:
                self._begin_new_game_from_item()
                self.flow_phase = "WAIT_READY"
                self._wait_ready_grace = _WAIT_READY_GRACE_SAMPLES
                phase = "WAIT_READY"
                if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                    self.log_view.append(
                        f"フロー: G{self.flow_game_index} WAIT_ITEM → WAIT_READY (raw={raw_scene})"
                    )

        if phase == "WAIT_READY" and self._can_detect_scene_once_per_game("ready", phase):
            if self._wait_ready_grace > 0:
                self._wait_ready_grace -= 1
            ready_strength = self._pregame_hit_strength(
                raw_scene,
                ranked,
                use_cnn=use_scene_cnn,
                target="ready",
                margin_max=_CNN_READY_MARGIN_MAX,
                frame_image=frame_image,
                scene_feature=scene_feature,
                frame_index=frame_index,
            )
            ready_hit_frame = ready_strength != "none"
            if ready_hit_frame:
                self._pregame_ready_pending = True
            self._pregame_ready_streak = self._bump_pregame_streak(
                ready_hit_frame,
                streak_attr="_pregame_ready_streak",
                gap_attr="_pregame_ready_none_gap",
            )
            if self._pregame_ready_streak == 0:
                self._pregame_ready_pending = False
                if not (
                    self._pregame_ready_latched
                    and self._analysis_scene_confirm_enabled("ready")
                ):
                    self._pregame_ready_latched = False
            ready_need = _PREGAME_READY_STREAK_WEAK
            if ready_strength == "strong":
                ready_need = 1
            if self._wait_ready_grace > 0 or self._pregame_ready_pending:
                ready_need = 1
            if (
                ready_hit_frame
                and self._pregame_ready_pending
                and self._pregame_ready_streak >= ready_need
            ):
                newly_latched = not self._pregame_ready_latched
                scene = "ready"
                self._pregame_ready_latched = True
                if not self._analysis_scene_confirm_enabled("ready"):
                    self._confirm_pregame_ready(position_ms, frame_index)
                elif newly_latched:
                    self._log_pregame_scene_detect(
                        "ready",
                        position_ms,
                        frame_index,
                        f" strength={ready_strength} streak={self._pregame_ready_streak}"
                        " → 確認モーダルで「はい」を押すと go へ進みます",
                    )
            elif self._pregame_ready_latched and not self._ready_confirmed_this_game:
                scene = "ready"
        elif phase == "WAIT_GO":
            if self._can_detect_scene_once_per_game("ready", phase):
                late_ready = self._pregame_hit_strength(
                    raw_scene,
                    ranked,
                    use_cnn=use_scene_cnn,
                    target="ready",
                    margin_max=_CNN_READY_MARGIN_MAX,
                    frame_image=frame_image,
                    scene_feature=scene_feature,
                    frame_index=frame_index,
                )
                if late_ready != "none":
                    self._pregame_ready_pending = True
                self._pregame_ready_streak = self._bump_pregame_streak(
                    late_ready != "none",
                    streak_attr="_pregame_ready_streak",
                    gap_attr="_pregame_ready_none_gap",
                )
                if self._pregame_ready_streak == 0:
                    self._pregame_ready_pending = False
                if self._pregame_ready_pending and self._pregame_ready_streak >= 1:
                    scene = "ready"
                    if not self._analysis_scene_confirm_enabled("ready"):
                        self._confirm_pregame_ready(position_ms, frame_index)
            if self._can_detect_scene_once_per_game("go", phase):
                go_strength = self._pregame_hit_strength(
                    raw_scene,
                    ranked,
                    use_cnn=use_scene_cnn,
                    target="go",
                    margin_max=_CNN_GO_MARGIN_MAX,
                    frame_image=frame_image,
                    scene_feature=scene_feature,
                )
                go_hit_frame = go_strength != "none"
                if go_hit_frame:
                    self._pregame_go_pending = True
                self._pregame_go_streak = self._bump_pregame_streak(
                    go_hit_frame,
                    streak_attr="_pregame_go_streak",
                    gap_attr="_pregame_go_none_gap",
                )
                if self._pregame_go_streak == 0:
                    self._pregame_go_pending = False
                    if not (
                        self._pregame_go_latched
                        and self._analysis_scene_confirm_enabled("go")
                    ):
                        self._pregame_go_latched = False
                go_need = 1 if go_strength == "strong" else _PREGAME_GO_STREAK_WEAK
                if (
                    go_hit_frame
                    and self._pregame_go_pending
                    and self._pregame_go_streak >= go_need
                    and self._ready_confirmed_this_game
                    and scene != "ready"
                ):
                    newly_latched = not self._pregame_go_latched
                    scene = "go"
                    self._pregame_go_latched = True
                    if not self._analysis_scene_confirm_enabled("go"):
                        self._confirm_pregame_go(position_ms, frame_index)
                    elif newly_latched:
                        self._log_pregame_scene_detect(
                            "go",
                            position_ms,
                            frame_index,
                            f" strength={go_strength} streak={self._pregame_go_streak}"
                            " → 確認モーダルで「はい」を押すと IN_GAME へ",
                        )
                    self._ingame_entry_none_streak = 0
                elif (
                    self._pregame_go_latched
                    and self._ready_confirmed_this_game
                    and scene != "ready"
                ):
                    scene = "go"
                    self._ingame_entry_none_streak = 0
                elif (
                    self._ready_confirmed_this_game
                    and scene != "ready"
                    and not self._pregame_go_pending
                    and raw_scene == "none"
                    and go_strength == "none"
                ):
                    self._ingame_entry_none_streak = min(self._ingame_entry_none_streak + 1, 30)
                    if self._ingame_entry_none_streak >= _FEVER_INGAME_ENTRY_NONE_STREAK:
                        if not self._analysis_scene_confirm_enabled("go"):
                            self._confirm_pregame_go(position_ms, frame_index)
                        elif not self._pregame_go_latched:
                            scene = "go"
                            self._pregame_go_latched = True
                            self._log_pregame_scene_detect(
                                "go",
                                position_ms,
                                frame_index,
                                " go未検知救済 → 確認モーダルで「はい」を押すと IN_GAME へ",
                            )
                else:
                    self._ingame_entry_none_streak = 0
        elif phase == "IN_GAME":
            if self._ingame_go_grace > 0:
                self._ingame_go_grace -= 1
            if self._fever_reject_cooldown > 0:
                self._fever_reject_cooldown -= 1
            if self._timeup_reject_cooldown > 0:
                self._timeup_reject_cooldown -= 1
            if self._analysis_fever_enabled() and self._in_game_fever_warmup > 0:
                self._in_game_fever_warmup -= 1
            gates_ok = (
                self._fever_cnn_gates_ok(
                    ranked or [],
                    fever_strong_hit=fever_strong_hit,
                    fever_top1=fever_top1,
                )
                if fever_simple
                else self._fever_reject_cooldown <= 0
            )
            loose_ok = self._analysis_fever_enabled() and self._in_game_fever_warmup <= 0 and gates_ok
            fever_latched = False
            cnn_timeup_candidate = (
                fever_simple
                and self._timeup_cnn_detected(ranked)
                and self._timeup_reject_cooldown <= 0
            )
            if fever_simple and self._analysis_fever_enabled():
                fever_latched = self._update_cnn_fever_latch(
                    fever_hit=self._fever_cnn_top1_detected(ranked),
                    fever_strong_hit=self._fever_cnn_top1_detected(ranked),
                    fever_top1=fever_top1,
                    ranked=ranked,
                    gates_ok=gates_ok,
                )
            if (
                not self._go_confirmed_this_game
                and self._ingame_go_grace > 0
                and self._pregame_class_hit(
                raw_scene, ranked, use_cnn=use_scene_cnn, target="go"
            )
            ):
                scene = "go"
            elif (
                self._can_detect_scene_once_per_game("timeup", phase)
                and cnn_timeup_candidate
                and not (
                    (frame_index >= 0 and self._timeup_frame_vetoed(frame_index))
                    or self._timeup_blocked_by_session_veto(
                        frame_image, scene_feature, ranked=ranked
                    )
                )
            ):
                scene = "timeup"
                if not self._analysis_scene_confirm_enabled("timeup"):
                    self._confirm_timeup_detection(position_ms)
            elif (
                self._can_detect_scene_once_per_game("timeup", phase)
                and not fever_simple
                and timeup_signal
                and not fever_hit
                and not fever_latched
            ):
                scene = "timeup"
                if not self._analysis_scene_confirm_enabled("timeup"):
                    self._confirm_timeup_detection(position_ms)
            elif (
                self._analysis_fever_enabled()
                and fever_latched
                and self._in_game_fever_warmup <= 0
                and self._ingame_fever_allowed(
                    frame_image,
                    scene_feature,
                    ranked=ranked,
                    fever_strong_hit=fever_strong_hit,
                    fever_top1=fever_top1,
                    frame_index=frame_index,
                )
            ):
                scene = "fever"
                self._fever_raw_streak = _FEVER_WEAK_STREAK_REQUIRED
                self._fever_none_gap_used = 0
            elif (
                self._analysis_fever_enabled()
                and not fever_simple
                and (fever_strong_hit or fever_top1)
                and gates_ok
                and self._ingame_fever_allowed(
                    frame_image,
                    scene_feature,
                    ranked=ranked,
                    fever_strong_hit=fever_strong_hit,
                    fever_top1=fever_top1,
                    frame_index=frame_index,
                )
            ):
                scene = "fever"
                self._fever_raw_streak = _FEVER_WEAK_STREAK_REQUIRED
                self._fever_none_gap_used = 0
            elif self._analysis_fever_enabled() and not fever_simple and fever_hit and loose_ok:
                self._fever_raw_streak = min(self._fever_raw_streak + 1, 30)
                if (
                    self._fever_raw_streak >= _FEVER_WEAK_STREAK_REQUIRED
                    and self._ingame_fever_allowed(
                        frame_image,
                        scene_feature,
                        ranked=ranked,
                        fever_strong_hit=fever_strong_hit,
                        fever_top1=fever_top1,
                        frame_index=frame_index,
                    )
                ):
                    scene = "fever"
                    self._fever_none_gap_used = 0
            elif (
                self._analysis_fever_enabled()
                and not fever_simple
                and raw_scene == "timeup"
                and not fever_hit
                and self._fever_raw_streak >= _FEVER_WEAK_STREAK_REQUIRED - 1
                and self._last_raw_scene_in_game == "fever"
                and self._ingame_fever_allowed(
                    frame_image,
                    scene_feature,
                    ranked=ranked,
                    fever_strong_hit=fever_strong_hit,
                    fever_top1=fever_top1,
                    frame_index=frame_index,
                )
            ):
                self._fever_raw_streak = _FEVER_WEAK_STREAK_REQUIRED
                scene = "fever"
            elif self._analysis_fever_enabled() and fever_signal == "none" and self._fever_raw_streak > 0:
                if self._fever_none_gap_used < _FEVER_NONE_GAP_ALLOW:
                    self._fever_none_gap_used += 1
                    if (
                        self._fever_raw_streak >= _FEVER_WEAK_STREAK_REQUIRED
                        and self._ingame_fever_allowed(
                            frame_image,
                            scene_feature,
                            ranked=ranked,
                            fever_strong_hit=fever_strong_hit,
                            fever_top1=fever_top1,
                            frame_index=frame_index,
                        )
                    ):
                        scene = "fever"
                else:
                    self._fever_raw_streak = 0
                    self._fever_none_gap_used = 0
            else:
                if not fever_hit and not self._fever_episode_latched:
                    self._fever_raw_streak = 0
                    self._fever_none_gap_used = 0
                if raw_scene in ("go", "item", "ready") or (
                    raw_scene in ("bonus", "coin", "result")
                    and not bonus_hit
                    and not coin_hit
                    and not result_hit
                ):
                    self._fever_raw_streak = 0
                    self._fever_none_gap_used = 0
                    if fever_simple:
                        self._reset_fever_latch()
                elif skill_detected_now and not fever_hit and not fever_top1:
                    if not self._fever_episode_latched:
                        self._fever_raw_streak = 0
                        self._fever_none_gap_used = 0
            self._last_raw_scene_in_game = fever_signal if fever_signal == "fever" else raw_scene
        elif phase == "WAIT_BONUS":
            self._reset_fever_latch()
            if not self._analysis_bonus_enabled():
                if (
                    self._can_detect_scene_once_per_game("coin", phase)
                    and self._coin_scene_detected(
                        raw_scene=raw_scene,
                        coin_hit=coin_hit,
                        use_scene_cnn=use_scene_cnn,
                        ranked=ranked,
                        frame_index=frame_index,
                        scene_feature=scene_feature,
                        frame_image=frame_image,
                    )
                ):
                    self._mark_coin_cnn_detected()
                    scene = "coin"
                    if not self._analysis_scene_confirm_enabled("coin"):
                        self._maybe_finalize_coin_confirmation()
            else:
                if self._bonus_reject_cooldown > 0:
                    self._bonus_reject_cooldown -= 1
                bonus_detected = False
                if use_scene_cnn:
                    bonus_detected = (
                        self._bonus_cnn_detected(ranked)
                        and self._bonus_reject_cooldown <= 0
                        and not (
                            (frame_index >= 0 and self._bonus_frame_vetoed(frame_index))
                            or self._bonus_blocked_by_session_veto(
                                frame_image, scene_feature, ranked=ranked
                            )
                        )
                    )
                elif raw_scene == "bonus" or bonus_hit:
                    bonus_detected = True
                if bonus_detected:
                    scene = "bonus"
                    if not self._analysis_scene_confirm_enabled("bonus"):
                        self._confirm_bonus_detection()
                elif (
                    self._can_detect_scene_once_per_game("coin", phase)
                    and self._coin_scene_detected(
                        raw_scene=raw_scene,
                        coin_hit=coin_hit,
                        use_scene_cnn=use_scene_cnn,
                        ranked=ranked,
                        frame_index=frame_index,
                        scene_feature=scene_feature,
                        frame_image=frame_image,
                    )
                ):
                    # bonus 無し: timeup 後に coin が来たら WAIT_COIN へ
                    self._confirm_bonus_detection()
                    self._mark_coin_cnn_detected()
                    scene = "coin"
                    if not self._analysis_scene_confirm_enabled("coin"):
                        self._maybe_finalize_coin_confirmation()
        elif phase == "WAIT_COIN":
            coin_detected = (
                self._can_detect_scene_once_per_game("coin", phase)
                and self._coin_scene_detected(
                    raw_scene=raw_scene,
                    coin_hit=coin_hit,
                    use_scene_cnn=use_scene_cnn,
                    ranked=ranked,
                    frame_index=frame_index,
                    scene_feature=scene_feature,
                    frame_image=frame_image,
                )
            )
            if coin_detected:
                self._mark_coin_cnn_detected()
                scene = "coin"
                if not self._analysis_scene_confirm_enabled("coin"):
                    self._maybe_finalize_coin_confirmation()
        elif phase == "WAIT_RESULT":
            self._reset_fever_latch()
            if self._result_reject_cooldown > 0:
                self._result_reject_cooldown -= 1
            result_detected = False
            if use_scene_cnn:
                result_detected = (
                    (
                        self._result_cnn_detected(ranked)
                        or raw_scene == "result"
                        or result_hit
                    )
                    and self._result_reject_cooldown <= 0
                    and not (
                        (frame_index >= 0 and self._result_frame_vetoed(frame_index))
                        or self._result_blocked_by_session_veto(
                            frame_image, scene_feature, ranked=ranked
                        )
                    )
                )
            elif raw_scene == "result" or result_hit:
                result_detected = True
            if result_detected and self._can_detect_scene_once_per_game("result", phase):
                scene = "result"
                if not self._analysis_scene_confirm_enabled("result"):
                    self._confirm_result_detection(position_ms, frame_image)

        if hasattr(self, "counter_analysis_state_label"):
            self.counter_analysis_state_label.setText(f"解析状態: G{self.flow_game_index} {self.flow_phase}")
        self._on_flow_scene_confirmed(scene, now_ms)
        return scene

    def _load_crop_positions(self) -> dict:
        path = self.project_root / "app/models/main_model/crop_positions.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _exec_scene_confirm_dialog(self, frame_image, scene_label: str) -> int:
        """プレビュー付き。1=はい、2=いいえ（修正へ）、0=×閉じる（誤検知破棄）。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("シーン判定の確認")
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)
        img = QLabel()
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setPixmap(
            _preview_pixmap_for_scene_modal(
                frame_image, _SCENE_CONFIRM_PREVIEW_MAX_W, _SCENE_CONFIRM_PREVIEW_MAX_H
            )
        )
        layout.addWidget(img)
        txt = QLabel(f"シーンを「{scene_label}」と判定しました。\nこの判定は合っていますか？")
        txt.setWordWrap(True)
        layout.addWidget(txt)
        skill_save_cb, skill_tsum_combo, skill_cat_combo = self._build_skill_save_group(
            layout,
            show_fever_hint=(scene_label == "fever"),
            default_tsum_dir=self._default_skill_tsum_dir(),
        )
        row = QHBoxLayout()
        row.addStretch(1)
        yes_btn = QPushButton("はい")
        no_btn = QPushButton("いいえ")
        row.addWidget(yes_btn)
        row.addWidget(no_btn)
        layout.addLayout(row)
        yes_btn.setDefault(True)

        yes_btn.clicked.connect(lambda: dlg.done(1))
        no_btn.clicked.connect(lambda: dlg.done(2))
        rc = dlg.exec()
        if rc == 1:
            self._try_skill_save_from_dialog(frame_image, skill_save_cb, skill_tsum_combo, skill_cat_combo)
        return int(rc)

    def _exec_scene_correction_dialog(
        self, frame_image, scene_label: str, *, default_choice: str = "none"
    ) -> tuple[Optional[str], bool]:
        """プレビュー付きクラス選択。（choice, accepted）。キャンセル時 choice は None。"""
        dlg = QDialog(self)
        dlg.setWindowTitle("正しいクラス")
        dlg.setModal(True)
        layout = QVBoxLayout(dlg)
        img = QLabel()
        img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img.setPixmap(
            _preview_pixmap_for_scene_modal(
                frame_image, _SCENE_CONFIRM_PREVIEW_MAX_W, _SCENE_CONFIRM_PREVIEW_MAX_H
            )
        )
        layout.addWidget(img)
        layout.addWidget(
            QLabel(
                "正しいクラスを選ぶと train/val に保存されます。"
                "誤検知の fever などは none を選ぶと train/none へ入ります。"
            )
        )
        combo = QComboBox()
        combo.addItems(list(SimpleTrainer.CLASSES))
        prefer = default_choice if default_choice in SimpleTrainer.CLASSES else "none"
        combo.setCurrentIndex(SimpleTrainer.CLASSES.index(prefer))
        layout.addWidget(combo)
        skill_save_cb, skill_tsum_combo, skill_cat_combo = self._build_skill_save_group(
            layout,
            show_fever_hint=True,
            default_tsum_dir=self._default_skill_tsum_dir(),
        )
        bbox = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        layout.addWidget(bbox)
        bbox.accepted.connect(dlg.accept)
        bbox.rejected.connect(dlg.reject)
        if dlg.exec() != 1:  # QDialog.Accepted
            return None, False
        self._try_skill_save_from_dialog(frame_image, skill_save_cb, skill_tsum_combo, skill_cat_combo)
        return combo.currentText(), True

    def _maybe_modal_skill_confirm(self, tsum_dir: str, frame_image, dist: float) -> bool:
        """スキル発動候補の確認。はい=True（確定）、いいえ/×=False（誤検知として破棄）。"""
        if not self.analysis_running or not tsum_dir:
            return False

        display = self.use_tsum_classifier._display_map.get(tsum_dir, tsum_dir)
        was_cv = getattr(self, "use_opencv_for_video", False)
        was_playing = self._is_player_playing()
        if was_playing:
            if was_cv:
                self._cv_pause()
            elif hasattr(self, "player") and _is_alive_qobject(self.player):
                self.player.pause()

        try:
            dlg = QDialog(self)
            dlg.setWindowTitle("スキル発動の確認")
            dlg.setModal(True)
            layout = QVBoxLayout(dlg)
            img = QLabel()
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img.setPixmap(
                _preview_pixmap_for_scene_modal(
                    frame_image, _SCENE_CONFIRM_PREVIEW_MAX_W, _SCENE_CONFIRM_PREVIEW_MAX_H
                )
            )
            layout.addWidget(img)
            txt = QLabel(
                f"スキル発動を検知しました（{display} / {tsum_dir}）。\n"
                "この判定は合っていますか？\n"
                "・はい … スキル1回としてカウント\n"
                "・いいえ … 誤検知として取り消し（カウントしません）"
            )
            txt.setWordWrap(True)
            layout.addWidget(txt)
            skill_save_cb, skill_tsum_combo, skill_cat_combo = self._build_skill_save_group(
                layout,
                show_fever_hint=True,
                default_tsum_dir=tsum_dir,
            )
            row = QHBoxLayout()
            yes_btn = QPushButton("はい")
            no_btn = QPushButton("いいえ")
            row.addStretch(1)
            row.addWidget(yes_btn)
            row.addWidget(no_btn)
            layout.addLayout(row)
            yes_btn.setDefault(True)
            yes_btn.clicked.connect(lambda: dlg.done(1))
            no_btn.clicked.connect(lambda: dlg.done(2))
            rc = dlg.exec()
            confirmed = rc == 1
            if confirmed:
                self._try_skill_save_from_dialog(frame_image, skill_save_cb, skill_tsum_combo, skill_cat_combo)
            return confirmed
        finally:
            if was_playing and self.analysis_running:
                if was_cv:
                    self._cv_play()
                elif hasattr(self, "player") and _is_alive_qobject(self.player):
                    self.player.play()

    def _on_train_skill_only_clicked(self) -> None:
        if self.train_busy:
            self._train_log("学習・保存処理中です。完了後に実行してください。")
            return
        self.train_busy = True
        self._train_busy_task = "skill"
        self.train_started_at = time.time()
        self._set_train_ui_state(status_text="状態: スキル・使用ツム 再学習中...")
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar):
            bar.setStyleSheet(
                "QProgressBar::chunk { background-color: #66BB6A; }"
            )
            bar.setRange(0, 0)
            bar.setFormat("スキル・使用ツム 再学習中…")
            bar.setVisible(True)
        self._train_log("スキル・使用ツムモデルの再学習を開始します…")
        self.train_poll_timer.start()

        def run_skill() -> None:
            q = self.train_message_queue
            emit = lambda msg: q.put(msg)
            try:
                emit("使用ツムモデルを更新中…")
                self._train_use_tsum_models(log=emit)
                emit("スキルモデルを更新中…")
                self._train_skill_models(log=emit)
                self.skill_classifier_pool.reload()
                self.video_analyzer.item_skill_classifier.reload()
                q.put("__SKILL_DONE__")
            except Exception as exc:
                q.put(f"__SKILL_ERROR__:{exc}")

        self.train_thread = threading.Thread(target=run_skill, daemon=True)
        self.train_thread.start()

    def _clear_analysis_confirm_edge_prefix(self, prefix: str) -> None:
        key = getattr(self, "_analysis_confirm_edge_key", "")
        if key.startswith(f"{prefix}:"):
            self._analysis_confirm_edge_key = ""

    def _confirm_scene_by_label(
        self,
        confirm_label: str,
        frame_image=None,
        *,
        position_ms: int = 0,
        frame_index: int = -1,
    ) -> None:
        if confirm_label == "ready":
            self._confirm_pregame_ready(position_ms, frame_index)
        elif confirm_label == "go":
            self._confirm_pregame_go(position_ms, frame_index)
        elif confirm_label == "timeup":
            self._confirm_timeup_detection(position_ms)
        elif confirm_label == "bonus":
            self._confirm_bonus_detection()
        elif confirm_label == "coin":
            self._acknowledge_coin_scene(frame_image)
        elif confirm_label == "result":
            self._confirm_result_detection(position_ms, frame_image)
        elif confirm_label == "fever":
            self._register_fever_count(position_ms)
        elif confirm_label == "item":
            self._confirm_item_detection()

    def _reject_scene_by_label(
        self,
        confirm_label: str,
        frame_image=None,
        frame_index: int = -1,
        *,
        record_veto: bool = False,
    ) -> None:
        if confirm_label == "ready":
            self._reject_pregame_ready(frame_image, frame_index, record_veto=record_veto)
        elif confirm_label == "go":
            self._reject_pregame_go()
        elif confirm_label == "timeup":
            self._reject_timeup_detection(frame_image, frame_index)
        elif confirm_label == "bonus":
            self._reject_bonus_detection(frame_image, frame_index)
        elif confirm_label == "coin":
            self._reject_coin_detection(frame_image, frame_index)
        elif confirm_label == "result":
            self._reject_result_detection(frame_image, frame_index)
        elif confirm_label == "fever":
            self._reject_fever_detection(frame_image, frame_index)
        elif confirm_label == "item":
            self._reject_item_detection(frame_image, frame_index)

    def _record_scene_none_rejection(
        self, confirm_label: str, frame_image, frame_index: int
    ) -> None:
        if confirm_label == "fever":
            self._record_fever_none_rejection(frame_image, frame_index)
        elif confirm_label == "timeup":
            self._record_timeup_none_rejection(frame_image, frame_index)
        elif confirm_label == "bonus":
            self._record_bonus_none_rejection(frame_image, frame_index)
        elif confirm_label == "coin":
            self._record_coin_none_rejection(frame_image, frame_index)
        elif confirm_label == "result":
            self._record_result_none_rejection(frame_image, frame_index)
        elif confirm_label == "ready":
            self._record_ready_rejection(frame_image, frame_index)

    def _maybe_modal_scene_confirm(
        self,
        confirm_label: str,
        frame_image,
        *,
        flow_scene: str = "",
        ready_confirmed: bool = False,
        frame_index: int = -1,
        position_ms: int = 0,
    ) -> None:
        """チェックされたシーンへ遷移した最初のフレームで確認（プレビュー付き）。誤りならクラス選択して保存。"""
        checks = getattr(self, "analysis_scene_confirm_checks", None)
        if checks is None or not self.analysis_running:
            return
        if confirm_label == "go" and not ready_confirmed and not self._ready_confirmed_this_game:
            return
        if confirm_label == "item" and self._item_confirmed_this_game:
            return
        if confirm_label == "item" and self._item_scan_suppressed:
            return
        if confirm_label == "ready" and self._ready_confirmed_this_game:
            return
        if confirm_label == "go" and self._go_confirmed_this_game:
            return
        if confirm_label == "timeup" and self._timeup_confirmed_this_game:
            return
        if confirm_label == "result" and self._result_confirmed_this_game:
            return
        if confirm_label == "coin" and self._coin_confirmed_this_game:
            return
        if confirm_label == "coin" and self._coin_modal_acknowledged:
            return
        if confirm_label == "fever" and frame_index >= 0:
            if self._fever_frame_vetoed(frame_index):
                return
        if confirm_label == "timeup" and frame_index >= 0:
            if self._timeup_frame_vetoed(frame_index):
                return
        if confirm_label == "bonus" and frame_index >= 0:
            if self._bonus_frame_vetoed(frame_index):
                return
        if confirm_label == "coin" and frame_index >= 0:
            if self._coin_frame_vetoed(frame_index):
                return
        if confirm_label == "result" and frame_index >= 0:
            if self._result_frame_vetoed(frame_index):
                return
        watched = [
            c
            for c in _ANALYSIS_SCENE_CONFIRM_LABELS
            if c in checks and _is_alive_qobject(checks[c]) and checks[c].isChecked()
        ]
        if confirm_label not in watched:
            return
        flow_scene = flow_scene or confirm_label
        if confirm_label == "fever" and frame_index >= 0:
            edge_key = f"fever:{frame_index}"
        elif confirm_label == "timeup" and frame_index >= 0:
            edge_key = f"timeup:{frame_index}"
        elif confirm_label == "bonus" and frame_index >= 0:
            edge_key = f"bonus:{frame_index}"
        elif confirm_label == "coin" and frame_index >= 0:
            edge_key = f"coin:{frame_index}"
        elif confirm_label == "item" and frame_index >= 0:
            edge_key = f"item:{frame_index}"
        elif confirm_label == "result" and frame_index >= 0:
            edge_key = f"result:{frame_index}"
        elif confirm_label in ("ready", "go"):
            edge_key = f"{confirm_label}:G{int(self.flow_game_index)}"
        else:
            edge_key = f"{confirm_label}:{flow_scene}"
        prev_key = getattr(self, "_analysis_confirm_edge_key", "")
        if edge_key == prev_key:
            return

        was_cv = getattr(self, "use_opencv_for_video", False)
        was_playing = self._is_player_playing()
        if was_playing:
            if was_cv:
                self._cv_pause()
            elif hasattr(self, "player") and _is_alive_qobject(self.player):
                self.player.pause()

        confirmed_yes = False
        try:
            confirm_rc = self._exec_scene_confirm_dialog(frame_image, confirm_label)
            if confirm_rc == 1:
                confirmed_yes = True
                if confirm_label in ("ready", "go"):
                    self._analysis_confirm_edge_key = edge_key
                if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                    self.log_view.append(f"シーン確認: {confirm_label} → はい")
                self._confirm_scene_by_label(
                    confirm_label,
                    frame_image,
                    position_ms=position_ms,
                    frame_index=frame_index,
                )
                self._maybe_save_scene_class_on_confirm_yes(frame_image, confirm_label)
            elif confirm_rc == 2:
                if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                    self.log_view.append(f"シーン確認: {confirm_label} → いいえ（正しいクラスを選択）")
                choice, ok = self._exec_scene_correction_dialog(
                    frame_image, confirm_label, default_choice="none"
                )
                if not ok:
                    self._reject_scene_by_label(
                        confirm_label, frame_image, frame_index, record_veto=False
                    )
                    if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                        self.log_view.append(
                            f"シーン確認: {confirm_label} の修正をキャンセル → 保留（再検知可）"
                        )
                elif choice:
                    if choice == "none":
                        self._record_scene_none_rejection(confirm_label, frame_image, frame_index)
                    save_ok, msg = self._save_frame_to_scene_class(frame_image, choice)
                    skipped_dup = msg.startswith(SAVE_SKIP_DUPLICATE_PREFIX)
                    if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                        if skipped_dup:
                            self.log_view.append(
                                "修正保存スキップ（同一画像）: "
                                f"{msg[len(SAVE_SKIP_DUPLICATE_PREFIX):]}"
                            )
                        else:
                            self.log_view.append(
                                ("修正保存: " if save_ok else "修正保存失敗: ") + msg
                            )
                    if save_ok and not skipped_dup:
                        self._train_log(f"解析修正保存: {msg}")
                        self._refit_scene_model_after_correction(choice, frame_image)
                        if choice == "none":
                            if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                                if confirm_label == "fever":
                                    if self.video_analyzer.scene_classifier.uses_cnn():
                                        self.log_view.append(
                                            "fever 誤検知として train/none に保存しました。"
                                            "学習タブで「学習開始」→「モデル保存」で反映してください。"
                                        )
                                    else:
                                        self.log_view.append(
                                            "この画面を誤検知リストに追加しました（同じ見た目はしばらく fever しません）。"
                                        )
                                else:
                                    self.log_view.append(
                                        f"{confirm_label} 誤検知として train/none に保存しました。"
                                        "学習タブで「学習開始」→「モデル保存」で反映してください。"
                                    )
                    elif skipped_dup:
                        self._train_log(
                            "解析修正保存スキップ（同一画像）: "
                            f"{msg[len(SAVE_SKIP_DUPLICATE_PREFIX):]}"
                        )
                    if choice == confirm_label:
                        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                            self.log_view.append(
                                f"シーン確認: 修正後も「{choice}」→ {confirm_label} として確定"
                            )
                        self._confirm_scene_by_label(
                            confirm_label,
                            frame_image,
                            position_ms=position_ms,
                            frame_index=frame_index,
                        )
                        confirmed_yes = True
                        if confirm_label in ("ready", "go"):
                            self._analysis_confirm_edge_key = edge_key
                    else:
                        self._reject_scene_by_label(
                            confirm_label,
                            frame_image,
                            frame_index,
                            record_veto=False,
                        )
                        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                            if choice == "none":
                                self.log_view.append(
                                    f"シーン確認: {confirm_label} を誤検知として破棄（none 保存）"
                                )
                            else:
                                self.log_view.append(
                                    f"シーン確認: {confirm_label} を誤検知として破棄"
                                    f"（正解は {choice} として保存）"
                                )
            else:
                self._reject_scene_by_label(
                    confirm_label, frame_image, frame_index, record_veto=False
                )
                if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                    self.log_view.append(
                        f"シーン確認: {confirm_label} を×で閉じた → 保留（再検知可）"
                    )
        finally:
            self._analysis_confirm_edge_prev = confirm_label
            if confirm_label in ("ready", "go"):
                if not confirmed_yes:
                    self._clear_analysis_confirm_edge_prefix(confirm_label)
            elif confirm_label == "coin":
                if confirmed_yes:
                    self._analysis_confirm_edge_key = edge_key
                else:
                    self._clear_analysis_confirm_edge_prefix("coin")
            else:
                if confirmed_yes:
                    self._analysis_confirm_edge_key = edge_key
            resume = was_playing and self.analysis_running
            if resume:
                if was_cv:
                    self._cv_play()
                elif hasattr(self, "player") and _is_alive_qobject(self.player):
                    self.player.play()

    def _append_analysis_log(
        self,
        result,
        selected_targets=None,
        use_tsum_detected: str = "-",
        item_detected: bool = False,
        item_debug: str = "",
        skill_detected: bool = False,
        skill_tsum_dir: str = "",
    ) -> None:
        if not hasattr(self, "log_view"):
            return
        if selected_targets is None:
            selected_targets = []
        seconds = result.timestamp_ms / 1000.0
        selected_text = ",".join(selected_targets) if selected_targets else "-"
        used_item_keys = _item_select_keys_from(selected_targets)
        used_item_names = [self.crop_target_display.get(k, k) for k in used_item_keys]
        used_items_text = ",".join(used_item_names) if used_item_names else "-"
        if self.locked_item_fixed and self.locked_item_targets:
            locked_item_names = [self.crop_target_display.get(k, k) for k in self.locked_item_targets]
            display_used_items = ",".join(locked_item_names)
            display_use_tsum = self.locked_use_tsum
        elif (
            result.scene_label == "item"
            and not self._item_scan_suppressed
            and used_items_text != "-"
        ):
            display_used_items = used_items_text
            display_use_tsum = use_tsum_detected if use_tsum_detected not in {"-", ""} else "-"
        else:
            display_used_items = "-"
            display_use_tsum = (
                use_tsum_detected
                if result.scene_label == "item" and not self._item_scan_suppressed
                else "-"
            )
        if hasattr(self, "counter_use_item_label"):
            self.counter_use_item_label.setText(f"使用アイテム: {display_used_items}")
        if hasattr(self, "counter_use_tsum_label"):
            self.counter_use_tsum_label.setText(f"使用ツム: {display_use_tsum}")
        # Log output is scene-based (avoid noisy fixed values on non-item scenes).
        log_used_items = (
            used_items_text
            if result.scene_label == "item" and not self._item_scan_suppressed
            else "-"
        )
        log_use_tsum = (
            use_tsum_detected
            if result.scene_label == "item" and not self._item_scan_suppressed
            else "-"
        )
        # item_detect comes from detection flow in _on_player_position_changed.
        effective_item_detected = (
            item_detected
            if result.scene_label == "item" and not self._item_scan_suppressed
            else False
        )
        item_detected_text = "YES" if effective_item_detected else "NO"
        scene_label = result.scene_label
        prefix = f"t={seconds:7.2f}s frame={result.frame_index:6d} scene="
        if result.scene_label == "item":
            short_suffix = f" item_detect={item_detected_text} used_items={log_used_items}"
            detail_suffix = (
                f" item_detect={item_detected_text} "
                f"item={result.item_skill_label} used_items={log_used_items} "
                f"selected={selected_text} use_tsum={log_use_tsum}"
            )
        elif result.scene_label == "coin" and self._coin_gain_best is not None:
            short_suffix = f" coin_gain={self._coin_gain_best}"
            detail_suffix = short_suffix
        else:
            short_suffix = f" [{self.flow_phase}]"
            detail_suffix = short_suffix
        skill_label = skill_tsum_dir if skill_detected else ""
        if hasattr(self, "detail_log_check") and not self.detail_log_check.isChecked():
            self._append_log_colored_scene(prefix, scene_label, short_suffix, skill_label)
            if item_debug and result.scene_label == "item":
                self.log_view.append(f"  item_debug: {item_debug}")
            return
        self._append_log_colored_scene(prefix, scene_label, detail_suffix, skill_label)
        if item_debug and result.scene_label == "item":
            self.log_view.append(f"  item_debug: {item_debug}")

    def _log_pregame_scene_detect(
        self,
        scene: str,
        position_ms: int,
        frame_index: int,
        note: str,
    ) -> None:
        """ready/go 検知（候補・確定）を scene= 行・オレンジで出す（候補だけ別行 plain にはしない）。"""
        if scene not in ("ready", "go"):
            return
        seconds = max(0, int(position_ms)) / 1000.0
        prefix = f"t={seconds:7.2f}s frame={max(0, int(frame_index)):6d} scene="
        suffix = f" [{self.flow_phase}]{note}"
        self._append_log_colored_scene(prefix, scene, suffix)
        self._analysis_last_logged_scene = scene

    def _append_log_colored_scene(
        self, prefix: str, scene_label: str, suffix: str, skill_label: str = ""
    ) -> None:
        """Log one line; scene name ready/go in orange via QTextCharFormat (HTML is unreliable in QTextEdit)."""
        if not hasattr(self, "log_view"):
            return
        batch_log = getattr(self, "analysis_running", False)
        if batch_log:
            self.log_view.setUpdatesEnabled(False)
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        mono = QTextCharFormat()
        mono.setFontFamilies(["monospace"])
        orange = QTextCharFormat()
        orange.setFontFamilies(["monospace"])
        orange.setForeground(QColor("#B45309"))
        cursor.setCharFormat(mono)
        cursor.insertText(prefix)
        if scene_label in {"ready", "go", "coin"}:
            cursor.setCharFormat(orange)
            cursor.insertText(scene_label)
            cursor.setCharFormat(mono)
        elif scene_label == "fever":
            fever_fmt = QTextCharFormat()
            fever_fmt.setFontFamilies(["monospace"])
            fever_fmt.setForeground(QColor("#DC2626"))
            cursor.setCharFormat(fever_fmt)
            cursor.insertText(scene_label)
            cursor.setCharFormat(mono)
        else:
            cursor.insertText(scene_label)
        if skill_label:
            skill_fmt = QTextCharFormat()
            skill_fmt.setFontFamilies(["monospace"])
            skill_fmt.setForeground(QColor("#7C3AED"))
            cursor.setCharFormat(skill_fmt)
            cursor.insertText(f" skill={skill_label}")
            cursor.setCharFormat(mono)
        cursor.insertText(suffix + "\n")
        self.log_view.setTextCursor(cursor)
        if batch_log:
            self.log_view.setUpdatesEnabled(True)
        # 解析中は毎行 ensureCursorVisible すると QTextEdit のスクロールが重い
        if not getattr(self, "analysis_running", False):
            self.log_view.ensureCursorVisible()
        else:
            self._analysis_log_scroll_counter += 1
            if self._analysis_log_scroll_counter % 16 == 0:
                self.log_view.ensureCursorVisible()

    def _refresh_item_counter_labels(self) -> None:
        if hasattr(self, "counter_use_item_label") and _is_alive_qobject(self.counter_use_item_label):
            if self.locked_item_fixed and self.locked_item_targets:
                keys = self.locked_item_targets
            elif self._item_scan_suppressed:
                keys = []
            elif self._item_scan_targets:
                keys = _item_select_keys_from(list(self._item_scan_targets))
            elif self._last_completed_round:
                keys = list(self._last_completed_round.get("item_targets") or [])
            else:
                keys = []
            if keys:
                names = [self.crop_target_display.get(k, k) for k in keys]
                text = ",".join(names)
            else:
                text = "--"
            self.counter_use_item_label.setText(f"使用アイテム: {text}")
        if hasattr(self, "counter_use_tsum_label") and _is_alive_qobject(self.counter_use_tsum_label):
            if self.locked_item_fixed:
                tsum = self.locked_use_tsum
            elif self._item_scan_suppressed:
                tsum = "--"
            elif self._item_use_tsum_pending not in {"-", "unknown", ""}:
                tsum = self._item_use_tsum_pending
            elif self._last_completed_round:
                tsum = self._last_completed_round.get("use_tsum", "-")
            else:
                tsum = "--"
            if tsum in {"-", ""}:
                tsum = "--"
            self.counter_use_tsum_label.setText(f"使用ツム: {tsum}")

    def _format_item_debug(self, evaluations: dict[str, dict], selected_targets: list[str]) -> str:
        parts = []
        for key in _ITEM_SELECT_KEYS:
            ev = evaluations.get(key)
            if ev is None:
                parts.append(f"{self.crop_target_display.get(key, key)}:NA")
                continue
            y = float(ev.get("yellow_ratio", 0.0))
            b = float(ev.get("blue_ratio", 0.0))
            mark = "ON" if key in _item_select_keys_from(selected_targets) else "OFF"
            parts.append(f"{self.crop_target_display.get(key, key)} Y={y:.3f} B={b:.3f} {mark}")
        return " | ".join(parts)

    def _detect_selected_targets(self, image) -> list[str]:
        if image is None or not self.crop_positions_for_analysis:
            return []
        if image.isNull():
            return []

        detected, _evaluations = self._evaluate_targets(image, self.crop_positions_for_analysis)
        return detected

    def _evaluate_targets(self, image, positions: dict) -> tuple[list[str], dict[str, dict]]:
        width = image.width()
        height = image.height()
        detected = []
        evaluations: dict[str, dict] = {}
        threshold_map = {
            "bomb": 0.03,
            "five_to_four": 0.03,
            "time": 0.03,
            "score": 0.07,
            "coin": 0.07,
            "exp": 0.07,
            "combo": 0.07,
        }
        for target, rect in positions.items():
            if not isinstance(rect, list) or len(rect) != 4:
                continue
            if target in {
                "use_tsum",
                "skill",
                "coin_gain",
                "score_gain",
                "remaining_time",
                *_RESULT_GAIN_CROP_KEYS,
            }:
                continue
            try:
                nx, ny, nw, nh = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
            except Exception:
                continue

            x = max(0, min(int(nx * width), width - 1))
            y = max(0, min(int(ny * height), height - 1))
            w = max(1, min(int(nw * width), width - x))
            h = max(1, min(int(nh * height), height - y))
            yellow_ratio, blue_ratio = self._yellow_blue_ratios(image, x, y, w, h)
            threshold = threshold_map.get(target, 0.07)
            evaluations[target] = {
                "yellow_ratio": yellow_ratio,
                "blue_ratio": blue_ratio,
                "threshold": threshold,
                "rect_px": (x, y, w, h),
            }
            if target in {"bomb", "five_to_four", "time"}:
                # item系は後段で3種を相対比較して同時ONを判定する。
                continue
            if yellow_ratio >= threshold:
                detected.append(target)

        # Item icons (bomb/5>4/time): relative comparison for multi-select support.
        item_keys = ["bomb", "five_to_four", "time"]
        item_scores: dict[str, float] = {}
        for key in item_keys:
            ev = evaluations.get(key)
            if ev is None:
                continue
            y = float(ev["yellow_ratio"])
            b = float(ev["blue_ratio"])
            # active-like score: yellow dominance over blue
            item_scores[key] = y - (b * 0.85)

        if item_scores:
            top_score = max(item_scores.values())
            for key, score in item_scores.items():
                ev = evaluations[key]
                y = float(ev["yellow_ratio"])
                b = float(ev["blue_ratio"])
                # Absolute floor + relative-to-best condition to allow multiple active items.
                if y >= 0.02 and y > (b * 1.03) and score >= max(0.008, top_score * 0.70):
                    detected.append(key)
        return detected, evaluations

    def _yellow_blue_ratios(self, image, x: int, y: int, w: int, h: int) -> tuple[float, float]:
        # Sample grid-based pixels for lightweight per-frame yellow detection.
        step_x = max(1, w // 24)
        step_y = max(1, h // 24)
        total = 0
        yellow = 0
        blue = 0
        for yy in range(y, y + h, step_y):
            for xx in range(x, x + w, step_x):
                color = image.pixelColor(xx, yy)
                r = color.red() / 255.0
                g = color.green() / 255.0
                b = color.blue() / 255.0
                h_hsv, s_hsv, v_hsv = colorsys.rgb_to_hsv(r, g, b)
                total += 1
                hsv_yellowish = (0.10 <= h_hsv <= 0.18 and s_hsv >= 0.30 and v_hsv >= 0.35)
                hsv_bluish = (0.52 <= h_hsv <= 0.72 and s_hsv >= 0.28 and v_hsv >= 0.25)
                if hsv_yellowish:
                    yellow += 1
                if hsv_bluish:
                    blue += 1
        if total == 0:
            return (0.0, 0.0)
        return (yellow / total, blue / total)

    def _resolve_tsum_dir(self, label: str) -> str:
        return resolve_tsum_dir(
            label,
            self.use_tsum_classifier._display_map,
            list(self.use_tsum_classifier._prototypes.keys()),
        )

    def _predict_skill_hit(self, frame_image, tsum_dir: str) -> tuple[bool, float]:
        if frame_image is None or frame_image.isNull() or not self.skill_classifier_pool.has_model(tsum_dir):
            return (False, float("inf"))
        skill_rect = None
        rect = self.crop_positions_for_analysis.get("skill")
        if isinstance(rect, list) and len(rect) == 4:
            try:
                skill_rect = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            except Exception:
                skill_rect = None
        return self.skill_classifier_pool.predict(tsum_dir, frame_image, skill_rect)

    def _probe_skill_new_episode(self, frame_image, tsum_dir: str) -> Optional[float]:
        """IN_GAME 中のスキル候補。新規エピソードなら距離を返す（まだカウントしない）。"""
        raw_hit, dist = self._predict_skill_hit(frame_image, tsum_dir)
        if raw_hit:
            self._skill_raw_streak = min(self._skill_raw_streak + 1, 30)
            self._skill_off_streak = 0
        else:
            self._skill_raw_streak = 0
            self._skill_off_streak += 1
            if self._skill_off_streak >= 2:
                self._skill_episode_active = False
            return None
        if self._skill_raw_streak < 3:
            return None
        if self._skill_episode_active:
            return None
        return dist

    def _commit_skill_episode(self, tsum_dir: str, dist: float) -> None:
        self._skill_episode_active = True
        self._skill_count += 1
        if hasattr(self, "counter_skill_count_label") and _is_alive_qobject(self.counter_skill_count_label):
            self.counter_skill_count_label.setText(f"スキル回数: {self._skill_count}")
        if hasattr(self, "log_view") and hasattr(self, "detail_log_check") and self.detail_log_check.isChecked():
            self.log_view.append(f"  skill_debug: {tsum_dir} dist={dist:.3f}")

    def _reject_skill_episode(self) -> None:
        """誤検知: カウントせず、同じ誤判定が連続しないよう streak を切る。"""
        self._skill_episode_active = False
        self._skill_raw_streak = 0

    def _detect_use_tsum(self, image) -> str:
        if image is None or image.isNull():
            return "-"
        rect = self.crop_positions_for_analysis.get("use_tsum")
        if not isinstance(rect, list) or len(rect) != 4:
            return "-"
        width = image.width()
        height = image.height()
        try:
            nx, ny, nw, nh = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
        except Exception:
            return "-"
        x = max(0, min(int(nx * width), width - 1))
        y = max(0, min(int(ny * height), height - 1))
        w = max(1, min(int(nw * width), width - x))
        h = max(1, min(int(nh * height), height - y))
        roi = image.copy(x, y, w, h)
        label, _distance = self.use_tsum_classifier.predict(roi)
        if hasattr(self, "log_view") and hasattr(self, "detail_log_check") and self.detail_log_check.isChecked():
            self.log_view.append(f"  use_tsum_debug: label={label} dist={_distance:.2f}")
        return label

    def _train_log(self, message: str) -> None:
        if hasattr(self, "log_view"):
            self.log_view.append(message)

    def _train_model_meta_path(self) -> Path:
        version = "version_1"
        if hasattr(self, "analysis_model_version_combo") and _is_alive_qobject(
            self.analysis_model_version_combo
        ):
            version = self.analysis_model_version_combo.currentText() or version
        return self.project_root / "app/models/main_model" / version / "model_meta.json"

    def _on_train_shortage_check_clicked(self) -> None:
        ok, summary, report = self._refresh_train_dataset_advice(log_result=True)
        if not ok:
            self._train_log(
                "不足画像を確認: 学習タブの表示が無効です。"
                " いったん別タブへ移動してから「学習」を開き直してください。"
            )
            return
        n_warn = len([s for s in report.shortages if s.cls != "_balance"])
        n_balance = len([s for s in report.shortages if s.cls == "_balance"])
        self._train_log(
            f"不足画像を確認: 完了 train={summary.train_total} val={summary.val_total}"
            f" 注意クラス={n_warn}"
            + (f" 偏り警告={n_balance}" if n_balance else "")
        )
        for line in report.lines:
            if line.startswith("【") or line.startswith(" ・") or line.startswith("※"):
                self._train_log(line)

    def _refresh_train_dataset_advice(
        self, *, log_result: bool = False
    ) -> tuple[bool, object, object]:
        """データ状況ラベルを更新。詳細は右の学習ログへ。 (成功, summary, report) を返す。"""
        empty_summary = None
        empty_report = None
        if not hasattr(self, "train_data_advice_status") or not _is_alive_qobject(
            self.train_data_advice_status
        ):
            return False, empty_summary, empty_report
        summary = self.trainer.summarize_dataset()
        report = analyze_scene_dataset(summary, model_meta_path=self._train_model_meta_path())
        text = format_advice_report_text(report)
        skill_root = self.project_root / "app/assets/images/skills"
        skill_total, skill_low = summarize_skill_activation_images(skill_root)
        if skill_total > 0 or skill_low:
            text += "\n\n--- スキル発動画像 ---\n"
            text += f"合計 {skill_total} 枚\n"
            if skill_low:
                text += "追加推奨:\n" + "\n".join(f" ・{line}" for line in skill_low[:12])
                if len(skill_low) > 12:
                    text += f"\n  …他 {len(skill_low) - 12} 件"
            else:
                text += "各ツム 10 枚以上あります。"
        scanned = datetime.now().strftime("%H:%M:%S")
        if report.has_critical:
            status_bg = "#FFEBEE"
            level = "要対応"
        elif report.shortages:
            status_bg = "#FFF8E1"
            level = "追加推奨あり"
        else:
            status_bg = "#E8F5E9"
            level = "良好"
        n_warn = len([s for s in report.shortages if s.cls != "_balance"])
        self.train_data_advice_status.setStyleSheet(
            f"color: #333; font-size: 11px; padding: 4px 6px;"
            f"background: {status_bg}; border: 1px solid #E0E0E0; border-radius: 4px;"
        )
        self.train_data_advice_status.setText(
            f"データ状況: {level}（{scanned}）"
            f" train={summary.train_total} val={summary.val_total}"
            f" 注意={n_warn}クラス → 詳細は右の学習ログ"
        )
        if log_result:
            self._train_log(f"--- データ状況 ({scanned}) ---")
            for line in text.splitlines():
                if line.strip():
                    self._train_log(line)
        return True, summary, report

    def _on_train_data_check_clicked(self) -> None:
        summary = self.trainer.summarize_dataset()
        self._train_log(f"データ確認: train={summary.train_total}, val={summary.val_total}")
        self._train_log(f"train内訳: {summary.per_class_train}")
        self._train_log(f"val内訳: {summary.per_class_val}")
        self._refresh_train_dataset_advice()
        report = analyze_scene_dataset(summary, model_meta_path=self._train_model_meta_path())
        for line in report.lines:
            if line.startswith("【") or line.startswith(" ・"):
                self._train_log(line)

    def _scene_images_root(self) -> Path:
        return self.project_root / "app/assets/images"

    def _run_scene_dedup(self, *, dry_run: bool) -> None:
        if self.train_busy:
            self._train_log("学習・保存処理中です。完了後に実行してください。")
            return
        images_root = self._scene_images_root()
        label = "重複確認" if dry_run else "重複削除"
        self.train_busy = True
        self._set_train_ui_state(status_text=f"状態: {label}中...")
        self._train_log(f"{label}を開始します（train/val 全体）…")

        def work() -> None:
            q = self.train_message_queue

            def emit(msg: str) -> None:
                q.put(msg)

            try:
                report = deduplicate_scene_dataset(
                    images_root,
                    dry_run=dry_run,
                    log=emit,
                )
                for line in report.summary_lines():
                    emit(line)
                if dry_run and report.removed_paths:
                    emit(
                        f"削除対象 {len(report.removed_paths)}枚。"
                        "実行するには「重複削除」を押してください。"
                    )
                elif not dry_run and report.files_removed:
                    emit("重複削除が完了しました。必要なら「学習開始」で再学習してください。")
                q.put("__DEDUP_DONE__")
            except Exception as exc:
                q.put(f"__DEDUP_ERROR__:{exc}")

        self.train_poll_timer.start()
        threading.Thread(target=work, daemon=True).start()

    def _dataset_review_predict(self, path: Path):
        cnn = self.trainer.scene_cnn
        if not cnn.is_loaded():
            return None
        ranked = cnn.ranked_path(path, top_k=1)
        if not ranked:
            return None
        return ranked[0]

    def _on_train_dataset_review_clicked(self) -> None:
        if self.train_busy:
            self._train_log("学習・保存処理中は画像レビューを開けません。")
            return
        dlg = SceneDatasetReviewDialog(
            self._scene_images_root(),
            predict_fn=self._dataset_review_predict,
            parent=self,
        )
        dlg.exec()
        if dlg.deleted_count <= 0:
            return
        self._trainer_scene_dirty = True
        self._train_log(f"画像レビュー: {dlg.deleted_count} 枚削除しました。再学習→モデル保存で反映してください。")
        self._refresh_train_dataset_advice()

    def _on_train_dedup_preview_clicked(self) -> None:
        self._run_scene_dedup(dry_run=True)

    def _on_train_dedup_clicked(self) -> None:
        preview = deduplicate_scene_dataset(self._scene_images_root(), dry_run=True)
        if not preview.removed_paths:
            self._train_log("重複削除: 削除対象の完全同一画像はありません。")
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("重複削除")
        box.setText(
            f"完全同一の画像が {preview.duplicate_groups} グループ"
            f"（{len(preview.removed_paths)} 枚）あります。削除しますか？"
        )
        box.setInformativeText(
            "train/val 全体で1枚にまとめます。\n"
            "残す優先: ファイル名とクラス一致 → train → 先に保存した枚。\n"
            "クラスが違う重複はファイル名が合う方を残します。"
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        box.setDefaultButton(QMessageBox.StandardButton.No)
        if box.exec() != QMessageBox.StandardButton.Yes:
            self._train_log("重複削除をキャンセルしました。")
            return
        self._run_scene_dedup(dry_run=False)

    def _on_train_start_clicked(self) -> None:
        if self.train_busy:
            self._train_log("学習はすでに実行中です。")
            return

        self.train_busy = True
        self.train_started_at = time.time()
        self._set_train_ui_state(status_text="状態: 学習中...")
        self._train_log("学習をバックグラウンドで開始します...")
        self.train_poll_timer.start()

        def run_train() -> None:
            try:
                self.trainer.start(lambda msg: self.train_message_queue.put(msg))
            finally:
                self.train_message_queue.put("__TRAIN_DONE__")

        self.train_thread = threading.Thread(target=run_train, daemon=True)
        self.train_thread.start()

    def _on_train_stop_clicked(self) -> None:
        if self.train_busy:
            self._train_log("現在の学習処理は停止できません。完了を待ってください。")
            self._set_train_ui_state(status_text="状態: 学習中(停止待ち)")
            return
        self.trainer.stop(self._train_log)
        self._set_train_ui_state(status_text="状態: 待機中")

    def _trainer_scene_save_ready(self, version: str) -> bool:
        if self.trainer.scene_cnn.is_loaded() or self.trainer.scene_model.centroids:
            return True
        out_dir = self.project_root / "app/models/main_model" / version
        return (out_dir / "scene_cnn.pt").exists() or (out_dir / "scene_model.json").exists()

    def _on_train_save_clicked(self) -> None:
        if self.train_busy:
            self._train_log("学習・保存処理中です。完了後に実行してください。")
            return
        model_version = self._selected_analysis_model_version()
        if not self._trainer_scene_save_ready(model_version):
            self._train_log(
                "保存できません。先に「学習開始」でシーン CNN を学習するか、"
                "保存済み scene_cnn.pt がある版を選んでください。"
                "獲得コイン DL だけ必要なら「コイン桁 CNN だけ保存」を使ってください。"
            )
            return

        self.train_busy = True
        self._train_busy_task = "model_save"
        self.train_started_at = time.time()
        self._set_train_ui_state(status_text="状態: モデル保存中...")
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar):
            bar.setStyleSheet("")
            bar.setRange(0, 0)
            bar.setFormat("モデル保存中…")
            bar.setVisible(True)
        self._train_log("モデル保存をバックグラウンドで開始します…")
        self.train_poll_timer.start()

        def run_save() -> None:
            q = self.train_message_queue
            emit = lambda msg: q.put(msg)
            try:
                emit("シーンモデルをファイルへ保存中…")
                self.trainer.save(emit, version=model_version)
                emit("解析用モデルを読み込み中…")
                self._trainer_scene_dirty = False
                self.video_analyzer.reload_model(version=model_version)
                images_root = self.project_root / "app/assets/images"
                centroid = self.video_analyzer.scene_classifier.model
                if centroid is not None:
                    try:
                        self.trainer.scene_model.fit_from_dataset(images_root)
                        self.video_analyzer.apply_scene_centroids(
                            self.trainer.scene_model.centroids,
                            self.trainer.scene_model.fever_calib,
                        )
                    except Exception:
                        pass
                    n_none = centroid.rebuild_none_veto_exemplars(images_root)
                    if n_none:
                        emit(
                            f"ready誤検知抑制: train/none {n_none} 枚を解析に反映しました。"
                        )
                emit("使用ツムモデルを更新中…")
                self._train_use_tsum_models(log=emit)
                emit("スキルモデルを更新中…")
                self._train_skill_models(log=emit)
                self.skill_classifier_pool.reload()
                emit("コイン桁 CNN を解析に反映中…")
                from app.services.coin_digit_cnn import reload_coin_digit_classifier

                reload_coin_digit_classifier()
                emit("コイン桁 CNN を反映しました。")
                q.put("__SAVE_DONE__")
            except Exception as exc:
                q.put(f"__SAVE_ERROR__:{exc}")

        self.train_thread = threading.Thread(target=run_save, daemon=True)
        self.train_thread.start()

    def _on_train_save_coin_digit_only_clicked(self) -> None:
        if self.train_busy:
            self._train_log("学習・保存処理中です。完了後に実行してください。")
            return
        self.train_busy = True
        self._train_busy_task = "coin_digit"
        self.train_started_at = time.time()
        self._set_train_ui_state(status_text="状態: コイン桁 CNN 学習・保存中...")
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar):
            bar.setStyleSheet(
                "QProgressBar::chunk { background-color: #FFB300; }"
            )
            bar.setRange(0, 0)
            bar.setFormat("コイン桁 CNN 学習・保存中…")
            bar.setVisible(True)
        self._train_log("コイン桁 CNN の学習・保存を開始します…")
        self.train_poll_timer.start()
        model_version = self._selected_analysis_model_version()

        def run_coin_save() -> None:
            q = self.train_message_queue
            emit = lambda msg: q.put(msg)
            try:
                self.trainer.save_coin_digit_only(emit, version=model_version)
                from app.services.coin_digit_cnn import reload_coin_digit_classifier

                reload_coin_digit_classifier()
                acc = float(getattr(self.trainer, "coin_digit_val_accuracy", 0.0))
                emit(f"コイン桁 CNN を解析に反映しました（検証精度 {acc:.1%}）。")
                q.put("__SAVE_DONE__")
            except Exception as exc:
                q.put(f"__SAVE_ERROR__:{exc}")

        self.train_thread = threading.Thread(target=run_coin_save, daemon=True)
        self.train_thread.start()

    def _on_train_save_result_digit_only_clicked(self) -> None:
        if self.train_busy:
            self._train_log("学習・保存処理中です。完了後に実行してください。")
            return
        self.train_busy = True
        self._train_busy_task = "result_digit"
        self.train_started_at = time.time()
        self._set_train_ui_state(status_text="状態: 結果桁 CNN 学習・保存中...")
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar):
            bar.setStyleSheet(
                "QProgressBar::chunk { background-color: #42A5F5; }"
            )
            bar.setRange(0, 0)
            bar.setFormat("結果桁 CNN 学習・保存中…")
            bar.setVisible(True)
        self._train_log("結果桁 CNN の学習・保存を開始します…")
        self.train_poll_timer.start()
        model_version = self._selected_analysis_model_version()

        def run_result_save() -> None:
            q = self.train_message_queue
            emit = lambda msg: q.put(msg)
            try:
                self.trainer.save_result_digit_only(emit, version=model_version)
                from app.services.result_digit_cnn import reload_result_digit_classifier

                reload_result_digit_classifier()
                acc = float(getattr(self.trainer, "result_digit_val_accuracy", 0.0))
                emit(f"結果桁 CNN を解析に反映しました（検証精度 {acc:.1%}）。")
                q.put("__SAVE_DONE__")
            except Exception as exc:
                q.put(f"__SAVE_ERROR__:{exc}")

        self.train_thread = threading.Thread(target=run_result_save, daemon=True)
        self.train_thread.start()

    def _flash_train_save_complete(self, label: str = "保存完了") -> None:
        """保存成功をプログレスバーで明示してから消す。"""
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar):
            bar.setRange(0, 100)
            bar.setValue(100)
            bar.setFormat(label)
            bar.setStyleSheet("QProgressBar::chunk { background-color: #66BB6A; }")
            bar.setVisible(True)
        QTimer.singleShot(3200, self._reset_train_progress_bar)

    def _reset_train_progress_bar(self) -> None:
        bar = getattr(self, "train_progress_bar", None)
        if bar is None or not _is_alive_qobject(bar):
            return
        bar.setStyleSheet("")
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setFormat("")
        bar.setVisible(False)

    def _train_use_tsum_models(self, log: Optional[Callable[[str], None]] = None) -> None:
        emit = log if log is not None else self._train_log
        images_root = self.project_root / "app/assets/images/use_tsums"
        models_root = self.project_root / "app/models/use_tsum"
        crop_positions = self._load_crop_positions()
        use_tsum_rect = None
        rect = crop_positions.get("use_tsum")
        if isinstance(rect, list) and len(rect) == 4:
            try:
                use_tsum_rect = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            except Exception:
                use_tsum_rect = None
        counts = UseTsumClassifier.build_models(images_root, models_root, use_tsum_rect)
        self.use_tsum_classifier.reload()
        if not counts:
            emit("使用ツムモデル学習: 対象画像なし")
            return
        summary = ", ".join(f"{k}:{v}枚" for k, v in sorted(counts.items()))
        if use_tsum_rect is None:
            emit("使用ツムモデル学習: use_tsum範囲未設定のため画像全体から作成")
        else:
            emit(f"使用ツムモデル学習: use_tsum範囲で切抜き作成 {use_tsum_rect}")
        emit(f"使用ツムモデル学習: {summary}")

    def _train_skill_models(self, log: Optional[Callable[[str], None]] = None) -> None:
        emit = log if log is not None else self._train_log
        images_root = self.project_root / "app/assets/images/skills"
        models_root = self.project_root / "app/models/skill"
        crop_positions = self._load_crop_positions()
        skill_rect = None
        rect = crop_positions.get("skill")
        if isinstance(rect, list) and len(rect) == 4:
            try:
                skill_rect = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            except Exception:
                skill_rect = None
        counts = SkillClassifierPool.build_models(images_root, models_root, skill_rect)
        self.skill_classifier_pool.reload()
        if not counts:
            emit(
                "スキルモデル学習: 対象画像なし"
                f"（app/assets/images/skills/<dir名>/{SKILL_IMAGE_CATEGORY_ACTIVATION}/ に発動時画像）"
            )
            return
        summary = ", ".join(f"{k}:{v}枚" for k, v in sorted(counts.items()))
        if skill_rect is None:
            emit("スキルモデル学習: skill範囲未設定のため画像全体から作成")
        else:
            emit(f"スキルモデル学習: skill範囲で切抜き作成 {skill_rect}")
        emit(f"スキルモデル学習: {summary}")

    def _on_train_timer_tick(self) -> None:
        # Legacy timer: no-op (training now runs in background thread).
        self.train_timer.stop()

    def _poll_train_messages(self) -> None:
        while not self.train_message_queue.empty():
            msg = self.train_message_queue.get()
            if msg == "__SAVE_DONE__":
                self.train_busy = False
                self.train_poll_timer.stop()
                elapsed = int(max(0.0, time.time() - self.train_started_at))
                task = getattr(self, "_train_busy_task", "")
                self._train_busy_task = ""
                if task == "coin_digit":
                    acc = float(getattr(self.trainer, "coin_digit_val_accuracy", 0.0))
                    self._train_log(
                        f"=== コイン桁 CNN の学習・保存が完了しました === "
                        f"所要 {elapsed}s / 検証精度 {acc:.1%}"
                    )
                    self._set_train_ui_state(
                        status_text=f"状態: コイン桁 CNN 保存完了 ({elapsed}s)"
                    )
                    self._flash_train_save_complete("コイン桁 CNN 保存完了")
                elif task == "result_digit":
                    acc = float(getattr(self.trainer, "result_digit_val_accuracy", 0.0))
                    self._train_log(
                        f"=== 結果桁 CNN の学習・保存が完了しました === "
                        f"所要 {elapsed}s / 検証精度 {acc:.1%}"
                    )
                    self._set_train_ui_state(
                        status_text=f"状態: 結果桁 CNN 保存完了 ({elapsed}s)"
                    )
                    self._flash_train_save_complete("結果桁 CNN 保存完了")
                else:
                    self._train_log(f"モデル保存が完了しました。所要時間: {elapsed}s")
                    self._set_train_ui_state(status_text=f"状態: モデル保存完了 ({elapsed}s)")
                    self._flash_train_save_complete("モデル保存完了")
                self._refresh_train_dataset_advice()
                continue
            if msg.startswith("__SAVE_ERROR__:"):
                self.train_busy = False
                self.train_poll_timer.stop()
                err = msg.split(":", 1)[1] if ":" in msg else msg
                task = getattr(self, "_train_busy_task", "")
                self._train_busy_task = ""
                if task == "coin_digit":
                    self._train_log(f"コイン桁 CNN 保存エラー: {err}")
                    self._set_train_ui_state(status_text="状態: コイン桁 CNN 保存失敗")
                elif task == "result_digit":
                    self._train_log(f"結果桁 CNN 保存エラー: {err}")
                    self._set_train_ui_state(status_text="状態: 結果桁 CNN 保存失敗")
                else:
                    self._train_log(f"モデル保存エラー: {err}")
                    self._set_train_ui_state(status_text="状態: 保存失敗")
                self._reset_train_progress_bar()
                continue
            if msg == "__SKILL_DONE__":
                self.train_busy = False
                self.train_poll_timer.stop()
                elapsed = int(max(0.0, time.time() - self.train_started_at))
                self._train_busy_task = ""
                self._train_log(
                    f"=== スキル・使用ツムモデルの再学習が完了しました === 所要 {elapsed}s"
                )
                self._set_train_ui_state(
                    status_text=f"状態: スキル・使用ツム 再学習完了 ({elapsed}s)"
                )
                self._flash_train_save_complete("スキル・使用ツム 再学習完了")
                continue
            if msg.startswith("__SKILL_ERROR__:"):
                self.train_busy = False
                self.train_poll_timer.stop()
                err = msg.split(":", 1)[1] if ":" in msg else msg
                self._train_busy_task = ""
                self._train_log(f"スキル・使用ツム 再学習エラー: {err}")
                self._set_train_ui_state(status_text="状態: スキル・使用ツム 再学習失敗")
                self._reset_train_progress_bar()
                continue
            if msg == "__DEDUP_DONE__":
                self.train_busy = False
                self.train_poll_timer.stop()
                self._set_train_ui_state(status_text="状態: 待機中")
                self._refresh_train_dataset_advice()
                continue
            if msg.startswith("__DEDUP_ERROR__:"):
                self.train_busy = False
                self.train_poll_timer.stop()
                err = msg.split(":", 1)[1] if ":" in msg else msg
                self._train_log(f"重複処理エラー: {err}")
                self._set_train_ui_state(status_text="状態: 重複処理失敗")
                continue
            if msg == "__TRAIN_DONE__":
                self.train_busy = False
                self.train_poll_timer.stop()
                elapsed = int(max(0.0, time.time() - self.train_started_at))
                self._train_log(f"学習処理が完了しました。所要時間: {elapsed}s")
                self._trainer_scene_dirty = True
                if self.trainer.scene_cnn.is_loaded():
                    self.video_analyzer.reload_model()
                    self._train_log(
                        "解析用 CNN を更新しました。ファイルへ反映するには「モデル保存」を押してください。"
                    )
                elif self.trainer.scene_model.centroids:
                    self.video_analyzer.apply_scene_centroids(
                        self.trainer.scene_model.centroids,
                        self.trainer.scene_model.fever_calib,
                    )
                    self._train_log(
                        "解析用モデル（centroid）を更新しました。"
                        "ファイルへ反映するには「モデル保存」を押してください。"
                    )
                self._set_train_ui_state(status_text=f"状態: 学習完了 ({elapsed}s)")
                self._reset_train_progress_bar()
                self._refresh_train_dataset_advice()
                continue
            self._train_log(msg)

    def _make_train_action_button(self, kind: str, handler: Callable[[], None]) -> QPushButton:
        meta = _TRAIN_ACTION_META[kind]
        btn = QPushButton(meta["title"])
        btn.setProperty("train_action_kind", kind)
        btn.setToolTip(f"{meta['tooltip']}\n{meta['subtitle']}")
        btn.setFixedHeight(34)
        btn.clicked.connect(handler)
        return btn

    def _train_action_button_text(self, kind: str, *, active: bool) -> str:
        meta = _TRAIN_ACTION_META[kind]
        if active:
            return f"{meta['title']}  ▶"
        return meta["title"]

    def _train_action_button_style(self, kind: str, *, active: bool, enabled: bool) -> str:
        palettes = {
            "coin_digit": ("#FFF8E1", "#FFB300", "#E65100", "#FFE082"),
            "result_digit": ("#E3F2FD", "#42A5F5", "#0D47A1", "#90CAF9"),
            "skill": ("#E8F5E9", "#66BB6A", "#1B5E20", "#A5D6A7"),
        }
        bg, border, text, active_bg = palettes[kind]
        base = (
            " padding: 4px 10px; text-align: left;"
            " border-radius: 4px; font-size: 12px;"
        )
        if active:
            return (
                f"QPushButton {{"
                f" background: {active_bg}; color: {text};"
                f" border: 1px solid {border}; border-left: 5px solid {border};"
                f" font-weight: bold;{base}"
                f"}}"
                f"QPushButton:disabled {{"
                f" background: {active_bg}; color: {text};"
                f" border: 1px solid {border}; border-left: 5px solid {border};"
                f"}}"
            )
        if not enabled:
            return (
                "QPushButton {"
                " background: #F5F5F5; color: #9E9E9E; border: 1px solid #E0E0E0;"
                f"{base}"
                "}"
            )
        hover_bg = {
            "coin_digit": "#FFECB3",
            "result_digit": "#BBDEFB",
            "skill": "#C8E6C9",
        }[kind]
        return (
            f"QPushButton {{"
            f" background: {bg}; color: {text}; border: 1px solid {border};"
            f"{base}"
            f"}}"
            f"QPushButton:hover {{"
            f" background: {hover_bg}; border: 2px solid {border};"
            f" padding: 3px 9px;"
            f"}}"
        )

    def _refresh_train_action_buttons(self) -> None:
        task = getattr(self, "_train_busy_task", "")
        buttons = (
            ("coin_digit", getattr(self, "train_coin_digit_save_button", None)),
            ("result_digit", getattr(self, "train_result_digit_save_button", None)),
            ("skill", getattr(self, "train_skill_only_button", None)),
        )
        for kind, btn in buttons:
            if btn is None or not _is_alive_qobject(btn):
                continue
            active = self.train_busy and task == kind
            btn.setEnabled(not self.train_busy)
            btn.setText(self._train_action_button_text(kind, active=active))
            btn.setStyleSheet(
                self._train_action_button_style(
                    kind, active=active, enabled=not self.train_busy
                )
            )

    def _set_train_ui_state(self, status_text: Optional[str] = None) -> None:
        if hasattr(self, "train_start_button"):
            self.train_start_button.setEnabled(not self.train_busy)
        if hasattr(self, "train_save_button"):
            self.train_save_button.setEnabled(not self.train_busy)
        if hasattr(self, "train_stop_button"):
            self.train_stop_button.setEnabled(self.train_busy)
        if hasattr(self, "train_dedup_button"):
            self.train_dedup_button.setEnabled(not self.train_busy)
        if hasattr(self, "train_status_label"):
            if status_text is not None:
                self.train_status_label.setText(status_text)
            else:
                self.train_status_label.setText("状態: 学習中..." if self.train_busy else "状態: 待機中")
            if self.train_busy:
                self.train_status_label.setStyleSheet("color: #1565C0; font-weight: bold;")
            else:
                self.train_status_label.setStyleSheet("")
        self._refresh_train_action_buttons()
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar) and self.train_busy:
            bar.setRange(0, 0)
            status = self.train_status_label.text() if hasattr(self, "train_status_label") else ""
            task = getattr(self, "_train_busy_task", "")
            meta = _TRAIN_ACTION_META.get(task)
            if meta is not None:
                bar.setFormat(meta["progress"])
                bar.setStyleSheet(
                    f"QProgressBar::chunk {{ background-color: {meta['bar_color']}; }}"
                )
            elif "コイン桁" in status:
                bar.setFormat("コイン桁 CNN 学習・保存中…")
                bar.setStyleSheet("QProgressBar::chunk { background-color: #FFB300; }")
            elif "結果桁" in status:
                bar.setFormat("結果桁 CNN 学習・保存中…")
                bar.setStyleSheet("QProgressBar::chunk { background-color: #42A5F5; }")
            elif "保存" in status:
                bar.setFormat("保存中…")
                bar.setStyleSheet("")
            else:
                bar.setFormat("学習処理中…")
                bar.setStyleSheet("")
            bar.setVisible(True)

    def _on_create_use_tsum_clicked(self) -> None:
        name = self.use_tsum_name_input.text().strip() if hasattr(self, "use_tsum_name_input") else ""
        dir_name = self.use_tsum_dir_input.text().strip() if hasattr(self, "use_tsum_dir_input") else ""
        if not name or not dir_name:
            self._train_log("使用ツム追加エラー: 表示名とdir名を入力してください。")
            return

        model_dir = self.project_root / "app/models/use_tsum" / dir_name
        image_dir = self.project_root / "app/assets/images/use_tsums" / dir_name
        skill_model_dir = self.project_root / "app/models/skill" / dir_name
        skill_image_dir = self.project_root / "app/assets/images/skills" / dir_name
        skill_activation_dir = skill_image_dir / SKILL_IMAGE_CATEGORY_ACTIVATION
        model_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        skill_model_dir.mkdir(parents=True, exist_ok=True)
        skill_activation_dir.mkdir(parents=True, exist_ok=True)

        registry_path = self.project_root / "app/models/use_tsum/registry.json"
        registry: dict[str, str] = {}
        if registry_path.exists():
            try:
                loaded = json.loads(registry_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    registry = {str(k): str(v) for k, v in loaded.items()}
            except Exception:
                registry = {}
        registry[dir_name] = name
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

        self._train_log(f"使用ツム追加: {name} ({dir_name})")
        self._train_log(f"作成: {model_dir}")
        self._train_log(f"作成: {image_dir}")
        self._train_log(f"作成: {skill_model_dir}")
        self._train_log(f"作成: {skill_activation_dir} （発動時画像用）")
        self._train_log(f"紐付け保存: {registry_path}")
        self.use_tsum_classifier.reload()
        self.skill_classifier_pool.reload()

    def _update_playback_indicators(self, position_ms: int) -> None:
        if not hasattr(self, "seek_slider"):
            return
        if self.analysis_running:
            if position_ms - self._playback_ui_last_ms < 300:
                return
            self._playback_ui_last_ms = position_ms
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(max(position_ms, 0))
        self.seek_slider.blockSignals(False)

        current_text = self._format_time(position_ms)
        total_text = self._format_time(self.media_duration_ms)
        self.time_label.setText(f"{current_text} / {total_text}")

        current_frame = int((max(position_ms, 0) / 1000.0) * self.estimated_fps)
        total_frame = int((self.media_duration_ms / 1000.0) * self.estimated_fps)
        self.frame_label.setText(f"{current_frame} / {total_frame}")

    def _format_time(self, milliseconds: int) -> str:
        total_seconds = max(milliseconds, 0) // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def _show_about(self) -> None:
        QMessageBox.information(
            self,
            "Analyzer_TsumTsum について",
            "Analyzer_TsumTsum\nPySide6製デスクトップアプリの土台です。",
        )

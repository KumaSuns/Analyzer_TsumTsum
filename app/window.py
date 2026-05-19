import os
import platform
import json
import colorsys
import threading
import queue
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QRect, Qt, QRunnable, QThreadPool, QUrl, QSettings, Signal, QTimer
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
from app.services.image_save import prepare_training_image, save_training_png
from app.services.scene_model import SceneCentroidModel
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
from app.services.file_video import FileVideoSource, is_file_video_available

# 解析テスト: item〜result を「シーンに入った瞬間」で確認モーダルする対象にできる（none 除く）
_ANALYSIS_SCENE_CONFIRM_LABELS = tuple(c for c in SimpleTrainer.CLASSES if c != "none")

_SCENE_CONFIRM_PREVIEW_MAX_W = 560
_SCENE_CONFIRM_PREVIEW_MAX_H = 315


class _AsyncPngSaveSignals(QObject):
    finished = Signal(bool, str)


class _AsyncPngSaveTask(QRunnable):
    def __init__(self, image: QImage, path: Path) -> None:
        super().__init__()
        self._image = image
        self._path = path
        self.signals: Optional[_AsyncPngSaveSignals] = None

    def run(self) -> None:
        from app.services.image_save import PNG_SAVE_COMPRESSION_FAST

        ok = self._image.save(str(self._path), "PNG", PNG_SAVE_COMPRESSION_FAST)
        if self.signals is not None:
            self.signals.finished.emit(ok, str(self._path))


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


def _media_error_extra_hints(error_string: str) -> str:
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
    if bl == "windows":
        lines.append(
            "WMF(windows) は形式によっては未対応になります。**まず ffmpeg を試してください**（下の1行目）。\n"
        )
    lines.append(f'ffmpeg で試す: $env:QT_MEDIA_BACKEND = "{other}"; python -m app.main\n')
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
        self._rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rubber_band.setStyleSheet("border: 2px solid #00E5FF; background: rgba(0, 229, 255, 40);")
        self._drag_origin = QPoint()
        self._crop_enabled = False
        self._selected_rect = QRect()
        self._source_width = 0
        self._source_height = 0

    def set_crop_enabled(self, enabled: bool) -> None:
        self._crop_enabled = enabled
        if not enabled:
            self._rubber_band.hide()

    def set_source_size(self, width: int, height: int) -> None:
        self._source_width = max(0, width)
        self._source_height = max(0, height)

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

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._crop_enabled and event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            # Only allow cropping when press starts inside displayed video area.
            if self._content_rect().contains(pos):
                self._drag_origin = pos
                self._rubber_band.setGeometry(QRect(self._drag_origin, self._drag_origin))
                self._rubber_band.show()
            # In crop mode, never treat click as play/pause toggle.
            super().mousePressEvent(event)
            return
        elif event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._crop_enabled and self._rubber_band.isVisible():
            current = event.position().toPoint()
            self._rubber_band.setGeometry(QRect(self._drag_origin, current).normalized())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._crop_enabled and event.button() == Qt.MouseButton.LeftButton and self._rubber_band.isVisible():
            selected = self._rubber_band.geometry()
            content_rect = self._content_rect()
            clipped = selected.intersected(content_rect)
            if clipped.width() > 2 and clipped.height() > 2 and content_rect.width() > 0 and content_rect.height() > 0:
                self._selected_rect = clipped
                self._rubber_band.setGeometry(clipped)
                self._rubber_band.show()
                nx = (clipped.x() - content_rect.x()) / content_rect.width()
                ny = (clipped.y() - content_rect.y()) / content_rect.height()
                nw = clipped.width() / content_rect.width()
                nh = clipped.height() / content_rect.height()
                self.cropSelected.emit(nx, ny, nw, nh)
                self.update()
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self._selected_rect.isNull():
            return
        painter = QPainter(self)
        pen = QPen(QColor("#00E5FF"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self._selected_rect)

    def set_selected_normalized_rect(self, nx: float, ny: float, nw: float, nh: float) -> None:
        content_rect = self._content_rect()
        if content_rect.width() <= 0 or content_rect.height() <= 0:
            return
        x = int(content_rect.x() + nx * content_rect.width())
        y = int(content_rect.y() + ny * content_rect.height())
        w = int(nw * content_rect.width())
        h = int(nh * content_rect.height())
        self._selected_rect = QRect(x, y, w, h).intersected(content_rect)
        self._rubber_band.setGeometry(self._selected_rect)
        self._rubber_band.show()
        self.update()

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
        self.video_analyzer = VideoAnalyzer(sample_every_frames=10)
        self.analysis_running = False
        self.analysis_warmup_until_ms = 0
        self.analysis_frame_seq = 0
        self._analysis_log_scroll_counter = 0
        self._analysis_confirm_edge_prev = ""
        self.flow_phase = "WAIT_ITEM"
        self.flow_game_index = 1
        self.timeup_confirm_count = 0
        self._fever_raw_streak = 0
        self._last_raw_scene_in_game = ""
        self._fever_count = 0
        self._fever_episode_active = False
        self._skill_count = 0
        self._skill_episode_active = False
        self._skill_off_streak = 0
        self._skill_raw_streak = 0
        self.locked_item_targets: list[str] = []
        self.locked_use_tsum = "-"
        self.locked_item_fixed = False
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
            ("UseTsum", "use_tsum"),
        ]
        self.crop_target_display = {key: display for display, key in self.crop_targets}
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
        self.train_started_at = 0.0
        self.use_opencv_for_video = False
        self.cv_source: Optional[FileVideoSource] = FileVideoSource() if is_file_video_available() else None
        self._cv_position_ms = 0
        self._cv_playing = False
        self._cv_playback_rate = 1.0
        self.cv_play_timer = QTimer(self)
        self.cv_play_timer.timeout.connect(self._cv_timer_tick)
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
            f"Version_00_00_01  {self.os_name}  {self.compute_status}",
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
        self._render_feature_ui(feature_id)

    def _on_video_tool_quick_mode_clicked(self, mode_id: int) -> None:
        """動画ツール: 1=右列・トリミング、2=右列・シーン保存、3〜6=右列を空に。"""
        if self.button_group.checkedId() != 2:
            return
        stack = getattr(self, "_video_tool_right_stack", None)
        if stack is None or not _is_alive_qobject(stack):
            return
        if mode_id == 1:
            stack.setCurrentIndex(0)
            stack.setVisible(True)
        elif mode_id == 2:
            stack.setCurrentIndex(1)
            stack.setVisible(True)
        else:
            stack.setVisible(False)

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
        """動画ツール2: スキル発動画像を skills/<dir>/<種類>/ へ保存（非同期）。"""
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
        out_dir = self._skills_images_root() / str(tsum_dir) / str(category)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_index = int((max(self._playback_position_ms(), 0) / 1000.0) * float(self.estimated_fps or 30.0))
        out_path = out_dir / f"skill_{category}_{timestamp}_f{frame_index}.png"
        rel_msg = f"skills/{tsum_dir}/{category}/{out_path.name}"

        prepared = prepare_training_image(image)
        if prepared.isNull():
            if status is not None and _is_alive_qobject(status):
                status.setStyleSheet("color: #C62828;")
                status.setText("画像の準備に失敗しました。")
            return

        if save_btn is not None and _is_alive_qobject(save_btn):
            save_btn.setEnabled(False)
        if status is not None and _is_alive_qobject(status):
            status.setStyleSheet("color: #555;")
            status.setText("スキル保存中…")

        signals = _AsyncPngSaveSignals()
        task = _AsyncPngSaveTask(prepared, out_path)

        def _on_skill_save_done(ok: bool, path_str: str) -> None:
            if save_btn is not None and _is_alive_qobject(save_btn):
                save_btn.setEnabled(True)
            if status is None or not _is_alive_qobject(status):
                return
            if ok:
                status.setStyleSheet("color: #2E7D32;")
                status.setText(f"スキル保存 → {rel_msg}")
            else:
                status.setStyleSheet("color: #C62828;")
                status.setText(f"スキル保存失敗: {path_str}")

        signals.finished.connect(_on_skill_save_done)
        task.signals = signals
        QThreadPool.globalInstance().start(task)

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
        if hasattr(self, "item_tsum_combo"):
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
        split = self._choose_scene_train_val_split(choice)
        out_dir = self.project_root / "app/assets/images" / split / choice
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_index = int((max(self._playback_position_ms(), 0) / 1000.0) * float(self.estimated_fps or 30.0))
        out_path = out_dir / f"scene_{choice}_{timestamp}_f{frame_index}.png"
        if save_training_png(image, out_path):
            return True, f"{split}/{choice}/{out_path.name}"
        return False, str(out_path)

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
            self.sample_frame_spin.setValue(10)
            self.sample_frame_spin.setFixedHeight(compact_h)
            self.sample_frame_spin.valueChanged.connect(self._on_sample_frame_changed)
            sample_row_layout.addWidget(QLabel("間隔"))
            sample_row_layout.addWidget(self.sample_frame_spin, 1)
            self.left_layout.addWidget(sample_row)

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

            self.left_layout.addWidget(QLabel("シーン判定の確認（入った瞬間・モーダル）"))
            confirm_wrap = QWidget()
            confirm_layout = QVBoxLayout(confirm_wrap)
            confirm_layout.setContentsMargins(0, 0, 0, 0)
            confirm_layout.setSpacing(1)
            self.analysis_scene_confirm_checks = {}
            for scene_key in _ANALYSIS_SCENE_CONFIRM_LABELS:
                cb = QCheckBox(scene_key)
                cb.setChecked(False)
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
                target_grid_layout = QHBoxLayout(target_grid)
                target_grid_layout.setContentsMargins(0, 0, 0, 0)
                target_grid_layout.setSpacing(2)
                for display, key in self.crop_targets:
                    btn = QPushButton(display)
                    btn.setCheckable(True)
                    btn.setFixedHeight(22)
                    btn.setStyleSheet(
                        "QPushButton { border:1px solid #888; background:#F5F5F5; padding:0 6px; }"
                        "QPushButton:checked { background:#FFD54F; border:1px solid #C9A227; }"
                    )
                    btn.toggled.connect(self._on_crop_target_selection_changed)
                    self.crop_target_buttons[key] = btn
                    target_grid_layout.addWidget(btn)
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
                    "左の「2」で表示。シーンはクラスを選んで「画像を保存」。"
                    "スキルはツム・種類を選んで「スキル発動画像を保存」。"
                )
                self._video_tool_scene_status.setWordWrap(True)
                self._video_tool_scene_status.setStyleSheet("color: #555;")
                sv.addWidget(self._video_tool_scene_status)
                sv.addStretch(1)

                self._video_tool_right_stack = QStackedWidget()
                self._video_tool_right_stack.addWidget(trim_page)
                self._video_tool_right_stack.addWidget(scene_page)
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
            video_tool_btn_row.setMaximumWidth(88)
            btn_row_layout = QVBoxLayout(video_tool_btn_row)
            btn_row_layout.setContentsMargins(4, 4, 4, 4)
            btn_row_layout.setSpacing(4)
            self._video_tool_mode_group = QButtonGroup(self)
            self._video_tool_mode_group.setExclusive(True)
            for i in range(6):
                btn = QPushButton(str(i + 1))
                btn.setCheckable(True)
                btn.setFixedHeight(28)
                btn.setStyleSheet(video_tool_btn_style)
                if i == 0:
                    btn.setToolTip("右列: トリミング・位置保存など")
                elif i == 1:
                    btn.setToolTip("右列: シーン画像・スキル発動画像の保存")
                else:
                    btn.setToolTip("右列を空にする（未割当）")
                btn_row_layout.addWidget(btn)
                self._video_tool_mode_group.addButton(btn, i + 1)
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
            self.counter_frame.setFixedHeight(72)
            self.counter_frame.setStyleSheet("border: none; background: transparent;")
            counter_layout = QVBoxLayout(self.counter_frame)
            counter_layout.setContentsMargins(0, 0, 0, 0)
            counter_layout.setSpacing(2)
            self.counter_use_tsum_label = QLabel("使用ツム: --")
            self.counter_use_item_label = QLabel("使用アイテム: --")
            self.counter_fever_count_label = QLabel("fever回数: 0")
            self.counter_skill_count_label = QLabel("スキル回数: 0")
            self.counter_analysis_state_label = QLabel("解析状態: 停止")
            self.counter_progress_label = QLabel("進行: --")
            counter_layout.addWidget(self.counter_use_tsum_label)
            counter_layout.addWidget(self.counter_use_item_label)
            counter_layout.addWidget(self.counter_fever_count_label)
            counter_layout.addWidget(self.counter_skill_count_label)
            counter_layout.addWidget(self.counter_analysis_state_label)
            counter_layout.addWidget(self.counter_progress_label)
            self.right_layout.addWidget(self.counter_frame)
            self.right_layout.addWidget(self.log_view, 1)
        if feature_id in (1, 2):
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
                    labels = ["仮1", "解析", "仮3", "仮4"] if feature_id == 1 else ["仮1", "仮2", "仮3", "仮4"]
                    for label in labels:
                        button = QPushButton(label, section)
                        button.setStyleSheet(analyze_row_button_style)
                        button.setFixedSize(88 if label == "解析" else 64, 24)
                        if label == "解析":
                            self.analyze_button = button
                            self.analyze_button.setText("解析開始")
                            button.clicked.connect(self._on_analyze_clicked)
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

            train_page = QFrame()
            train_page_layout = QVBoxLayout(train_page)
            train_page_layout.setContentsMargins(12, 12, 12, 12)
            train_page_layout.setSpacing(8)
            train_page_layout.addWidget(QLabel("学習ページ (button3)"))
            data_check_button = QPushButton("データ確認")
            data_check_button.clicked.connect(self._on_train_data_check_clicked)
            train_page_layout.addWidget(data_check_button)

            train_start_button = QPushButton("学習開始")
            train_start_button.clicked.connect(self._on_train_start_clicked)
            self.train_start_button = train_start_button
            train_page_layout.addWidget(train_start_button)

            train_stop_button = QPushButton("学習停止")
            train_stop_button.clicked.connect(self._on_train_stop_clicked)
            self.train_stop_button = train_stop_button
            train_page_layout.addWidget(train_stop_button)

            train_save_button = QPushButton("モデル保存")
            train_save_button.clicked.connect(self._on_train_save_clicked)
            self.train_save_button = train_save_button
            train_page_layout.addWidget(train_save_button)
            self.train_status_label = QLabel("状態: 待機中")
            train_page_layout.addWidget(self.train_status_label)
            self.train_progress_bar = QProgressBar()
            self.train_progress_bar.setFixedHeight(14)
            self.train_progress_bar.setRange(0, 100)
            self.train_progress_bar.setValue(0)
            self.train_progress_bar.setTextVisible(True)
            self.train_progress_bar.setFormat("")
            self.train_progress_bar.setVisible(False)
            train_page_layout.addWidget(self.train_progress_bar)

            train_page_layout.addWidget(QLabel("使用ツム登録"))
            use_tsum_row_1 = QFrame()
            use_tsum_row_1_layout = QHBoxLayout(use_tsum_row_1)
            use_tsum_row_1_layout.setContentsMargins(0, 0, 0, 0)
            use_tsum_row_1_layout.setSpacing(6)
            use_tsum_row_1_layout.addWidget(QLabel("表示名"))
            self.use_tsum_name_input = QLineEdit()
            self.use_tsum_name_input.setPlaceholderText("例: ナミネ")
            use_tsum_row_1_layout.addWidget(self.use_tsum_name_input, 1)
            train_page_layout.addWidget(use_tsum_row_1)

            use_tsum_row_2 = QFrame()
            use_tsum_row_2_layout = QHBoxLayout(use_tsum_row_2)
            use_tsum_row_2_layout.setContentsMargins(0, 0, 0, 0)
            use_tsum_row_2_layout.setSpacing(6)
            use_tsum_row_2_layout.addWidget(QLabel("dir名"))
            self.use_tsum_dir_input = QLineEdit()
            self.use_tsum_dir_input.setPlaceholderText("例: namine")
            use_tsum_row_2_layout.addWidget(self.use_tsum_dir_input, 1)
            train_page_layout.addWidget(use_tsum_row_2)

            create_use_tsum_button = QPushButton("使用ツム追加")
            create_use_tsum_button.clicked.connect(self._on_create_use_tsum_clicked)
            train_page_layout.addWidget(create_use_tsum_button)

            train_page_layout.addWidget(QLabel("スキル発動モデル（ツム別）"))
            skill_train_hint = QLabel(
                "画像: app/assets/images/skills/<dir>/activation/\n"
                "「モデル保存」で使用ツム・スキル・シーンをまとめて学習します。"
                "スキルだけなら学習開始は不要です。"
            )
            skill_train_hint.setWordWrap(True)
            skill_train_hint.setStyleSheet("color: #444;")
            train_page_layout.addWidget(skill_train_hint)
            skill_save_only_btn = QPushButton("スキル・使用ツムだけ再学習")
            skill_save_only_btn.setToolTip("シーンを触らず use_tsum / skill モデルだけ更新")
            skill_save_only_btn.clicked.connect(self._on_train_skill_only_clicked)
            train_page_layout.addWidget(skill_save_only_btn)

            self.center_layout.addWidget(train_page, 1)
            self._set_train_ui_state()

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

    def _switch_to_qt_video_and_load(self, resolved: str) -> bool:
        """OpenCV / imageio 経路が失敗したとき、同一ファイルを Qt Multimedia で開き直す。"""
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
                if self._switch_to_qt_video_and_load(resolved):
                    if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                        self.log_view.append(
                            f"ファイル動画モードで開けなかったため、Qt で再試行しました: {resolved}"
                        )
                    return
                if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                    self.log_view.append(f"動画を開けませんでした: {resolved}")
                drive_hint = ""
                if "マイドライブ" in resolved or "My Drive" in resolved:
                    drive_hint = (
                        "\n\nマイドライブ上のファイルは、開くときにバックグラウンドで順次ダウンロードします。"
                        "初回は容量・回線状況により数十秒〜かかることがあります。別アプリでファイルを開きっぱなしにしていると失敗することがあります。"
                    )
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Icon.Warning)
                box.setWindowTitle("動画を開けません")
                box.setText(
                    "動画の取り込みまたはデコードに失敗しました（順次読み・ffmpeg コピー・OpenCV・FFmpeg・H.264 トランスコードを試行済み）。"
                    "Qt への切り替えもできないか、Qt でも読み込めませんでした。"
                    "ネットワーク・ディスク容量・他アプリのロックを確認してください。"
                    + drive_hint
                )
                if detail.strip():
                    box.setDetailedText(detail.strip())
                    QApplication.clipboard().setText(detail.strip())
                    box.setInformativeText("詳細は「詳細を表示」、またはクリップボードにコピー済みです。")
                box.exec()
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
                if self._switch_to_qt_video_and_load(resolved):
                    if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                        self.log_view.append(
                            f"最初のフレーム取得に失敗したため、Qt で再試行しました: {resolved}"
                        )
                    return
                QMessageBox.warning(self, "動画を開けません", "最初のフレームを読み取れませんでした。")
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
        fps = max(float(self.estimated_fps), 1.0)
        self.cv_play_timer.setInterval(max(int(1000.0 / fps), 1))

    def _cv_push_frame(self, image: QImage) -> None:
        if image is None or image.isNull():
            return
        self.current_video_frame_image = image
        if self.current_video_container is not None:
            self.current_video_container.set_source_size(image.width(), image.height())
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
            lbl.setPixmap(
                QPixmap.fromImage(image).scaled(
                    tw,
                    th,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._on_player_position_changed(self._cv_position_ms)
        self._video_frame_counter += 1
        if self._video_frame_counter % 2 != 0:
            return
        self.analysis_frame_seq += 1
        if self.analysis_running:
            self._run_analysis_step(self._cv_position_ms, image)

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

    def _cv_timer_tick(self) -> None:
        if not getattr(self, "use_opencv_for_video", False) or not self._cv_playing:
            return
        if self.cv_source is None or not self.cv_source.is_open:
            return
        if self.crop_playback_lock:
            return
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
        self._cv_push_frame(img)

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

    def _on_end_of_media_cleanup(self) -> None:
        if self.analysis_running:
            self.analysis_running = False
            self.analysis_warmup_until_ms = 0
            self.analysis_frame_seq = 0
            if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
                self.counter_analysis_state_label.setText("解析状態: 完了")
            if hasattr(self, "analyze_button") and _is_alive_qobject(self.analyze_button):
                self.analyze_button.setText("解析開始")
            if hasattr(self, "log_view"):
                self.log_view.append("解析完了: 動画終端に到達しました。")
        self._update_step_buttons_enabled()

    def _on_player_error(self, error, error_string: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        msg = (error_string or "").strip() or "動画を読み込めませんでした（コーデックまたは形式の問題の可能性があります）。"
        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
            self.log_view.append(f"メディアエラー: {msg}")
        hints = _media_error_extra_hints(msg)
        details = f"{msg}{hints}\n{_qt_multimedia_environment_report()}"
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
        if hasattr(self, "counter_progress_label") and _is_alive_qobject(self.counter_progress_label):
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
        if hasattr(self, "analyze_button") and _is_alive_qobject(self.analyze_button):
            self.analyze_button.setText("解析開始")

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
                if hasattr(self, "analyze_button") and _is_alive_qobject(self.analyze_button):
                    self.analyze_button.setText("解析開始")

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

    def _step_backward(self, frames: int) -> None:
        if not self._is_player_paused():
            return
        step_ms = int((frames / self.estimated_fps) * 1000)
        target = max(self._playback_position_ms() - step_ms, 0)
        if getattr(self, "use_opencv_for_video", False):
            self._cv_seek_and_show(target)
        else:
            self.player.setPosition(target)

    def _set_playback_rate(self, rate: float) -> None:
        self._cv_playback_rate = max(rate, 0.05)
        if not getattr(self, "use_opencv_for_video", False):
            self.player.setPlaybackRate(rate)
        elif self._cv_playing:
            self._sync_cv_timer_interval()

    def _on_analyze_clicked(self) -> None:
        if self.analysis_running:
            self.analysis_running = False
            self.analysis_warmup_until_ms = 0
            self.analysis_frame_seq = 0
            self._reset_analysis_flow()
            if getattr(self, "use_opencv_for_video", False):
                self._cv_pause()
            else:
                self.player.pause()
            if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
                self.counter_analysis_state_label.setText("解析状態: 停止")
            if hasattr(self, "analyze_button") and _is_alive_qobject(self.analyze_button):
                self.analyze_button.setText("解析開始")
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
        self.video_analyzer.reload_model()
        self.video_analyzer.reset()
        self.analysis_running = True
        self.analysis_frame_seq = 0
        self._analysis_log_scroll_counter = 0
        self._analysis_confirm_edge_prev = ""
        self._reset_analysis_flow()
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
            self.counter_analysis_state_label.setText("解析状態: 実行中")
        if hasattr(self, "analyze_button") and _is_alive_qobject(self.analyze_button):
            self.analyze_button.setText("解析停止")
        if hasattr(self, "start_from_zero_check") and self.start_from_zero_check.isChecked():
            if getattr(self, "use_opencv_for_video", False):
                self._cv_seek_and_show(0)
            else:
                self.player.setPosition(0)
        # Ignore unstable first frames right after start/seek.
        self.analysis_warmup_until_ms = max(self._playback_position_ms(), 0) + 500
        if getattr(self, "use_opencv_for_video", False):
            self._cv_play()
        else:
            self.player.play()
        if hasattr(self, "log_view"):
            self.log_view.clear()
            if hasattr(self, "counter_use_item_label"):
                self.counter_use_item_label.setText("使用アイテム: --")
            self.log_view.append(
                f"解析開始: {self.video_analyzer.sample_every_frames}フレームごとに判定します。 model={self.video_analyzer.active_model_version}"
            )
            self.log_view.append(
                "モデル状態: "
                f"loaded={self.video_analyzer.scene_model_loaded} "
                f"class_count={self.video_analyzer.scene_class_count}"
            )
            if self.video_analyzer.scene_model_loaded:
                labels = sorted(self.video_analyzer.scene_classifier.model.centroids.keys())
                self.log_view.append(
                    "シーン centroid のクラス（学習で train に画像があったもののみ）: " + ", ".join(labels)
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
                self.log_view.append("警告: scene_model.json が未読込です。学習タブで学習→モデル保存を先に実施してください。")

    def _on_sample_frame_changed(self, value: int) -> None:
        self.video_analyzer.sample_every_frames = max(1, value)

    def _on_crop_selected(self, x: float, y: float, w: float, h: float) -> None:
        self.pending_crop_rect = [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
        self._update_crop_controls_from_pending()
        self._refresh_crop_preview()

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
            self._apply_pending_rect_to_video()
            self._refresh_crop_preview()
        else:
            self.crop_rect_label.setText("範囲: 未選択")

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
        for key in selected_keys:
            data[key] = self.pending_crop_rect
        path = self.project_root / "app/models/main_model/crop_positions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
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
        self._apply_pending_rect_to_video()
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
        if hasattr(self, "crop_status_label"):
            self.crop_status_label.setText("状態: 選択中" if enabled else "状態: 停止")
        if hasattr(self, "crop_start_button"):
            self.crop_start_button.setText("トリミング終了" if enabled else "トリミング開始")

    def _apply_pending_rect_to_video(self) -> None:
        if self.pending_crop_rect is None or self.current_video_container is None:
            return
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
        if self._video_frame_counter % 2 != 0:
            return
        try:
            image = frame.toImage()
        except Exception:
            return
        if image is None or image.isNull():
            return
        self.current_video_frame_image = image
        self.analysis_frame_seq += 1
        if self.current_video_container is not None:
            self.current_video_container.set_source_size(image.width(), image.height())
        if self.analysis_running:
            self._run_analysis_step(self._playback_position_ms(), image)

    def _run_analysis_step(self, position_ms: int, frame_image) -> None:
        if position_ms < self.analysis_warmup_until_ms:
            return
        # crop_positions は解析開始時に読み込み済み（毎フレームの disk I/O を避ける）
        selected_tsum = "auto"
        if hasattr(self, "item_tsum_combo"):
            selected_tsum = self.item_tsum_combo.currentText()
        use_tsum_dir = ""
        if self.locked_item_fixed:
            use_tsum_dir = self._resolve_tsum_dir(self.locked_use_tsum)
        elif selected_tsum != "auto":
            use_tsum_dir = self._resolve_tsum_dir(selected_tsum)
        results = self.video_analyzer.process_frame(
            self.analysis_frame_seq,
            position_ms,
            selected_tsum,
            frame_image,
            use_tsum_dir=use_tsum_dir,
        )
        for result in results:
            evaluations: dict[str, dict] = {}
            item_debug = ""
            if self.flow_phase == "WAIT_ITEM":
                selected_targets, evaluations = self._evaluate_targets(frame_image, self.crop_positions_for_analysis)
                used_item_keys = [k for k in selected_targets if k != "use_tsum"]
                item_detected = len(used_item_keys) > 0
                use_tsum_detected = "-"
                if selected_tsum == "auto":
                    use_tsum_detected = self._detect_use_tsum(frame_image)
                else:
                    use_tsum_detected = selected_tsum
                scene_label = self._apply_scene_flow(result.scene_label, item_detected, use_tsum_detected)
                if scene_label == "item":
                    # Fix use-item/use-tsum for current game.
                    self.locked_item_targets = used_item_keys
                    self.locked_use_tsum = use_tsum_detected
                    self.locked_item_fixed = True
                item_debug = self._format_item_debug(evaluations, selected_targets)
            else:
                # After item is established, never re-detect until next game.
                selected_targets = list(self.locked_item_targets)
                item_detected = self.locked_item_fixed
                use_tsum_detected = self.locked_use_tsum if self.locked_item_fixed else "-"
                scene_label = self._apply_scene_flow(result.scene_label, item_detected, use_tsum_detected)
                item_debug = "item_lock:ON"
            result.scene_label = scene_label
            skill_tsum_dir = (
                self._resolve_tsum_dir(self.locked_use_tsum) if self.locked_item_fixed else use_tsum_dir
            )
            skill_new = False
            skill_log_dir = ""
            if self.locked_item_fixed and skill_tsum_dir and self.flow_phase == "IN_GAME":
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
            self._maybe_modal_scene_confirm(scene_label, frame_image)
            self._append_analysis_log(
                result,
                selected_targets,
                use_tsum_detected,
                item_detected,
                item_debug,
                skill_detected=skill_new,
                skill_tsum_dir=skill_log_dir,
            )
            if scene_label == "timeup":
                self.analysis_running = False
                self.analysis_warmup_until_ms = 0
                self.analysis_frame_seq = 0
                if getattr(self, "use_opencv_for_video", False):
                    self._cv_pause()
                else:
                    self.player.pause()
                if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
                    self.counter_analysis_state_label.setText("解析状態: 完了(timeup)")
                if hasattr(self, "analyze_button") and _is_alive_qobject(self.analyze_button):
                    self.analyze_button.setText("解析開始")
                if hasattr(self, "log_view"):
                    self.log_view.append("解析完了: timeupを検知したため停止しました。")
                break

    def _reset_analysis_flow(self) -> None:
        self.flow_phase = "WAIT_ITEM"
        self.flow_game_index = 1
        self.timeup_confirm_count = 0
        self._fever_raw_streak = 0
        self._last_raw_scene_in_game = ""
        self._fever_count = 0
        self._fever_episode_active = False
        self._skill_count = 0
        self._skill_episode_active = False
        self._skill_off_streak = 0
        self._skill_raw_streak = 0
        self.locked_item_targets = []
        self.locked_use_tsum = "-"
        self.locked_item_fixed = False
        if hasattr(self, "counter_fever_count_label") and _is_alive_qobject(self.counter_fever_count_label):
            self.counter_fever_count_label.setText("fever回数: 0")
        if hasattr(self, "counter_skill_count_label") and _is_alive_qobject(self.counter_skill_count_label):
            self.counter_skill_count_label.setText("スキル回数: 0")

    def _on_flow_scene_confirmed(self, scene: str) -> None:
        """1プレイ内の fever エピソードごとにカウンタを1回だけ増やす。"""
        if scene == "fever":
            if not self._fever_episode_active:
                self._fever_episode_active = True
                self._fever_count += 1
                if hasattr(self, "counter_fever_count_label") and _is_alive_qobject(self.counter_fever_count_label):
                    self.counter_fever_count_label.setText(f"fever回数: {self._fever_count}")
        elif scene in ("timeup", "bonus", "result"):
            self._fever_episode_active = False

    def _apply_scene_flow(self, raw_scene: str, item_detected: bool, use_tsum_detected: str) -> str:
        """Apply game-order constraints:
        item -> ready -> go -> fever/timeup -> bonus -> result -> next game.
        """
        phase = self.flow_phase
        has_tsum = use_tsum_detected not in {"-", "unknown", ""}
        item_established = item_detected or has_tsum

        scene = "none"
        if phase == "WAIT_ITEM":
            if item_established:
                scene = "item"
                self.flow_phase = "WAIT_READY"
        elif phase == "WAIT_READY":
            if raw_scene == "ready":
                scene = "ready"
                self.flow_phase = "WAIT_GO"
        elif phase == "WAIT_GO":
            if raw_scene == "go":
                scene = "go"
                self.flow_phase = "IN_GAME"
                self._last_raw_scene_in_game = ""
        elif phase == "IN_GAME":
            if raw_scene == "fever":
                # 1フレームだけの誤分類で none→fever になるのを抑える（連続2回 raw fever で確定）
                self._fever_raw_streak = min(self._fever_raw_streak + 1, 30)
                if self._fever_raw_streak >= 2:
                    scene = "fever"
                    self.timeup_confirm_count = 0
            elif raw_scene == "timeup":
                # 直前サンプルが fever の直後に 1 回だけ timeup → 多くは誤認（その次からは raw で連続判定）
                if self._last_raw_scene_in_game == "fever":
                    scene = "fever"
                    self.timeup_confirm_count = 0
                else:
                    self.timeup_confirm_count += 1
                    if self.timeup_confirm_count >= 3:
                        scene = "timeup"
                        self.flow_phase = "WAIT_BONUS"
                        self.timeup_confirm_count = 0
            else:
                # none など誤判定の挟み込みでは streak を維持（fever 表示が出にくい問題の対策）
                if raw_scene != "none":
                    self._fever_raw_streak = 0
                self.timeup_confirm_count = 0
            self._last_raw_scene_in_game = raw_scene
        elif phase == "WAIT_BONUS":
            if raw_scene == "bonus":
                scene = "bonus"
                self.flow_phase = "WAIT_RESULT"
        elif phase == "WAIT_RESULT":
            if raw_scene == "result":
                scene = "result"
                self.flow_game_index += 1
                self.flow_phase = "WAIT_ITEM"
                self.timeup_confirm_count = 0
                self.locked_item_targets = []
                self.locked_use_tsum = "-"
                self.locked_item_fixed = False

        if hasattr(self, "counter_analysis_state_label"):
            self.counter_analysis_state_label.setText(f"解析状態: G{self.flow_game_index} {self.flow_phase}")
        self._on_flow_scene_confirmed(scene)
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

    def _exec_scene_confirm_dialog(self, frame_image, scene_label: str) -> bool:
        """プレビュー付き。はい・×閉じる＝判定で合っている。いいえ＝修正へ。"""
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
        if rc != 2:
            self._try_skill_save_from_dialog(frame_image, skill_save_cb, skill_tsum_combo, skill_cat_combo)
        return rc != 2

    def _exec_scene_correction_dialog(self, frame_image, scene_label: str) -> tuple[Optional[str], bool]:
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
            QLabel("保存するクラス（none は保存しません／他は train・val 自動）")
        )
        combo = QComboBox()
        combo.addItems(list(SimpleTrainer.CLASSES))
        cur = 0
        if scene_label in SimpleTrainer.CLASSES:
            cur = SimpleTrainer.CLASSES.index(scene_label)
        combo.setCurrentIndex(cur)
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
            self._train_log("学習中は実行できません。")
            return
        self._train_use_tsum_models()
        self._train_skill_models()
        self.skill_classifier_pool.reload()
        self.video_analyzer.item_skill_classifier.reload()
        self._train_log("スキル・使用ツムモデルの再学習が完了しました。")

    def _maybe_modal_scene_confirm(self, scene_label: str, frame_image) -> None:
        """チェックされたシーンへ遷移した最初のフレームで確認（プレビュー付き）。誤りならクラス選択して保存。"""
        checks = getattr(self, "analysis_scene_confirm_checks", None)
        if checks is None or not self.analysis_running:
            return
        watched = [
            c
            for c in _ANALYSIS_SCENE_CONFIRM_LABELS
            if c in checks and _is_alive_qobject(checks[c]) and checks[c].isChecked()
        ]
        prev = getattr(self, "_analysis_confirm_edge_prev", "")
        if scene_label not in watched:
            self._analysis_confirm_edge_prev = scene_label
            return
        if scene_label == prev:
            return

        was_cv = getattr(self, "use_opencv_for_video", False)
        was_playing = self._is_player_playing()
        if was_playing:
            if was_cv:
                self._cv_pause()
            elif hasattr(self, "player") and _is_alive_qobject(self.player):
                self.player.pause()

        try:
            if not self._exec_scene_confirm_dialog(frame_image, scene_label):
                choice, ok = self._exec_scene_correction_dialog(frame_image, scene_label)
                if ok and choice:
                    if choice == "none":
                        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                            self.log_view.append("修正: none（画像は保存しません）")
                    else:
                        save_ok, msg = self._save_frame_to_scene_class(frame_image, choice)
                        if hasattr(self, "log_view") and _is_alive_qobject(self.log_view):
                            self.log_view.append(("修正保存: " if save_ok else "修正保存失敗: ") + msg)
                        if save_ok:
                            self._train_log(f"解析修正保存: {msg}")
        finally:
            self._analysis_confirm_edge_prev = scene_label
            resume = was_playing and self.analysis_running and scene_label != "timeup"
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
        used_item_keys = [k for k in selected_targets if k != "use_tsum"]
        used_item_names = [self.crop_target_display.get(k, k) for k in used_item_keys]
        used_items_text = ",".join(used_item_names) if used_item_names else "-"
        if self.locked_item_fixed:
            locked_item_names = [self.crop_target_display.get(k, k) for k in self.locked_item_targets]
            display_used_items = ",".join(locked_item_names) if locked_item_names else "-"
            display_use_tsum = self.locked_use_tsum
        else:
            display_used_items = used_items_text if result.scene_label == "item" else "-"
            display_use_tsum = use_tsum_detected if result.scene_label == "item" else "-"
        if hasattr(self, "counter_use_item_label"):
            self.counter_use_item_label.setText(f"使用アイテム: {display_used_items}")
        if hasattr(self, "counter_use_tsum_label"):
            self.counter_use_tsum_label.setText(f"使用ツム: {display_use_tsum}")
        # Log output is scene-based (avoid noisy fixed values on non-item scenes).
        log_used_items = used_items_text if result.scene_label == "item" else "-"
        log_use_tsum = use_tsum_detected if result.scene_label == "item" else "-"
        # item_detect comes from detection flow in _on_player_position_changed.
        effective_item_detected = item_detected if result.scene_label == "item" else False
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
        else:
            short_suffix = ""
            detail_suffix = ""
        skill_label = skill_tsum_dir if skill_detected else ""
        if hasattr(self, "detail_log_check") and not self.detail_log_check.isChecked():
            self._append_log_colored_scene(prefix, scene_label, short_suffix, skill_label)
            if item_debug and result.scene_label == "item":
                self.log_view.append(f"  item_debug: {item_debug}")
            return
        self._append_log_colored_scene(prefix, scene_label, detail_suffix, skill_label)
        if item_debug and result.scene_label == "item":
            self.log_view.append(f"  item_debug: {item_debug}")

    def _append_log_colored_scene(
        self, prefix: str, scene_label: str, suffix: str, skill_label: str = ""
    ) -> None:
        """Log one line; scene name ready/go in orange via QTextCharFormat (HTML is unreliable in QTextEdit)."""
        if not hasattr(self, "log_view"):
            return
        cursor = self.log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        mono = QTextCharFormat()
        mono.setFontFamilies(["monospace"])
        orange = QTextCharFormat()
        orange.setFontFamilies(["monospace"])
        orange.setForeground(QColor("#B45309"))
        cursor.setCharFormat(mono)
        cursor.insertText(prefix)
        if scene_label in {"ready", "go"}:
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
        # 解析中は毎行 ensureCursorVisible すると QTextEdit のスクロールが重い
        if not getattr(self, "analysis_running", False):
            self.log_view.ensureCursorVisible()
        else:
            self._analysis_log_scroll_counter += 1
            if self._analysis_log_scroll_counter % 16 == 0:
                self.log_view.ensureCursorVisible()

    def _format_item_debug(self, evaluations: dict[str, dict], selected_targets: list[str]) -> str:
        parts = []
        for key in ["bomb", "five_to_four", "time"]:
            ev = evaluations.get(key)
            if ev is None:
                parts.append(f"{self.crop_target_display.get(key, key)}:NA")
                continue
            y = float(ev.get("yellow_ratio", 0.0))
            b = float(ev.get("blue_ratio", 0.0))
            mark = "ON" if key in selected_targets else "OFF"
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

    def _probe_skill_new_episode(self, frame_image, tsum_dir: str) -> Optional[float]:
        """IN_GAME 中のスキル候補。新規エピソードなら距離を返す（まだカウントしない）。"""
        if frame_image is None or frame_image.isNull() or not self.skill_classifier_pool.has_model(tsum_dir):
            return None
        skill_rect = None
        rect = self.crop_positions_for_analysis.get("skill")
        if isinstance(rect, list) and len(rect) == 4:
            try:
                skill_rect = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            except Exception:
                skill_rect = None
        raw_hit, dist = self.skill_classifier_pool.predict(tsum_dir, frame_image, skill_rect)
        if raw_hit:
            self._skill_raw_streak = min(self._skill_raw_streak + 1, 30)
            self._skill_off_streak = 0
        else:
            self._skill_raw_streak = 0
            self._skill_off_streak += 1
            if self._skill_off_streak >= 2:
                self._skill_episode_active = False
            return None
        if self._skill_raw_streak < 2:
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

    def _on_train_data_check_clicked(self) -> None:
        summary = self.trainer.summarize_dataset()
        self._train_log(f"データ確認: train={summary.train_total}, val={summary.val_total}")
        self._train_log(f"train内訳: {summary.per_class_train}")
        self._train_log(f"val内訳: {summary.per_class_val}")

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

    def _on_train_save_clicked(self) -> None:
        if self.train_busy:
            self._train_log("学習中は保存できません。完了後に実行してください。")
            return
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar):
            bar.setStyleSheet("")
            bar.setRange(0, 0)
            bar.setFormat("保存中…")
            bar.setVisible(True)
        try:
            self._set_train_ui_state(status_text="状態: モデル保存中...")
            self.trainer.save(self._train_log, version="version_1")
            self.video_analyzer.reload_model()
            self._train_use_tsum_models()
            self._train_skill_models()
            self.skill_classifier_pool.reload()
        except Exception as exc:
            self._train_log(f"モデル保存エラー: {exc}")
            self._set_train_ui_state(status_text="状態: 保存失敗")
            self._reset_train_progress_bar()
            return
        self._set_train_ui_state(status_text="状態: モデル保存完了")
        self._flash_train_save_complete()

    def _flash_train_save_complete(self) -> None:
        """保存成功をプログレスバーで明示してから消す。"""
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar):
            bar.setRange(0, 100)
            bar.setValue(100)
            bar.setFormat("保存完了")
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

    def _train_use_tsum_models(self) -> None:
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
            self._train_log("使用ツムモデル学習: 対象画像なし")
            return
        summary = ", ".join(f"{k}:{v}枚" for k, v in sorted(counts.items()))
        if use_tsum_rect is None:
            self._train_log("使用ツムモデル学習: use_tsum範囲未設定のため画像全体から作成")
        else:
            self._train_log(f"使用ツムモデル学習: use_tsum範囲で切抜き作成 {use_tsum_rect}")
        self._train_log(f"使用ツムモデル学習: {summary}")

    def _train_skill_models(self) -> None:
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
            self._train_log(
                "スキルモデル学習: 対象画像なし"
                f"（app/assets/images/skills/<dir名>/{SKILL_IMAGE_CATEGORY_ACTIVATION}/ に発動時画像）"
            )
            return
        summary = ", ".join(f"{k}:{v}枚" for k, v in sorted(counts.items()))
        if skill_rect is None:
            self._train_log("スキルモデル学習: skill範囲未設定のため画像全体から作成")
        else:
            self._train_log(f"スキルモデル学習: skill範囲で切抜き作成 {skill_rect}")
        self._train_log(f"スキルモデル学習: {summary}")

    def _on_train_timer_tick(self) -> None:
        # Legacy timer: no-op (training now runs in background thread).
        self.train_timer.stop()

    def _poll_train_messages(self) -> None:
        while not self.train_message_queue.empty():
            msg = self.train_message_queue.get()
            if msg == "__TRAIN_DONE__":
                self.train_busy = False
                self.train_poll_timer.stop()
                elapsed = int(max(0.0, time.time() - self.train_started_at))
                self._train_log(f"学習処理が完了しました。所要時間: {elapsed}s")
                self._set_train_ui_state(status_text=f"状態: 学習完了 ({elapsed}s)")
                self._reset_train_progress_bar()
                continue
            self._train_log(msg)

    def _set_train_ui_state(self, status_text: Optional[str] = None) -> None:
        if hasattr(self, "train_start_button"):
            self.train_start_button.setEnabled(not self.train_busy)
        if hasattr(self, "train_save_button"):
            self.train_save_button.setEnabled(not self.train_busy)
        if hasattr(self, "train_stop_button"):
            self.train_stop_button.setEnabled(self.train_busy)
        if hasattr(self, "train_status_label"):
            if status_text is not None:
                self.train_status_label.setText(status_text)
            else:
                self.train_status_label.setText("状態: 学習中..." if self.train_busy else "状態: 待機中")
        bar = getattr(self, "train_progress_bar", None)
        if bar is not None and _is_alive_qobject(bar) and self.train_busy:
            bar.setRange(0, 0)
            bar.setFormat("学習処理中…")
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

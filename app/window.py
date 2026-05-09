import platform
import json
import colorsys
import threading
import queue
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

from PySide6.QtCore import QPoint, QRect, Qt, QUrl, QSettings, Signal, QTimer
from PySide6.QtGui import QAction, QColor, QPainter, QPen, QPixmap, QTextCharFormat, QTextCursor
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRubberBand,
    QSlider,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.services.analyzer import VideoAnalyzer
from app.services.trainer import SimpleTrainer
from app.services.tsum_registry import TsumRegistry
from app.services.use_tsum_classifier import UseTsumClassifier


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
        self.video_widget.setGeometry(QRect(x, y, best_w, best_h))

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
        video_rect = self.video_widget.geometry()
        local_rect = QRect(
            self._selected_rect.x() - video_rect.x(),
            self._selected_rect.y() - video_rect.y(),
            self._selected_rect.width(),
            self._selected_rect.height(),
        )
        snapshot = self.video_widget.grab(local_rect)
        return snapshot

    def _content_rect(self) -> QRect:
        video_rect = self.video_widget.geometry()
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
        self.flow_phase = "WAIT_ITEM"
        self.flow_game_index = 1
        self.timeup_confirm_count = 0
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

    def _render_feature_ui(self, feature_id: int) -> None:
        self._clear_layout(self.left_layout)
        self._clear_layout(self.center_layout)
        self._clear_layout(self.right_layout)
        if feature_id == 2:
            self._release_counter_and_log_widgets()

        # 動画ツール(tab2)のみ: 左=クイックボタン列、右=トリミングのみ（列幅入替）。ログ・カウンターは置かない。
        if feature_id == 2:
            self._left_section_frame.setFixedWidth(520)
            self._right_section_frame.setFixedWidth(300)
        else:
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
            self.left_layout.addStretch(1)
        else:
            if feature_id == 2:
                self.right_layout.setContentsMargins(2, 2, 2, 2)
                self.right_layout.setSpacing(2)
                self.right_layout.addWidget(QLabel("トリミング"))
                self.right_layout.addWidget(QLabel("対象(複数選択)"))
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
                self.right_layout.addWidget(target_grid)
                # Default selection to avoid "saved but no target selected" confusion.
                if "score" in self.crop_target_buttons:
                    self.crop_target_buttons["score"].setChecked(True)

                self.crop_rect_label = QLabel("範囲: 未選択")
                self.right_layout.addWidget(self.crop_rect_label)

                self.crop_start_button = QPushButton("トリミング開始")
                self.crop_start_button.setCheckable(True)
                self.crop_start_button.toggled.connect(self._on_toggle_crop_mode)
                self.right_layout.addWidget(self.crop_start_button)

                self.crop_status_label = QLabel("状態: 停止")
                self.right_layout.addWidget(self.crop_status_label)

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
                self.right_layout.addWidget(adjust_row1)

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
                self.right_layout.addWidget(adjust_row2)

                save_crop_button = QPushButton("位置を保存")
                save_crop_button.clicked.connect(self._on_save_crop_clicked)
                self.right_layout.addWidget(save_crop_button)
                save_dataset_button = QPushButton("対象画像保存(自動train/val)")
                save_dataset_button.clicked.connect(self._on_save_target_images_clicked)
                self.right_layout.addWidget(save_dataset_button)
                check_crop_button = QPushButton("停止画で切り抜き確認")
                check_crop_button.clicked.connect(self._on_check_crop_preview_clicked)
                self.right_layout.addWidget(check_crop_button)
                self.right_layout.addWidget(QLabel("動画上をドラッグで指定"))

                self.crop_preview_label = QLabel("プレビューなし")
                self.crop_preview_label.setFixedSize(180, 100)
                self.crop_preview_label.setStyleSheet("border:1px solid #888; background:#111; color:#DDD;")
                self.crop_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.right_layout.addWidget(self.crop_preview_label)
                self.crop_check_label = QLabel("確認プレビューなし")
                self.crop_check_label.setFixedSize(280, 180)
                self.crop_check_label.setStyleSheet("border:1px solid #888; background:#111; color:#DDD;")
                self.crop_check_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.right_layout.addWidget(self.crop_check_label)
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
            )
            video_tool_btn_row = QFrame()
            video_tool_btn_row.setStyleSheet("background: transparent;")
            btn_row_layout = QHBoxLayout(video_tool_btn_row)
            btn_row_layout.setContentsMargins(4, 4, 4, 2)
            btn_row_layout.setSpacing(4)
            for i in range(6):
                btn = QPushButton(str(i + 1))
                btn.setFixedHeight(28)
                btn.setStyleSheet(video_tool_btn_style)
                btn.setToolTip(f"動画ツール クイックボタン {i + 1}（未割当）")
                btn_row_layout.addWidget(btn, 1)
            self.left_layout.addWidget(video_tool_btn_row)
            self.left_layout.addStretch(1)
        else:
            self.log_view = QTextEdit()
            self.log_view.setReadOnly(True)
            self.log_view.setAcceptRichText(True)
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
            self.player.setVideoOutput(video_container.video_widget)
            sink = video_container.video_widget.videoSink()
            if sink is not None:
                sink.videoFrameChanged.connect(self._on_video_frame_changed)
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

        dialog = QFileDialog(self, "動画ファイルを選択")
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("動画ファイル (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)")
        dialog.setDirectory(initial_path)
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)

        if self.last_video_path and Path(self.last_video_path).exists():
            dialog.selectFile(self.last_video_path)

        if not dialog.exec():
            return

        selected_files = dialog.selectedFiles()
        if not selected_files:
            return
        self._load_video(selected_files[0])

    def _load_video(self, path: str) -> None:
        self.last_video_path = path
        self.settings.setValue("last_video_dir", str(Path(path).parent))
        self.video_file_label.setText(Path(path).name)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.pause()
        self.analysis_running = False
        self.analysis_warmup_until_ms = 0
        self.analysis_frame_seq = 0
        self._reset_analysis_flow()
        self.video_analyzer.reset()
        self._on_playback_state_changed(self.player.playbackState())

    def _on_mute_toggled(self, checked: bool) -> None:
        self.audio_output.setMuted(checked)
        self.settings.setValue("is_muted", checked)

    def _on_player_duration_changed(self, duration_ms: int) -> None:
        self.media_duration_ms = max(duration_ms, 0)
        if hasattr(self, "seek_slider"):
            self.seek_slider.setMaximum(self.media_duration_ms)
        self._update_playback_indicators(self.player.position())

    def _on_player_position_changed(self, position_ms: int) -> None:
        self._update_playback_indicators(position_ms)
        if hasattr(self, "counter_progress_label"):
            frame_index = int((max(position_ms, 0) / 1000.0) * self.estimated_fps)
            self.counter_progress_label.setText(f"進行: {position_ms/1000.0:.2f}s / f{frame_index}")

    def _on_slider_moved(self, position_ms: int) -> None:
        self.player.setPosition(position_ms)

    def _on_play_pause_clicked(self) -> None:
        crop_checked = False
        if hasattr(self, "crop_start_button") and _is_alive_qobject(self.crop_start_button):
            crop_checked = self.crop_start_button.isChecked()
        if self.crop_playback_lock or crop_checked:
            # Never toggle playback while cropping.
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_stop_clicked(self) -> None:
        self.player.stop()
        self.analysis_running = False
        self.analysis_warmup_until_ms = 0
        self.analysis_frame_seq = 0
        self._reset_analysis_flow()
        if hasattr(self, "counter_analysis_state_label"):
            self.counter_analysis_state_label.setText("解析状態: 停止")
        if hasattr(self, "analyze_button"):
            self.analyze_button.setText("解析開始")

    def _on_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if self.crop_playback_lock and state == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
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

        pause_only_enabled = state == QMediaPlayer.PlaybackState.PausedState
        if hasattr(self, "step_small_left_button"):
            self.step_huge_left_button.setEnabled(pause_only_enabled)
            self.step_small_left_button.setEnabled(pause_only_enabled)
            self.step_large_left_button.setEnabled(pause_only_enabled)
            self.step_small_right_button.setEnabled(pause_only_enabled)
            self.step_large_right_button.setEnabled(pause_only_enabled)
            self.step_huge_right_button.setEnabled(pause_only_enabled)

    def _step_forward(self, frames: int) -> None:
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PausedState:
            return
        step_ms = int((frames / self.estimated_fps) * 1000)
        target = min(self.player.position() + step_ms, self.media_duration_ms)
        self.player.setPosition(target)

    def _step_backward(self, frames: int) -> None:
        if self.player.playbackState() != QMediaPlayer.PlaybackState.PausedState:
            return
        step_ms = int((frames / self.estimated_fps) * 1000)
        target = max(self.player.position() - step_ms, 0)
        self.player.setPosition(target)

    def _set_playback_rate(self, rate: float) -> None:
        self.player.setPlaybackRate(rate)

    def _on_analyze_clicked(self) -> None:
        if self.analysis_running:
            self.analysis_running = False
            self.analysis_warmup_until_ms = 0
            self.analysis_frame_seq = 0
            self._reset_analysis_flow()
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
        self.video_analyzer.reload_model()
        self.video_analyzer.reset()
        self.analysis_running = True
        self.analysis_frame_seq = 0
        self._reset_analysis_flow()
        if hasattr(self, "counter_analysis_state_label") and _is_alive_qobject(self.counter_analysis_state_label):
            self.counter_analysis_state_label.setText("解析状態: 実行中")
        if hasattr(self, "analyze_button") and _is_alive_qobject(self.analyze_button):
            self.analyze_button.setText("解析停止")
        if hasattr(self, "start_from_zero_check") and self.start_from_zero_check.isChecked():
            self.player.setPosition(0)
        # Ignore unstable first frames right after start/seek.
        self.analysis_warmup_until_ms = max(self.player.position(), 0) + 500
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
            self.log_view.append(f"トリミング読込: {list(self.crop_positions_for_analysis.keys())}")
            if not self.video_analyzer.scene_model_loaded:
                self.log_view.append("警告: scene_model.json が未読込です。button3で学習→モデル保存を先に実施してください。")

    def _on_sample_frame_changed(self, value: int) -> None:
        self.video_analyzer.sample_every_frames = max(1, value)

    def _on_crop_selected(self, x: float, y: float, w: float, h: float) -> None:
        self.pending_crop_rect = [round(x, 4), round(y, 4), round(w, 4), round(h, 4)]
        self._update_crop_controls_from_pending()
        self._refresh_crop_preview()

    def _on_crop_target_selection_changed(self) -> None:
        if not hasattr(self, "crop_target_buttons") or not hasattr(self, "crop_rect_label"):
            return
        selected = [k for k, b in self.crop_target_buttons.items() if b.isChecked()]
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
        selected_keys = [k for k, b in self.crop_target_buttons.items() if b.isChecked()]
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
        selected_keys = [k for k, b in self.crop_target_buttons.items() if b.isChecked()]
        if not selected_keys:
            self._train_log("保存失敗: 対象を1つ以上選択してください。")
            return

        crop_data = self._load_crop_positions()
        image = self.current_video_frame_image
        width = image.width()
        height = image.height()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        frame_index = int((max(self.player.position(), 0) / 1000.0) * self.estimated_fps) if hasattr(self, "player") else 0
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
            if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                count += 1
        return count

    def _on_crop_spin_changed(self, _value: float) -> None:
        if not all(
            hasattr(self, name)
            for name in ["crop_x_spin", "crop_y_spin", "crop_w_spin", "crop_h_spin"]
        ):
            return
        self.pending_crop_rect = [
            round(self.crop_x_spin.value(), 4),
            round(self.crop_y_spin.value(), 4),
            round(self.crop_w_spin.value(), 4),
            round(self.crop_h_spin.value(), 4),
        ]
        if hasattr(self, "crop_rect_label"):
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
            selected_keys = [k for k, b in self.crop_target_buttons.items() if b.isChecked()]
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
        if hasattr(self, "crop_rect_label"):
            self.crop_rect_label.setText(f"範囲: {self.pending_crop_rect}")
        for spin_name, value in [
            ("crop_x_spin", self.pending_crop_rect[0]),
            ("crop_y_spin", self.pending_crop_rect[1]),
            ("crop_w_spin", self.pending_crop_rect[2]),
            ("crop_h_spin", self.pending_crop_rect[3]),
        ]:
            if hasattr(self, spin_name):
                spin = getattr(self, spin_name)
                spin.blockSignals(True)
                spin.setValue(float(value))
                spin.blockSignals(False)

    def _on_toggle_crop_mode(self, enabled: bool) -> None:
        self.crop_playback_lock = enabled
        if enabled and hasattr(self, "player"):
            if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
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
        if not hasattr(self, "crop_preview_label"):
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
            self._run_analysis_step(self.player.position(), image)

    def _run_analysis_step(self, position_ms: int, frame_image) -> None:
        if position_ms < self.analysis_warmup_until_ms:
            return
        # Keep analysis consistent with latest saved crop settings.
        self.crop_positions_for_analysis = self._load_crop_positions()
        selected_tsum = "auto"
        if hasattr(self, "item_tsum_combo"):
            selected_tsum = self.item_tsum_combo.currentText()
        results = self.video_analyzer.process_frame(
            self.analysis_frame_seq,
            position_ms,
            selected_tsum,
            frame_image,
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
            self._append_analysis_log(result, selected_targets, use_tsum_detected, item_detected, item_debug)
            if scene_label == "timeup":
                self.analysis_running = False
                self.analysis_warmup_until_ms = 0
                self.analysis_frame_seq = 0
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
        self.locked_item_targets = []
        self.locked_use_tsum = "-"
        self.locked_item_fixed = False

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
        elif phase == "IN_GAME":
            if raw_scene == "fever":
                scene = "fever"
                self.timeup_confirm_count = 0
            elif raw_scene == "timeup":
                self.timeup_confirm_count += 1
                if self.timeup_confirm_count >= 2:
                    scene = "timeup"
                    self.flow_phase = "WAIT_BONUS"
                    self.timeup_confirm_count = 0
            else:
                self.timeup_confirm_count = 0
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

    def _append_analysis_log(
        self,
        result,
        selected_targets=None,
        use_tsum_detected: str = "-",
        item_detected: bool = False,
        item_debug: str = "",
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
            detail_suffix = f" item={result.item_skill_label}"
        if hasattr(self, "detail_log_check") and not self.detail_log_check.isChecked():
            self._append_log_colored_scene(prefix, scene_label, short_suffix)
            if item_debug and result.scene_label == "item":
                self.log_view.append(f"  item_debug: {item_debug}")
            return
        self._append_log_colored_scene(prefix, scene_label, detail_suffix)
        if item_debug and result.scene_label == "item":
            self.log_view.append(f"  item_debug: {item_debug}")

    def _append_log_colored_scene(self, prefix: str, scene_label: str, suffix: str) -> None:
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
        else:
            cursor.insertText(scene_label)
        cursor.insertText(suffix + "\n")
        self.log_view.setTextCursor(cursor)
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
        self._set_train_ui_state(status_text="状態: モデル保存中...")
        self.trainer.save(self._train_log, version="version_1")
        self._train_use_tsum_models()
        self._set_train_ui_state(status_text="状態: モデル保存完了")

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

    def _on_create_use_tsum_clicked(self) -> None:
        name = self.use_tsum_name_input.text().strip() if hasattr(self, "use_tsum_name_input") else ""
        dir_name = self.use_tsum_dir_input.text().strip() if hasattr(self, "use_tsum_dir_input") else ""
        if not name or not dir_name:
            self._train_log("使用ツム追加エラー: 表示名とdir名を入力してください。")
            return

        model_dir = self.project_root / "app/models/use_tsum" / dir_name
        image_dir = self.project_root / "app/assets/images/use_tsums" / dir_name
        model_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

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
        self._train_log(f"紐付け保存: {registry_path}")

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

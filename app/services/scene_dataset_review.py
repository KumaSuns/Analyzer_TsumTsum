"""学習データ画像をスライドショー表示しながら削除するダイアログ。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.services.scene_model import SCENE_CLASSES, iter_scene_dataset_images

PredictFn = Callable[[Path], Optional[Tuple[str, float]]]


def collect_dataset_image_paths(
    images_root: Path,
    *,
    split: str,
    cls: str,
) -> List[Path]:
    """split: train | val | both、cls: クラス名 or all。"""
    splits = ["train", "val"] if split == "both" else [split]
    classes = list(SCENE_CLASSES) if cls == "all" else [cls]
    paths: List[Path] = []
    for sp in splits:
        for name in classes:
            cls_dir = images_root / sp / name
            if not cls_dir.is_dir():
                continue
            for p in sorted(iter_scene_dataset_images(cls_dir), key=lambda x: (str(x.parent), x.name)):
                paths.append(p)
    return paths


def _path_split_class(path: Path, images_root: Path) -> Tuple[str, str]:
    try:
        rel = path.relative_to(images_root)
        parts = rel.parts
        if len(parts) >= 2:
            return parts[0], parts[1]
    except ValueError:
        pass
    return "?", "?"


class SceneDatasetReviewDialog(QDialog):
    """train/val シーン画像のレビュー・削除。"""

    def __init__(
        self,
        images_root: Path,
        *,
        predict_fn: Optional[PredictFn] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.images_root = images_root
        self.predict_fn = predict_fn
        self.deleted_count = 0
        self._paths: List[Path] = []
        self._index = 0

        self.setWindowTitle("学習画像レビュー")
        self.resize(920, 720)
        self.setMinimumSize(640, 480)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("分割"))
        self.split_combo = QComboBox()
        self.split_combo.addItem("train", "train")
        self.split_combo.addItem("val", "val")
        self.split_combo.addItem("train + val", "both")
        filter_row.addWidget(self.split_combo, 1)

        filter_row.addWidget(QLabel("クラス"))
        self.class_combo = QComboBox()
        self.class_combo.addItem("全クラス", "all")
        for name in SCENE_CLASSES:
            self.class_combo.addItem(name, name)
        filter_row.addWidget(self.class_combo, 1)

        self.misclassified_check = QCheckBox("CNN top1 がフォルダと違うものだけ")
        self.misclassified_check.setEnabled(predict_fn is not None)
        self.misclassified_check.setToolTip("保存済み CNN が読み込まれているときのみ")
        filter_row.addWidget(self.misclassified_check)
        root.addLayout(filter_row)

        self.image_label = QLabel("画像なし")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(420)
        self.image_label.setStyleSheet("background:#111; color:#AAA; border:1px solid #555;")
        root.addWidget(self.image_label, 1)

        self.info_label = QLabel("")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        root.addWidget(self.info_label)

        self.pred_label = QLabel("")
        self.pred_label.setWordWrap(True)
        self.pred_label.setStyleSheet("color: #1565C0; font-size: 11px;")
        root.addWidget(self.pred_label)

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 前 (←)")
        self.prev_btn.clicked.connect(self._go_prev)
        nav_row.addWidget(self.prev_btn)

        self.index_label = QLabel("0 / 0")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self.index_label, 1)

        self.next_btn = QPushButton("次 ▶ (→)")
        self.next_btn.clicked.connect(self._go_next)
        nav_row.addWidget(self.next_btn)
        root.addLayout(nav_row)

        action_row = QHBoxLayout()
        self.delete_btn = QPushButton("削除 (Del)")
        self.delete_btn.setStyleSheet("QPushButton { color: #B71C1C; }")
        self.delete_btn.clicked.connect(self._delete_current)
        action_row.addWidget(self.delete_btn)

        self.autoplay_check = QCheckBox("自動送り")
        action_row.addWidget(self.autoplay_check)

        action_row.addWidget(QLabel("間隔(秒)"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 30)
        self.interval_spin.setValue(2)
        action_row.addWidget(self.interval_spin)

        action_row.addStretch(1)

        close_btn = QPushButton("閉じる")
        close_btn.clicked.connect(self.accept)
        action_row.addWidget(close_btn)
        root.addLayout(action_row)

        hint = QLabel(
            "← → で移動 / Del で削除（確認あり）/ 削除後は次の画像へ。"
            " 変更後は「学習開始」→「モデル保存」で CNN に反映してください。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #555; font-size: 10px;")
        root.addWidget(hint)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_autoplay_tick)

        self.split_combo.currentIndexChanged.connect(self._reload_paths)
        self.class_combo.currentIndexChanged.connect(self._reload_paths)
        self.misclassified_check.toggled.connect(self._reload_paths)
        self.autoplay_check.toggled.connect(self._update_autoplay)
        self.interval_spin.valueChanged.connect(self._update_autoplay)

        self._reload_paths()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in (Qt.Key.Key_Left, Qt.Key.Key_A):
            self._go_prev()
            event.accept()
            return
        if key in (Qt.Key.Key_Right,):
            self._go_next()
            event.accept()
            return
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._delete_current()
            event.accept()
            return
        super().keyPressEvent(event)

    def _reload_paths(self) -> None:
        split = str(self.split_combo.currentData())
        cls = str(self.class_combo.currentData())
        paths = collect_dataset_image_paths(self.images_root, split=split, cls=cls)
        if self.misclassified_check.isChecked() and self.predict_fn is not None:
            paths = [p for p in paths if self._is_misclassified(p)]
        self._paths = paths
        self._index = min(self._index, max(len(self._paths) - 1, 0))
        if self._paths and self._index < 0:
            self._index = 0
        self._refresh_view()
        self._update_autoplay()

    def _is_misclassified(self, path: Path) -> bool:
        if self.predict_fn is None:
            return False
        split, folder_cls = _path_split_class(path, self.images_root)
        pred = self.predict_fn(path)
        if pred is None:
            return False
        top1, _score = pred
        return top1 != folder_cls

    def _refresh_view(self) -> None:
        total = len(self._paths)
        if total == 0:
            self.image_label.setText("該当画像がありません")
            self.image_label.setPixmap(QPixmap())
            self.info_label.setText("")
            self.pred_label.setText("")
            self.index_label.setText("0 / 0")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
            return

        self._index = max(0, min(self._index, total - 1))
        path = self._paths[self._index]
        split, folder_cls = _path_split_class(path, self.images_root)

        image = QImage(str(path))
        if image.isNull():
            self.image_label.setText(f"読込失敗: {path.name}")
            self.image_label.setPixmap(QPixmap())
        else:
            self.image_label.setText("")
            max_w = max(self.image_label.width() - 8, 400)
            max_h = max(self.image_label.height() - 8, 300)
            pix = QPixmap.fromImage(image).scaled(
                max_w,
                max_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(pix)

        self.info_label.setText(
            f"[{self._index + 1}/{total}]  {split}/{folder_cls}\n"
            f"{path.name}\n"
            f"{path}"
        )

        if self.predict_fn is not None:
            pred = self.predict_fn(path)
            if pred is None:
                self.pred_label.setText("CNN: 未読込")
            else:
                top1, score = pred
                mark = " ✓" if top1 == folder_cls else " ← フォルダと不一致"
                self.pred_label.setText(f"CNN top1: {top1}  (score={score:.3f}){mark}")
        else:
            self.pred_label.setText("")

        self.index_label.setText(f"{self._index + 1} / {total}")
        self.prev_btn.setEnabled(self._index > 0)
        self.next_btn.setEnabled(self._index < total - 1)
        self.delete_btn.setEnabled(True)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._paths:
            self._refresh_view()

    def _go_prev(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._refresh_view()

    def _go_next(self) -> None:
        if self._index < len(self._paths) - 1:
            self._index += 1
            self._refresh_view()

    def _delete_current(self) -> None:
        if not self._paths:
            return
        path = self._paths[self._index]
        split, folder_cls = _path_split_class(path, self.images_root)
        reply = QMessageBox.question(
            self,
            "画像を削除",
            f"削除しますか？\n\n{split}/{folder_cls}/{path.name}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            path.unlink()
        except OSError as exc:
            QMessageBox.warning(self, "削除失敗", str(exc))
            return
        self.deleted_count += 1
        del self._paths[self._index]
        if self._index >= len(self._paths) and self._index > 0:
            self._index -= 1
        self._refresh_view()

    def _update_autoplay(self) -> None:
        if self.autoplay_check.isChecked() and len(self._paths) > 1:
            self._timer.start(max(1, self.interval_spin.value()) * 1000)
        else:
            self._timer.stop()

    def _on_autoplay_tick(self) -> None:
        if len(self._paths) <= 1:
            self._timer.stop()
            return
        if self._index < len(self._paths) - 1:
            self._index += 1
        else:
            self._index = 0
        self._refresh_view()

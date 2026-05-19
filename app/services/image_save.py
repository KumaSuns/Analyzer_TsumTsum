from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

# 学習は内部で 18px 程度に縮小するため、保存も控えめに（容量・速度）
TRAINING_IMAGE_MAX_SIDE = 540
# Qt PNG: 0=高圧縮で遅い, 9=低圧縮で速い
PNG_SAVE_COMPRESSION_FAST = 9


def prepare_training_image(image: QImage, max_side: int = TRAINING_IMAGE_MAX_SIDE) -> QImage:
    """学習用保存向けに縮小（巨大スクショのコピー・PNG化を避ける）。"""
    if image.isNull():
        return QImage()
    w, h = image.width(), image.height()
    if w < 1 or h < 1:
        return QImage()
    if max(w, h) > max_side:
        if w >= h:
            nw = max_side
            nh = max(1, int(round(h * max_side / w)))
        else:
            nh = max_side
            nw = max(1, int(round(w * max_side / h)))
        image = image.scaled(
            nw,
            nh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
    return image.convertToFormat(QImage.Format.Format_RGB888)


def save_training_png(
    image: QImage,
    path: Path | str,
    *,
    max_side: int = TRAINING_IMAGE_MAX_SIDE,
) -> bool:
    prepared = prepare_training_image(image, max_side)
    if prepared.isNull():
        return False
    return prepared.save(str(path), "PNG", PNG_SAVE_COMPRESSION_FAST)

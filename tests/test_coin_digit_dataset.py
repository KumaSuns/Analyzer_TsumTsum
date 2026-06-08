"""coin_digit_dataset の単体テスト。"""
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage

from app.services.coin_digit_dataset import (
    build_saved_crop_training_samples,
    parse_crop_label,
    save_labeled_crop,
)


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


def test_parse_crop_label() -> None:
    assert parse_crop_label(Path("coin_gain_3065_20260101_123.png")) == 3065
    assert parse_crop_label(Path("coin_gain_99_x.png")) is None


def test_save_and_build_training_samples(tmp_path: Path) -> None:
    ref = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "assets"
        / "images"
        / "coin_hud_ref_5395.png"
    )
    if not ref.exists():
        return
    bgr = cv2.imread(str(ref))
    assert bgr is not None
    canvas = np.full((45, 461, 3), (18, 16, 12), np.uint8)
    rh = min(bgr.shape[0], 45)
    small = cv2.resize(bgr, (bgr.shape[1], rh))
    y0 = (45 - rh) // 2
    canvas[y0 : y0 + rh, 20 : 20 + bgr.shape[1]] = small
    image = _bgr_to_qimage(canvas)
    assets = tmp_path / "images"
    ok, msg = save_labeled_crop(image, 5395, assets_root=assets)
    assert ok, msg
    train, val = build_saved_crop_training_samples(assets)
    assert len(train) + len(val) >= 4

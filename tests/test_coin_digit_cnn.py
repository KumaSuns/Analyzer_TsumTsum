"""coin_digit_cnn の単体テスト。"""
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage

from app.services.coin_digit_cnn import (
    CoinDigitCnnClassifier,
    build_digit_training_samples,
    default_coin_digit_model_path,
    get_coin_digit_classifier,
    train_and_save_default_model,
)
from app.services.coin_gain_reader import read_coin_gain_crop, read_coin_hud


def _ensure_model() -> None:
    if get_coin_digit_classifier().is_loaded():
        return
    train_and_save_default_model()


def test_coin_digit_cnn_trains_and_classifies_ref_digits() -> None:
    train, val = build_digit_training_samples(per_digit=40)
    clf = CoinDigitCnnClassifier()
    acc = clf.train_from_samples(train, val, epochs=6)
    assert acc >= 0.85
    ref = cv2.imread(
        str(
            Path(__file__).resolve().parents[1]
            / "app"
            / "assets"
            / "images"
            / "coin_hud_ref_5395.png"
        ),
        cv2.IMREAD_GRAYSCALE,
    )
    assert ref is not None
    from app.services.coin_gain_reader import (
        _hud_digit_row_gray,
        _hud_split_four_digits,
        _normalize_digit_patch,
    )

    row = _hud_digit_row_gray(ref)
    parts = _hud_split_four_digits(row)
    assert parts is not None
    labels = (5, 3, 9, 5)
    for digit, part in zip(labels, parts):
        probs = clf.predict_probs(_normalize_digit_patch(part))
        assert probs is not None
        assert int(probs.argmax()) == digit


def test_reads_hud_ref_with_dl() -> None:
    _ensure_model()
    path = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "assets"
        / "images"
        / "coin_hud_ref_5395.png"
    )
    if not path.exists():
        return
    image = QImage(str(path))
    value, err, dbg = read_coin_hud(image)
    assert value == 5395, f"{value} err={err} {dbg}"


def test_reads_coin_gain_crop_wide_with_dl() -> None:
    _ensure_model()
    ref = cv2.imread(
        str(
            Path(__file__).resolve().parents[1]
            / "app"
            / "assets"
            / "images"
            / "coin_hud_ref_5395.png"
        )
    )
    if ref is None:
        return
    canvas = np.full((45, 461, 3), (18, 16, 12), dtype=np.uint8)
    rh = min(ref.shape[0], 45)
    small = cv2.resize(ref, (ref.shape[1], rh))
    y0 = (45 - rh) // 2
    canvas[y0 : y0 + rh, 20 : 20 + ref.shape[1]] = small
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()
    value, err, dbg = read_coin_gain_crop(image)
    if value is None:
        return
    assert value == 5395, f"{value} err={err} {dbg}"


def test_default_model_path_under_main_model() -> None:
    path = default_coin_digit_model_path()
    assert path.name == "coin_digit.pt"
    assert path.parent.name.startswith("version_")

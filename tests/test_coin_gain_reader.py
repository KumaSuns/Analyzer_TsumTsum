"""coin_gain_reader の単体テスト。"""
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtGui import QImage

from app.services.coin_gain_reader import (
    _suspicious_coin_value,
    bonus_coin_hud_rect,
    coin_gain_confirmed,
    coin_gain_confirmed_combined,
    coin_gain_confirmed_hud,
    consensus_coin_gain,
    plausible_coin_value,
    read_coin_gain,
    read_coin_gain_crop,
    read_coin_hud,
)
from app.services.coin_digit_cnn import get_coin_digit_classifier, train_and_save_default_model


def _ensure_coin_digit_model() -> None:
    if get_coin_digit_classifier().is_loaded():
        return
    train_and_save_default_model()


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()


def _make_synthetic_coin_roi(value: int, width: int = 220, height: int = 64) -> QImage:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (28, 24, 18)
    cv2.circle(canvas, (20, height // 2), 13, (0, 205, 255), -1)
    cv2.putText(
        canvas,
        str(value),
        (42, int(height * 0.72)),
        cv2.FONT_HERSHEY_DUPLEX,
        1.15,
        (0, 225, 255),
        2,
        cv2.LINE_AA,
    )
    return _bgr_to_qimage(canvas)


def _make_synthetic_coin_roi_comma(value: int, width: int = 220, height: int = 48) -> QImage:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (18, 16, 12)
    text = f"{value:,}"
    cv2.putText(
        canvas,
        text,
        (8, int(height * 0.78)),
        cv2.FONT_HERSHEY_DUPLEX,
        0.95,
        (0, 225, 255),
        2,
        cv2.LINE_AA,
    )
    return _bgr_to_qimage(canvas)


def test_suspicious_rejects_stroke_artifacts() -> None:
    assert _suspicious_coin_value(17111)
    assert _suspicious_coin_value(11111)
    assert _suspicious_coin_value(11513)
    assert _suspicious_coin_value(8885)
    assert not _suspicious_coin_value(7205)
    assert not _suspicious_coin_value(5395)


def test_bonus_coin_hud_rect_from_coin_gain() -> None:
    rect = bonus_coin_hud_rect({"coin_gain": [0.305, 0.195, 0.171, 0.055]})
    assert abs(rect[0] - 0.285) < 1e-6
    assert abs(rect[2] - 0.211) < 1e-6
    assert abs(rect[3] - 0.055) < 1e-6


def test_bonus_coin_hud_rect_prefers_coin_bonus() -> None:
    custom = [0.38, 0.215, 0.24, 0.030]
    rect = bonus_coin_hud_rect({"coin_bonus": custom, "coin_gain": [0.36, 0.215, 0.28, 0.025]})
    assert rect == tuple(custom)


def test_plausible_accepts_one_to_seven_digits() -> None:
    assert plausible_coin_value(1)
    assert plausible_coin_value(51)
    assert plausible_coin_value(859)
    assert plausible_coin_value(5395)
    assert plausible_coin_value(9_999_999)
    assert not plausible_coin_value(0)
    assert not plausible_coin_value(10_000_000)


def test_consensus_prefers_repeated_value() -> None:
    reads = [(7205, 20.0), (7205, 22.0), (17111, 5.0), (7205, 25.0)]
    assert consensus_coin_gain(reads) == 7205


def test_consensus_ignores_implausible_short_value() -> None:
    reads = [(5395, 18.0), (51, 4.0), (5395, 20.0)]
    assert consensus_coin_gain(reads) == 5395


def test_reads_sample_coin_strip() -> None:
    sample = Path("tmp_coin_debug/repro_bin3.png")
    if not sample.exists():
        return
    _ensure_coin_digit_model()
    gray = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image = _bgr_to_qimage(bgr)
    value, _err, dbg = read_coin_gain(image)
    assert value == 7205, dbg


def test_consensus_rejects_only_short_reads() -> None:
    reads = [(51, 3.0), (12, 2.0)]
    assert consensus_coin_gain(reads) is None


def test_coin_gain_not_confirmed_on_single_medium_err_read() -> None:
    assert not coin_gain_confirmed([(1376, 64.8, "strip valley_slots=4")])
    assert not coin_gain_confirmed([(7162, 36.8, "strip valley_slots=4")])
    assert not coin_gain_confirmed([(5395, 71.0, "strip n=4")])


def test_coin_gain_confirmed_on_repeated_reads() -> None:
    hud_pair = [
        (5395, 48.0, "hud_strip n=4"),
        (5395, 52.0, "hud_strip n=4"),
    ]
    assert coin_gain_confirmed_hud(hud_pair)
    gain_three = [
        (5395, 48.0, "strip n=4"),
        (5395, 52.0, "strip n=4"),
        (5395, 50.0, "n=4"),
    ]
    assert coin_gain_confirmed(gain_three)
    assert coin_gain_confirmed_combined(hud_pair, [])


def test_coin_gain_not_confirmed_on_slot_only_reads() -> None:
    slot_dbg = "ok err=36.8 strip valley_slots=4"
    assert not coin_gain_confirmed(
        [(7162, 36.8, slot_dbg), (7162, 36.8, slot_dbg), (7162, 36.8, slot_dbg)]
    )
    assert not coin_gain_confirmed_combined(
        [],
        [(7162, 36.8, slot_dbg), (7162, 36.8, slot_dbg), (7162, 36.8, slot_dbg)],
    )


def test_reads_synthetic_four_digit_value() -> None:
    _ensure_coin_digit_model()
    image = _make_synthetic_coin_roi(5395)
    value, err, dbg = read_coin_gain(image)
    assert value == 5395, f"{value} err={err} {dbg}"


def test_reads_real_hud_ref_5395_crop() -> None:
    _ensure_coin_digit_model()
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


def test_reads_coin_5395_debug_crop() -> None:
    _ensure_coin_digit_model()
    path = Path(__file__).resolve().parents[1] / "_coin_5395_crop.png"
    if not path.exists():
        return
    image = QImage(str(path))
    value, err, dbg = read_coin_hud(image)
    assert value == 5395, f"{value} err={err} {dbg}"


def test_reads_coin_gain_crop_wide_roi() -> None:
    import cv2
    import numpy as np

    _ensure_coin_digit_model()
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
    assert value == 5395, f"{value} err={err} {dbg}"


def test_reads_comma_formatted_hud_value() -> None:
    _ensure_coin_digit_model()
    image = _make_synthetic_coin_roi_comma(5395)
    value, err, dbg = read_coin_gain(image)
    assert value == 5395, f"{value} err={err} {dbg}"

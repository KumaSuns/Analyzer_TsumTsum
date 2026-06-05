"""coin_gain_reader の単体テスト。"""
from pathlib import Path

import cv2
from PySide6.QtGui import QImage

from app.services.coin_gain_reader import (
    _suspicious_coin_value,
    consensus_coin_gain,
    read_coin_gain,
)


def test_suspicious_rejects_stroke_artifacts() -> None:
    assert _suspicious_coin_value(17111)
    assert _suspicious_coin_value(11111)
    assert not _suspicious_coin_value(7205)


def test_consensus_prefers_repeated_value() -> None:
    reads = [(7205, 20.0), (7205, 22.0), (17111, 5.0), (7205, 25.0)]
    assert consensus_coin_gain(reads) == 7205


def test_reads_sample_coin_strip() -> None:
    sample = Path("tmp_coin_debug/repro_bin3.png")
    if not sample.exists():
        return
    gray = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
    assert gray is not None
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = QImage(
        rgb.data,
        rgb.shape[1],
        rgb.shape[0],
        rgb.strides[0],
        QImage.Format.Format_RGB888,
    ).copy()
    value, _err, dbg = read_coin_gain(image)
    assert value == 7205, dbg

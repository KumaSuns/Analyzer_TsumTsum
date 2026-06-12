"""result_digit_dataset の単体テスト。"""
from __future__ import annotations

from pathlib import Path

from app.services.result_digit_dataset import parse_crop_label


def test_parse_result_crop_label() -> None:
    assert parse_crop_label(Path("result_score_gain_12345_x.png")) == 12345
    assert parse_crop_label(Path("score_bonus_gain_88_x.png")) == 88
    assert parse_crop_label(Path("result_coin_gain_999_x.png")) == 999
    assert parse_crop_label(Path("coin_gain_100_x.png")) is None

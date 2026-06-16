"""result_digit_dataset の単体テスト。"""
from __future__ import annotations

from pathlib import Path

from app.services.result_digit_dataset import (
    _patch_score_acceptable,
    is_result_coin_crop,
    parse_crop_label,
)


def test_parse_result_crop_label() -> None:
    assert parse_crop_label(Path("result_score_gain_12345_x.png")) == 12345
    assert parse_crop_label(Path("score_bonus_gain_88_x.png")) == 88
    assert parse_crop_label(Path("result_coin_gain_999_x.png")) == 999
    assert parse_crop_label(Path("coin_gain_100_x.png")) is None


def test_is_result_coin_crop() -> None:
    assert is_result_coin_crop(Path("result_coin_gain_1234_x.png"))
    assert not is_result_coin_crop(Path("result_score_gain_1234_x.png"))


def test_patch_score_acceptable_relaxed_for_coin_training() -> None:
    assert _patch_score_acceptable(5, 6.0, decoded_match=False, relaxed=True)
    assert not _patch_score_acceptable(5, 6.0, decoded_match=False, relaxed=False)
    assert _patch_score_acceptable(5, 20.0, decoded_match=True, relaxed=False)

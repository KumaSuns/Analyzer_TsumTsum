"""result_gain_reader の単体テスト。"""
from __future__ import annotations

from app.services.result_gain_reader import (
    RESULT_GAIN_FIELD_KEYS,
    field_spec_for_key,
    result_gain_confirmed,
)


def test_result_gain_field_keys() -> None:
    assert len(RESULT_GAIN_FIELD_KEYS) == 4
    assert field_spec_for_key("result_score_gain") is not None
    assert field_spec_for_key("score_bonus_gain") is not None
    assert field_spec_for_key("result_exp_gain") is not None
    assert field_spec_for_key("result_coin_gain") is not None


def test_result_gain_confirmed_requires_reads() -> None:
    reads = [(1234, 40.0, "ok"), (1234, 41.0, "ok")]
    assert result_gain_confirmed(reads, field_key="result_score_gain")

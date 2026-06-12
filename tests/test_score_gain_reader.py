"""score_gain_reader の単体テスト。"""
from __future__ import annotations

from app.services.score_gain_reader import (
    consensus_score_gain,
    plausible_score_value,
    score_gain_confirmed,
)


def test_plausible_score_value_accepts_one_to_ten_digits() -> None:
    assert plausible_score_value(5)
    assert plausible_score_value(12_345_678_90)
    assert not plausible_score_value(0)
    assert not plausible_score_value(10_000_000_000)
    assert not plausible_score_value(None)


def test_consensus_score_gain_picks_majority() -> None:
    reads = [(1234, 40.0), (1234, 42.0), (5678, 30.0)]
    assert consensus_score_gain(reads) == 1234


def test_score_gain_confirmed_requires_two_reads() -> None:
    one = [(1234, 40.0, "ok")]
    two = [(1234, 40.0, "ok"), (1234, 41.0, "ok")]
    assert not score_gain_confirmed(one)
    assert score_gain_confirmed(two)

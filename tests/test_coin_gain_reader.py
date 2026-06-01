"""coin_gain_reader の単体テスト。"""
from app.services.coin_gain_reader import _suspicious_coin_value, consensus_coin_gain


def test_suspicious_rejects_stroke_artifacts() -> None:
    assert _suspicious_coin_value(17111)
    assert _suspicious_coin_value(11111)
    assert not _suspicious_coin_value(7205)


def test_consensus_prefers_repeated_value() -> None:
    reads = [(7205, 20.0), (7205, 22.0), (17111, 5.0), (7205, 25.0)]
    assert consensus_coin_gain(reads) == 7205

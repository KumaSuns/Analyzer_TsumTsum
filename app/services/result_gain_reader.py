"""リザルト画面の数値読取（result_digit CNN）。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from PySide6.QtGui import QImage

from app.services.coin_gain_reader import (
    plausible_coin_value,
    set_digit_classifier_factory,
)
from app.services.score_gain_reader import (
    _decode_up_to_ten_digits,
    consensus_score_gain,
    plausible_score_value,
    read_score_gain_crop,
    score_gain_confirmed,
)

_MIN_CONSENSUS_READS = 2
_MAX_CONSENSUS_AVG_ERR = 66.0


@dataclass(frozen=True)
class ResultGainFieldSpec:
    key: str
    label: str
    file_prefix: str
    plausible: Callable[[Optional[int]], bool]
    max_digits: int


RESULT_GAIN_FIELDS: tuple[ResultGainFieldSpec, ...] = (
    ResultGainFieldSpec(
        key="result_score_gain",
        label="最終獲得スコア",
        file_prefix="result_score_gain",
        plausible=plausible_score_value,
        max_digits=10,
    ),
    ResultGainFieldSpec(
        key="score_bonus_gain",
        label="スコアボーナス",
        file_prefix="score_bonus_gain",
        plausible=plausible_score_value,
        max_digits=10,
    ),
    ResultGainFieldSpec(
        key="result_exp_gain",
        label="獲得経験値",
        file_prefix="result_exp_gain",
        plausible=plausible_score_value,
        max_digits=10,
    ),
    ResultGainFieldSpec(
        key="result_coin_gain",
        label="最終獲得コイン",
        file_prefix="result_coin_gain",
        plausible=plausible_coin_value,
        max_digits=7,
    ),
)

RESULT_GAIN_FIELD_KEYS: tuple[str, ...] = tuple(f.key for f in RESULT_GAIN_FIELDS)


def field_spec_for_key(key: str) -> Optional[ResultGainFieldSpec]:
    for spec in RESULT_GAIN_FIELDS:
        if spec.key == key:
            return spec
    return None


def result_digit_cnn_ready() -> bool:
    try:
        from app.services.result_digit_cnn import result_digit_cnn_available

        return result_digit_cnn_available()
    except Exception:
        return False


@contextmanager
def use_result_digit_cnn():
    from app.services.result_digit_cnn import get_result_digit_classifier

    set_digit_classifier_factory(get_result_digit_classifier)
    try:
        yield
    finally:
        set_digit_classifier_factory(None)


def read_result_gain_crop(roi: QImage, *, field_key: str) -> Tuple[Optional[int], float, str]:
    """result 用切り抜き向け DL 読取（result_digit CNN）。"""
    spec = field_spec_for_key(field_key)
    if spec is None:
        return None, 1e9, "unknown_field"
    if not result_digit_cnn_ready():
        return None, 1e9, "result_digit未学習"
    with use_result_digit_cnn():
        if spec.max_digits <= 7:
            from app.services.coin_gain_reader import read_coin_gain_crop

            val, err, dbg = read_coin_gain_crop(roi)
        else:
            val, err, dbg = read_score_gain_crop(roi)
    if val is None:
        return val, err, dbg.replace("crop", spec.key).replace("score", spec.key)
    if not spec.plausible(val):
        return None, 1e9, f"{spec.key}_invalid"
    return val, err, dbg.replace("crop", spec.key).replace("score", spec.key)


def consensus_result_gain(
    reads: List[Tuple[int, float]], *, field_key: str
) -> Optional[int]:
    spec = field_spec_for_key(field_key)
    if spec is None:
        return None
    if spec.max_digits <= 7:
        from app.services.coin_gain_reader import consensus_coin_gain

        return consensus_coin_gain(
            [(v, e) for v, e in reads if spec.plausible(v)]
        )
    return consensus_score_gain([(v, e) for v, e in reads if spec.plausible(v)])


def result_gain_confirmed(
    reads: List[Tuple[int, float, str]], *, field_key: str
) -> bool:
    spec = field_spec_for_key(field_key)
    if spec is None or not reads:
        return False
    if spec.max_digits <= 7:
        from app.services.coin_gain_reader import coin_gain_confirmed

        filtered = [(v, e, d) for v, e, d in reads if spec.plausible(v)]
        return coin_gain_confirmed(filtered)
    return score_gain_confirmed([(v, e, d) for v, e, d in reads if spec.plausible(v)])


def patches_decode_context_for_value(value: int):
    """保存 crop の桁パッチ化用コンテキスト。"""
    if plausible_coin_value(value) and len(str(value)) <= 7:
        from contextlib import nullcontext

        return nullcontext()
    return _decode_up_to_ten_digits()

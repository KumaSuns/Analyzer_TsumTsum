"""獲得スコア画面の数字領域を読み取る（coin_digit CNN / DL を流用）。"""
from __future__ import annotations

from contextlib import contextmanager
from typing import List, Optional, Tuple

from PySide6.QtGui import QImage

import app.services.coin_gain_reader as _cgr
from app.services.coin_gain_reader import (
    _MAX_HUD_ACCEPTABLE_ERR,
    _candidate_rank_item,
    _coin_gain_strip_for_hud,
    _upscale_gray_for_hud,
    opencv_available,
    qimage_to_bgr,
    qimage_to_gray,
)

_MIN_SCORE_DIGITS = 1
_MAX_SCORE_DIGITS = 10
_MAX_SCORE_VALUE = 9_999_999_999
_MIN_SCORE_CONSENSUS_READS = 2
_MAX_SCORE_CONSENSUS_AVG_ERR = 66.0


def plausible_score_value(value: Optional[int]) -> bool:
    if value is None:
        return False
    if value < 1 or value > _MAX_SCORE_VALUE:
        return False
    n = len(str(value))
    return _MIN_SCORE_DIGITS <= n <= _MAX_SCORE_DIGITS


@contextmanager
def _decode_up_to_ten_digits():
    old_max = _cgr._MAX_DIGITS
    old_pref = _cgr._PREFERRED_DIGITS
    _cgr._MAX_DIGITS = _MAX_SCORE_DIGITS
    _cgr._PREFERRED_DIGITS = tuple(range(_MIN_SCORE_DIGITS, _MAX_SCORE_DIGITS + 1))
    try:
        yield
    finally:
        _cgr._MAX_DIGITS = old_max
        _cgr._PREFERRED_DIGITS = old_pref


def read_score_gain_crop(roi: QImage) -> Tuple[Optional[int], float, str]:
    """score_gain 切り抜き向け DL 読取（1〜10 桁）。"""
    if not opencv_available():
        return None, 1e9, "opencv未導入"
    if not _cgr._coin_digit_cnn_ready():
        return None, 1e9, "coin_digit未学習"
    if roi is None or roi.isNull() or roi.width() < 12 or roi.height() < 6:
        return None, 1e9, "crop_roi小"
    gray = qimage_to_gray(roi)
    if gray is None:
        return None, 1e9, "gray失敗"
    bgr = qimage_to_bgr(roi)
    candidates: List[Tuple[int, float, str]] = []
    with _decode_up_to_ten_digits():
        row_hud = _cgr._try_hud_digit_decode(gray)
        if row_hud is not None:
            _cgr._append_decode_candidate(
                candidates, row_hud[0], row_hud[1], f"score {row_hud[2]}", bonus=-36.0
            )
        prepared_full, _ = _upscale_gray_for_hud(gray, bgr)
        full_hud = _cgr._try_hud_digit_decode(prepared_full)
        if full_hud is not None:
            _cgr._append_decode_candidate(
                candidates, full_hud[0], full_hud[1], f"score {full_hud[2]}", bonus=-34.0
            )
        strip_hud = _cgr._try_hud_digit_decode(_coin_gain_strip_for_hud(gray, bgr))
        if strip_hud is not None:
            _cgr._append_decode_candidate(
                candidates, strip_hud[0], strip_hud[1], f"score {strip_hud[2]}", bonus=-26.0
            )
    if not candidates:
        return None, 1e9, "score_dl失敗"
    ranked = sorted(candidates, key=_candidate_rank_item)
    acceptable = [
        cand
        for cand in ranked
        if plausible_score_value(cand[0]) and cand[1] <= _MAX_HUD_ACCEPTABLE_ERR
    ]
    if not acceptable:
        top = ranked[0]
        return None, 1e9, f"score_err_high={top[1]:.1f} {top[2]}"
    best = acceptable[0]
    return best[0], best[1], f"ok err={best[1]:.1f} {best[2]}"


def consensus_score_gain(reads: List[Tuple[int, float]]) -> Optional[int]:
    if not reads:
        return None
    filtered = [(v, e) for v, e in reads if plausible_score_value(v)]
    if not filtered:
        return None
    buckets: dict[int, List[float]] = {}
    for value, err in filtered:
        buckets.setdefault(value, []).append(err)

    def rank(item: tuple[int, List[float]]) -> tuple[int, float, int]:
        value, errs = item
        n = len(errs)
        avg_err = sum(errs) / n
        digits = len(str(value))
        digit_penalty = 0.0 if 4 <= digits <= 8 else 6.0
        return (-n, avg_err + digit_penalty, -value)

    return min(buckets.items(), key=rank)[0]


def score_gain_confirmed(reads: List[Tuple[int, float, str]]) -> bool:
    if not reads:
        return False
    picked = consensus_score_gain([(value, err) for value, err, _dbg in reads])
    if picked is None:
        return False
    matching = [(err, dbg) for value, err, dbg in reads if value == picked]
    if len(matching) < _MIN_SCORE_CONSENSUS_READS:
        return False
    if not matching:
        return False
    avg_err = sum(err for err, _dbg in matching) / len(matching)
    return avg_err <= _MAX_SCORE_CONSENSUS_AVG_ERR

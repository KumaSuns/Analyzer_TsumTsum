"""リザルト画面の数値読取（result_digit CNN）。"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QImage

from app.services.coin_gain_reader import (
    _coin_gain_strip_for_hud,
    _coin_digit_cnn_ready,
    _digits_to_value,
    _get_active_digit_classifier,
    _gray_digit_strip,
    _hud_digit_row_gray,
    _hud_estimate_digit_count,
    _hud_split_n_digit_variants,
    _MAX_HUD_ACCEPTABLE_ERR,
    _patch_digit_errors,
    _upscale_gray_for_hud,
    opencv_available,
    plausible_coin_value,
    qimage_to_bgr,
    qimage_to_gray,
    set_digit_classifier_factory,
)
from app.services.score_gain_reader import (
    _decode_up_to_ten_digits,
    consensus_score_gain,
    plausible_score_value,
    score_gain_confirmed,
)

_MIN_CONSENSUS_READS = 2
_MAX_CONSENSUS_AVG_ERR = 66.0
_MAX_RESULT_DECODE_ERR = 35.0


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


def _count_result_commas(row: np.ndarray) -> int:
    """カンマ区切りらしい細い谷の数（桁数推定用）。"""
    if row is None or row.size == 0:
        return 0
    _, otsu = cv2.threshold(row, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = 255 - otsu
    proj = ink.sum(axis=0).astype(np.float32)
    peak = float(proj.max()) if proj.size else 0.0
    if peak <= 0:
        return 0
    norm = proj / peak
    h = row.shape[0]
    min_gap = max(2, int(h * 0.035))
    max_gap = max(min_gap + 1, int(h * 0.16))
    commas = 0
    in_gap = False
    gap_start = 0
    for x, v in enumerate(norm):
        if v < 0.12:
            if not in_gap:
                in_gap = True
                gap_start = x
            continue
        if in_gap:
            gap_w = x - gap_start
            if min_gap <= gap_w <= max_gap:
                left = float(norm[max(0, gap_start - 4) : gap_start].max()) if gap_start > 0 else 0.0
                right = float(norm[x : min(len(norm), x + 4)].max()) if x < len(norm) else 0.0
                if left > 0.35 and right > 0.35:
                    commas += 1
            in_gap = False
    return commas


def _result_digit_count_candidates(row: np.ndarray, *, max_digits: int) -> List[int]:
    width_est = _hud_estimate_digit_count(row)
    width_est = max(1, min(max_digits, width_est))
    comma_count = _count_result_commas(row)
    candidates: set[int] = {width_est}
    if comma_count > 0:
        for extra in (1, 2, 3):
            n = comma_count * 3 + extra
            if 1 <= n <= max_digits:
                candidates.add(n)
    return sorted(candidates, key=lambda n: (abs(n - width_est), -n))


def _digit_count_matches_commas(n_digits: int, comma_count: int) -> bool:
    if comma_count <= 0:
        return True
    for extra in (1, 2, 3):
        if n_digits == comma_count * 3 + extra:
            return True
    return False


def _suspicious_result_value(value: int) -> bool:
    text = str(value)
    if len(text) >= 5 and len(set(text)) <= 2:
        return True
    if len(text) >= 4 and text.count(text[0]) >= len(text) - 1:
        return True
    fives_sixes = sum(1 for ch in text if ch in "56")
    if len(text) >= 6 and fives_sixes >= int(len(text) * 0.62):
        return True
    return False


def _result_number_strip(gray: np.ndarray, bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """ラベル付き result 行 crop から、右側の大きい数字列だけを切り出す。"""
    if gray is None or gray.size == 0:
        return None
    if bgr is None:
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    prepared, bgr_up = _upscale_gray_for_hud(gray, bgr)
    ph, pw = prepared.shape[:2]
    if pw < 24 or ph < 8:
        return None
    aspect = pw / max(1, ph)
    if aspect >= 4.5:
        right = prepared[:, int(pw * 0.52) :]
        if right.shape[1] >= 16:
            return right
    from app.services.coin_gain_reader import _locator_masks

    best_mask: Optional[np.ndarray] = None
    best_count = 0
    for mask in _locator_masks(bgr_up):
        count = int(np.count_nonzero(mask))
        if count >= 24 and count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None:
        return None
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tall: List[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh >= ph * 0.40 and bw >= max(4, int(bh * 0.10)):
            tall.append((x, y, bw, bh))
    if not tall:
        return None
    max_h = max(b[3] for b in tall)
    big = [b for b in tall if b[3] >= max_h * 0.70]
    if len(big) >= 2:
        right_edge = max(b[0] + b[2] for b in big)
        right_cluster = [
            b for b in big if (b[0] + b[2]) >= right_edge - int(pw * 0.42)
        ]
        if len(right_cluster) >= 2:
            big = right_cluster
    if len(big) < 2:
        big = sorted(tall, key=lambda b: b[2] * b[3], reverse=True)[: max(2, len(tall))]
    x0 = max(0, min(b[0] for b in big) - 4)
    x1 = min(pw, max(b[0] + b[2] for b in big) + 4)
    y0 = max(0, min(b[1] for b in big) - 4)
    y1 = min(ph, max(b[1] + b[3] for b in big) + 4)
    if x1 - x0 < 16 or y1 - y0 < 10:
        return None
    return prepared[y0:y1, x0:x1]


def _decode_result_greedy(
    parts: List[np.ndarray],
    tag: str,
    *,
    plausible: Callable[[Optional[int]], bool],
) -> Optional[Tuple[int, float, str]]:
    if not _coin_digit_cnn_ready():
        return None
    cnn = _get_active_digit_classifier()
    if not cnn.is_loaded():
        return None
    n = len(parts)
    if n < 1:
        return None
    digits: List[int] = []
    err_sum = 0.0
    min_conf = 1.0
    for part in parts:
        probs = cnn.predict_probs(part)
        if probs is None:
            return None
        d = int(probs.argmax())
        conf = float(probs[d])
        if conf < 0.18:
            return None
        min_conf = min(min_conf, conf)
        digits.append(d)
        err_sum += 1.0 - conf
    value = _digits_to_value(digits)
    if not plausible(value) or len(str(value)) != n:
        return None
    mean_err = (err_sum / float(n)) * 100.0
    if mean_err > 85.0:
        return None
    bonus = 16.0 if tag == "comma" else 10.0
    bonus += min_conf * 10.0
    return value, mean_err - bonus, f"greedy n={n} {tag}"


def _decode_result_n_parts(
    parts: List[np.ndarray],
    tag: str,
    *,
    plausible: Callable[[Optional[int]], bool],
) -> Optional[Tuple[int, float, str]]:
    n = len(parts)
    if n < 1:
        return None
    part_errs = [_patch_digit_errors(p) for p in parts]
    best_val: Optional[int] = None
    best_err = 1e9
    threshold = 1.35

    def _try_value(digits: List[int], total: float) -> None:
        nonlocal best_val, best_err
        if total >= best_err:
            return
        value = _digits_to_value(digits)
        if not plausible(value) or len(str(value)) != n:
            return
        best_err = total
        best_val = value

    if n <= 6:
        stack: List[tuple[int, List[int], float]] = [(0, [], 0.0)]
        while stack:
            pos, digits, err = stack.pop()
            if pos == n:
                _try_value(digits, err)
                continue
            for d in range(10):
                if part_errs[pos][d] > threshold:
                    continue
                stack.append((pos + 1, digits + [d], err + part_errs[pos][d]))
    else:
        digits = []
        total = 0.0
        for pos in range(n):
            d = min(range(10), key=lambda i: part_errs[pos][i])
            if part_errs[pos][d] > threshold:
                return None
            digits.append(d)
            total += part_errs[pos][d]
        _try_value(digits, total)

    if best_val is None:
        return None
    mean_err = best_err / float(n)
    if mean_err > 1.35:
        return None
    bonus = 14.0 if tag == "comma" else 8.0
    return best_val, mean_err - bonus, f"dl n={n} {tag}"


def _pick_result_decode(
    decoded_list: List[Tuple[int, float, str]],
    *,
    plausible: Callable[[Optional[int]], bool],
    expected_ns: Optional[List[int]] = None,
) -> Optional[Tuple[int, float, str]]:
    expected = expected_ns or []
    valid = [
        item
        for item in decoded_list
        if plausible(item[0]) and not _suspicious_result_value(item[0])
    ]
    if not valid:
        return None

    def _rank(item: Tuple[int, float, str]) -> tuple[float, int, int, int]:
        val, err, dbg = item
        n = len(str(val))
        est_penalty = min((abs(n - e) for e in expected), default=0) * 9.0 if expected else 0.0
        tag = dbg.rsplit(" ", 1)[-1] if dbg else ""
        tag_order = {"comma": 0, "greedy": 1, "valley": 2, "ratio": 3, "ref": 4}.get(tag, 9)
        if dbg.startswith("greedy"):
            tag_order = 0
        return (err + est_penalty, tag_order, -n, -val)

    return min(valid, key=_rank)


def _try_result_digit_decode_gray(
    gray: np.ndarray,
    *,
    spec: ResultGainFieldSpec,
) -> Optional[Tuple[int, float, str]]:
    row = _hud_digit_row_gray(gray)
    if row is None:
        return None
    expected_ns = _result_digit_count_candidates(row, max_digits=spec.max_digits)
    decoded_list: List[Tuple[int, float, str]] = []
    for n in expected_ns:
        for parts, tag in _hud_split_n_digit_variants(row, n):
            greedy = _decode_result_greedy(parts, tag, plausible=spec.plausible)
            if greedy is not None:
                decoded_list.append(greedy)
            decoded = _decode_result_n_parts(parts, tag, plausible=spec.plausible)
            if decoded is not None:
                decoded_list.append(decoded)
    return _pick_result_decode(
        decoded_list, plausible=spec.plausible, expected_ns=expected_ns
    )


def _result_gray_sources(
    gray: np.ndarray, bgr: Optional[np.ndarray]
) -> List[np.ndarray]:
    sources: List[np.ndarray] = []
    seen: set[tuple[int, int, int]] = set()

    def _add(src: Optional[np.ndarray]) -> None:
        if src is None or src.size == 0:
            return
        prepared, _ = _upscale_gray_for_hud(src, bgr if src is gray else None)
        key = prepared.shape
        if key in seen:
            return
        seen.add(key)
        sources.append(prepared)

    _add(gray)
    _add(_result_number_strip(gray, bgr))
    _add(_gray_digit_strip(gray, bgr) if bgr is not None else None)
    _add(_coin_gain_strip_for_hud(gray, bgr))
    return sources


def read_result_gain_crop(roi: QImage, *, field_key: str) -> Tuple[Optional[int], float, str]:
    """result 用切り抜き向け DL 読取（result_digit CNN）。"""
    spec = field_spec_for_key(field_key)
    if spec is None:
        return None, 1e9, "unknown_field"
    if not result_digit_cnn_ready():
        return None, 1e9, "result_digit未学習"
    if not opencv_available():
        return None, 1e9, "opencv未導入"
    if roi is None or roi.isNull() or roi.width() < 12 or roi.height() < 6:
        return None, 1e9, f"{field_key}_roi小"
    if field_key == "result_coin_gain":
        from app.services.coin_gain_reader import read_coin_gain_crop

        with use_result_digit_cnn():
            val, err, dbg = read_coin_gain_crop(roi)
        if val is None:
            return val, err, dbg.replace("crop_", "result_coin_")
        return val, err, dbg.replace("crop ", "result_coin ")
    gray = qimage_to_gray(roi)
    if gray is None:
        return None, 1e9, "gray失敗"
    bgr = qimage_to_bgr(roi)

    candidates: List[Tuple[int, float, str, List[int]]] = []
    with use_result_digit_cnn(), _decode_up_to_ten_digits():
        for src in _result_gray_sources(gray, bgr):
            row = _hud_digit_row_gray(src)
            expected_ns = (
                _result_digit_count_candidates(row, max_digits=spec.max_digits)
                if row is not None
                else [spec.max_digits]
            )
            decoded = _try_result_digit_decode_gray(src, spec=spec)
            if decoded is not None:
                candidates.append((*decoded, expected_ns))

    if not candidates:
        return None, 1e9, f"{field_key}_dl失敗"

    flat = [item[:3] for item in candidates]
    expected_ns = candidates[0][3]
    best = _pick_result_decode(flat, plausible=spec.plausible, expected_ns=expected_ns)
    if best is None:
        return None, 1e9, f"{field_key}_信頼度不足（正解値を手入力して保存→CNN学習）"
    val, err, dbg = best
    if expected_ns and abs(len(str(val)) - expected_ns[0]) > 1:
        return None, 1e9, f"{field_key}_桁数不一致（正解値を手入力して保存→CNN学習）"
    row = _hud_digit_row_gray(_result_gray_sources(gray, bgr)[0])
    comma_n = _count_result_commas(row) if row is not None else 0
    if not _digit_count_matches_commas(len(str(val)), comma_n):
        return None, 1e9, f"{field_key}_桁数不一致（正解値を手入力して保存→CNN学習）"
    if err < 0 or err > _MAX_RESULT_DECODE_ERR or _suspicious_result_value(val):
        return None, 1e9, f"{field_key}_信頼度不足（正解値を手入力して保存→CNN学習）"
    return val, err, f"ok err={err:.1f} {spec.key} {dbg}"


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
    from app.services.score_gain_reader import _decode_up_to_ten_digits

    return _decode_up_to_ten_digits()

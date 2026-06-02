"""プレイ中 HUD の残り時間（秒）を ROI から読み取る。"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtGui import QImage

try:
    import cv2

    _CV2_OK = True
except ImportError:
    cv2 = None  # type: ignore

from app.services.coin_gain_reader import (
    _match_digit,
    qimage_to_bgr,
    qimage_to_gray,
)

_MIN_DIGITS = 1
_MAX_DIGITS = 3  # タイマー表示は最大3桁（例: 58, 105）
_MAX_SECONDS = 999
_DIGIT_MATCH_MAX = 78.0
_DIGIT_MATCH_MAX_SINGLE = 52.0
_MIN_DIGIT_H_RATIO = 0.22
_MAX_DIGIT_H_RATIO = 1.08
_MIN_DIGIT_W_OF_H = 0.18
_MAX_BOX_WIDTH_RATIO = 0.55
_MAX_BOX_AREA_RATIO = 0.35
_MIN_BRIGHT_PIXELS = 48
_MIN_ROI_MEAN = 28.0


def _upscale_gray(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    target_h = max(72, min(128, h * 5))
    scale = target_h / max(1, h)
    return cv2.resize(
        gray,
        (max(1, int(w * scale)), target_h),
        interpolation=cv2.INTER_CUBIC,
    )


def _binarize_variants(gray: np.ndarray) -> List[np.ndarray]:
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    out: List[np.ndarray] = []
    for inv in (False, True):
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if inv:
            otsu = 255 - otsu
        out.append(otsu)
        adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        if inv:
            adapt = 255 - adapt
        out.append(adapt)
    return out


def _bright_masks(bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    masks: List[np.ndarray] = []
    _, white = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    masks.append(white)
    masks.append(cv2.inRange(hsv, (12, 55, 100), (48, 255, 255)))
    masks.append(cv2.inRange(hsv, (0, 0, 150), (180, 70, 255)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(white, kernel, iterations=1)
    masks.append(eroded)
    return [cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1) for m in masks]


def _roi_has_timer_signal(gray: np.ndarray) -> bool:
    if float(gray.mean()) < _MIN_ROI_MEAN:
        return False
    return int(np.count_nonzero(gray > 175)) >= _MIN_BRIGHT_PIXELS


def _find_split_valley(proj: np.ndarray, lo_ratio: float, hi_ratio: float) -> Optional[int]:
    bw = len(proj)
    mid_lo = int(bw * lo_ratio)
    mid_hi = int(bw * hi_ratio)
    if mid_hi <= mid_lo:
        return None
    valley = mid_lo + int(np.argmin(proj[mid_lo:mid_hi]))
    if valley <= 2 or valley >= bw - 3:
        return None
    return valley


def _split_wide_box(
    binary: np.ndarray, box: Tuple[int, int, int, int]
) -> List[Tuple[int, int, int, int]]:
    """横に連結した 2〜3 桁を、投影の谷で分割する。"""
    x, y, bw, bh = box
    if bw < bh * 0.55:
        return [box]
    patch = binary[y : y + bh, x : x + bw]
    proj = patch.sum(axis=0).astype(np.float32)
    if proj.max() <= 0:
        return [box]

    target_parts = 3 if bw >= bh * 1.55 else 2
    cuts: List[int] = []
    segments = [(0, bw)]
    while len(cuts) < target_parts - 1 and segments:
        start, end = segments.pop(0)
        span = end - start
        if span < bh * 0.45:
            segments.insert(0, (start, end))
            break
        sub = proj[start:end]
        valley = _find_split_valley(sub, 0.22, 0.78)
        if valley is None:
            segments.insert(0, (start, end))
            break
        cut = start + valley
        cuts.append(cut)
        segments = [(start, cut), (cut, end)] + segments

    if not cuts:
        return [box]
    cuts = sorted(set(cuts))
    parts: List[Tuple[int, int, int, int]] = []
    prev = 0
    for cut in cuts:
        w = cut - prev
        if w >= 4:
            parts.append((x + prev, y, w, bh))
        prev = cut
    if bw - prev >= 4:
        parts.append((x + prev, y, bw - prev, bh))
    if len(parts) < 2 or len(parts) > _MAX_DIGITS:
        return [box]
    return parts


def _segment_digit_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < h * _MIN_DIGIT_H_RATIO or bh > h * _MAX_DIGIT_H_RATIO:
            continue
        if bw > w * _MAX_BOX_WIDTH_RATIO:
            continue
        if (bw * bh) > (w * h * _MAX_BOX_AREA_RATIO):
            continue
        if bw < max(2, int(bh * _MIN_DIGIT_W_OF_H)) or bw > w * 0.45:
            continue
        if bw < bh * 0.10:
            continue
        aspect = bh / max(1, bw)
        if aspect > 4.0 or aspect < 0.35:
            continue
        boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: b[0])
    merged: List[Tuple[int, int, int, int]] = []
    for box in boxes:
        if not merged:
            merged.append(box)
            continue
        px, py, pw, ph = merged[-1]
        x, y, bw, bh = box
        gap = max(2, int(pw * 0.12))
        if x <= px + pw + gap and (pw * ph + bw * bh) < (w * h * _MAX_BOX_AREA_RATIO):
            nx = min(px, x)
            ny = min(py, y)
            merged[-1] = (
                nx,
                ny,
                max(px + pw, x + bw) - nx,
                max(py + ph, y + bh) - ny,
            )
        else:
            merged.append(box)
    split: List[Tuple[int, int, int, int]] = []
    for box in merged:
        if box[2] > w * _MAX_BOX_WIDTH_RATIO or box[2] > box[3] * 0.55:
            split.extend(_split_wide_box(binary, box))
        else:
            split.append(box)
    split.sort(key=lambda b: b[0])
    return split


def _decode_binary(binary: np.ndarray) -> Tuple[Optional[int], float, str]:
    boxes = _segment_digit_boxes(binary)
    if not boxes:
        return None, 1e9, "digit_boxes=0"
    n = len(boxes)
    if n < _MIN_DIGITS or n > _MAX_DIGITS:
        return None, 1e9, f"digit_count={n}"
    max_err = _DIGIT_MATCH_MAX_SINGLE if n == 1 else _DIGIT_MATCH_MAX
    digits: List[str] = []
    total_err = 0.0
    for x, y, bw, bh in boxes:
        patch = binary[y : y + bh, x : x + bw]
        digit, err = _match_digit(patch)
        if digit is None or err > max_err:
            return None, 1e9, f"match_fail@{x}"
        digits.append(str(digit))
        total_err += err
    try:
        value = int("".join(digits))
    except ValueError:
        return None, 1e9, "parse_fail"
    if value < 0 or value > _MAX_SECONDS:
        return None, 1e9, "range"
    if n == 1 and value <= 9 and total_err / n > 45.0:
        return None, 1e9, "single_digit_low_conf"
    return value, total_err / len(digits), f"n={n}"


def _decode_sources(gray: np.ndarray, bgr: Optional[np.ndarray]) -> Tuple[Optional[int], float, str]:
    candidates: List[Tuple[int, float, str, int]] = []
    for binary in _binarize_variants(_upscale_gray(gray)):
        val, err, dbg = _decode_binary(binary)
        if val is not None:
            n = int(dbg.split("n=")[-1]) if "n=" in dbg else 1
            candidates.append((val, err, f"gray {dbg}", n))
    if bgr is not None:
        for mask in _bright_masks(bgr):
            val, err, dbg = _decode_binary(_upscale_gray(mask))
            if val is not None:
                n = int(dbg.split("n=")[-1]) if "n=" in dbg else 1
                candidates.append((val, err, f"mask {dbg}", n))
    if not candidates:
        return None, 1e9, "decode失敗"

    def score(item: Tuple[int, float, str, int]) -> Tuple[float, float, float]:
        val, err, _dbg, n = item
        # 桁数が多い候補を優先（最大3桁）。1桁だけの誤読を抑える。
        digit_penalty = {3: 0.0, 2: 6.0, 1: 22.0}.get(n, 30.0)
        single_penalty = 35.0 if n == 1 and val <= 9 else 0.0
        return (digit_penalty + single_penalty + err, err, -float(val))

    best = min(candidates, key=score)
    return best[0], best[1], best[2]


def read_remaining_seconds(roi: QImage) -> Tuple[Optional[int], str]:
    """残り時間（秒）を推定。失敗時は (None, debug)。"""
    if not _CV2_OK:
        return None, "opencv未導入"
    if roi is None or roi.isNull():
        return None, "roi空"
    gray = qimage_to_gray(roi)
    if gray is None:
        return None, "gray失敗"
    if gray.shape[0] < 4 or gray.shape[1] < 6:
        return None, f"roi小 {gray.shape[1]}x{gray.shape[0]}"
    if not _roi_has_timer_signal(gray):
        return None, (
            "タイマーが写っていません（トリミング位置を左上の秒数表示に合わせてください）"
        )

    bgr = qimage_to_bgr(roi)
    val, err, dbg = _decode_sources(gray, bgr)
    if val is None:
        return None, f"{dbg} size={roi.width()}x{roi.height()}"
    return val, f"ok err={err:.1f} {dbg}"

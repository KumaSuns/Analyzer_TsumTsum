"""コイン獲得画面の数字領域を読み取る（coin_digit CNN / DL）。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtGui import QImage

try:
    import cv2

    _CV2_OK = True
except ImportError:
    cv2 = None  # type: ignore
    _CV2_OK = False

# 1 桁あたりの平均差がこれ未満なら採用
_DIGIT_MATCH_MAX_MEAN = 95.0
_WIDE_SPLIT_MAX_MEAN = 105.0
_SLOT_MATCH_MAX_MEAN = 118.0
_MIN_DIGIT_HEIGHT_RATIO = 0.35
_MAX_DIGIT_HEIGHT_RATIO = 1.05
_MIN_DIGIT_WIDTH_RATIO = 0.28
_MIN_DIGITS = 1
_MAX_DIGITS = 7
_MAX_COIN_VALUE = 9_999_999
_PREFERRED_DIGITS = tuple(range(_MIN_DIGITS, _MAX_DIGITS + 1))
_MAX_ACCEPTABLE_ERR = 72.0
_MAX_HUD_ACCEPTABLE_ERR = 80.0
_MIN_HUD_CONSENSUS_READS = 2
_MAX_HUD_CONSENSUS_AVG_ERR = 62.0
_MAX_CONSENSUS_AVG_ERR = 66.0
_MIN_CONSENSUS_READS = 3
_BOX_OVER_SLOT_MARGIN = 22.0
_AMBIGUOUS_RANK_GAP = 10.0
_FIVE_DIGIT_EXTRA_ERR = 18.0
_SLOT_LOW_BOX_PENALTY = 24.0
_MIN_COIN_VALUE = 1


def bonus_coin_hud_rect(positions: dict) -> tuple[float, float, float, float]:
    """bonus 画面の中央コイン表示（5,395 等）。coin_gain の X 付近の横帯。"""
    if isinstance(positions, dict):
        bonus = positions.get("coin_bonus")
        if isinstance(bonus, list) and len(bonus) == 4:
            try:
                return tuple(float(bonus[i]) for i in range(4))  # type: ignore[return-value]
            except (TypeError, ValueError):
                pass
    cg = positions.get("coin_gain") if isinstance(positions, dict) else None
    if isinstance(cg, list) and len(cg) == 4:
        try:
            nx, ny, nw, nh = (float(cg[i]) for i in range(4))
            nh = max(nh, 0.055)
            nw = max(nw, 0.17)
            return (
                max(0.0, nx - 0.02),
                max(0.0, ny),
                min(0.50, nw + 0.04),
                nh,
            )
        except (TypeError, ValueError):
            pass
    return (0.36, 0.215, 0.28, 0.030)


def plausible_coin_value(value: Optional[int]) -> bool:
    """獲得コインとして妥当な値か（2桁の bookend 誤読などを除外）。"""
    if value is None:
        return False
    if value < _MIN_COIN_VALUE or value > _MAX_COIN_VALUE:
        return False
    n = len(str(value))
    return _MIN_DIGITS <= n <= _MAX_DIGITS


def opencv_available() -> bool:
    return _CV2_OK


def qimage_to_bgr(image: QImage) -> Optional[np.ndarray]:
    if image is None or image.isNull():
        return None
    rgb = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = rgb.width(), rgb.height()
    if w <= 0 or h <= 0:
        return None
    bpl = int(rgb.bytesPerLine())
    buf = rgb.constBits()
    if buf is None:
        return None
    nbytes = bpl * h
    arr = np.frombuffer(buf, dtype=np.uint8, count=nbytes).reshape(h, bpl)[:, : w * 3]
    if arr.size < w * h * 3:
        return None
    bgr = cv2.cvtColor(arr.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
    return np.ascontiguousarray(bgr)


def qimage_to_gray(image: QImage) -> Optional[np.ndarray]:
    if image is None or image.isNull():
        return None
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = gray.width(), gray.height()
    if w <= 0 or h <= 0:
        return None
    bpl = int(gray.bytesPerLine())
    buf = gray.constBits()
    if buf is None:
        return None
    arr = np.frombuffer(buf, dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, :w]
    return np.ascontiguousarray(arr.copy())


def _suspicious_coin_value(value: int) -> bool:
    """黄色マスクの縦線化けや余分桁（11513 等）を弾く。"""
    s = str(value)
    if len(s) < 4:
        return False
    ones = s.count("1")
    if ones >= len(s) - 1:
        return True
    if s.startswith("17") and ones >= 2:
        return True
    if len(s) == 5 and s.startswith("11"):
        return True
    if len(s) == 5 and ones >= 3:
        return True
    for digit in "0123456789":
        if s.count(digit) >= 3:
            return True
    return False


def detect_coin_hud_rect(bgr: np.ndarray) -> Optional[tuple[float, float, float, float]]:
    """コインアイコン付き HUD 数字列 ROI を推定。"""
    if not _CV2_OK or bgr is None or bgr.size == 0:
        return None
    fh, fw = bgr.shape[:2]
    x0, x1 = int(fw * 0.08), int(fw * 0.68)
    y0, y1 = int(fh * 0.12), int(fh * 0.32)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    row = _hud_digit_row_gray(patch_gray)
    if row is None:
        return None
    span = _hud_trimmed_ink_span(row)
    if span is None:
        return None
    rx0, rx1, _proj = span
    row_h = row.shape[0]
    pad_x = max(4, int(row_h * 0.45))
    pad_y = max(2, int(row_h * 0.18))
    left = max(0, x0 + rx0 - pad_x)
    top = max(0, y0 + int((y1 - y0) * 0.12) - pad_y)
    width = min(fw - left, (rx1 - rx0) + pad_x * 2 + int(row_h * 0.55))
    height = min(fh - top, int(row_h * 1.15) + pad_y * 2)
    nw = width / max(1, fw)
    nh = height / max(1, fh)
    if not (0.12 <= nw <= 0.24 and 0.038 <= nh <= 0.09):
        return None
    return (left / fw, top / fh, nw, nh)


@lru_cache(maxsize=1)
def _digit_templates() -> List[Tuple[int, np.ndarray]]:
    if not _CV2_OK:
        return []
    out: List[Tuple[int, np.ndarray]] = []
    for digit in range(10):
        for scale in (0.75, 0.9, 1.0, 1.15, 1.3):
            for thickness in (2, 3):
                h, w = 40, 28
                canvas = np.zeros((h, w), dtype=np.uint8)
                cv2.putText(
                    canvas,
                    str(digit),
                    (2, 33),
                    cv2.FONT_HERSHEY_DUPLEX,
                    scale,
                    255,
                    thickness,
                    cv2.LINE_AA,
                )
                out.append((digit, canvas))
    return out


def _locator_masks(bgr: np.ndarray) -> List[np.ndarray]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (14, 70, 110), (42, 255, 255)),
        cv2.inRange(hsv, (10, 40, 80), (50, 255, 255)),
        cv2.inRange(hsv, (0, 0, 120), (180, 70, 255)),
    ]
    _, white = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return [cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1) for m in masks]


def _digit_strip_start_x(mask: np.ndarray, x0: int, x1: int) -> int:
    """コインアイコン分だけ左を飛ばし、先頭桁を切らない。"""
    crop = mask[:, max(0, x0) : min(mask.shape[1], x1 + 1)]
    if crop.size == 0:
        return x0
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    icon_end = 0
    crop_w = crop.shape[1]
    for cnt in contours:
        bx, _by, bw, bh = cv2.boundingRect(cnt)
        if _is_coin_icon_box(bx, bw, bh, crop_w):
            icon_end = max(icon_end, bx + bw + 2)
    if icon_end > 0:
        return x0 + icon_end
    span_x = max(1, x1 - x0)
    return x0 + min(int(span_x * 0.14), max(8, int(span_x * 0.22)))


def _gray_digit_strip(gray: np.ndarray, bgr: np.ndarray) -> Optional[np.ndarray]:
    """明るい領域の位置から数字列だけのグレー帯を切り出す。"""
    best_mask: Optional[np.ndarray] = None
    best_count = 0
    for mask in _locator_masks(bgr):
        count = int(np.count_nonzero(mask))
        if count >= 24 and count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None:
        return None
    ys, xs = np.where(best_mask > 0)
    if len(xs) < 24:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    dx0 = _digit_strip_start_x(best_mask, x0, x1)
    if dx0 >= x1 - 8:
        dx0 = x0
    pad_y = max(6, int((y1 - y0) * 0.8))
    y_a = max(0, y0 - pad_y)
    y_b = min(gray.shape[0], y1 + pad_y + 1)
    x_b = min(gray.shape[1], x1 + 3)
    strip = gray[y_a:y_b, dx0:x_b]
    if strip.size == 0 or strip.shape[0] < 10 or strip.shape[1] < 16:
        return None
    return strip


def _prepare_gray(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    target_h = max(72, min(120, h * 5))
    scale = target_h / max(1, h)
    resized = cv2.resize(
        gray,
        (max(1, int(w * scale)), target_h),
        interpolation=cv2.INTER_CUBIC,
    )
    tile_w = max(2, min(8, resized.shape[1] // 8))
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, tile_w))
    return clahe.apply(resized)


def _binarize_variants(gray: np.ndarray) -> List[np.ndarray]:
    prepared = _prepare_gray(gray)
    h, w = prepared.shape
    target_h = max(48, min(96, h * 3))
    scale = target_h / max(1, h)
    resized = cv2.resize(
        prepared,
        (max(1, int(w * scale)), target_h),
        interpolation=cv2.INTER_CUBIC,
    )
    blurred = cv2.GaussianBlur(resized, (3, 3), 0)
    variants: List[np.ndarray] = []
    for inv in (False, True):
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if inv:
            otsu = 255 - otsu
        variants.append(otsu)
        adapt = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        if inv:
            adapt = 255 - adapt
        variants.append(adapt)
    return variants


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


def _is_coin_icon_box(x: int, bw: int, bh: int, w: int) -> bool:
    return x < int(w * 0.06) and bw >= int(bh * 0.75) and (x + bw) <= int(w * 0.22)


def _is_comma_box(bw: int, bh: int) -> bool:
    return bw < max(2, int(bh * 0.22))


def _is_comma_column(patch: np.ndarray) -> bool:
    if patch.size == 0:
        return True
    bh, bw = patch.shape
    if bw > max(3, int(bh * 0.16)):
        return False
    ink = float(np.count_nonzero(patch))
    return ink < bh * bw * 0.42


def _digit_boxes_for_decode(
    binary: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    boxes = _select_primary_row(_segment_digit_boxes(binary))
    return [b for b in boxes if not _is_comma_box(b[2], b[3])]


def _segment_digit_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if _is_coin_icon_box(x, bw, bh, w):
            continue
        if _is_comma_box(bw, bh):
            continue
        if bh < h * _MIN_DIGIT_HEIGHT_RATIO or bh > h * _MAX_DIGIT_HEIGHT_RATIO:
            continue
        if bw < max(4, int(bh * _MIN_DIGIT_WIDTH_RATIO)) or bw > w * 0.45:
            continue
        aspect = bh / max(1, bw)
        if aspect > 3.2 or aspect < 0.45:
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
        gap = x - (px + pw)
        if gap <= 0 or gap <= max(3, int(pw * 0.12)):
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
    return merged


def _stroke_artifact_boxes(boxes: List[Tuple[int, int, int, int]]) -> bool:
    if len(boxes) < 3:
        return False
    thin = sum(1 for _, _, bw, bh in boxes if bw < bh * 0.32)
    return thin >= len(boxes) - 1


def _select_primary_row(
    boxes: List[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    if len(boxes) <= _MAX_DIGITS:
        return boxes
    rows: dict[int, List[Tuple[int, int, int, int]]] = {}
    for box in boxes:
        y_mid = box[1] + box[3] // 2
        bucket = y_mid // max(8, box[3] // 2)
        rows.setdefault(bucket, []).append(box)
    best: List[Tuple[int, int, int, int]] = []
    best_score = -1
    for row_boxes in rows.values():
        row_boxes = sorted(row_boxes, key=lambda b: b[0])
        width = sum(b[2] for b in row_boxes)
        score = len(row_boxes) * 1000 + width
        if score > best_score:
            best_score = score
            best = row_boxes
    return sorted(best, key=lambda b: b[0])


def _match_digit_template(
    patch: np.ndarray, max_mean: float = _DIGIT_MATCH_MAX_MEAN
) -> Tuple[Optional[int], float]:
    templates = _digit_templates()
    if patch.size == 0 or not templates:
        return None, 1e9
    best_digit: Optional[int] = None
    best_err = 1e9
    for digit, tmpl in templates:
        th, tw = tmpl.shape
        resized = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
        err = float(np.mean(np.abs(resized.astype(np.float32) - tmpl.astype(np.float32))))
        if err < best_err:
            best_err = err
            best_digit = digit
    if best_digit is None or best_err > max_mean:
        return None, best_err
    return best_digit, best_err


_digit_classifier_factory = None


def set_digit_classifier_factory(factory) -> None:
    global _digit_classifier_factory
    _digit_classifier_factory = factory


def _get_active_digit_classifier():
    if _digit_classifier_factory is not None:
        return _digit_classifier_factory()
    from app.services.coin_digit_cnn import get_coin_digit_classifier

    return get_coin_digit_classifier()


def _match_digit(
    patch: np.ndarray, max_mean: float = _DIGIT_MATCH_MAX_MEAN
) -> Tuple[Optional[int], float]:
    if patch.size == 0:
        return None, 1e9
    if not _coin_digit_cnn_ready():
        return None, 1e9
    try:
        cnn = _get_active_digit_classifier()
        if cnn.is_loaded():
            return cnn.match_digit(patch, max_mean=max_mean)
    except Exception:
        pass
    return None, 1e9


def _expand_box_to_digits(
    binary: np.ndarray, box: Tuple[int, int, int, int]
) -> List[Tuple[int, float]]:
    x, y, bw, bh = box
    patch = binary[y : y + bh, x : x + bw]
    digit, err = _match_digit(patch)
    if digit is not None:
        return [(digit, err)]
    if bw >= bh * 0.55:
        return _multi_split_expand(binary, box)
    return []


def _tail_digit_penalty(digits: List[int]) -> float:
    penalty = 0.0
    text = "".join(str(d) for d in digits)
    if "11" in text:
        penalty += 18.0
    if len(digits) >= 3 and digits[-1] == digits[-2]:
        penalty += 12.0
    return penalty


def _multi_split_expand(
    binary: np.ndarray,
    box: Tuple[int, int, int, int],
    *,
    hint_digit: Optional[int] = None,
) -> List[Tuple[int, float]]:
    x, y, bw, bh = box
    if bw < bh * 0.55:
        return []
    patch = binary[y : y + bh, x : x + bw]
    proj = patch.sum(axis=0).astype(np.float32)
    min_cut = max(4, int(bh * 0.35))
    candidates: List[List[Tuple[int, float]]] = []

    def consider(parts: List[np.ndarray]) -> None:
        seq: List[Tuple[int, float]] = []
        for part in parts:
            digit, err = _match_digit(part, _WIDE_SPLIT_MAX_MEAN)
            if digit is None:
                return
            seq.append((digit, err))
        if len(seq) >= 2:
            candidates.append(seq)

    cut_candidates: List[int] = []
    if bw > min_cut * 2:
        mid_lo = min_cut
        mid_hi = bw - min_cut
        if mid_hi > mid_lo:
            cut_candidates.append(mid_lo + int(np.argmin(proj[mid_lo:mid_hi])))
        valley = _find_split_valley(proj, 0.2, 0.8)
        if valley is not None:
            cut_candidates.append(valley)
        for ratio in (0.33, 0.45, 0.55, 0.67):
            cut_candidates.append(int(bw * ratio))
    for cut in sorted(set(c for c in cut_candidates if min_cut <= c <= bw - min_cut)):
        consider([patch[:, :cut], patch[:, cut:]])

    for c1 in sorted(set(c for c in cut_candidates if min_cut <= c <= bw - min_cut * 2)):
        rest = patch[:, c1:]
        rest_proj = rest.sum(axis=0).astype(np.float32)
        inner_cuts = [int(len(rest_proj) * r) for r in (0.35, 0.5, 0.65)]
        valley = _find_split_valley(rest_proj, 0.2, 0.8)
        if valley is not None:
            inner_cuts.append(valley)
        for c2 in sorted(set(c for c in inner_cuts if min_cut <= c <= len(rest_proj) - min_cut)):
            consider([patch[:, :c1], patch[:, c1 : c1 + c2], patch[:, c1 + c2 :]])

    if not candidates:
        return []
    if hint_digit is not None:
        hinted = [seq for seq in candidates if seq[0][0] == hint_digit]
        if hinted:
            three_part = [seq for seq in hinted if len(seq) >= 3]
            pool = three_part if three_part else hinted
            return min(
                pool,
                key=lambda seq: (
                    sum(err for _digit, err in seq)
                    + _tail_digit_penalty([digit for digit, _err in seq]),
                    -len(seq),
                ),
            )
    return min(candidates, key=lambda seq: sum(err for _digit, err in seq))


def _try_bookend_decode(
    binary: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, float, str]]:
    if len(boxes) < 2:
        return None
    head = _expand_box_to_digits(binary, boxes[0])
    if not head:
        return None
    hint_digit: Optional[int] = None
    if len(boxes) >= 3:
        middle = _expand_box_to_digits(binary, boxes[1])
        if middle:
            hint_digit = middle[0][0]
    tail = _multi_split_expand(binary, boxes[-1], hint_digit=hint_digit)
    if not tail:
        return None
    head_digit, head_err = head[0]
    tail_digits = [digit for digit, _err in tail]
    tail_err = sum(err for _digit, err in tail)
    value = int(f"{head_digit}{''.join(map(str, tail_digits))}")
    total_digits = len(str(value))
    if total_digits < _MIN_DIGITS or not plausible_coin_value(value):
        return None
    err = (head_err + tail_err) / (1 + len(tail_digits))
    penalty = 0.0
    if len(boxes) >= 3 and len(tail_digits) < 3:
        penalty += 40.0
    if total_digits not in _PREFERRED_DIGITS:
        penalty += 12.0
    return value, err + penalty, f"bookend n={1 + len(tail_digits)}"


def _decode_box_group(
    binary: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    *,
    max_mean: float = _DIGIT_MATCH_MAX_MEAN,
) -> Tuple[Optional[int], float, str]:
    digits: List[str] = []
    total_err = 0.0
    for box in boxes:
        x, y, bw, bh = box
        patch = binary[y : y + bh, x : x + bw]
        digit, err = _match_digit(patch, max_mean)
        if digit is None:
            expanded = _expand_box_to_digits(binary, box)
            if not expanded:
                return None, 1e9, f"match_fail@{box[0]}"
            for d, e in expanded:
                digit, err = d, e
                digits.append(str(digit))
                total_err += err
            continue
        digits.append(str(digit))
        total_err += err
    if len(digits) < _MIN_DIGITS:
        return None, 1e9, f"digit_count={len(digits)}"
    try:
        value = int("".join(digits))
    except ValueError:
        return None, 1e9, "parse_fail"
    if not plausible_coin_value(value):
        return None, 1e9, "range"
    if _suspicious_coin_value(value):
        return None, 1e9, "suspicious"
    mean_err = total_err / len(digits)
    penalty = 0.0 if len(digits) in _PREFERRED_DIGITS else 12.0
    return value, mean_err + penalty, f"n={len(digits)}"


def _row_bounds_from_binary(
    binary: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    boxes = _select_primary_row(_segment_digit_boxes(binary))
    if boxes:
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        return (x0, y0, x1 - x0, y1 - y0)
    proj_y = binary.sum(axis=1).astype(np.float32)
    proj_x = binary.sum(axis=0).astype(np.float32)
    if proj_y.max() <= 0 or proj_x.max() <= 0:
        return None
    ys = np.where(proj_y >= proj_y.max() * 0.22)[0]
    xs = np.where(proj_x >= proj_x.max() * 0.18)[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    x0, x1 = int(xs[0]), int(xs[-1])
    y0, y1 = int(ys[0]), int(ys[-1])
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    if bw < 12 or bh < 6:
        return None
    return (x0, y0, bw, bh)


def _valley_slot_ranges(
    row: np.ndarray, n_slots: int
) -> Optional[List[Tuple[int, int]]]:
    """数字列の投影谷でスロット境界を決める（等幅より頑健）。"""
    usable = row.shape[1]
    bh = row.shape[0]
    min_seg = max(4, int(bh * 0.22))
    if usable < min_seg * n_slots:
        return None
    proj = row.sum(axis=0).astype(np.float32)
    segments: List[Tuple[int, int]] = [(0, usable)]
    while len(segments) < n_slots:
        segments.sort(key=lambda seg: seg[1] - seg[0], reverse=True)
        start, end = segments.pop(0)
        span = end - start
        if span < min_seg * 2:
            segments.insert(0, (start, end))
            break
        sub = proj[start:end]
        valley = _find_split_valley(sub, 0.12, 0.88)
        if valley is None:
            segments.insert(0, (start, end))
            break
        cut = start + valley
        segments = [(start, cut), (cut, end)] + segments
    if len(segments) != n_slots:
        return None
    segments.sort(key=lambda seg: seg[0])
    return segments


def _decode_slot_ranges(
    row: np.ndarray, ranges: List[Tuple[int, int]], *, tag: str
) -> Tuple[Optional[int], float, str]:
    digits: List[str] = []
    errs: List[float] = []
    bh = row.shape[0]
    for sx, ex in ranges:
        if ex <= sx:
            digits = []
            break
        patch = row[:, sx:ex]
        if _is_comma_column(patch):
            continue
        digit, err = _match_digit(patch, _SLOT_MATCH_MAX_MEAN)
        if digit is None:
            digits = []
            break
        digits.append(str(digit))
        errs.append(err)
    if len(digits) < _MIN_DIGITS:
        return None, 1e9, f"{tag}_short"
    try:
        value = int("".join(digits))
    except ValueError:
        return None, 1e9, f"{tag}_parse"
    if not plausible_coin_value(value) or _suspicious_coin_value(value):
        return None, 1e9, f"{tag}_implausible"
    mean_err = sum(errs) / len(errs)
    penalty = 0.0 if len(digits) in _PREFERRED_DIGITS else 10.0
    return value, mean_err + penalty, f"{tag}={len(digits)}"


def _try_two_box_head_tail(
    binary: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, float, str]]:
    """「5,395」のように 1桁 + 3桁に分かれた行を復元する。"""
    if len(boxes) != 2:
        return None
    left = _expand_box_to_digits(binary, boxes[0])
    right = _multi_split_expand(binary, boxes[1])
    if not left or not right or len(left) != 1 or len(right) < 3:
        return None
    head_digit, head_err = left[0]
    tail_digits = [digit for digit, _err in right[:3]]
    tail_err = sum(err for _digit, err in right[:3])
    value = int(f"{head_digit}{''.join(map(str, tail_digits))}")
    if not plausible_coin_value(value) or _suspicious_coin_value(value):
        return None
    err = (head_err + tail_err) / 4.0
    return value, err - 4.0, "head_tail n=4"


def _decode_four_box_row(
    binary: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
) -> Tuple[Optional[int], float, str]:
    if len(boxes) != 4:
        return None, 1e9, "four_box_count"
    digits: List[int] = []
    errs: List[float] = []
    for box in boxes:
        x, y, bw, bh = box
        patch = binary[y : y + bh, x : x + bw]
        digit, err = _match_digit(patch, _DIGIT_MATCH_MAX_MEAN)
        if digit is None:
            digit, err = _match_digit(patch, _SLOT_MATCH_MAX_MEAN)
        if digit is None:
            return None, 1e9, "four_box_match_fail"
        digits.append(digit)
        errs.append(err)
    value = int("".join(str(d) for d in digits))
    if not plausible_coin_value(value) or _suspicious_coin_value(value):
        return None, 1e9, "four_box_implausible"
    mean_err = sum(errs) / len(errs)
    return value, mean_err - 8.0, "n=4"


def _decode_slot_partition(
    binary: np.ndarray,
) -> Tuple[Optional[int], float, str]:
    bounds = _row_bounds_from_binary(binary)
    if bounds is None:
        return None, 1e9, "slot_no_bounds"
    x, y, bw, bh = bounds
    h, w = binary.shape
    icon_skip = 0
    for bx, _by, bbw, bbh in _segment_digit_boxes(binary):
        if _is_coin_icon_box(bx, bbw, bbh, w):
            icon_skip = max(icon_skip, bx + bbw + 2)
    x_start = max(x, icon_skip, int(w * 0.04))
    x_end = min(w, x + bw)
    usable = x_end - x_start
    if usable < max(16, int(bh * 1.6)):
        return None, 1e9, "slot_narrow"

    best: Optional[Tuple[int, float, str]] = None
    row = binary[y : y + bh, x_start:x_end]
    box_count = len(_select_primary_row(_segment_digit_boxes(binary)))
    for n_slots in (4, 5):
        attempts: List[Tuple[List[Tuple[int, int]], str]] = []
        valley_ranges = _valley_slot_ranges(row, n_slots)
        if valley_ranges is not None:
            attempts.append((valley_ranges, "valley_slots"))
        slot_w = usable / n_slots
        if slot_w >= bh * 0.22:
            equal_ranges: List[Tuple[int, int]] = []
            for i in range(n_slots):
                sx = int(i * slot_w)
                ex = int((i + 1) * slot_w) if i < n_slots - 1 else usable
                equal_ranges.append((sx, ex))
            attempts.append((equal_ranges, "slots"))
        for ranges, tag in attempts:
            val, err, dbg = _decode_slot_ranges(row, ranges, tag=tag)
            if val is None:
                continue
            if box_count < 3:
                err += _SLOT_LOW_BOX_PENALTY
            if best is None or err < best[1]:
                best = (val, err, dbg)
    if best is None:
        return None, 1e9, "slot_fail"
    return best


def _decode_binary(binary: np.ndarray) -> Tuple[Optional[int], float, str]:
    boxes = _digit_boxes_for_decode(binary)
    if not boxes:
        return None, 1e9, "digit_boxes=0"
    if _stroke_artifact_boxes(boxes):
        return None, 1e9, "stroke_artifact"

    best: Optional[Tuple[int, float, str]] = None
    four_box = _decode_four_box_row(binary, boxes)
    if four_box[0] is not None:
        best = four_box
    head_tail = _try_two_box_head_tail(binary, boxes)
    if head_tail is not None and (best is None or head_tail[1] < best[1]):
        best = head_tail
    bookend = _try_bookend_decode(binary, boxes)
    if bookend is not None and (best is None or bookend[1] < best[1]):
        best = bookend
    decode_attempts: List[Tuple[float, List[Tuple[int, int, int, int]]]] = []
    seen_windows: set[tuple[float, tuple[tuple[int, int, int, int], ...]]] = set()

    def _queue_window(
        window: List[Tuple[int, int, int, int]], max_mean: float
    ) -> None:
        key = (max_mean, tuple(window))
        if key in seen_windows:
            return
        seen_windows.add(key)
        decode_attempts.append((max_mean, window))

    _queue_window(boxes, _DIGIT_MATCH_MAX_MEAN)
    if len(boxes) in _PREFERRED_DIGITS:
        _queue_window(boxes, _WIDE_SPLIT_MAX_MEAN)
    for start in range(len(boxes)):
        for end in range(start + _MIN_DIGITS, min(start + _MAX_DIGITS + 1, len(boxes) + 1)):
            _queue_window(boxes[start:end], _DIGIT_MATCH_MAX_MEAN)
    for max_mean, window in decode_attempts:
        val, err, dbg = _decode_box_group(binary, window, max_mean=max_mean)
        if val is None:
            continue
        relaxed_penalty = 6.0 if max_mean > _DIGIT_MATCH_MAX_MEAN else 0.0
        err += relaxed_penalty
        dbg = f"{dbg} mean<={max_mean:.0f}"
        if best is None or err < best[1]:
            best = (val, err, dbg)
    slot = _decode_slot_partition(binary)
    if slot[0] is not None:
        slot_err = slot[1]
        if four_box[0] is not None and slot[0] != four_box[0]:
            slot_err += 20.0
        if best is None or slot_err < best[1]:
            best = (slot[0], slot_err, slot[2])
    if best is None:
        n = len(boxes)
        return None, 1e9, f"digit_count={n}"
    return best


def _right_digit_strip(gray: np.ndarray, skip_ratio: float = 0.22) -> Optional[np.ndarray]:
    x_skip = int(gray.shape[1] * skip_ratio)
    if gray.shape[1] - x_skip < 16:
        return None
    return gray[:, x_skip:]


def _gray_source_variants(gray: np.ndarray, bgr: Optional[np.ndarray]) -> List[Tuple[np.ndarray, str]]:
    sources: List[Tuple[np.ndarray, str]] = []
    if bgr is not None:
        strip = _gray_digit_strip(gray, bgr)
        if strip is not None:
            sources.append((strip, "strip"))
    if gray.shape[0] >= 18:
        bottom = gray[int(gray.shape[0] * 0.42) :]
        if bottom.shape[0] >= 8:
            sources.append((bottom, "bottom"))
            right = _right_digit_strip(bottom)
            if right is not None:
                sources.append((right, "bottom-right"))
    right = _right_digit_strip(gray)
    if right is not None:
        sources.append((right, "right"))
    sources.append((gray, "full"))
    return sources


def _enhance_gray_variants(gray: np.ndarray) -> List[Tuple[np.ndarray, str]]:
    variants: List[Tuple[np.ndarray, str]] = [(gray, "raw")]
    if float(gray.mean()) >= 40:
        return variants
    if float(gray.max()) >= 200 and len(np.unique(gray)) <= 16:
        return variants
    tile_w = max(2, min(8, gray.shape[1] // 8))
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, tile_w))
    enhanced = clahe.apply(gray)
    variants.append((enhanced, "clahe"))
    boosted = np.clip(enhanced.astype(np.float32) * 2.2, 0, 255).astype(np.uint8)
    variants.append((boosted, "boost"))
    return variants


def _append_decode_candidate(
    out: List[Tuple[int, float, str]],
    val: Optional[int],
    err: float,
    dbg: str,
    *,
    bonus: float = 0.0,
) -> None:
    if val is None or not plausible_coin_value(val) or _suspicious_coin_value(val):
        return
    out.append((val, err + bonus, dbg))


def _decode_gray_sources(gray: np.ndarray, bgr: Optional[np.ndarray]) -> List[Tuple[int, float, str]]:
    out: List[Tuple[int, float, str]] = []
    if float(gray.max()) >= 200 and len(np.unique(gray)) <= 16:
        val, err, dbg = _decode_binary(gray)
        _append_decode_candidate(out, val, err - 8.0, f"binary {dbg}")
        slot = _decode_slot_partition(gray)
        _append_decode_candidate(out, slot[0], slot[1] - 6.0, f"binary {slot[2]}")
    for base_gray, base_tag in _enhance_gray_variants(gray):
        for src, tag in _gray_source_variants(base_gray, bgr):
            src_tag = tag if base_tag == "raw" else f"{base_tag}/{tag}"
            for binary in _binarize_variants(src):
                val, err, dbg = _decode_binary(binary)
                bonus = 0.0
                if tag == "strip":
                    bonus -= 10.0
                elif tag in {"bottom-right", "right"}:
                    bonus += 6.0
                elif tag == "bottom":
                    bonus += 3.0
                if base_tag == "clahe":
                    bonus -= 1.0
                _append_decode_candidate(out, val, err, f"{src_tag} {dbg}", bonus=bonus)
                slot = _decode_slot_partition(binary)
                _append_decode_candidate(out, slot[0], slot[1], f"{src_tag} {slot[2]}", bonus=bonus - 2.0)
    return out


def _candidate_rank_item(item: Tuple[int, float, str]) -> tuple[float, float]:
    val, err, dbg = item
    if not plausible_coin_value(val):
        return (1e9, err)
    score = err
    try:
        from app.services.coin_digit_cnn import coin_digit_cnn_available

        if coin_digit_cnn_available() and (
            "slots" in dbg or "valley_slots" in dbg or "slot_" in dbg
        ):
            score += 90.0
    except Exception:
        pass
    digits = len(str(val))
    if digits not in _PREFERRED_DIGITS:
        score += 15.0
    if digits == 5:
        score += _FIVE_DIGIT_EXTRA_ERR
    if "bookend" in dbg:
        score -= 3.0
    if dbg.startswith("n=4") or " n=4" in dbg or "dl n=4" in dbg:
        score -= 12.0
    if " strip " in f" {dbg} " or dbg.startswith("strip "):
        score -= 8.0
    if "strip" in dbg and "valley_slots" in dbg:
        score -= 6.0
    if "bottom-right" in dbg or " bottom " in f" {dbg} ":
        score += 10.0
    if "valley_slots" in dbg:
        score += 8.0
    if "slots" in dbg and not dbg.startswith("n=4"):
        score += 10.0
    if _suspicious_coin_value(val):
        score += 50.0
    return (score, err)


def _is_box_decode_dbg(dbg: str) -> bool:
    return (
        dbg.startswith("n=4")
        or " n=4" in dbg
        or "dl n=4" in dbg
        or "head_tail" in dbg
    )


def _is_slot_decode_dbg(dbg: str) -> bool:
    return "slots" in dbg


def _pick_best_candidate(
    candidates: List[Tuple[int, float, str]],
) -> Tuple[Optional[int], float, str]:
    if not candidates:
        return None, 1e9, "no_candidates"
    ranked = sorted(candidates, key=_candidate_rank_item)
    acceptable = [
        cand
        for cand in ranked
        if plausible_coin_value(cand[0]) and cand[1] <= _MAX_ACCEPTABLE_ERR
    ]
    if not acceptable:
        top = ranked[0]
        return None, 1e9, f"err_high={top[1]:.1f} {top[2]}"
    best = acceptable[0]
    box_cands = [c for c in acceptable if _is_box_decode_dbg(c[2])]
    slot_cands = [c for c in acceptable if _is_slot_decode_dbg(c[2])]
    if box_cands and slot_cands:
        box_best = min(box_cands, key=_candidate_rank_item)
        slot_best = min(slot_cands, key=_candidate_rank_item)
        if box_best[0] != slot_best[0]:
            box_rank = _candidate_rank_item(box_best)[0]
            slot_rank = _candidate_rank_item(slot_best)[0]
            if box_rank <= slot_rank + _BOX_OVER_SLOT_MARGIN:
                best = box_best
    if len(acceptable) >= 2 and acceptable[0][0] != acceptable[1][0]:
        gap = (
            _candidate_rank_item(acceptable[1])[0]
            - _candidate_rank_item(acceptable[0])[0]
        )
        if gap < _AMBIGUOUS_RANK_GAP:
            return (
                None,
                1e9,
                f"ambiguous {acceptable[0][0]} vs {acceptable[1][0]} "
                f"({acceptable[0][2]} / {acceptable[1][2]})",
            )
    return best


@lru_cache(maxsize=1)
def _hud_soft_digit_templates() -> List[Tuple[int, np.ndarray]]:
    """ゲーム HUD 向け（薄い白文字・ややぼかし）。"""
    if not _CV2_OK:
        return []
    out: List[Tuple[int, np.ndarray]] = []
    for digit in range(10):
        for scale in (0.5, 0.62, 0.74, 0.86):
            h, w = 48, 34
            canvas = np.full((h, w), 22, dtype=np.uint8)
            cv2.putText(
                canvas,
                str(digit),
                (1, 38),
                cv2.FONT_HERSHEY_DUPLEX,
                scale,
                215,
                1,
                cv2.LINE_AA,
            )
            canvas = cv2.GaussianBlur(canvas, (7, 7), 1.6)
            out.append((digit, canvas))
    return out


def _normalize_digit_patch(patch: np.ndarray) -> np.ndarray:
    return cv2.resize(patch, (28, 40), interpolation=cv2.INTER_CUBIC)


def _match_patch_to_ref(patch: np.ndarray, ref: np.ndarray) -> float:
    a = _normalize_digit_patch(patch).astype(np.float32)
    b = ref.astype(np.float32)
    if b.shape != a.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_CUBIC)
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.mean(np.abs(a - b)))


def _synth_digit_errors(patch: np.ndarray) -> List[float]:
    errs = [1e9] * 10
    if patch is None or patch.size == 0 or patch.shape[0] < 3 or patch.shape[1] < 2:
        return errs
    tmpl_groups = [_digit_templates(), _hud_soft_digit_templates()]
    for templates in tmpl_groups:
        for digit in range(10):
            for tmpl_digit, tmpl in templates:
                if tmpl_digit != digit:
                    continue
                th, tw = tmpl.shape
                for interp in (cv2.INTER_AREA, cv2.INTER_CUBIC):
                    resized = cv2.resize(patch, (tw, th), interpolation=interp)
                    for norm in (False, True):
                        a = resized.astype(np.float32)
                        b = tmpl.astype(np.float32)
                        if norm:
                            a = (a - a.mean()) / (a.std() + 1e-6)
                            b = (b - b.mean()) / (b.std() + 1e-6)
                        errs[digit] = min(
                            errs[digit], float(np.mean(np.abs(a - b)))
                        )
    return errs


@lru_cache(maxsize=1)
def _game_hud_ref_templates() -> dict[int, List[np.ndarray]]:
    """実機 HUD（5,395）参照画像から抽出した桁テンプレート。"""
    if not _CV2_OK:
        return {}
    path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "images"
        / "coin_hud_ref_5395.png"
    )
    if not path.exists():
        return {}
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return {}
    row = _hud_digit_row_gray(gray)
    parts = _hud_split_four_digits(row) if row is not None else None
    if not parts:
        return {}
    labels = (5, 3, 9, 5)
    out: dict[int, List[np.ndarray]] = {}
    for digit, part in zip(labels, parts):
        if part.size == 0:
            continue
        out.setdefault(digit, []).append(_normalize_digit_patch(part).copy())
    return out


def _patch_digit_errors(patch: np.ndarray) -> List[float]:
    """各桁 0-9 の DL 分類誤差（coin_digit CNN + HUD テンプレ補助）。"""
    cnn_errs: Optional[List[float]] = None
    cnn_conf: Optional[float] = None
    if _coin_digit_cnn_ready():
        try:
            cnn = _get_active_digit_classifier()
            if cnn.is_loaded():
                probs = cnn.predict_probs(patch)
                if probs is not None:
                    cnn_errs = cnn.digit_errors(patch)
                    cnn_conf = float(probs.max())
        except Exception:
            pass
    synth_errs = _synth_digit_errors(patch)
    if cnn_errs is None:
        return synth_errs
    if cnn_conf is not None and cnn_conf >= 0.42:
        return cnn_errs
    out: List[float] = []
    for digit in range(10):
        cnn_e = cnn_errs[digit]
        synth_e = synth_errs[digit]
        err = cnn_e
        if synth_e < 0.68:
            err = min(err, synth_e * 0.86 + 0.05)
        elif cnn_e > 0.82 and synth_e < cnn_e:
            err = min(err, synth_e * 0.94 + 0.08)
        if digit in (0, 6) and synth_e < 0.82:
            err = min(err, synth_e * 0.70 + 0.04)
        out.append(err)
    return out


def _hud_ink_span(row: np.ndarray) -> Optional[tuple[int, int, np.ndarray]]:
    if row is None or row.size == 0:
        return None
    mask = cv2.inRange(row, 112, 255)
    proj = mask.sum(axis=0).astype(np.float32)
    ink = np.where(proj > row.shape[0] * 5)[0]
    if len(ink) < 10:
        return None
    x0, x1 = int(ink[0]), int(ink[-1]) + 1
    return x0, x1, proj


def _hud_icon_skip_x(row: np.ndarray, x0: int, x1: int) -> int:
    """コインアイコン分を除き、数字列の左端を返す。"""
    h = row.shape[0]
    mask = cv2.inRange(row, 112, 255)
    skip = _digit_strip_start_x(mask, x0, x1)
    span_x = max(1, x1 - x0)
    char_w = max(10, int(h * 0.42))
    if skip - x0 >= char_w * 0.85:
        return skip
    proj = mask.sum(axis=0).astype(np.float32)
    peak = float(proj[x0:x1].max()) if x1 > x0 else 0.0
    if peak <= 0:
        return skip
    left_peak = float(proj[x0 : min(x1, x0 + char_w)].max())
    if left_peak < peak * 0.22:
        return skip
    if x0 > char_w * 0.2:
        return skip
    low = (mask.sum(axis=0) <= h * 4).astype(np.uint8)
    low = cv2.dilate(low.reshape(1, -1), np.ones((1, 7), np.uint8)).flatten()
    search_hi = min(x1, x0 + int(char_w * 1.45))
    best_len = 0
    best_end = skip
    run = 0
    for idx in range(x0 + 4, search_hi):
        if low[idx]:
            run += 1
        else:
            if run > best_len:
                best_len = run
                best_end = idx
            run = 0
    if run > best_len:
        best_end = search_hi
    if best_len >= 6 and best_end > skip:
        return best_end
    crop = mask[:, max(0, x0) : min(mask.shape[1], x1 + 1)]
    if crop.size == 0:
        return skip
    icon_end = 0
    contours, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        bx, _by, bw, bh = cv2.boundingRect(cnt)
        if bx > int(span_x * 0.08):
            continue
        if bh >= max(8, int(h * 0.24)) and bw >= max(8, int(bh * 0.45)):
            icon_end = max(icon_end, bx + bw + 2)
    if icon_end > skip - x0:
        return x0 + icon_end
    return skip


def _hud_right_digit_edge(proj: np.ndarray, x0: int, x1: int) -> int:
    """右端の薄いインクを落とし、数字列の終端に合わせる。"""
    if x1 <= x0:
        return x1
    span = proj[x0:x1]
    if len(span) < 8:
        return x1
    peak = float(span.max())
    if peak <= 0:
        return x1
    thresh = peak * 0.14
    for rel in range(len(span) - 1, -1, -1):
        if span[rel] >= thresh:
            return x0 + rel + 1
    return x1


def _hud_trimmed_ink_span(row: np.ndarray) -> Optional[tuple[int, int, np.ndarray]]:
    """HUD 数字列だけに span を絞る（アイコン・右側ノイズを除外）。"""
    span = _hud_ink_span(row)
    if span is None:
        return None
    x0, x1, proj = span
    x0 = _hud_icon_skip_x(row, x0, x1)
    h = row.shape[0]
    char_w = max(10, int(h * 0.42))
    max_w = int(char_w * 8.2)
    x1 = min(x1, x0 + max_w)
    x1 = _hud_right_digit_edge(proj, x0, x1)
    if x1 - x0 < char_w * 2:
        return None
    return x0, x1, proj


def _hud_parts_from_cut_list(row: np.ndarray, cuts: List[int]) -> Optional[List[np.ndarray]]:
    if len(cuts) < 3:
        return None
    parts = [row[:, cuts[i] : cuts[i + 1]] for i in range(len(cuts) - 1)]
    if any(p.shape[1] < 3 for p in parts):
        return None
    if not _hud_parts_plausible(row, parts):
        return None
    return parts


def _hud_parts_from_cuts(row: np.ndarray, cuts: List[int]) -> Optional[List[np.ndarray]]:
    if len(cuts) != 5:
        return None
    return _hud_parts_from_cut_list(row, cuts)


def _hud_max_part_width(row: np.ndarray) -> int:
    return max(12, int(row.shape[0] * 0.75))


def _hud_parts_plausible(row: np.ndarray, parts: List[np.ndarray]) -> bool:
    max_w = _hud_max_part_width(row)
    return all(4 <= part.shape[1] <= max_w for part in parts)


def _hud_comma_parts_plausible(row: np.ndarray, parts: List[np.ndarray]) -> bool:
    if len(parts) != 4 or not _hud_parts_plausible(row, parts[1:]):
        return False
    char_w = max(10, int(row.shape[0] * 0.42))
    return parts[0].shape[1] <= max(12, int(char_w * 1.6))


def _hud_split_four_digits(row: np.ndarray) -> Optional[List[np.ndarray]]:
    """インク列を 4 桁分に分割（参照 5,395 と同型レイアウト）。"""
    span = _hud_trimmed_ink_span(row)
    if span is None:
        return None
    x0, x1, _proj = span
    width = max(1, x1 - x0)
    ratios = (0.10, 0.20, 0.30, 0.50, 0.72)
    cuts = [x0 + int(width * ratio) for ratio in ratios]
    cuts[0] = max(x0 + 1, cuts[0])
    cuts[-1] = min(x1, max(cuts[-2] + 4, cuts[-1]))
    return _hud_parts_from_cuts(row, cuts)


@lru_cache(maxsize=1)
def _hud_ref_cut_ratios() -> tuple[float, float, float, float, float]:
    """参照 5,395 画像の谷位置から得た分割比率。"""
    path = (
        Path(__file__).resolve().parents[1]
        / "assets"
        / "images"
        / "coin_hud_ref_5395.png"
    )
    if not path.exists():
        return (0.10, 0.20, 0.30, 0.50, 0.72)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return (0.10, 0.20, 0.30, 0.50, 0.72)
    row = _hud_digit_row_gray(gray)
    if row is None:
        return (0.10, 0.20, 0.30, 0.50, 0.72)
    span = _hud_ink_span(row)
    if span is None:
        return (0.10, 0.20, 0.30, 0.50, 0.72)
    x0, x1, proj = span
    cuts = _hud_valley_cut_indices(proj, x0, x1)
    if cuts is None:
        return (0.10, 0.20, 0.30, 0.50, 0.72)
    width = max(1, x1 - x0)
    return tuple((cuts[i] - x0) / width for i in range(5))  # type: ignore[return-value]


def _hud_split_four_digits_ref(row: np.ndarray) -> Optional[List[np.ndarray]]:
    span = _hud_trimmed_ink_span(row)
    if span is None:
        return None
    x0, x1, _proj = span
    width = max(1, x1 - x0)
    ratios = _hud_ref_cut_ratios()
    cuts = [x0 + int(width * ratio) for ratio in ratios]
    cuts[0] = max(x0 + 1, cuts[0])
    cuts[-1] = min(x1, max(cuts[-2] + 4, cuts[-1]))
    return _hud_parts_from_cuts(row, cuts)


def _hud_valley_cut_indices(proj: np.ndarray, x0: int, x1: int) -> Optional[List[int]]:
    width = max(1, x1 - x0)
    span = proj[x0:x1]
    if len(span) < 16:
        return None
    smooth = np.convolve(span, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    margin = max(3, int(width * 0.07))
    valleys: List[tuple[float, int]] = []
    for idx in range(margin, len(smooth) - margin):
        left = smooth[idx - 1]
        mid = smooth[idx]
        right = smooth[idx + 1]
        if mid <= left and mid <= right:
            valleys.append((float(mid), idx))
    if not valleys:
        return None
    valleys.sort()
    min_gap = max(4, int(width * 0.11))
    picked: List[int] = []
    for _depth, rel_idx in valleys:
        if any(abs(rel_idx - p) < min_gap for p in picked):
            continue
        picked.append(rel_idx)
        if len(picked) == 3:
            break
    if len(picked) != 3:
        return None
    picked.sort()
    return [x0, x0 + picked[0], x0 + picked[1], x0 + picked[2], x1]


def _hud_valley_n_part_cuts(
    proj: np.ndarray, x0: int, x1: int, *, valleys_needed: int
) -> Optional[List[int]]:
    """谷を valleys_needed 個拾い (valleys_needed+1) 分割の切り位置を返す。"""
    width = max(1, x1 - x0)
    span = proj[x0:x1]
    if len(span) < 12:
        return None
    smooth = np.convolve(span, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    margin = max(2, int(width * 0.05))
    valleys: List[tuple[float, int]] = []
    for idx in range(margin, len(smooth) - margin):
        left = smooth[idx - 1]
        mid = smooth[idx]
        right = smooth[idx + 1]
        if mid <= left and mid <= right:
            valleys.append((float(mid), idx))
    if not valleys:
        return None
    valleys.sort()
    min_gap = max(3, int(width * 0.08))
    picked: List[int] = []
    for _depth, rel_idx in valleys:
        if any(abs(rel_idx - p) < min_gap for p in picked):
            continue
        picked.append(rel_idx)
        if len(picked) == valleys_needed:
            break
    if len(picked) != valleys_needed:
        return None
    picked.sort()
    cuts = [x0]
    cuts.extend(x0 + p for p in picked)
    cuts.append(x1)
    return cuts


def _hud_subpatch_ink_span(row: np.ndarray) -> Optional[tuple[int, int, np.ndarray]]:
    """単桁・尾桁など部分切り出し向け（全行用の最小幅制約なし）。"""
    span = _hud_ink_span(row)
    if span is None:
        return None
    x0, x1, proj = span
    if x1 - x0 < 4:
        return None
    return x0, x1, proj


def _hud_find_leading_comma_idx(span_proj: np.ndarray, width: int) -> Optional[int]:
    """1桁+カンマ+3桁（5,395 / 3,065）の先頭カンマ直後（尾3桁開始）。"""
    peak = float(span_proj.max())
    if peak <= 0 or width < 20:
        return None
    thresh = peak * 0.10
    first_digit_w = max(6, int(width * 0.14))
    if float(span_proj[:first_digit_w].max()) < peak * 0.30:
        return None
    lead_end = max(5, int(width * 0.05))
    search = max(14, int(width * 0.52))
    run_start: Optional[int] = None
    best_end: Optional[int] = None
    best_len = 0
    for idx in range(lead_end, min(search, len(span_proj) - 3)):
        if float(span_proj[idx]) <= thresh:
            if run_start is None:
                run_start = idx
        elif run_start is not None:
            run_len = idx - run_start
            if run_len >= 2 and float(span_proj[idx:].max()) >= peak * 0.30:
                if run_len > best_len:
                    best_len = run_len
                    best_end = idx
            run_start = None
    return best_end


def _hud_tail_three_digit_parts(
    tail: np.ndarray, tx0: int, tx1: int, proj: np.ndarray
) -> Optional[List[np.ndarray]]:
    """カンマ右の3桁を切り出す（谷優先、ダメなら等分）。"""
    width = max(1, tx1 - tx0)
    if width < 12:
        return None
    cuts = _hud_valley_n_part_cuts(proj, tx0, tx1, valleys_needed=2)
    if cuts is not None and len(cuts) == 4:
        parts = [tail[:, cuts[i] : cuts[i + 1]] for i in range(3)]
        widths = [p.shape[1] for p in parts]
        if all(w >= 4 for w in widths) and max(widths) <= min(widths) * 3.2:
            return parts
    c1 = tx0 + int(width * 0.33)
    c2 = tx0 + int(width * 0.66)
    parts = [tail[:, tx0:c1], tail[:, c1:c2], tail[:, c2:tx1]]
    if any(p.shape[1] < 4 for p in parts):
        return None
    return parts


def _hud_leading_digit_patch(row_left: np.ndarray) -> np.ndarray:
    """カンマ左の先頭1桁だけを切り出す（カンマ列を除外）。"""
    span = _hud_subpatch_ink_span(row_left)
    if span is None:
        return row_left
    x0, x1, proj = span
    width = max(1, x1 - x0)
    sub = proj[x0:x1].astype(np.float32)
    peak = float(sub.max())
    if peak <= 0:
        return row_left[:, x0:x1]
    min_w = max(4, int(width * 0.28))
    for rel in range(width - 2, min_w, -1):
        if sub[rel] <= peak * 0.18:
            cut = x0 + rel
            if cut - x0 >= 4:
                return row_left[:, x0:cut]
    cut = x0 + max(4, int(width * 0.52))
    if cut - x0 >= 4:
        return row_left[:, x0:cut]
    return row_left[:, x0:x1]


def _hud_comma_split_variant_list(row: np.ndarray) -> List[List[np.ndarray]]:
    """カンマ検出 + 尾3桁の分割比率を複数試す。"""
    span = _hud_trimmed_ink_span(row)
    if span is None:
        return []
    x0, x1, proj = span
    span_proj = proj[x0:x1]
    if len(span_proj) < 16:
        return []
    width = max(1, x1 - x0)
    comma_rel = _hud_find_leading_comma_idx(span_proj, width)
    if comma_rel is None:
        return []
    comma_candidates = sorted(
        set(
            comma_rel + delta
            for delta in range(-3, 4)
            if 8 <= comma_rel + delta <= width - 14
        )
    )
    ratio_pairs = ((0.28, 0.58), (0.30, 0.62), (0.33, 0.66), (0.35, 0.68))
    out: List[List[np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()
    for comma_rel in comma_candidates:
        comma_x = x0 + comma_rel
        left = row[:, x0:comma_x]
        tail = row[:, comma_x:x1]
        if left.shape[1] < 4 or tail.shape[1] < 12:
            continue
        left_trim = _hud_subpatch_ink_span(left)
        if left_trim is not None:
            lx0, lx1, _ = left_trim
            left = left[:, lx0:lx1]
        left = _hud_leading_digit_patch(left)
        tail_span = _hud_subpatch_ink_span(tail)
        if tail_span is None:
            continue
        tx0, tx1, tproj = tail_span
        tail_w = max(1, tx1 - tx0)
        attempts: List[List[np.ndarray]] = []
        valley_parts = _hud_tail_three_digit_parts(tail, tx0, tx1, tproj)
        if valley_parts is not None:
            attempts.append(valley_parts)
        for r1, r2 in ratio_pairs:
            c1 = tx0 + int(tail_w * r1)
            c2 = tx0 + int(tail_w * r2)
            if c2 - c1 < 4 or c1 - tx0 < 4 or tx1 - c2 < 4:
                continue
            attempts.append(
                [tail[:, tx0:c1], tail[:, c1:c2], tail[:, c2:tx1]]
            )
        for tail_parts in attempts:
            parts = [left, tail_parts[0], tail_parts[1], tail_parts[2]]
            if not _hud_comma_parts_plausible(row, parts):
                continue
            key = tuple(p.shape[1] for p in parts)
            if key in seen:
                continue
            seen.add(key)
            out.append(parts)
    return out


def _hud_split_comma_four_digits(row: np.ndarray) -> Optional[List[np.ndarray]]:
    """5,395 / 3,065 型（1桁+カンマ+3桁）向け分割。"""
    variants = _hud_comma_split_variant_list(row)
    return variants[0] if variants else None


def _hud_split_four_digits_valley(row: np.ndarray) -> Optional[List[np.ndarray]]:
    """垂直投影の谷で 4 桁分割（カンマ位置も谷として扱う）。"""
    span = _hud_trimmed_ink_span(row)
    if span is None:
        return None
    x0, x1, proj = span
    cuts = _hud_valley_cut_indices(proj, x0, x1)
    if cuts is None:
        return None
    return _hud_parts_from_cuts(row, cuts)


def _hud_split_four_digit_variants(row: np.ndarray) -> List[tuple[List[np.ndarray], str]]:
    variants: List[tuple[List[np.ndarray], str]] = []
    seen: set[tuple[int, ...]] = set()
    for parts in _hud_comma_split_variant_list(row):
        key = tuple(p.shape[1] for p in parts)
        if key in seen:
            continue
        seen.add(key)
        variants.append((parts, "comma"))
    for parts, tag in (
        (_hud_split_four_digits_valley(row), "valley"),
        (_hud_split_four_digits_ref(row), "ref"),
        (_hud_split_four_digits(row), "ratio"),
    ):
        if parts is None:
            continue
        key = tuple(p.shape[1] for p in parts)
        if key in seen:
            continue
        seen.add(key)
        variants.append((parts, tag))
    return variants


def _hud_split_n_digits_valley(row: np.ndarray, n: int) -> Optional[List[np.ndarray]]:
    if n < _MIN_DIGITS or n > _MAX_DIGITS:
        return None
    span = _hud_trimmed_ink_span(row)
    if span is None:
        return None
    x0, x1, proj = span
    cuts = _hud_valley_n_part_cuts(proj, x0, x1, valleys_needed=n - 1)
    if cuts is None:
        return None
    return _hud_parts_from_cut_list(row, cuts)


def _hud_split_n_digits_ratio(row: np.ndarray, n: int) -> Optional[List[np.ndarray]]:
    if n < _MIN_DIGITS or n > _MAX_DIGITS:
        return None
    span = _hud_trimmed_ink_span(row)
    if span is None:
        return None
    x0, x1, _proj = span
    width = max(1, x1 - x0)
    cuts = [x0 + int(width * i / n) for i in range(n)] + [x1]
    cuts[0] = max(x0 + 1, cuts[0])
    cuts[-1] = min(x1, max(cuts[-2] + 3, cuts[-1]))
    return _hud_parts_from_cut_list(row, cuts)


def _hud_split_subrow_n_digits(sub: np.ndarray, n: int) -> Optional[List[np.ndarray]]:
    if n < 1 or n > 4:
        return None
    span = _hud_subpatch_ink_span(sub)
    if span is None:
        return None
    x0, x1, proj = span
    if n == 1:
        part = sub[:, x0:x1]
        return [part] if part.shape[1] >= 3 else None
    cuts = _hud_valley_n_part_cuts(proj, x0, x1, valleys_needed=n - 1)
    if cuts is not None:
        parts = _hud_parts_from_cut_list(sub, cuts)
        if parts is not None and len(parts) == n:
            return parts
    width = max(1, x1 - x0)
    cuts = [x0 + int(width * i / n) for i in range(n)] + [x1]
    parts = _hud_parts_from_cut_list(sub, cuts)
    if parts is not None and len(parts) == n:
        return parts
    return None


def _hud_comma_n_digit_variants(row: np.ndarray, n: int) -> List[List[np.ndarray]]:
    """カンマ区切り表示向け N 桁分割（4=1+3, 5=2+3, 6=3+3, 7=1+3+3）。"""
    if n not in (4, 5, 6, 7):
        return []
    if n == 4:
        return _hud_comma_split_variant_list(row)

    span = _hud_trimmed_ink_span(row)
    if span is None:
        return []
    x0, x1, proj = span
    span_proj = proj[x0:x1]
    width = max(1, x1 - x0)
    if len(span_proj) < 16:
        return []

    out: List[List[np.ndarray]] = []
    seen: set[tuple[int, ...]] = set()

    def _append(parts: List[np.ndarray]) -> None:
        if len(parts) != n or not _hud_parts_plausible(row, parts):
            return
        key = tuple(p.shape[1] for p in parts)
        if key in seen:
            return
        seen.add(key)
        out.append(parts)

    if n in (5, 6):
        comma_rel = _hud_find_leading_comma_idx(span_proj, width)
        if comma_rel is None:
            return out
        for delta in range(-4, 5):
            comma_x = x0 + comma_rel + delta
            if comma_x - x0 < 8 or x1 - comma_x < 12:
                continue
            left = row[:, x0:comma_x]
            right = row[:, comma_x:x1]
            left_n = 2 if n == 5 else 3
            right_n = 3
            left_parts = _hud_split_subrow_n_digits(left, left_n)
            right_span = _hud_subpatch_ink_span(right)
            if left_parts is None or right_span is None:
                continue
            rx0, rx1, rproj = right_span
            right_parts = _hud_tail_three_digit_parts(right, rx0, rx1, rproj)
            if right_parts is None and right_n == 3:
                right_parts = _hud_split_subrow_n_digits(right[:, rx0:rx1], 3)
            if right_parts is None:
                continue
            _append(left_parts + right_parts)

    if n == 7:
        comma_rel = _hud_find_leading_comma_idx(span_proj, width)
        if comma_rel is None:
            return out
        for delta in range(-4, 5):
            comma_x = x0 + comma_rel + delta
            if comma_x - x0 < 6 or x1 - comma_x < 20:
                continue
            lead = _hud_leading_digit_patch(row[:, x0:comma_x])
            rest = row[:, comma_x:x1]
            rest_span = _hud_subpatch_ink_span(rest)
            if rest_span is None:
                continue
            rx0, rx1, rproj = rest_span
            rest_trim = rest[:, rx0:rx1]
            inner_proj = rproj[rx0:rx1]
            inner_w = max(1, rx1 - rx0)
            inner_comma = _hud_find_leading_comma_idx(inner_proj, inner_w)
            if inner_comma is None:
                continue
            for delta2 in range(-3, 4):
                mid_x = rx0 + inner_comma + delta2
                if mid_x - rx0 < 10 or rx1 - mid_x < 10:
                    continue
                mid = rest_trim[:, : mid_x - rx0]
                tail = rest_trim[:, mid_x - rx0 :]
                mid_parts = _hud_split_subrow_n_digits(mid, 3)
                tail_span = _hud_subpatch_ink_span(tail)
                if mid_parts is None or tail_span is None:
                    continue
                tx0, tx1, tproj = tail_span
                tail_parts = _hud_tail_three_digit_parts(tail, tx0, tx1, tproj)
                if tail_parts is None:
                    tail_parts = _hud_split_subrow_n_digits(tail[:, tx0:tx1], 3)
                if tail_parts is None:
                    continue
                _append([lead, mid_parts[0], mid_parts[1], mid_parts[2], tail_parts[0], tail_parts[1], tail_parts[2]])

    return out


def _hud_split_n_digit_variants(row: np.ndarray, n: int) -> List[tuple[List[np.ndarray], str]]:
    if n == 4:
        return _hud_split_four_digit_variants(row)
    variants: List[tuple[List[np.ndarray], str]] = []
    seen: set[tuple[int, ...]] = set()

    def _add(parts: Optional[List[np.ndarray]], tag: str) -> None:
        if parts is None or len(parts) != n:
            return
        key = tuple(p.shape[1] for p in parts)
        if key in seen:
            return
        seen.add(key)
        variants.append((parts, tag))

    for parts in _hud_comma_n_digit_variants(row, n):
        _add(parts, "comma")
    _add(_hud_split_n_digits_valley(row, n), "valley")
    _add(_hud_split_n_digits_ratio(row, n), "ratio")
    return variants


def _hud_estimate_digit_count(row: np.ndarray) -> int:
    h = max(1, row.shape[0])
    char_w = max(10, int(h * 0.42))
    est = int(round(row.shape[1] / char_w))
    return max(_MIN_DIGITS, min(_MAX_DIGITS, est))


def _digits_to_value(digits: List[int]) -> int:
    value = 0
    for d in digits:
        value = value * 10 + d
    return value


def _decode_hud_n_parts(
    parts: List[np.ndarray], tag: str
) -> Optional[Tuple[int, float, str]]:
    n = len(parts)
    if n < _MIN_DIGITS or n > _MAX_DIGITS:
        return None
    part_errs = [_patch_digit_errors(p) for p in parts]
    best_val: Optional[int] = None
    best_err = 1e9
    threshold = 1.25

    def _try_value(digits: List[int], total: float) -> None:
        nonlocal best_val, best_err
        if total >= best_err:
            return
        value = _digits_to_value(digits)
        if not plausible_coin_value(value) or _suspicious_coin_value(value):
            return
        if len(str(value)) != n:
            return
        best_err = total
        best_val = value

    if n <= 5:
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
        digits: List[int] = []
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
    if mean_err > 1.2:
        return None
    bonus = 20.0 if n == 4 else 12.0
    return best_val, mean_err - bonus, f"hud n={n} {tag}"


def _decode_hud_four_parts(
    parts: List[np.ndarray], tag: str
) -> Optional[Tuple[int, float, str]]:
    if len(parts) != 4:
        return None
    return _decode_hud_n_parts(parts, tag)


def _hud_digit_row_gray(gray: np.ndarray) -> Optional[np.ndarray]:
    h, w = gray.shape[:2]
    target_h = 120.0
    if h < target_h * 0.92:
        scale = target_h / max(1.0, h)
        up = cv2.resize(
            gray,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_LANCZOS4,
        )
    else:
        up = gray
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)).apply(up)
    h = clahe.shape[0]
    row = clahe[int(h * 0.12) : int(h * 0.90), :]
    trim = _hud_trimmed_ink_span(row)
    if trim is None:
        return None
    x0, x1, _proj = trim
    if x1 - x0 < 12:
        return None
    return row[:, x0:x1]


def _decode_hud_greedy_cnn(
    parts: List[np.ndarray], tag: str
) -> Optional[Tuple[int, float, str]]:
    """各パッチの CNN argmax をそのまま採用（分割が合っていれば最も安定）。"""
    if not _coin_digit_cnn_ready():
        return None
    cnn = _get_active_digit_classifier()
    if not cnn.is_loaded():
        return None
    n = len(parts)
    if n < _MIN_DIGITS or n > _MAX_DIGITS:
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
        if conf < 0.22:
            return None
        min_conf = min(min_conf, conf)
        digits.append(d)
        err_sum += 1.0 - conf
    value = _digits_to_value(digits)
    if not plausible_coin_value(value) or _suspicious_coin_value(value):
        return None
    if len(str(value)) != n:
        return None
    mean_err = (err_sum / float(n)) * 100.0
    if mean_err > 78.0:
        return None
    bonus = 28.0 if tag == "comma" and n in (4, 5) else 18.0 if n == 4 else 10.0
    bonus += min_conf * 12.0
    return value, mean_err - bonus, f"greedy n={n} {tag}"


def _decode_hud_n_parts_cnn(
    parts: List[np.ndarray], tag: str
) -> Optional[Tuple[int, float, str]]:
    """CNN N 桁復元（ルール補正なし）。"""
    decoded = _decode_hud_n_parts(parts, tag)
    if decoded is None:
        return None
    val, err, dbg = decoded
    n = len(parts)
    return val, err, dbg.replace(f"hud n={n}", f"dl n={n}")


def _decode_hud_four_parts_cnn(
    parts: List[np.ndarray], tag: str
) -> Optional[Tuple[int, float, str]]:
    return _decode_hud_n_parts_cnn(parts, tag)


def _hud_decode_rank(item: Tuple[int, float, str], *, prefer_ref_split: bool) -> tuple[int, float, float]:
    """分割方式の優先度。低解像度 crop では ref/valley、通常は ratio を優先。"""
    _val, err, dbg = item
    tag = dbg.rsplit(" ", 1)[-1] if dbg else ""
    if prefer_ref_split:
        order = {"comma": 0, "ref": 1, "valley": 2, "ratio": 3}.get(tag, 9)
    else:
        order = {"comma": 0, "ratio": 1, "ref": 2, "valley": 3}.get(tag, 9)
    return (order, err, err)


def _coin_digit_cnn_ready() -> bool:
    try:
        return _get_active_digit_classifier().is_loaded()
    except Exception:
        return False


def _try_hud_digit_decode(gray: np.ndarray) -> Optional[Tuple[int, float, str]]:
    """実機 HUD 向け: 1〜7 桁分割 + coin_digit CNN。"""
    if not _coin_digit_cnn_ready():
        return None
    row = _hud_digit_row_gray(gray)
    if row is None:
        return None
    est_n = _hud_estimate_digit_count(row)
    try_ns = sorted(
        range(_MIN_DIGITS, _MAX_DIGITS + 1),
        key=lambda n: (abs(n - est_n), n),
    )
    decoded_list: List[Tuple[int, float, str]] = []
    for n in try_ns:
        for parts, tag in _hud_split_n_digit_variants(row, n):
            greedy = _decode_hud_greedy_cnn(parts, tag)
            if greedy is not None:
                decoded_list.append(greedy)
            decoded = _decode_hud_n_parts_cnn(parts, tag)
            if decoded is None:
                continue
            decoded_list.append(decoded)
    if not decoded_list:
        return None

    def _pick_key(item: Tuple[int, float, str]) -> tuple:
        val, err, dbg = item
        n = len(str(val))
        tag = dbg.rsplit(" ", 1)[-1] if dbg else ""
        is_greedy = dbg.startswith("greedy")
        tag_order = {"comma": 0, "ref": 1, "valley": 2, "ratio": 3}.get(tag, 9)
        if is_greedy:
            tag_order = 0
        digit_pref = 0.0 if 4 <= n <= 6 else 6.0
        return (err + digit_pref, tag_order, -n, -val)

    return min(decoded_list, key=_pick_key)


def _try_hud_four_digit_decode(gray: np.ndarray) -> Optional[Tuple[int, float, str]]:
    return _try_hud_digit_decode(gray)


def _upscale_gray_for_hud(gray: np.ndarray, bgr: Optional[np.ndarray]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = gray.shape[:2]
    if h >= 56:
        return gray, bgr
    scale = max(8.0, 120.0 / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    gray_up = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    bgr_up = None
    if bgr is not None:
        bgr_up = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return gray_up, bgr_up


def _coin_gain_strip_for_hud(gray: np.ndarray, bgr: Optional[np.ndarray]) -> np.ndarray:
    """横長 coin_gain ROI を DL 4 桁読取向けに数字列だけへ絞る。"""
    if gray is None or gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    aspect = w / max(1, h)
    if aspect <= 3.0:
        prepared, _ = _upscale_gray_for_hud(gray, bgr)
        return prepared
    strip = _gray_digit_strip(gray, bgr) if bgr is not None else None
    if strip is None or strip.size == 0 or strip.shape[1] < 16:
        prepared, _ = _upscale_gray_for_hud(gray, bgr)
        return prepared
    sh, sw = strip.shape[:2]
    target_h = max(96, min(120, sh * 4)) if sh < 40 else max(72, min(120, sh * 4))
    scale = target_h / max(1, sh)
    return cv2.resize(
        strip,
        (max(1, int(sw * scale)), max(1, int(sh * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def read_coin_gain_crop(roi: QImage) -> Tuple[Optional[int], float, str]:
    """coin_gain 切り抜き向け DL 読取（coin_digit CNN）。"""
    if not _CV2_OK:
        return None, 1e9, "opencv未導入"
    if not _coin_digit_cnn_ready():
        return None, 1e9, "coin_digit未学習"
    if roi is None or roi.isNull() or roi.width() < 12 or roi.height() < 6:
        return None, 1e9, "crop_roi小"
    gray = qimage_to_gray(roi)
    if gray is None:
        return None, 1e9, "gray失敗"
    bgr = qimage_to_bgr(roi)
    candidates: List[Tuple[int, float, str]] = []
    row_hud = _try_hud_digit_decode(gray)
    if row_hud is not None:
        _append_decode_candidate(
            candidates, row_hud[0], row_hud[1], f"crop {row_hud[2]}", bonus=-36.0
        )
    prepared_full, _ = _upscale_gray_for_hud(gray, bgr)
    full_hud = _try_hud_digit_decode(prepared_full)
    if full_hud is not None:
        _append_decode_candidate(
            candidates, full_hud[0], full_hud[1], f"crop {full_hud[2]}", bonus=-34.0
        )
    strip_hud = _try_hud_digit_decode(_coin_gain_strip_for_hud(gray, bgr))
    if strip_hud is not None:
        _append_decode_candidate(
            candidates, strip_hud[0], strip_hud[1], f"crop {strip_hud[2]}", bonus=-26.0
        )
    if not candidates:
        return None, 1e9, "crop_dl失敗"
    ranked = sorted(candidates, key=_candidate_rank_item)
    if len(ranked) >= 2:
        top_val, top_err, top_dbg = ranked[0]
        alt_val, alt_err, alt_dbg = ranked[1]
        if (
            str(top_val).startswith("6")
            and not str(alt_val).startswith("6")
            and len(str(top_val)) == len(str(alt_val))
            and alt_err <= top_err + 18.0
        ):
            ranked = [ranked[1], ranked[0]] + ranked[2:]
    acceptable = [
        cand
        for cand in ranked
        if plausible_coin_value(cand[0])
        and not _suspicious_coin_value(cand[0])
        and cand[1] <= _MAX_HUD_ACCEPTABLE_ERR
    ]
    if not acceptable:
        top = ranked[0]
        return None, 1e9, f"crop_err_high={top[1]:.1f} {top[2]}"
    best = acceptable[0]
    return best[0], best[1], f"ok err={best[1]:.1f} {best[2]}"


def read_coin_hud(roi: QImage) -> Tuple[Optional[int], float, str]:
    """bonus HUD 等向け DL 読取。"""
    val, err, dbg = read_coin_gain_crop(roi)
    if val is None:
        return val, err, dbg.replace("crop_", "hud_").replace("crop ", "hud ")
    return val, err, dbg.replace("crop ", "hud ")


def read_coin_gain(roi: QImage) -> Tuple[Optional[int], float, str]:
    """獲得コイン ROI 向け DL 読取（read_coin_gain_crop と同一）。"""
    return read_coin_gain_crop(roi)


def _reads_pass_quality_gate(matching: List[Tuple[float, str]], *, max_avg_err: float) -> bool:
    if not matching:
        return False
    if (sum(err for err, _dbg in matching) / len(matching)) > max_avg_err:
        return False
    if all(
        ("valley_slots" in dbg or "slots=" in dbg) and not _is_box_decode_dbg(dbg)
        for _err, dbg in matching
    ):
        return False
    return True


def coin_gain_confirmed_hud(reads: List[Tuple[int, float, str]]) -> bool:
    """bonus 画面 HUD 向け（2フレーム一致で可）。"""
    if not reads:
        return False
    picked = consensus_coin_gain([(value, err) for value, err, _dbg in reads])
    if picked is None:
        return False
    matching = [(err, dbg) for value, err, dbg in reads if value == picked]
    if len(matching) < _MIN_HUD_CONSENSUS_READS:
        return False
    return _reads_pass_quality_gate(matching, max_avg_err=_MAX_HUD_CONSENSUS_AVG_ERR)


def coin_gain_confirmed(reads: List[Tuple[int, float, str]]) -> bool:
    """コイン獲得画面向け（通常3フレーム / 高品質2フレーム一致）。"""
    if not reads:
        return False
    picked = consensus_coin_gain([(value, err) for value, err, _dbg in reads])
    if picked is None:
        return False
    matching = [(err, dbg) for value, err, dbg in reads if value == picked]
    min_reads = _MIN_CONSENSUS_READS
    if matching and all(
        _is_box_decode_dbg(dbg) or "n=4" in dbg or "head_tail" in dbg
        for _err, dbg in matching
    ):
        min_reads = 2
    if len(matching) < min_reads:
        return False
    return _reads_pass_quality_gate(matching, max_avg_err=_MAX_CONSENSUS_AVG_ERR)


def coin_gain_confirmed_combined(
    hud_reads: List[Tuple[int, float, str]],
    gain_reads: List[Tuple[int, float, str]],
) -> bool:
    return coin_gain_confirmed_hud(hud_reads) or coin_gain_confirmed(gain_reads)


def consensus_coin_gain(reads: List[Tuple[int, float]]) -> Optional[int]:
    """複数フレームの読み取りから代表値を選ぶ（max ではなく多数決＋低誤差）。"""
    if not reads:
        return None
    filtered = [
        (v, e)
        for v, e in reads
        if plausible_coin_value(v) and not _suspicious_coin_value(v)
    ]
    if not filtered:
        return None
    buckets: dict[int, List[float]] = {}
    for value, err in filtered:
        buckets.setdefault(value, []).append(err)
    if not buckets:
        return None

    def rank(item: tuple[int, List[float]]) -> tuple[int, float, int]:
        value, errs = item
        n = len(errs)
        avg_err = sum(errs) / n
        digits = len(str(value))
        digit_penalty = 0.0
        if digits not in _PREFERRED_DIGITS:
            digit_penalty += 8.0
        if digits == 5:
            digit_penalty += 12.0
        return (-n, avg_err + digit_penalty, -value)

    return min(buckets.items(), key=rank)[0]

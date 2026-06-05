"""コイン獲得画面の数字領域を読み取る（OpenCV テンプレート照合）。"""
from __future__ import annotations

from functools import lru_cache
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
_MIN_DIGIT_HEIGHT_RATIO = 0.35
_MAX_DIGIT_HEIGHT_RATIO = 1.05
_MIN_DIGIT_WIDTH_RATIO = 0.28
_MIN_DIGITS = 3
_MAX_DIGITS = 6
_PREFERRED_DIGITS = (4, 5)


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
    """黄色マスクの縦線化け（11111 / 17111 等）を弾く。"""
    s = str(value)
    if len(s) < 4:
        return False
    ones = s.count("1")
    if ones >= len(s) - 1:
        return True
    if s.startswith("17") and ones >= 2:
        return True
    return False


@lru_cache(maxsize=1)
def _digit_templates() -> List[Tuple[int, np.ndarray]]:
    if not _CV2_OK:
        return []
    out: List[Tuple[int, np.ndarray]] = []
    for digit in range(10):
        for scale in (0.9, 1.0, 1.15):
            h, w = 36, 26
            canvas = np.zeros((h, w), dtype=np.uint8)
            cv2.putText(
                canvas,
                str(digit),
                (2, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                255,
                2,
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
    span_x = max(1, x1 - x0)
    dx0 = x0 + int(span_x * 0.28)
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
    return bw < max(2, int(bh * 0.18))


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
        if x <= px + pw + max(4, int(pw * 0.15)):
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


def _match_digit(
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
    err = (head_err + tail_err) / (1 + len(tail_digits))
    penalty = 0.0
    total_digits = len(str(value))
    if len(boxes) >= 3 and len(tail_digits) < 3:
        penalty += 40.0
    if total_digits not in _PREFERRED_DIGITS:
        penalty += 12.0
    return value, err + penalty, f"bookend n={1 + len(tail_digits)}"


def _decode_box_group(
    binary: np.ndarray, boxes: List[Tuple[int, int, int, int]]
) -> Tuple[Optional[int], float, str]:
    digits: List[str] = []
    total_err = 0.0
    for box in boxes:
        expanded = _expand_box_to_digits(binary, box)
        if not expanded:
            return None, 1e9, f"match_fail@{box[0]}"
        for digit, err in expanded:
            digits.append(str(digit))
            total_err += err
    if len(digits) < _MIN_DIGITS:
        return None, 1e9, f"digit_count={len(digits)}"
    try:
        value = int("".join(digits))
    except ValueError:
        return None, 1e9, "parse_fail"
    if value < 1 or value > 999_999:
        return None, 1e9, "range"
    if _suspicious_coin_value(value):
        return None, 1e9, "suspicious"
    mean_err = total_err / len(digits)
    penalty = 0.0 if len(digits) in _PREFERRED_DIGITS else 12.0
    return value, mean_err + penalty, f"n={len(digits)}"


def _decode_binary(binary: np.ndarray) -> Tuple[Optional[int], float, str]:
    boxes = _select_primary_row(_segment_digit_boxes(binary))
    if not boxes:
        return None, 1e9, "digit_boxes=0"
    if _stroke_artifact_boxes(boxes):
        return None, 1e9, "stroke_artifact"

    best: Optional[Tuple[int, float, str]] = None
    bookend = _try_bookend_decode(binary, boxes)
    if bookend is not None:
        best = bookend
    for start in range(len(boxes)):
        for end in range(start + _MIN_DIGITS, min(start + _MAX_DIGITS + 1, len(boxes) + 1)):
            window = boxes[start:end]
            val, err, dbg = _decode_box_group(binary, window)
            if val is None:
                continue
            if best is None or err < best[1]:
                best = (val, err, dbg)
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


def _decode_gray_sources(gray: np.ndarray, bgr: Optional[np.ndarray]) -> List[Tuple[int, float, str]]:
    out: List[Tuple[int, float, str]] = []
    if float(gray.max()) >= 200 and len(np.unique(gray)) <= 16:
        val, err, dbg = _decode_binary(gray)
        if val is not None:
            out.append((val, err - 8.0, f"binary {dbg}"))
    for base_gray, base_tag in _enhance_gray_variants(gray):
        for src, tag in _gray_source_variants(base_gray, bgr):
            src_tag = tag if base_tag == "raw" else f"{base_tag}/{tag}"
            for binary in _binarize_variants(src):
                val, err, dbg = _decode_binary(binary)
                if val is not None:
                    bonus = 0.0
                    if tag in {"strip", "bottom-right", "right"}:
                        bonus -= 4.0
                    elif tag == "bottom":
                        bonus -= 2.0
                    if base_tag == "clahe":
                        bonus -= 1.0
                    out.append((val, err + bonus, f"{src_tag} {dbg}"))
    return out


def read_coin_gain(roi: QImage) -> Tuple[Optional[int], float, str]:
    """トリミング ROI から獲得コイン数を推定。 (値, 誤差, debug)。"""
    if not _CV2_OK:
        return None, 1e9, "opencv未導入"
    if roi is None or roi.isNull():
        return None, 1e9, "roi空"
    if roi.width() < 12 or roi.height() < 8:
        return None, 1e9, "roi小"

    gray = qimage_to_gray(roi)
    if gray is None:
        return None, 1e9, "gray失敗"
    bgr = qimage_to_bgr(roi)
    candidates = _decode_gray_sources(gray, bgr)
    if not candidates:
        return None, 1e9, "decode失敗"

    def _candidate_rank(item: Tuple[int, float, str]) -> tuple[float, float]:
        val, err, dbg = item
        score = err
        if len(str(val)) not in _PREFERRED_DIGITS:
            score += 15.0
        if "bookend" in dbg:
            score -= 6.0
        if _suspicious_coin_value(val):
            score += 50.0
        return (score, err)

    best = min(candidates, key=_candidate_rank)
    return best[0], best[1], f"ok err={best[1]:.1f} {best[2]}"


def consensus_coin_gain(reads: List[Tuple[int, float]]) -> Optional[int]:
    """複数フレームの読み取りから代表値を選ぶ（max ではなく多数決＋低誤差）。"""
    if not reads:
        return None
    filtered = [(v, e) for v, e in reads if not _suspicious_coin_value(v)]
    if not filtered:
        return None
    buckets: dict[int, List[float]] = {}
    for value, err in filtered:
        if value < 1 or value > 999_999:
            continue
        buckets.setdefault(value, []).append(err)
    if not buckets:
        return None

    def rank(item: tuple[int, List[float]]) -> tuple[int, float, int]:
        value, errs = item
        n = len(errs)
        avg_err = sum(errs) / n
        digit_bonus = 0 if len(str(value)) in _PREFERRED_DIGITS else 1
        return (-n, avg_err + digit_bonus * 8.0, -value)

    return min(buckets.items(), key=rank)[0]

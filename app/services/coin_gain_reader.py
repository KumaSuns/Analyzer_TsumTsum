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
_DIGIT_MATCH_MAX_MEAN = 85.0
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


def _yellow_mask(bgr: np.ndarray) -> Optional[np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (14, 70, 110), (42, 255, 255))
    if int(np.count_nonzero(mask)) < 40:
        return None
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)


def _gray_digit_strip(gray: np.ndarray, bgr: np.ndarray) -> Optional[np.ndarray]:
    """金色領域の位置から数字列だけのグレー帯を切り出す（黄色二値化は使わない）。"""
    mask = _yellow_mask(bgr)
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
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


def _binarize_variants(gray: np.ndarray) -> List[np.ndarray]:
    h, w = gray.shape
    target_h = max(48, min(96, h * 3))
    scale = target_h / max(1, h)
    resized = cv2.resize(gray, (max(1, int(w * scale)), target_h), interpolation=cv2.INTER_CUBIC)
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


def _segment_digit_boxes(binary: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
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


def _match_digit(patch: np.ndarray) -> Tuple[Optional[int], float]:
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
    if best_digit is None or best_err > _DIGIT_MATCH_MAX_MEAN:
        return None, best_err
    return best_digit, best_err


def _decode_binary(binary: np.ndarray) -> Tuple[Optional[int], float, str]:
    boxes = _select_primary_row(_segment_digit_boxes(binary))
    if not boxes:
        return None, 1e9, "digit_boxes=0"
    if _stroke_artifact_boxes(boxes):
        return None, 1e9, "stroke_artifact"
    n = len(boxes)
    if n < _MIN_DIGITS or n > _MAX_DIGITS:
        return None, 1e9, f"digit_count={n}"
    digits: List[str] = []
    total_err = 0.0
    for x, y, bw, bh in boxes:
        patch = binary[y : y + bh, x : x + bw]
        digit, err = _match_digit(patch)
        if digit is None:
            return None, 1e9, f"match_fail@{x}"
        digits.append(str(digit))
        total_err += err
    if not digits:
        return None, 1e9, "no_digits"
    try:
        value = int("".join(digits))
    except ValueError:
        return None, 1e9, "parse_fail"
    if value < 1 or value > 999_999:
        return None, 1e9, "range"
    if _suspicious_coin_value(value):
        return None, 1e9, "suspicious"
    mean_err = total_err / len(digits)
    penalty = 0.0
    if n not in _PREFERRED_DIGITS:
        penalty += 12.0
    return value, mean_err + penalty, f"n={n}"


def _decode_gray_sources(gray: np.ndarray, bgr: Optional[np.ndarray]) -> List[Tuple[int, float, str]]:
    out: List[Tuple[int, float, str]] = []
    sources: List[Tuple[np.ndarray, str]] = [(gray, "full")]
    if bgr is not None:
        strip = _gray_digit_strip(gray, bgr)
        if strip is not None:
            sources.append((strip, "strip"))
    for src, tag in sources:
        for binary in _binarize_variants(src):
            val, err, dbg = _decode_binary(binary)
            if val is not None:
                bonus = -4.0 if tag == "strip" else 0.0
                out.append((val, err + bonus, f"{tag} {dbg}"))
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
    best = min(candidates, key=lambda c: c[1])
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

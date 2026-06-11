"""UI から保存した coin_gain 切り抜き（正解ラベル付き）の管理。"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QImage

from app.services.coin_gain_reader import plausible_coin_value
from app.services.image_save import save_training_png

_LABEL_RE = re.compile(r"^coin_gain_(\d{1,7})_")


def _imread_crop(path: Path, flags: int) -> np.ndarray | None:
    """Windows 日本語パスでも cv2.imread できるよう imdecode 経由で読む。"""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def coin_digits_root(assets_root: Path | None = None) -> Path:
    base = assets_root or (Path(__file__).resolve().parents[1] / "assets" / "images")
    return base / "coin_digits"


def _iter_png_files(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        return
    for path in sorted(directory.glob("*.png")):
        if path.is_file():
            yield path


def parse_crop_label(path: Path) -> int | None:
    match = _LABEL_RE.match(path.name)
    if not match:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    if not plausible_coin_value(value):
        return None
    return value


def choose_train_val_split(root: Path) -> str:
    train_dir = root / "train"
    val_dir = root / "val"
    train_count = sum(1 for _ in _iter_png_files(train_dir))
    val_count = sum(1 for _ in _iter_png_files(val_dir))
    total = train_count + val_count
    expected_val_after = int((total + 1) * 0.2)
    if val_count < expected_val_after:
        return "val"
    return "train"


def count_saved_crops(root: Path | None = None) -> tuple[int, int]:
    base = coin_digits_root(root)
    train_n = sum(1 for _ in _iter_png_files(base / "train"))
    val_n = sum(1 for _ in _iter_png_files(base / "val"))
    return train_n, val_n


def iter_saved_labeled_crops(
    root: Path | None = None,
) -> Iterator[Tuple[str, Path, int]]:
    base = coin_digits_root(root)
    for split in ("train", "val"):
        for path in _iter_png_files(base / split):
            label = parse_crop_label(path)
            if label is not None:
                yield split, path, label


def save_labeled_crop(
    image: QImage,
    value: int,
    *,
    assets_root: Path | None = None,
    frame_index: int = 0,
) -> tuple[bool, str]:
    """coin_gain 切り抜きを train/val に保存。ファイル名に正解値を埋め込む。"""
    if image is None or image.isNull():
        return False, "画像がありません"
    if not plausible_coin_value(value):
        return False, "獲得コインは 1〜9999999（1〜7桁）の範囲で指定してください"
    root = coin_digits_root(assets_root)
    split = choose_train_val_split(root)
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"coin_gain_{value}_{timestamp}_f{frame_index}.png"
    if not save_training_png(image, out_path):
        return False, f"保存失敗: {out_path.name}"
    return True, f"{split}/{out_path.name}"


def _patches_from_gray_source(
    gray: np.ndarray, value: int
) -> tuple[List[Tuple[np.ndarray, int]] | None, float]:
    """読取と同じ _hud_digit_row_gray 経路で分割を選ぶ（正解ラベルでスコア）。"""
    from app.services.coin_digit_cnn import _normalize_digit_patch
    from app.services.coin_gain_reader import (
        _hud_digit_row_gray,
        _hud_split_n_digit_variants,
        _patch_digit_errors,
    )

    digits = [int(ch) for ch in str(value)]
    n = len(digits)
    row = _hud_digit_row_gray(gray)
    if row is None:
        return None, 1e9
    best_parts = None
    best_score = 1e9
    best_decoded_match = False
    for parts, _tag in _hud_split_n_digit_variants(row, n):
        if len(parts) != n:
            continue
        part_errs = [_patch_digit_errors(p) for p in parts]
        decoded = [int(np.argmin(part_errs[i])) for i in range(n)]
        score = sum(part_errs[i][digits[i]] for i in range(n))
        decoded_match = decoded == digits
        rank = (0 if decoded_match else 1, score)
        best_rank = (0 if best_decoded_match else 1, best_score)
        if rank < best_rank:
            best_score = score
            best_parts = parts
            best_decoded_match = decoded_match
    threshold = max(2.5, 0.45 * n + 1.0)
    if best_parts is None or best_score > threshold:
        return None, best_score
    out: List[Tuple[np.ndarray, int]] = []
    for digit, part in zip(digits, best_parts):
        if part is None or part.size == 0:
            return None, best_score
        out.append((_normalize_digit_patch(part), digit))
    return out, best_score


def extract_digit_patches_from_crop(
    crop_path: Path, value: int
) -> List[Tuple[np.ndarray, int]] | None:
    """保存済み crop から N 桁パッチを切り出す（1〜7 桁、読取と同前処理）。"""
    from app.services.coin_gain_reader import _coin_gain_strip_for_hud

    if not plausible_coin_value(value):
        return None
    gray = _imread_crop(crop_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    bgr = _imread_crop(crop_path, cv2.IMREAD_COLOR)

    candidates: List[tuple[List[Tuple[np.ndarray, int]], float]] = []
    strip = _coin_gain_strip_for_hud(gray, bgr)
    if strip is not None and strip.size > 0:
        patches, score = _patches_from_gray_source(strip, value)
        if patches is not None:
            candidates.append((patches, score))
    patches, score = _patches_from_gray_source(gray, value)
    if patches is not None:
        candidates.append((patches, score))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[1])[0]


def build_saved_crop_training_samples(
    root: Path | None = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """保存済み coin_gain crop から (patch, digit) 学習サンプルを構築。"""
    train: List[Tuple[np.ndarray, int]] = []
    val: List[Tuple[np.ndarray, int]] = []
    failed: List[str] = []
    for split, path, value in iter_saved_labeled_crops(root):
        patches = extract_digit_patches_from_crop(path, value)
        if not patches:
            failed.append(path.name)
            continue
        bucket = train if split == "train" else val
        bucket.extend(patches)
    if log and failed:
        preview = ", ".join(failed[:5])
        if len(failed) > 5:
            preview += " …"
        log(f"コイン桁 パッチ化失敗 {len(failed)}枚: {preview}")
    return train, val

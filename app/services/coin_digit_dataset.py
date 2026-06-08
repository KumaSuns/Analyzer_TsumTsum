"""UI から保存した coin_gain 切り抜き（正解ラベル付き）の管理。"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np
from PySide6.QtGui import QImage

from app.services.image_save import save_training_png

_LABEL_RE = re.compile(r"^coin_gain_(\d{3,6})_")


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
    if value < 100 or value > 999_999:
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
    if value < 100 or value > 999_999:
        return False, "獲得コインは 100〜999999 の範囲で指定してください"
    root = coin_digits_root(assets_root)
    split = choose_train_val_split(root)
    out_dir = root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = out_dir / f"coin_gain_{value}_{timestamp}_f{frame_index}.png"
    if not save_training_png(image, out_path):
        return False, f"保存失敗: {out_path.name}"
    return True, f"{split}/{out_path.name}"


def extract_digit_patches_from_crop(
    crop_path: Path, value: int
) -> List[Tuple[np.ndarray, int]] | None:
    """保存済み crop から 4 桁パッチを切り出す。"""
    from app.services.coin_digit_cnn import _normalize_digit_patch
    from app.services.coin_gain_reader import (
        _hud_digit_row_gray,
        _hud_split_four_digit_variants,
        _patch_digit_errors,
    )

    digits = [int(ch) for ch in str(value)]
    if len(digits) not in (4, 5, 6):
        return None
    if len(digits) != 4:
        return None
    gray = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return None
    row = _hud_digit_row_gray(gray)
    if row is None:
        return None
    best_parts = None
    best_score = 1e9
    for parts, _tag in _hud_split_four_digit_variants(row):
        if len(parts) != 4:
            continue
        part_errs = [_patch_digit_errors(p) for p in parts]
        score = sum(part_errs[i][digits[i]] for i in range(4))
        if score < best_score:
            best_score = score
            best_parts = parts
    if best_parts is None or best_score > 2.5:
        return None
    out: List[Tuple[np.ndarray, int]] = []
    for digit, part in zip(digits, best_parts):
        if part is None or part.size == 0:
            return None
        out.append((_normalize_digit_patch(part), digit))
    return out


def build_saved_crop_training_samples(
    root: Path | None = None,
) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    """保存済み coin_gain crop から (patch, digit) 学習サンプルを構築。"""
    train: List[Tuple[np.ndarray, int]] = []
    val: List[Tuple[np.ndarray, int]] = []
    for split, path, value in iter_saved_labeled_crops(root):
        patches = extract_digit_patches_from_crop(path, value)
        if not patches:
            continue
        bucket = train if split == "train" else val
        bucket.extend(patches)
    return train, val

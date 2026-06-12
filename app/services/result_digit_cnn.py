"""リザルト画面 桁分類 CNN（0-9）。"""
from __future__ import annotations

import random
from functools import lru_cache
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from app.services.coin_digit_cnn import (
    CoinDigitCnnClassifier,
    _assets_root,
    _augment_patch,
    _lowres_augmented_patch,
    _model_root,
    _synthetic_digit_patch,
)

WEIGHTS_NAME = "result_digit.pt"


def default_result_digit_model_path() -> Path:
    active_file = _model_root() / "ACTIVE_VERSION"
    version = (
        active_file.read_text(encoding="utf-8").strip() if active_file.exists() else "version_1"
    )
    return _model_root() / version / WEIGHTS_NAME


def build_result_digit_training_samples(
    *,
    per_digit: int = 320,
    seed: int = 42,
    assets_root: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    from app.services.result_digit_dataset import (
        build_saved_crop_training_samples,
        count_saved_crops,
        iter_saved_labeled_crops,
    )

    def _log(msg: str) -> None:
        if log:
            log(msg)

    rng = random.Random(seed)
    saved_train, saved_val = build_saved_crop_training_samples(assets_root, log=_log)
    train_n, val_n = count_saved_crops(assets_root)
    usable = sum(1 for _ in iter_saved_labeled_crops(assets_root))
    raw_patch_n = len(saved_train) + len(saved_val)
    _log(
        f"結果桁 保存 crop: train={train_n} val={val_n}枚 "
        f"(桁パッチ化 {raw_patch_n} 件)"
    )
    if usable == 0:
        _log(
            "結果桁: UI 保存データなし。"
            " 動画ツール「結果」で4項目の切り抜きを保存してください。"
        )

    train: List[Tuple[np.ndarray, int]] = []
    real_patches = saved_train + saved_val
    for patch, digit in real_patches:
        for i in range(6):
            train.append(
                (patch, digit) if i == 0 else (_augment_patch(patch, rng), digit)
            )
    val: List[Tuple[np.ndarray, int]] = []

    if usable == 0:
        target_train_per_digit = per_digit
        target_val_per_digit = max(20, per_digit // 5)
    elif usable < 10:
        target_train_per_digit = max(32, per_digit // 2)
        target_val_per_digit = max(16, per_digit // 8)
    else:
        target_train_per_digit = max(32, per_digit // 8)
        target_val_per_digit = max(24, per_digit // 8)

    for digit in range(10):
        have_train = sum(1 for _, d in train if d == digit)
        have_val = sum(1 for _, d in val if d == digit)
        for i in range(max(0, target_train_per_digit - have_train)):
            train.append((_synthetic_digit_patch(digit, rng, hud_style=(i % 2 == 0)), digit))
        for i in range(max(0, target_val_per_digit - have_val)):
            val.append((_synthetic_digit_patch(digit, rng, hud_style=True), digit))
    rng.shuffle(train)
    rng.shuffle(val)
    _log(
        f"結果桁 学習構成: train={len(train)} val={len(val)} "
        f"(実機由来~{len(real_patches) * 6 / max(1, len(train)):.0%} of train)"
    )
    return train, val


def evaluate_saved_result_crop_reads(
    clf: CoinDigitCnnClassifier,
    *,
    assets_root: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    from app.services.coin_gain_reader import set_digit_classifier_factory
    from app.services.result_digit_dataset import (
        extract_digit_patches_from_crop,
        iter_saved_labeled_crops,
    )

    def _log(msg: str) -> None:
        if log:
            log(msg)

    if not clf.is_loaded():
        return 0, 0
    set_digit_classifier_factory(get_result_digit_classifier)
    try:
        ok = total = 0
        for _split, path, label in iter_saved_labeled_crops(assets_root):
            patches = extract_digit_patches_from_crop(path, label)
            if not patches:
                continue
            total += 1
            match = True
            for patch, digit in patches:
                probs = clf.predict_probs(patch)
                if probs is None or int(probs.argmax()) != digit:
                    match = False
                    break
            if match:
                ok += 1
        if total:
            _log(f"結果桁 保存 crop 桁一致: {ok}/{total} ({ok / total:.0%})")
        return ok, total
    finally:
        set_digit_classifier_factory(None)


@lru_cache(maxsize=1)
def get_result_digit_classifier() -> CoinDigitCnnClassifier:
    clf = CoinDigitCnnClassifier()
    path = default_result_digit_model_path()
    if not clf.load(path):
        clf.load(_model_root() / "version_1" / WEIGHTS_NAME)
    return clf


def reload_result_digit_classifier() -> CoinDigitCnnClassifier:
    get_result_digit_classifier.cache_clear()
    return get_result_digit_classifier()


def result_digit_cnn_available() -> bool:
    return get_result_digit_classifier().is_loaded()

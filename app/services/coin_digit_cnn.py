"""コイン HUD 桁分類 CNN（0-9）。"""
from __future__ import annotations

import json
import pickle
import random
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services.scene_cnn import (
    CNN_WEIGHTS_FORMAT,
    pick_torch_device,
    resolve_training_device,
    torch_available,
)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = None  # type: ignore

DIGIT_W = 28
DIGIT_H = 40
NUM_CLASSES = 10
WEIGHTS_NAME = "coin_digit.pt"


def _assets_root() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


def _model_root() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "main_model"


def default_coin_digit_model_path() -> Path:
    active_file = _model_root() / "ACTIVE_VERSION"
    version = active_file.read_text(encoding="utf-8").strip() if active_file.exists() else "version_1"
    return _model_root() / version / WEIGHTS_NAME


def _normalize_digit_patch(patch: np.ndarray) -> np.ndarray:
    if patch is None or patch.size == 0:
        return np.zeros((DIGIT_H, DIGIT_W), dtype=np.uint8)
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.resize(patch, (DIGIT_W, DIGIT_H), interpolation=cv2.INTER_CUBIC)


def _patch_to_tensor(patch: np.ndarray) -> "torch.Tensor":
    gray = _normalize_digit_patch(patch).astype(np.float32) / 255.0
    x = gray.reshape(1, DIGIT_H, DIGIT_W)
    return torch.from_numpy(x)


if _TORCH_OK:

    class _DigitPatchDataset(Dataset):
        def __init__(self, samples: List[Tuple[np.ndarray, int]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            patch, label = self.samples[index]
            return _patch_to_tensor(patch), label

    class SmallDigitCNN(nn.Module):
        def __init__(self, num_classes: int = NUM_CLASSES) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(128, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.net(x)
            x = torch.flatten(x, 1)
            return self.head(x)


def _augment_patch(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    out = _normalize_digit_patch(patch)
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    if rng.random() < 0.7:
        scale = rng.uniform(0.85, 1.15)
        nh = max(8, int(DIGIT_H * scale))
        nw = max(8, int(DIGIT_W * scale))
        resized = cv2.resize(out, (nw, nh), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((DIGIT_H, DIGIT_W), dtype=np.uint8)
        y0 = max(0, (DIGIT_H - nh) // 2)
        x0 = max(0, (DIGIT_W - nw) // 2)
        canvas[y0 : y0 + nh, x0 : x0 + nw] = resized[: min(nh, DIGIT_H - y0), : min(nw, DIGIT_W - x0)]
        out = canvas
    if rng.random() < 0.6:
        alpha = rng.uniform(0.65, 1.35)
        beta = rng.randint(-24, 24)
        out = np.clip(out.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.45:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.3, 1.2))
    if rng.random() < 0.35:
        noise = np.random.randint(-18, 19, size=out.shape, dtype=np.int16)
        out = np.clip(out.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return out


def _synthetic_digit_patch(digit: int, rng: random.Random, *, hud_style: bool) -> np.ndarray:
    canvas = np.zeros((DIGIT_H, DIGIT_W), dtype=np.uint8)
    if hud_style:
        canvas[:] = rng.randint(12, 28)
        color = rng.randint(180, 245)
        scale = rng.uniform(0.55, 0.95)
        thickness = 1
        font = cv2.FONT_HERSHEY_DUPLEX
    else:
        color = 255
        scale = rng.uniform(0.75, 1.25)
        thickness = rng.choice([2, 3])
        font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(
        canvas,
        str(digit),
        (2, DIGIT_H - 6),
        font,
        scale,
        int(color),
        thickness,
        cv2.LINE_AA,
    )
    if hud_style and rng.random() < 0.5:
        canvas = cv2.GaussianBlur(canvas, (5, 5), 1.0)
    return _augment_patch(canvas, rng)


def _extract_labeled_ref_digit_patches(
    ref_path: Path, labels: Tuple[int, ...]
) -> Dict[int, List[np.ndarray]]:
    from app.services.coin_gain_reader import (
        _hud_digit_row_gray,
        _hud_split_four_digit_variants,
        _patch_digit_errors,
    )

    out: Dict[int, List[np.ndarray]] = {d: [] for d in range(10)}
    gray = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return out
    row = _hud_digit_row_gray(gray)
    if row is None or len(labels) != 4:
        return out
    best_parts: Optional[List[np.ndarray]] = None
    best_score = 1e9
    for parts, _tag in _hud_split_four_digit_variants(row):
        if len(parts) != 4:
            continue
        part_errs = [_patch_digit_errors(p) for p in parts]
        score = sum(part_errs[i][labels[i]] for i in range(4))
        if score < best_score:
            best_score = score
            best_parts = parts
    if best_parts is None:
        return out
    for digit, part in zip(labels, best_parts):
        if part is not None and part.size > 0:
            out[digit].append(_normalize_digit_patch(part))
    return out


def _extract_ref_digit_patches(ref_path: Path) -> Dict[int, List[np.ndarray]]:
    return _extract_labeled_ref_digit_patches(ref_path, (5, 3, 9, 5))


def _extract_wide_crop_digit_patches(
    ref_path: Path, rng: random.Random, *, n_variants: int = 80
) -> Dict[int, List[np.ndarray]]:
    """横長 coin_gain 切り抜きを模した画像から、正解ラベル付き桁パッチを抽出。"""
    from app.services.coin_gain_reader import (
        _gray_digit_strip,
        _hud_digit_row_gray,
        _hud_split_four_digits,
        _hud_split_four_digits_ref,
    )

    out: Dict[int, List[np.ndarray]] = {d: [] for d in range(10)}
    ref = cv2.imread(str(ref_path))
    if ref is None:
        return out
    labels = (5, 3, 9, 5)
    for _ in range(n_variants):
        canvas_h = rng.choice([32, 38, 45, 50])
        canvas_w = rng.randint(420, 480)
        canvas = np.full((canvas_h, canvas_w, 3), (18, 16, 12), np.uint8)
        rh = min(ref.shape[0], canvas_h)
        rw = ref.shape[1]
        small = cv2.resize(ref, (rw, rh), interpolation=cv2.INTER_AREA)
        x0 = rng.randint(12, 36)
        y0 = max(0, (canvas_h - rh) // 2)
        canvas[y0 : y0 + rh, x0 : x0 + rw] = small
        if rng.random() < 0.4:
            canvas = cv2.GaussianBlur(canvas, (3, 3), rng.uniform(0.2, 0.9))
        gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        strip = _gray_digit_strip(gray, canvas)
        if strip is None or strip.shape[0] < 8:
            continue
        target_h = 120
        scale = target_h / max(1, strip.shape[0])
        big = cv2.resize(
            strip,
            (max(1, int(strip.shape[1] * scale)), target_h),
            interpolation=cv2.INTER_CUBIC,
        )
        row = _hud_digit_row_gray(big)
        if row is None:
            continue
        parts = _hud_split_four_digits_ref(row) or _hud_split_four_digits(row)
        if parts is None or len(parts) != 4:
            continue
        for digit, part in zip(labels, parts):
            if part is None or part.size == 0:
                continue
            out[digit].append(_augment_patch(_normalize_digit_patch(part), rng))
    return out


def _lowres_augmented_patch(base: np.ndarray, rng: random.Random) -> np.ndarray:
    """実機の横長 crop 相当の低解像度→拡大を模倣。"""
    base = _normalize_digit_patch(base)
    small_h = rng.randint(8, 16)
    small_w = max(6, int(small_h * DIGIT_W / DIGIT_H))
    tiny = cv2.resize(base, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(tiny, (DIGIT_W, DIGIT_H), interpolation=cv2.INTER_CUBIC)
    return _augment_patch(up, rng)


def build_digit_training_samples(
    *,
    per_digit: int = 320,
    seed: int = 42,
    log: Optional[Callable[[str], None]] = None,
) -> Tuple[List[Tuple[np.ndarray, int]], List[Tuple[np.ndarray, int]]]:
    from app.services.coin_digit_dataset import (
        build_saved_crop_training_samples,
        count_saved_crops,
        iter_saved_labeled_crops,
    )

    def _log(msg: str) -> None:
        if log:
            log(msg)

    rng = random.Random(seed)
    saved_train, saved_val = build_saved_crop_training_samples()
    train_n, val_n = count_saved_crops()
    usable = sum(1 for _ in iter_saved_labeled_crops())
    _log(
        f"コイン桁 保存 crop: train={train_n} val={val_n}枚 "
        f"(桁パッチ化 {len(saved_train) + len(saved_val)} 件)"
    )
    if usable == 0:
        _log(
            "コイン桁: UI 保存データなし。"
            " 獲得コイン確定ダイアログまたは動画ツールで切り抜きを保存してください。"
        )

    train: List[Tuple[np.ndarray, int]] = list(saved_train)
    val: List[Tuple[np.ndarray, int]] = list(saved_val)

    ref_path = _assets_root() / "images" / "coin_hud_ref_5395.png"
    ref_patches = _extract_ref_digit_patches(ref_path) if ref_path.exists() else {}
    wide_patches = (
        _extract_wide_crop_digit_patches(ref_path, rng, n_variants=40)
        if ref_path.exists() and usable < 3
        else {}
    )

    synth_train = max(40, per_digit // 3) if usable >= 3 else per_digit
    synth_val = max(12, per_digit // 10) if usable >= 3 else max(20, per_digit // 5)
    for digit in range(10):
        refs = ref_patches.get(digit, []) + wide_patches.get(digit, [])
        have_train = sum(1 for _, d in train if d == digit)
        have_val = sum(1 for _, d in val if d == digit)
        for i in range(max(0, synth_train - have_train)):
            if refs and i % 3 != 2:
                base = refs[i % len(refs)]
                patch = (
                    _lowres_augmented_patch(base, rng)
                    if i % 4 == 0
                    else _augment_patch(base, rng)
                )
            else:
                patch = _synthetic_digit_patch(digit, rng, hud_style=(i % 2 == 0))
            train.append((patch, digit))
        for i in range(max(0, synth_val - have_val)):
            if refs and i % 2 == 0:
                patch = _augment_patch(refs[i % len(refs)], rng)
            else:
                patch = _synthetic_digit_patch(digit, rng, hud_style=True)
            val.append((patch, digit))
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class CoinDigitCnnClassifier:
    def __init__(self) -> None:
        self.model: Optional["nn.Module"] = None
        self.device = pick_torch_device() if _TORCH_OK else None
        self.val_accuracy = 0.0

    def is_loaded(self) -> bool:
        return self.model is not None

    def train_from_samples(
        self,
        train_samples: List[Tuple[np.ndarray, int]],
        val_samples: List[Tuple[np.ndarray, int]],
        *,
        epochs: int = 24,
        batch_size: int = 64,
        lr: float = 1e-3,
        log: Optional[Callable[[str], None]] = None,
    ) -> float:
        if not _TORCH_OK:
            raise RuntimeError("PyTorch が未インストールです。")
        if not train_samples:
            raise RuntimeError("学習データがありません。")

        def _log(msg: str) -> None:
            if log:
                log(msg)

        self.device = resolve_training_device(_log)
        train_ds = _DigitPatchDataset(train_samples)
        val_ds = _DigitPatchDataset(val_samples) if val_samples else None
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None
        )

        net = SmallDigitCNN(NUM_CLASSES).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()
        best_acc = 0.0
        best_state = None

        for epoch in range(1, epochs + 1):
            net.train()
            total_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                logits = net(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
                n_batches += 1
            val_acc = 0.0
            if val_loader is not None:
                net.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device)
                        yb = yb.to(self.device)
                        pred = net(xb).argmax(dim=1)
                        correct += int((pred == yb).sum().item())
                        total += int(yb.shape[0])
                val_acc = correct / max(1, total)
                if val_acc >= best_acc:
                    best_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            _log(
                f"coin_digit epoch {epoch}/{epochs} loss={total_loss / max(1, n_batches):.4f}"
                + (f" val_acc={val_acc:.3f}" if val_loader else "")
            )

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.model = net
        self.val_accuracy = best_acc
        return best_acc

    def predict_probs(self, patch: np.ndarray) -> Optional[np.ndarray]:
        if not self.is_loaded() or not _TORCH_OK:
            return None
        x = _patch_to_tensor(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)  # type: ignore[operator]
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        return probs.astype(np.float32)

    def match_digit(
        self, patch: np.ndarray, max_mean: float = 95.0
    ) -> Tuple[Optional[int], float]:
        probs = self.predict_probs(patch)
        if probs is None:
            return None, 1e9
        digit = int(probs.argmax())
        conf = float(probs[digit])
        err = (1.0 - conf) * 100.0
        if err > max_mean:
            return None, err
        return digit, err

    def digit_errors(self, patch: np.ndarray) -> List[float]:
        probs = self.predict_probs(patch)
        if probs is None:
            return [1e9] * NUM_CLASSES
        return [float(1.0 - probs[d]) for d in range(NUM_CLASSES)]

    @staticmethod
    def _state_dict_to_numpy(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}

    @staticmethod
    def _state_dict_from_numpy(arrays: Dict[str, np.ndarray]) -> Dict[str, "torch.Tensor"]:
        return {k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items()}

    def save(self, path: Path) -> None:
        if not self.is_loaded():
            raise RuntimeError("coin_digit モデルが未学習です。")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": CNN_WEIGHTS_FORMAT,
            "state_dict": self._state_dict_to_numpy(self.model.state_dict()),  # type: ignore[union-attr]
            "classes": [str(i) for i in range(NUM_CLASSES)],
            "img_size": [DIGIT_W, DIGIT_H],
            "val_accuracy": self.val_accuracy,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=4)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "classes": payload["classes"],
                    "img_size": payload["img_size"],
                    "val_accuracy": self.val_accuracy,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self, path: Path) -> bool:
        if not _TORCH_OK or not path.exists():
            self.model = None
            return False
        try:
            with path.open("rb") as f:
                payload = pickle.load(f)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        raw_sd = payload.get("state_dict")
        if not isinstance(raw_sd, dict):
            return False
        net = SmallDigitCNN(NUM_CLASSES)
        if payload.get("format") == CNN_WEIGHTS_FORMAT:
            state_dict = self._state_dict_from_numpy(raw_sd)  # type: ignore[arg-type]
        else:
            state_dict = raw_sd  # type: ignore[assignment]
        net.load_state_dict(state_dict)
        self.device = pick_torch_device()
        net.to(self.device)
        self.model = net.eval()
        self.val_accuracy = float(payload.get("val_accuracy", 0.0))
        return True


@lru_cache(maxsize=1)
def get_coin_digit_classifier() -> CoinDigitCnnClassifier:
    clf = CoinDigitCnnClassifier()
    path = default_coin_digit_model_path()
    if not clf.load(path):
        clf.load(_model_root() / "version_1" / WEIGHTS_NAME)
    return clf


def reload_coin_digit_classifier() -> CoinDigitCnnClassifier:
    get_coin_digit_classifier.cache_clear()
    return get_coin_digit_classifier()


def coin_digit_cnn_available() -> bool:
    return get_coin_digit_classifier().is_loaded()


def train_and_save_default_model(log: Optional[Callable[[str], None]] = None) -> Path:
    train_samples, val_samples = build_digit_training_samples()
    clf = CoinDigitCnnClassifier()
    clf.train_from_samples(train_samples, val_samples, log=log)
    out = default_coin_digit_model_path()
    clf.save(out)
    get_coin_digit_classifier.cache_clear()
    return out


if __name__ == "__main__":
    import sys

    def _print(msg: str) -> None:
        print(msg, flush=True)

    path = train_and_save_default_model(log=_print)
    _print(f"saved: {path}")

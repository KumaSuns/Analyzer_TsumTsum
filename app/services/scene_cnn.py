"""シーン分類用の小さな CNN（train/val 画像フォルダから学習）。"""
from __future__ import annotations

import json
import math
import os
import pickle
import platform
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from app.services.scene_model import SCENE_CLASSES, iter_scene_dataset_images

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

IMG_SIZE = 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CNN_WEIGHTS_FORMAT = "numpy_state_dict_v1"


def torch_available() -> bool:
    return _TORCH_OK


def _env_forced_torch_device() -> Optional[str]:
    """ANALYZER_TORCH_DEVICE=cpu|cuda|mps で明示指定（未設定なら自動）。"""
    name = os.environ.get("ANALYZER_TORCH_DEVICE", "").strip().lower()
    if name in ("cpu", "cuda", "mps"):
        return name
    return None


def _mps_usable() -> bool:
    """MPS は Apple Silicon 向け。Intel Mac + AMD では Metal エラーになるため使わない。"""
    if platform.machine() != "arm64":
        return False
    return bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[union-attr]
    )


def pick_torch_device() -> "torch.device":
    """CUDA → Apple Silicon MPS → CPU（Intel Mac は常に CPU）。"""
    if not _TORCH_OK:
        return torch.device("cpu")  # type: ignore[union-attr]
    forced = _env_forced_torch_device()
    if forced == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if forced == "mps":
        return torch.device("mps" if _mps_usable() else "cpu")
    if forced == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_usable():
        return torch.device("mps")
    return torch.device("cpu")


def verify_torch_install() -> Optional[str]:
    """壊れた PyTorch（torch.save 不可など）を検出。問題なければ None。"""
    if not _TORCH_OK:
        return "PyTorch が未インストールです。"
    try:
        import io

        buf = io.BytesIO()
        torch.save({"probe": torch.tensor(0)}, buf)
    except ModuleNotFoundError as exc:
        if "torch.utils.serialization" in str(exc):
            return (
                "PyTorch のインストールが壊れています。"
                " プロジェクトの venv で CUDA 版を入れ直してください:"
                " .venv\\Scripts\\pip install torch torchvision"
                " --index-url https://download.pytorch.org/whl/cu124"
            )
        return f"PyTorch の保存機能が使えません: {exc}"
    except Exception as exc:
        return f"PyTorch の動作確認に失敗しました: {exc}"
    return None


def _probe_torch_device(device: "torch.device") -> bool:
    """学習前の簡易動作確認。Metal 障害時は False。"""
    if not _TORCH_OK or device.type == "cpu":
        return True
    try:
        conv = nn.Conv2d(3, 4, kernel_size=1).to(device)  # type: ignore[union-attr]
        x = torch.zeros(2, 3, 16, 16, device=device)  # type: ignore[union-attr]
        y = conv(x)
        return math.isfinite(float(y.sum().item()))
    except Exception:
        return False


def resolve_training_device(log: Optional[Callable[[str], None]] = None) -> "torch.device":
    """学習用デバイス。Intel Mac は CPU、MPS はプローブ失敗時 CPU にフォールバック。"""
    if not _TORCH_OK:
        return torch.device("cpu")  # type: ignore[union-attr]
    dev = pick_torch_device()
    if platform.machine() != "arm64" and dev.type != "cpu":
        if log:
            log(
                f"警告: {describe_torch_device(dev)} は Intel Mac では使えません。"
                " CPU で学習します。"
            )
        dev = torch.device("cpu")
    elif dev.type == "mps" and not _probe_torch_device(dev):
        if log:
            log("警告: MPS の動作確認に失敗したため CPU で学習します。")
        dev = torch.device("cpu")
    return dev


def describe_torch_device(device: Optional["torch.device"] = None) -> str:
    if not _TORCH_OK:
        return "PyTorch 未インストール"
    dev = device or pick_torch_device()
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        return f"cuda ({name})"
    if dev.type == "mps":
        return "mps (Apple Silicon GPU)"
    if platform.machine() != "arm64":
        return "cpu (Intel Mac: PyTorch は CPU 学習)"
    return "cpu"


def _load_rgb_array(path: Path) -> Optional[np.ndarray]:
    try:
        import cv2  # type: ignore

        bgr = cv2.imread(str(path))
        if bgr is not None and bgr.size > 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return rgb
    except Exception:
        pass
    try:
        from PIL import Image

        with Image.open(path) as im:
            return np.array(im.convert("RGB"))
    except Exception:
        return None


def _resize_rgb(arr: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    try:
        import cv2  # type: ignore

        return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    except Exception:
        from PIL import Image

        im = Image.fromarray(arr, mode="RGB")
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        return np.array(im.resize((size, size), resample))


def array_to_tensor(arr: np.ndarray) -> "torch.Tensor":
    x = arr.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def qimage_to_rgb_array(image) -> Optional[np.ndarray]:
    if image is None or image.isNull():
        return None
    from PySide6.QtGui import QImage

    img = image.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    if w < 1 or h < 1:
        return None
    ptr = img.constBits()
    if ptr is None:
        return None
    bpl = int(img.bytesPerLine())
    buf = np.frombuffer(ptr, dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, : w * 3]
    return buf.reshape(h, w, 3).copy()


if _TORCH_OK:

    class _SceneImageDataset(Dataset):
        def __init__(
            self,
            images_root: Path,
            split: str,
            class_to_idx: Dict[str, int],
            augment: bool = False,
        ) -> None:
            self.samples: List[Tuple[Path, int]] = []
            root = images_root / split
            for cls, idx in class_to_idx.items():
                cls_dir = root / cls
                if not cls_dir.is_dir():
                    continue
                for p in iter_scene_dataset_images(cls_dir):
                    self.samples.append((p, idx))
            self.augment = augment

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            path, label = self.samples[index]
            arr = _load_rgb_array(path)
            if arr is None:
                arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            if self.augment and torch.rand(1).item() > 0.5:
                arr = np.flip(arr, axis=1).copy()
            arr = _resize_rgb(arr, IMG_SIZE)
            return array_to_tensor(arr), label

    class SmallSceneCNN(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
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
                nn.MaxPool2d(2),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(128, num_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = self.net(x)
            x = torch.flatten(x, 1)
            return self.head(x)


class SceneCnnClassifier:
    """学習済み CNN の保存・読込・推論。"""

    def __init__(self) -> None:
        self.classes: List[str] = list(SCENE_CLASSES)
        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}
        self.model: Optional["nn.Module"] = None
        self.device = pick_torch_device() if _TORCH_OK else None
        self.val_accuracy = 0.0
        self.img_size = IMG_SIZE

    def is_loaded(self) -> bool:
        return self.model is not None

    def train_from_dataset(
        self,
        images_root: Path,
        *,
        epochs: int = 25,
        batch_size: int = 16,
        lr: float = 1e-3,
        early_stop_patience: int = 8,
        log: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, float]:
        if not _TORCH_OK:
            raise RuntimeError(
                "PyTorch が未インストールです。venv で pip install torch torchvision を実行してください。"
            )

        def _log(msg: str) -> None:
            if log:
                log(msg)

        present = [
            c
            for c in self.classes
            if (images_root / "train" / c).is_dir()
            and any(iter_scene_dataset_images(images_root / "train" / c))
        ]
        if not present:
            raise RuntimeError("train に学習可能な画像がありません。")

        self.classes = present
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        train_ds = _SceneImageDataset(images_root, "train", self.class_to_idx, augment=True)
        val_ds = _SceneImageDataset(images_root, "val", self.class_to_idx, augment=False)
        if len(train_ds) == 0:
            raise RuntimeError("train 画像を読み込めませんでした。")

        self.device = resolve_training_device(_log)
        use_cuda = self.device.type == "cuda"
        if use_cuda:
            batch_size = max(batch_size, 32)
            torch.backends.cudnn.benchmark = True

        loader_kw: Dict[str, object] = {"num_workers": 0}
        if use_cuda:
            loader_kw["pin_memory"] = True

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **loader_kw
        )
        val_loader = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kw)
            if len(val_ds) > 0
            else None
        )

        net = SmallSceneCNN(len(self.classes))
        net.to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

        from collections import Counter

        label_counts = Counter(label for _, label in train_ds.samples)
        n_train = max(len(train_ds), 1)
        n_classes = len(self.classes)
        class_weights = [
            min(n_train / (n_classes * max(label_counts.get(i, 0), 1)), 12.0)
            for i in range(n_classes)
        ]
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        imbalanced = max(class_weights) / max(min(class_weights), 1e-6)
        if imbalanced > 3.0:
            none_idx = self.class_to_idx.get("none")
            none_n = label_counts.get(none_idx, 0) if none_idx is not None else 0
            _log(
                f"  クラス不均衡: 最大/最小 weight={imbalanced:.1f} "
                f"(none={none_n}枚など、少数クラスに重み付け)"
            )

        _log(f"CNN 学習: クラス={self.classes}")
        _log(f"  train={len(train_ds)} val={len(val_ds)} device={describe_torch_device(self.device)}")
        if self.device.type == "cpu":
            if platform.machine() != "arm64":
                _log(
                    "  注意: Intel Mac では CPU 学習です（AMD GPU の MPS は未対応・Metal エラー回避）。"
                )
            elif torch.cuda.is_available() is False:
                _log(
                    "  注意: CPU で学習中です。GPU を使うには CUDA 版 PyTorch が必要です。"
                    " pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124"
                )

        best_acc = 0.0
        best_epoch = 0
        best_state: Optional[Dict[str, "torch.Tensor"]] = None
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            net.train()
            running = 0.0
            for xb, yb in train_loader:
                xb = xb.to(self.device, non_blocking=use_cuda)
                yb = yb.to(self.device, non_blocking=use_cuda)
                opt.zero_grad()
                loss = criterion(net(xb), yb)
                loss.backward()
                opt.step()
                loss_val = float(loss.item())
                if not math.isfinite(loss_val):
                    raise RuntimeError(
                        f"学習 loss が異常です (device={self.device})。"
                        " アプリを再起動し、Intel Mac では CPU 表示になることを確認してください。"
                    )
                running += loss_val

            val_acc = 0.0
            if val_loader is not None:
                net.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(self.device, non_blocking=use_cuda)
                        yb = yb.to(self.device, non_blocking=use_cuda)
                        pred = net(xb).argmax(dim=1)
                        correct += int((pred == yb).sum().item())
                        total += int(yb.size(0))
                val_acc = (correct / total) if total > 0 else 0.0
                if not math.isfinite(val_acc) or val_acc > 1.0:
                    raise RuntimeError(
                        f"検証精度が異常です (val_acc={val_acc}, device={self.device})。"
                        " 学習を中止しました。アプリを再起動して CPU で再実行してください。"
                    )
                if val_acc > best_acc + 1e-6:
                    best_acc = val_acc
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    stale_epochs = 0
                else:
                    stale_epochs += 1

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                mark = ""
                if val_loader is not None and epoch == best_epoch:
                    mark = " *best*"
                _log(
                    f"  epoch {epoch:02d}/{epochs:02d} loss={running / max(len(train_loader), 1):.4f} "
                    f"val_acc={val_acc:.3f}{mark}"
                )

            if val_loader is not None and stale_epochs >= early_stop_patience and best_state is not None:
                _log(
                    f"  early stop: epoch {epoch:02d}（best val={best_acc:.3f} @ epoch {best_epoch:02d}）"
                )
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        self.model = net.eval()
        self.val_accuracy = best_acc if best_state is not None else 0.0
        per_class_val = self.evaluate_val(images_root) if val_loader else {}
        fever_total = per_class_val.get("fever", (0, 0))[1]
        fever_ok = per_class_val.get("fever", (0, 0))[0]
        if best_epoch:
            _log(
                f"CNN 学習完了: 保存する重み=epoch {best_epoch:02d} val精度={self.val_accuracy:.3f}"
            )
        else:
            _log(f"CNN 学習完了: 保存する重み=epoch - val精度={self.val_accuracy:.3f}")
        weak = [
            f"{cls}:{ok}/{total}"
            for cls, (ok, total) in sorted(per_class_val.items())
            if total and ok / total < 0.7
        ]
        if weak:
            _log(f"  val で弱いクラス: {', '.join(weak)}")
        if fever_total:
            _log(f"  fever(val): {fever_ok}/{fever_total}")
        return {"val_accuracy": best_acc, "per_class_val": per_class_val}

    def evaluate_val(self, images_root: Path) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, Tuple[int, int]] = {}
        if not self.is_loaded():
            return out
        for cls in self.classes:
            cls_dir = images_root / "val" / cls
            if not cls_dir.is_dir():
                continue
            ok = 0
            total = 0
            for p in iter_scene_dataset_images(cls_dir):
                pred = self.predict_path(p)
                total += 1
                if pred == cls:
                    ok += 1
            if total:
                out[cls] = (ok, total)
        return out

    def predict_path(self, path: Path) -> str:
        arr = _load_rgb_array(path)
        if arr is None:
            return "none"
        return self.predict_rgb(arr)

    def predict_rgb(self, arr: np.ndarray) -> str:
        if not self.is_loaded():
            return "none"
        arr = _resize_rgb(arr, self.img_size)
        x = array_to_tensor(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)  # type: ignore[operator]
            idx = int(logits.argmax(dim=1).item())
        if 0 <= idx < len(self.classes):
            return self.classes[idx]
        return "none"

    def predict_qimage(self, image) -> str:
        arr = qimage_to_rgb_array(image)
        if arr is None:
            return "none"
        return self.predict_rgb(arr)

    def ranked_qimage(self, image, top_k: int = 8) -> List[Tuple[str, float]]:
        if not self.is_loaded() or image is None or image.isNull():
            return [("none", 1e9)]
        arr = qimage_to_rgb_array(image)
        if arr is None:
            return [("none", 1e9)]
        arr = _resize_rgb(arr, self.img_size)
        x = array_to_tensor(arr).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)  # type: ignore[operator]
            probs = torch.softmax(logits, dim=1)[0]
        pairs = [(self.classes[i], float(1.0 - probs[i].item())) for i in range(len(self.classes))]
        pairs.sort(key=lambda t: t[1])
        return pairs[:top_k]

    @staticmethod
    def _state_dict_to_numpy(state_dict: Dict[str, "torch.Tensor"]) -> Dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy() for k, v in state_dict.items()}

    @staticmethod
    def _state_dict_from_numpy(arrays: Dict[str, np.ndarray]) -> Dict[str, "torch.Tensor"]:
        return {k: torch.from_numpy(np.asarray(v)) for k, v in arrays.items()}

    def _build_save_payload(self) -> Dict[str, object]:
        return {
            "format": CNN_WEIGHTS_FORMAT,
            "state_dict": self._state_dict_to_numpy(self.model.state_dict()),  # type: ignore[union-attr]
            "classes": self.classes,
            "img_size": self.img_size,
            "val_accuracy": self.val_accuracy,
        }

    def _write_weights_file(self, path: Path, payload: Dict[str, object]) -> None:
        """torch.save が壊れている環境向けに pickle（numpy のみ）で保存。"""
        with path.open("wb") as f:
            pickle.dump(payload, f, protocol=4)

    def _read_weights_file(self, path: Path) -> Optional[Dict[str, object]]:
        raw = path.read_bytes()
        if not raw:
            return None
        if raw[:1] in (b"\x80", b"\x7d") or raw[:2] == b"\x80\x04":
            try:
                data = pickle.loads(raw)
                if isinstance(data, dict):
                    return data
            except Exception:
                return None
        if not _TORCH_OK:
            return None
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        return data if isinstance(data, dict) else None

    def save(self, path: Path) -> None:
        if not self.is_loaded():
            raise RuntimeError("CNN モデルが未学習です。")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_weights_file(path, self._build_save_payload())
        meta = path.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "classes": self.classes,
                    "img_size": self.img_size,
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
        payload = self._read_weights_file(path)
        if not payload:
            self.model = None
            return False
        classes = payload.get("classes", SCENE_CLASSES)
        if not classes:
            return False
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.img_size = int(payload.get("img_size", IMG_SIZE))
        self.val_accuracy = float(payload.get("val_accuracy", 0.0))
        net = SmallSceneCNN(len(self.classes))
        raw_sd = payload.get("state_dict")
        if not isinstance(raw_sd, dict):
            return False
        if payload.get("format") == CNN_WEIGHTS_FORMAT:
            state_dict = self._state_dict_from_numpy(raw_sd)  # type: ignore[arg-type]
        else:
            state_dict = raw_sd  # type: ignore[assignment]
        net.load_state_dict(state_dict)
        self.device = pick_torch_device()
        net.to(self.device)
        self.model = net.eval()
        return True

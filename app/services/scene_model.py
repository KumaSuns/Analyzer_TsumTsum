from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtGui import QImage

# 推論時は先に縮小（1080p 全体をグレー化すると重い）
_INFERENCE_MAX_EDGE = 160

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    cv2 = None  # type: ignore
    np = None  # type: ignore


SCENE_CLASSES = ["none", "item", "ready", "go", "fever", "timeup", "bonus", "result"]
# 学習・評価で扱う画像（.gitkeep 等は除外）
SCENE_DATASET_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})


def is_scene_dataset_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SCENE_DATASET_IMAGE_SUFFIXES


def iter_scene_dataset_images(cls_dir: Path):
    """train/item 直下だけでなくサブフォルダ内の画像も列挙する。"""
    if not cls_dir.is_dir():
        return
    for p in cls_dir.rglob("*"):
        if is_scene_dataset_image(p):
            yield p


def _downscale_gray_max_edge(arr: np.ndarray, max_edge: int = _INFERENCE_MAX_EDGE) -> np.ndarray:
    """学習・推論で同じ前処理（大きい画像は長辺 max_edge に揃える）。"""
    if arr is None or arr.size == 0 or arr.ndim != 2:
        return arr
    h, w = arr.shape
    m = max(h, w)
    if m <= max_edge:
        return arr
    if w >= h:
        nw = max_edge
        nh = max(1, int(h * max_edge / w))
    else:
        nh = max_edge
        nw = max(1, int(w * max_edge / h))
    if cv2 is not None:
        try:
            return cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    try:
        from PIL import Image

        im = Image.fromarray(arr, mode="L")
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        im = im.resize((nw, nh), resample)
        return np.array(im, dtype=np.uint8)
    except Exception:
        return arr


def _resize_gray_to_size(arr: np.ndarray, size: int) -> Optional[np.ndarray]:
    """24x24 への縮小。OpenCV が無い環境では Pillow のみで行う。"""
    if arr is None or arr.size == 0 or arr.ndim != 2:
        return None
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if cv2 is not None:
        try:
            return cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
        except Exception:
            return None
    try:
        from PIL import Image

        im = Image.fromarray(arr, mode="L")
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        im = im.resize((size, size), resample)
        return np.array(im, dtype=np.uint8)
    except Exception:
        return None


def image_file_to_feature(path: Path, size: int = 24) -> Optional[List[float]]:
    """学習・検証用。バックグラウンドスレッドでも使える（QImage は GUI スレッド専用のため使わない）。"""
    if np is None:
        return None
    path_r = path.expanduser().resolve(strict=False)
    arr = None

    if cv2 is not None:
        arr = cv2.imread(str(path_r), cv2.IMREAD_GRAYSCALE)

    if arr is None:
        try:
            import imageio.v3 as iio  # type: ignore

            im = np.asarray(iio.imread(str(path_r)))
            if im.size == 0:
                im = np.array([])
            if im.size > 0:
                if im.ndim == 3:
                    if im.shape[2] >= 3:
                        arr = (
                            0.299 * im[..., 0].astype(np.float64)
                            + 0.587 * im[..., 1].astype(np.float64)
                            + 0.114 * im[..., 2].astype(np.float64)
                        ).clip(0, 255).astype(np.uint8)
                    else:
                        arr = im[..., 0].astype(np.uint8)
                else:
                    arr = im.astype(np.uint8)
        except Exception:
            try:
                import imageio as iio_legacy  # type: ignore

                im = np.asarray(iio_legacy.imread(str(path_r)))
                if im.size > 0:
                    if im.ndim == 3 and im.shape[2] >= 3:
                        arr = (
                            0.299 * im[..., 0].astype(np.float64)
                            + 0.587 * im[..., 1].astype(np.float64)
                            + 0.114 * im[..., 2].astype(np.float64)
                        ).clip(0, 255).astype(np.uint8)
                    elif im.ndim == 3:
                        arr = im[..., 0].astype(np.uint8)
                    else:
                        arr = im.astype(np.uint8)
            except Exception:
                arr = None

    if arr is None:
        try:
            from PIL import Image

            pil = Image.open(path_r).convert("L")
            if pil.size[0] < 1 or pil.size[1] < 1:
                arr = None
            else:
                arr = np.array(pil, dtype=np.uint8)
        except Exception:
            arr = None

    if arr is None:
        return None
    arr = _downscale_gray_max_edge(arr)
    resized = _resize_gray_to_size(arr, size)
    if resized is None:
        return None
    return (resized.astype(np.float64) / 255.0).flatten().tolist()


def describe_training_image_load_failure(images_root: Path) -> str:
    """特徴量0件時の切り分け用（最初に見つかった画像で各読み込みを試す）。"""
    root = images_root.expanduser().resolve(strict=False)
    sample: Optional[Path] = None
    for cls in SCENE_CLASSES:
        cls_dir = root / "train" / cls
        if not cls_dir.is_dir():
            continue
        for p in iter_scene_dataset_images(cls_dir):
            sample = p
            break
        if sample is not None:
            break
    if sample is None:
        return f"images_root={root} の train/*/ 以下に画像拡張子のファイルがありません（サブフォルダは検索します）。"

    parts = [f"サンプル: {sample}"]
    pr = sample.expanduser().resolve(strict=False)
    parts.append(f"exists={pr.is_file()} size_bytes={pr.stat().st_size if pr.is_file() else 0}")

    if cv2 is not None:
        a = cv2.imread(str(pr), cv2.IMREAD_GRAYSCALE)
        parts.append(f"cv2.imread(grayscale)={'OK' if a is not None and a.size else 'NG'}")
    else:
        parts.append("cv2=未インストール")

    try:
        import imageio.v3 as iio  # type: ignore

        im = np.asarray(iio.imread(str(pr)))
        parts.append(f"imageio.v3={'OK shape=' + str(im.shape) if im.size else 'NG empty'}")
    except Exception as exc:
        parts.append(f"imageio.v3=NG ({exc})")
        try:
            import imageio as iio2  # type: ignore

            im2 = np.asarray(iio2.imread(str(pr)))
            parts.append(f"imageio legacy={'OK' if im2.size else 'NG empty'}")
        except Exception as exc2:
            parts.append(f"imageio legacy=NG ({exc2})")

    try:
        from PIL import Image

        pil = Image.open(pr)
        parts.append(f"PIL open={'OK ' + str(pil.size)}")
    except Exception as exc:
        parts.append(f"PIL=NG ({exc})")

    feat = image_file_to_feature(sample)
    parts.append(f"image_file_to_feature={'OK' if feat else 'NG'}")
    if not feat:
        parts.append(
            "対処: プロジェクトで venv を有効にし `pip install -e .`（少なくとも Pillow または opencv-python-headless が必要です）"
        )
    return " | ".join(parts)


def qimage_to_gray_array(image: QImage) -> Optional[np.ndarray]:
    """QImage → グレースケール配列（学習時の image_file_to_feature と同系統）。"""
    if image.isNull() or np is None:
        return None
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = gray.width(), gray.height()
    if w < 1 or h < 1:
        return None
    bpl = int(gray.bytesPerLine())
    buf = gray.constBits()
    if buf is None:
        return None
    nbytes = bpl * h
    arr = np.frombuffer(buf, dtype=np.uint8, count=nbytes).reshape(h, bpl)[:, :w].copy()
    return arr


def image_to_feature(image: QImage, size: int = 24) -> List[float]:
    """推論用。image_file_to_feature と同じ縮小・正規化に揃える。"""
    if image.isNull() or np is None:
        return []
    arr = qimage_to_gray_array(image)
    if arr is None:
        return []
    arr = _downscale_gray_max_edge(arr)
    resized = _resize_gray_to_size(arr, size)
    if resized is None:
        return []
    return (resized.astype(np.float64) / 255.0).flatten().tolist()


def l1_distance(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1e9
    total = 0.0
    for i in range(n):
        total += abs(a[i] - b[i])
    return total / n


def _distance_map(ranked: List[Tuple[str, float]]) -> Dict[str, float]:
    return {cls: dist for cls, dist in ranked}


# fever 誤検知抑制: none/go に僅差のときは raw を fever にしない
_FEVER_MIN_LEAD_OVER_NONE = 0.008
_FEVER_MIN_LEAD_OVER_SECOND = 0.010
_FEVER_GO_CLEARLY_CLOSER = 0.003


def fever_passes_confidence_gate(ranked: List[Tuple[str, float]]) -> bool:
    """fever 1 位でも、none/go に近すぎる場合は none 扱い（通常プレイの誤検知を抑える）。"""
    if not ranked or ranked[0][0] != "fever":
        return False
    dist = _distance_map(ranked)
    df = dist.get("fever", 1e9)
    dg = dist.get("go", 1e9)
    dn = dist.get("none", 1e9)
    if dg < df - _FEVER_GO_CLEARLY_CLOSER:
        return False
    if dn <= df + _FEVER_MIN_LEAD_OVER_NONE:
        return False
    if len(ranked) >= 2:
        second_cls, second_dist = ranked[1]
        if second_cls in ("none", "go") and (second_dist - df) < _FEVER_MIN_LEAD_OVER_SECOND:
            return False
    return True


def resolve_scene_label_from_ranked(ranked: List[Tuple[str, float]]) -> str:
    """最近傍。fever は go との信頼度のみ確認（時間方向は window 側）。"""
    if not ranked:
        return "none"
    best_cls = ranked[0][0]
    if best_cls == "fever":
        return "fever" if fever_passes_confidence_gate(ranked) else "none"
    return best_cls


class SceneCentroidModel:
    def __init__(self) -> None:
        self.centroids: Dict[str, List[float]] = {}

    def fit_from_dataset(self, images_root: Path) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        self.centroids = {}
        for cls in SCENE_CLASSES:
            cls_dir = images_root / "train" / cls
            vectors: List[List[float]] = []
            if cls_dir.exists():
                for file in iter_scene_dataset_images(cls_dir):
                    feat = image_file_to_feature(file)
                    if feat is None:
                        continue
                    vectors.append(feat)
            counts[cls] = len(vectors)
            if not vectors:
                continue
            dim = len(vectors[0])
            centroid = [0.0] * dim
            for vec in vectors:
                for i in range(dim):
                    centroid[i] += vec[i]
            inv = 1.0 / len(vectors)
            for i in range(dim):
                centroid[i] *= inv
            self.centroids[cls] = centroid
        return counts

    def predict_from_feature(self, feat: List[float]) -> str:
        if not self.centroids or not feat:
            return "none"
        pairs = [(cls, l1_distance(feat, centroid)) for cls, centroid in self.centroids.items()]
        pairs.sort(key=lambda x: x[1])
        return resolve_scene_label_from_ranked(pairs)

    def ranked_distances(self, image: QImage) -> List[Tuple[str, float]]:
        """全クラスの距離を昇順（分類のあいまいさ解消に使う）。"""
        if image.isNull() or not self.centroids:
            return [("none", 1e9)]
        feat = image_to_feature(image)
        pairs = [(cls, l1_distance(feat, centroid)) for cls, centroid in self.centroids.items()]
        pairs.sort(key=lambda x: x[1])
        return pairs

    def predict(self, image: QImage) -> Tuple[str, float]:
        if image.isNull() or not self.centroids:
            return ("none", 1e9)
        ranked = self.ranked_distances(image)
        label = resolve_scene_label_from_ranked(ranked)
        dist = ranked[0][1]
        for cls, d in ranked:
            if cls == label:
                dist = d
                break
        return label, dist

    def evaluate_val(self, images_root: Path) -> Tuple[int, int]:
        total = 0
        correct = 0
        for cls in SCENE_CLASSES:
            cls_dir = images_root / "val" / cls
            if not cls_dir.exists():
                continue
            for file in iter_scene_dataset_images(cls_dir):
                feat = image_file_to_feature(file)
                if feat is None:
                    continue
                pred = self.predict_from_feature(feat)
                total += 1
                if pred == cls:
                    correct += 1
        return (correct, total)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"centroids": self.centroids}, ensure_ascii=False),
            encoding="utf-8",
        )

    def load(self, path: Path) -> bool:
        if not path.exists():
            self.centroids = {}
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            centroids = data.get("centroids", {})
            if isinstance(centroids, dict):
                self.centroids = {str(k): [float(v) for v in vals] for k, vals in centroids.items()}
                return bool(self.centroids)
        except Exception:
            pass
        self.centroids = {}
        return False

    @staticmethod
    def describe_load_failure(path: Path) -> str:
        """load() が False のときの切り分け用メッセージ。"""
        if not path.exists():
            return "ファイルがありません"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return f"JSON が不正です ({exc})"
        centroids = data.get("centroids", {})
        if not isinstance(centroids, dict) or not centroids:
            return (
                "centroids が空です。"
                "学習タブで「学習開始」→「モデル保存」をやり直してください"
                "（保存時に numpy / opencv が使える venv で起動しているかも確認）"
            )
        return "centroids の形式が不正です"

    def class_count(self) -> int:
        return len(self.centroids)

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


SCENE_CLASSES = ["none", "item", "ready", "go", "fever", "timeup", "bonus", "coin", "result"]
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


def feature_matches_none_exemplars(
    feat: List[float],
    exemplars: List[List[float]],
    max_dist: float,
) -> bool:
    if not feat or not exemplars:
        return False
    return min(l1_distance(feat, ex) for ex in exemplars) <= max_dist


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


_IN_GAME_FEVER_NEAR_TOP = 0.030
_IN_GAME_FEVER_NEAR_ENDGAME = 0.028
# train/none の実画像（新しい順）に近いフレームは fever 候補から外す（centroid 平均だけでは効かない誤検知向け）
_NONE_VETO_EXEMPLAR_MAX = 60
_NONE_VETO_MARGIN = 0.005
# CNN 利用時（centroid なし）: train/none  exemplar との L1 がこれ以下なら fever 抑制
_NONE_VETO_ABSOLUTE_MAX = 0.012
# 解析中にユーザーが fever 誤検知で none 保存したフレームのみ（ほぼ同一）
_NONE_VETO_SESSION_MAX = 0.007
DEFAULT_FEVER_CALIB: Dict[str, float] = {
    "top1_lead": 0.003,
    "none_go_gap": 0.015,
    "none_go_lead": 0.008,
    "endgame_gap": 0.030,
    "endgame_lead": 0.003,
    "weak_none_go_gap": 0.022,
    "weak_none_go_lead": 0.002,
    "weak_endgame_gap": 0.032,
    "weak_endgame_lead": 0.0,
}


def _fever_calib(calib: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    merged = dict(DEFAULT_FEVER_CALIB)
    if calib:
        for key, val in calib.items():
            if key in merged:
                merged[key] = float(val)
    return merged


def _fever_go_blocks(dist: Dict[str, float]) -> bool:
    dg = dist.get("go", 1e9)
    df = dist.get("fever", 1e9)
    return dg < df - _FEVER_GO_CLEARLY_CLOSER


def fever_strong_from_ranked(
    ranked: List[Tuple[str, float]],
    calib: Optional[Dict[str, float]] = None,
) -> bool:
    """はっきりした fever（1 サンプルで確定向け）。学習時に val から閾値を更新する。"""
    if not ranked:
        return False
    params = _fever_calib(calib)
    dist = _distance_map(ranked)
    if _fever_go_blocks(dist):
        return False
    top1, d1 = ranked[0]
    df = dist.get("fever", 1e9)
    dn = dist.get("none", 1e9)
    if top1 == "fever":
        return (dn - df) >= params["top1_lead"]
    if top1 in ("none", "go"):
        return df <= d1 + params["none_go_gap"] and (dn - df) >= params["none_go_lead"]
    if top1 in ("timeup", "bonus", "result"):
        return df <= d1 + params["endgame_gap"] and (dn - df) >= params["endgame_lead"]
    return False


def fever_weak_from_ranked(
    ranked: List[Tuple[str, float]],
    calib: Optional[Dict[str, float]] = None,
) -> bool:
    """弱い fever 候補（連続 2 サンプルで確定）。取りこぼし補完用。"""
    if not ranked or fever_strong_from_ranked(ranked, calib):
        return False
    params = _fever_calib(calib)
    dist = _distance_map(ranked)
    if _fever_go_blocks(dist):
        return False
    top1, d1 = ranked[0]
    df = dist.get("fever", 1e9)
    dn = dist.get("none", 1e9)
    if top1 == "fever":
        return (dn - df) >= params["weak_none_go_lead"]
    if top1 in ("none", "go"):
        return df <= d1 + params["weak_none_go_gap"] and (dn - df) >= params["weak_none_go_lead"]
    if top1 in ("timeup", "bonus", "result"):
        return df <= d1 + params["weak_endgame_gap"] and (dn - df) >= params["weak_endgame_lead"]
    return False


def none_go_blocks_loose_fever(ranked: List[Tuple[str, float]]) -> bool:
    """1 位が none/go で fever よりかなり近い → 緩い recall だけ止める（本物 fever は通す）。"""
    if not ranked:
        return False
    top1, _d1 = ranked[0]
    if top1 == "fever":
        return False
    dist = _distance_map(ranked)
    df = dist.get("fever", 1e9)
    if top1 == "none":
        return (dist.get("none", 1e9) - df) >= 0.012
    if top1 == "go":
        return (dist.get("go", 1e9) - df) >= 0.012
    return False


def in_game_fever_candidate_from_ranked(
    ranked: List[Tuple[str, float]],
    calib: Optional[Dict[str, float]] = None,
) -> bool:
    """IN_GAME: 学習時に調整した fever ゲート（strong / weak / recall）。"""
    if not ranked:
        return False
    if fever_strong_from_ranked(ranked, calib) or fever_weak_from_ranked(ranked, calib):
        return True
    if none_go_blocks_loose_fever(ranked):
        return False
    return fever_recall_from_ranked(ranked, calib)


def fever_recall_from_ranked(
    ranked: List[Tuple[str, float]],
    calib: Optional[Dict[str, float]] = None,
) -> bool:
    """取りこぼし防止の緩い fever 候補（強/弱の次に適用）。"""
    if not ranked:
        return False
    if fever_strong_from_ranked(ranked, calib) or fever_weak_from_ranked(ranked, calib):
        return True
    dist = _distance_map(ranked)
    if _fever_go_blocks(dist):
        return False
    top1, d1 = ranked[0]
    df = dist.get("fever", 1e9)
    if top1 == "fever":
        return True
    if top1 in ("none", "go") and df <= d1 + _IN_GAME_FEVER_NEAR_TOP:
        return True
    if top1 in ("timeup", "bonus", "result") and df <= d1 + _IN_GAME_FEVER_NEAR_ENDGAME + 0.006:
        return True
    near = (
        _IN_GAME_FEVER_NEAR_ENDGAME + 0.006
        if top1 in ("timeup", "bonus", "result")
        else _IN_GAME_FEVER_NEAR_TOP
    )
    for cls, _d in ranked[:3]:
        if cls == "fever" and df <= d1 + near:
            return True
    return False


def in_game_fever_live_from_ranked(
    ranked: List[Tuple[str, float]],
    calib: Optional[Dict[str, float]] = None,
) -> bool:
    """実動画向け（IN_GAME）。スキル UI 混在でも fever 画面を取りこぼしにくくする。"""
    if not ranked:
        return False
    dist = _distance_map(ranked)
    if "fever" not in dist:
        return False
    df = dist.get("fever", 1e9)
    dg = dist.get("go", 1e9)
    if dg < df - _FEVER_GO_CLEARLY_CLOSER:
        return False
    top1, d1 = ranked[0]
    if top1 == "fever":
        return True
    # fever が 2 位以内かつ 1 位との差が小さい → 実プレイの fever 取りこぼし防止
    margin = 0.045
    if top1 in ("timeup", "bonus", "result"):
        margin = 0.050
    if df <= d1 + margin:
        return True
    for cls, d_cls in ranked[:4]:
        if cls == "fever" and df <= d_cls + 0.010:
            return True
    return False


def in_game_fever_raw_from_ranked(
    ranked: List[Tuple[str, float]],
    calib: Optional[Dict[str, float]] = None,
) -> str:
    """フロー遷移用（go 取りこぼし時の IN_GAME 入り）。"""
    if fever_recall_from_ranked(ranked, calib):
        return "fever"
    return "none"


def calibrate_fever_gate(
    centroids: Dict[str, List[float]],
    images_root: Path,
) -> Dict[str, float]:
    """val 画像で fever 閾値を grid search し、centroid 更新のたびに合わせる。"""
    if not centroids or "fever" not in centroids:
        return dict(DEFAULT_FEVER_CALIB)

    cache: Dict[str, List[List[Tuple[str, float]]]] = {"fever": [], "none": [], "go": []}
    val_root = images_root / "val"
    for cls in cache:
        cls_dir = val_root / cls
        if not cls_dir.is_dir():
            continue
        for file in iter_scene_dataset_images(cls_dir):
            feat = image_file_to_feature(file)
            if feat is None:
                continue
            pairs = [(c, l1_distance(feat, centroids[c])) for c in centroids]
            pairs.sort(key=lambda x: x[1])
            cache[cls].append(pairs)

    if not cache["fever"]:
        return dict(DEFAULT_FEVER_CALIB)

    best: Optional[tuple[int, Dict[str, float], int, int, int]] = None
    for top1_lead in (0.0, 0.003, 0.005, 0.008):
        for ng_gap in (0.012, 0.015, 0.018, 0.022):
            for ng_lead in (0.005, 0.008, 0.010, 0.012):
                for end_gap in (0.028, 0.030, 0.032, 0.035):
                    calib = {
                        "top1_lead": top1_lead,
                        "none_go_gap": ng_gap,
                        "none_go_lead": ng_lead,
                        "endgame_gap": end_gap,
                        "endgame_lead": 0.003,
                        "weak_none_go_gap": min(ng_gap + 0.012, 0.035),
                        "weak_none_go_lead": 0.0,
                        "weak_endgame_gap": end_gap + 0.006,
                        "weak_endgame_lead": 0.0,
                    }
                    tp = sum(
                        1
                        for ranked in cache["fever"]
                        if fever_recall_from_ranked(ranked, calib)
                    )
                    strong_fp = sum(
                        1 for ranked in cache["none"] + cache["go"]
                        if fever_strong_from_ranked(ranked, calib)
                    )
                    weak_fp = sum(
                        1 for ranked in cache["none"] + cache["go"]
                        if fever_weak_from_ranked(ranked, calib)
                    )
                    # 弱候補は解析側で 2 連続が必要。none/go の strong 誤検知を強く罰する
                    score = tp * 100 - strong_fp * 20 - weak_fp * 4
                    if best is None or score > best[0]:
                        best = (score, calib, tp, strong_fp, weak_fp)

    return best[1] if best else dict(DEFAULT_FEVER_CALIB)


def evaluate_fever_gate(
    centroids: Dict[str, List[float]],
    images_root: Path,
    calib: Optional[Dict[str, float]] = None,
) -> Dict[str, int]:
    """fever ゲートの val 指標（学習ログ用）。"""
    params = _fever_calib(calib)
    out = {
        "fever_val_total": 0,
        "fever_val_strong": 0,
        "fever_val_weak_only": 0,
        "fever_val_any": 0,
        "false_none_go_strong": 0,
        "false_none_go_weak": 0,
    }
    val_root = images_root / "val"
    for cls, keys in (
        ("fever", ("fever_val_total", "fever_val_strong", "fever_val_weak_only", "fever_val_any")),
        ("none", ("false_none_go_strong", "false_none_go_weak")),
        ("go", ("false_none_go_strong", "false_none_go_weak")),
    ):
        cls_dir = val_root / cls
        if not cls_dir.is_dir():
            continue
        for file in iter_scene_dataset_images(cls_dir):
            feat = image_file_to_feature(file)
            if feat is None:
                continue
            pairs = [(c, l1_distance(feat, centroids[c])) for c in centroids]
            pairs.sort(key=lambda x: x[1])
            strong = fever_strong_from_ranked(pairs, params)
            weak = fever_weak_from_ranked(pairs, params)
            if cls == "fever":
                out["fever_val_total"] += 1
                recall = fever_recall_from_ranked(pairs, params)
                if strong:
                    out["fever_val_strong"] += 1
                if weak:
                    out["fever_val_weak_only"] += 1
                if recall:
                    out["fever_val_any"] += 1
            else:
                if strong:
                    out["false_none_go_strong"] += 1
                if weak:
                    out["false_none_go_weak"] += 1
    return out


class SceneCentroidModel:
    def __init__(self) -> None:
        self.centroids: Dict[str, List[float]] = {}
        self.fever_calib: Dict[str, float] = dict(DEFAULT_FEVER_CALIB)
        self.none_veto_exemplars: List[List[float]] = []

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
        self.fever_calib = calibrate_fever_gate(self.centroids, images_root)
        self.rebuild_none_veto_exemplars(images_root)
        return counts

    def rebuild_none_veto_exemplars(
        self,
        images_root: Path,
        max_n: int = _NONE_VETO_EXEMPLAR_MAX,
    ) -> int:
        """train/none の新しい画像をそのまま参照し、似たフレームは fever にしない。"""
        self.none_veto_exemplars = []
        cls_dir = images_root / "train" / "none"
        if not cls_dir.is_dir():
            return 0
        files = sorted(
            iter_scene_dataset_images(cls_dir),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in files[:max_n]:
            feat = image_file_to_feature(path)
            if feat is not None:
                self.none_veto_exemplars.append(feat)
        return len(self.none_veto_exemplars)

    def none_veto_blocks_fever(self, feat: List[float]) -> bool:
        """保存済み train/none に近いフレームは fever 扱いにしない（CNN でも有効）。"""
        if not feat or not self.none_veto_exemplars:
            return False
        best_none = min(l1_distance(feat, ex) for ex in self.none_veto_exemplars)
        if self.centroids and "fever" in self.centroids:
            df = l1_distance(feat, self.centroids["fever"])
            return best_none + 0.002 < df
        return best_none <= _NONE_VETO_ABSOLUTE_MAX

    def none_veto_blocks_timeup(
        self,
        feat: List[float],
        *,
        timeup_score: Optional[float] = None,
    ) -> bool:
        """保存済み train/none に近いフレームは timeup 扱いにしない（CNN でも有効）。"""
        if not feat or not self.none_veto_exemplars:
            return False
        if timeup_score is not None and timeup_score <= 0.12:
            # CNN が強く timeup を示す場合は抑制しない。
            return False
        best_none = min(l1_distance(feat, ex) for ex in self.none_veto_exemplars)
        if self.centroids and "timeup" in self.centroids:
            dt = l1_distance(feat, self.centroids["timeup"])
            margin = 0.002 if (timeup_score is None or timeup_score >= 0.22) else 0.006
            return best_none + margin < dt
        return best_none <= _NONE_VETO_ABSOLUTE_MAX

    def none_veto_blocks_ready(
        self,
        feat: List[float],
        *,
        ready_score: Optional[float] = None,
    ) -> bool:
        """保存済み train/none に近いフレームは ready 扱いにしない（CNN でも有効）。"""
        if not feat or not self.none_veto_exemplars:
            return False
        # CNN がはっきり ready と出しているときは centroid 比較で潰さない
        if ready_score is not None and ready_score < 0.45:
            return False
        best_none = min(l1_distance(feat, ex) for ex in self.none_veto_exemplars)
        if self.centroids and "ready" in self.centroids:
            dr = l1_distance(feat, self.centroids["ready"])
            margin = 0.002 if (ready_score is None or ready_score >= 0.22) else 0.006
            return best_none + margin < dr
        return best_none <= _NONE_VETO_ABSOLUTE_MAX

    def exemplar_blocks_fever(self, feat: List[float]) -> bool:
        """保存した none 実画像にほぼ同一のフレームだけ fever を止める（誤判定を広げない）。"""
        return self.none_veto_blocks_fever(feat)

    def in_game_fever_raw(self, ranked: List[Tuple[str, float]]) -> str:
        return in_game_fever_raw_from_ranked(ranked, self.fever_calib)

    def in_game_fever_live(self, ranked: List[Tuple[str, float]]) -> bool:
        return in_game_fever_live_from_ranked(ranked, self.fever_calib)

    def in_game_fever_candidate(self, ranked: List[Tuple[str, float]]) -> bool:
        return in_game_fever_candidate_from_ranked(ranked, self.fever_calib)

    def none_go_blocks_loose_fever(self, ranked: List[Tuple[str, float]]) -> bool:
        return none_go_blocks_loose_fever(ranked)

    def exemplar_blocks_fever_feature(self, feat: List[float]) -> bool:
        return self.exemplar_blocks_fever(feat)

    def fever_strong(self, ranked: List[Tuple[str, float]]) -> bool:
        return fever_strong_from_ranked(ranked, self.fever_calib)

    def fever_weak(self, ranked: List[Tuple[str, float]]) -> bool:
        return fever_weak_from_ranked(ranked, self.fever_calib)

    def fever_gate_metrics(self, images_root: Path) -> Dict[str, int]:
        return evaluate_fever_gate(self.centroids, images_root, self.fever_calib)

    def predict_from_feature(self, feat: List[float]) -> str:
        if not self.centroids or not feat:
            return "none"
        return resolve_scene_label_from_ranked(self.ranked_from_feature(feat))

    def ranked_from_feature(self, feat: List[float]) -> List[Tuple[str, float]]:
        if not self.centroids or not feat:
            return [("none", 1e9)]
        pairs = [(cls, l1_distance(feat, centroid)) for cls, centroid in self.centroids.items()]
        pairs.sort(key=lambda x: x[1])
        return pairs

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
            json.dumps(
                {"centroids": self.centroids, "fever_calib": self.fever_calib},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self, path: Path) -> bool:
        if not path.exists():
            self.centroids = {}
            self.fever_calib = dict(DEFAULT_FEVER_CALIB)
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            centroids = data.get("centroids", {})
            if isinstance(centroids, dict):
                self.centroids = {str(k): [float(v) for v in vals] for k, vals in centroids.items()}
                raw_calib = data.get("fever_calib", {})
                if isinstance(raw_calib, dict) and raw_calib:
                    self.fever_calib = _fever_calib(raw_calib)
                else:
                    self.fever_calib = dict(DEFAULT_FEVER_CALIB)
                return bool(self.centroids)
        except Exception:
            pass
        self.centroids = {}
        self.fever_calib = dict(DEFAULT_FEVER_CALIB)
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

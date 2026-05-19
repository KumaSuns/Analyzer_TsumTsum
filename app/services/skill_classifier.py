from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from PySide6.QtGui import QImage

from app.services.use_tsum_classifier import UseTsumClassifier

# ツムごとの skills/<dir>/<category>/ 。発動検出は activation（発動時）のみ使用。
SKILL_IMAGE_CATEGORY_ACTIVATION = "activation"
SKILL_TRAIN_CATEGORIES = (SKILL_IMAGE_CATEGORY_ACTIVATION,)

SKILL_CATEGORY_DISPLAY: Dict[str, str] = {
    SKILL_IMAGE_CATEGORY_ACTIVATION: "発動時",
}

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})


def list_skill_tsum_dirs(images_root: Path) -> List[str]:
    if not images_root.exists():
        return []
    return sorted(
        p.name for p in images_root.iterdir() if p.is_dir() and not p.name.startswith(".")
    )


def list_skill_categories(images_root: Path, tsum_dir: str) -> List[str]:
    """skills/<tsum_dir>/ 直下のサブフォルダ名（発動時など）。"""
    categories: List[str] = list(SKILL_TRAIN_CATEGORIES)
    tsum_path = images_root / tsum_dir
    if tsum_path.is_dir():
        for p in sorted(tsum_path.iterdir()):
            if p.is_dir() and not p.name.startswith(".") and p.name not in categories:
                categories.append(p.name)
    return categories


def skill_category_label(category: str) -> str:
    return SKILL_CATEGORY_DISPLAY.get(category, category)


def iter_skill_images(tsum_dir: Path):
    """skills/<tsum_dir>/<category>/ 以下の画像を列挙（直下ファイルは無視）。"""
    for category in SKILL_TRAIN_CATEGORIES:
        cat_dir = tsum_dir / category
        if not cat_dir.is_dir():
            continue
        for path in sorted(cat_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                yield path


def compute_match_max_dist(prototypes: List[List[float]]) -> float:
    """学習画像同士の距離から、誤検知を抑えたマッチ上限を推定する。"""
    if len(prototypes) < 2:
        return 0.065
    worst = 0.0
    for i, vec_a in enumerate(prototypes):
        for j, vec_b in enumerate(prototypes):
            if i == j:
                continue
            worst = max(worst, UseTsumClassifier._l1_distance(vec_a, vec_b))
    # 同じツムの発動画像のばらつきより少し余裕だけ見る（全画面系の誤検知は別途シーンで抑制）
    return min(0.065, max(0.05, worst * 1.1))


def resolve_tsum_dir(
    label: str,
    registry: Dict[str, str],
    known_dirs: List[str],
) -> str:
    """表示名・dir名・registry いずれから skill 用 dir 名（例: namine, c_bazu）へ。"""
    if not label or label in {"-", "unknown", "unknown_tsum"}:
        return ""
    if label in known_dirs:
        return label
    for dir_id, display in registry.items():
        if label == display or label == dir_id:
            return dir_id
    return label


class SkillClassifierPool:
    """使用ツムごとのスキル検出モデル（use_tsum とは別ディレクトリ）。"""

    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self._prototypes: Dict[str, List[List[float]]] = {}
        self._max_dist: Dict[str, float] = {}
        self.reload()

    def reload(self) -> None:
        self._prototypes = {}
        self._max_dist = {}
        if not self.models_root.exists():
            return
        for tsum_dir in sorted(p for p in self.models_root.iterdir() if p.is_dir()):
            model_file = tsum_dir / "model.json"
            if not model_file.exists():
                continue
            try:
                data = json.loads(model_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            raw = data.get("prototypes", [])
            feats: List[List[float]] = []
            if isinstance(raw, list):
                for vec in raw:
                    if isinstance(vec, list):
                        feats.append([float(v) for v in vec])
            if feats:
                self._prototypes[tsum_dir.name] = feats
                stored = data.get("match_max_dist")
                if isinstance(stored, (int, float)) and float(stored) > 0:
                    self._max_dist[tsum_dir.name] = float(stored)
                else:
                    self._max_dist[tsum_dir.name] = compute_match_max_dist(feats)

    def known_dirs(self) -> List[str]:
        return sorted(self._prototypes.keys())

    def has_model(self, tsum_dir: str) -> bool:
        return bool(tsum_dir) and tsum_dir in self._prototypes

    def match_threshold(self, tsum_dir: str) -> float:
        return self._max_dist.get(tsum_dir, 0.065)

    def predict(
        self,
        tsum_dir: str,
        image: QImage,
        crop_rect: Tuple[float, float, float, float] | None = None,
        *,
        max_dist: float | None = None,
    ) -> Tuple[bool, float]:
        protos = self._prototypes.get(tsum_dir)
        if not protos or image.isNull():
            return (False, float("inf"))
        roi = image
        if crop_rect is not None:
            cropped = UseTsumClassifier._crop_by_normalized_rect(image, crop_rect)
            if not cropped.isNull():
                roi = cropped
        feat = UseTsumClassifier._image_to_feature(roi)
        if not feat:
            return (False, float("inf"))
        dist = min(UseTsumClassifier._l1_distance(feat, proto) for proto in protos)
        limit = self.match_threshold(tsum_dir) if max_dist is None else max_dist
        return (dist <= limit, dist)

    @staticmethod
    def build_models(
        images_root: Path,
        models_root: Path,
        crop_rect: Tuple[float, float, float, float] | None = None,
    ) -> Dict[str, int]:
        models_root.mkdir(parents=True, exist_ok=True)
        result: Dict[str, int] = {}
        if not images_root.exists():
            return result

        for tsum_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
            feats: List[List[float]] = []
            for image_file in iter_skill_images(tsum_dir):
                image = QImage(str(image_file))
                if image.isNull():
                    continue
                if crop_rect is not None:
                    cropped = UseTsumClassifier._crop_by_normalized_rect(image, crop_rect)
                    if not cropped.isNull():
                        image = cropped
                feat = UseTsumClassifier._image_to_feature(image)
                if feat:
                    feats.append(feat)
            out_dir = models_root / tsum_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "model.json"
            match_max = compute_match_max_dist(feats) if feats else 0.065
            out_file.write_text(
                json.dumps(
                    {
                        "tsum_id": tsum_dir.name,
                        "sample_count": len(feats),
                        "match_max_dist": match_max,
                        "prototypes": feats,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            result[tsum_dir.name] = len(feats)
        return result

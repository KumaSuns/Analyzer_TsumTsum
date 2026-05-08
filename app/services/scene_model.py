from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PySide6.QtGui import QImage


SCENE_CLASSES = ["none", "item", "ready", "go", "fever", "timeup", "bonus", "result"]


def image_to_feature(image: QImage, size: int = 24) -> List[float]:
    gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
    small = gray.scaled(size, size)
    feat: List[float] = []
    for y in range(small.height()):
        for x in range(small.width()):
            feat.append(small.pixelColor(x, y).red() / 255.0)
    return feat


def l1_distance(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 1e9
    total = 0.0
    for i in range(n):
        total += abs(a[i] - b[i])
    return total / n


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
                for file in cls_dir.iterdir():
                    if not file.is_file():
                        continue
                    img = QImage(str(file))
                    if img.isNull():
                        continue
                    vectors.append(image_to_feature(img))
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

    def predict(self, image: QImage) -> Tuple[str, float]:
        if image.isNull() or not self.centroids:
            return ("none", 1e9)
        feat = image_to_feature(image)
        best_cls = "none"
        best_dist = 1e9
        for cls, centroid in self.centroids.items():
            d = l1_distance(feat, centroid)
            if d < best_dist:
                best_dist = d
                best_cls = cls
        return (best_cls, best_dist)

    def evaluate_val(self, images_root: Path) -> Tuple[int, int]:
        total = 0
        correct = 0
        for cls in SCENE_CLASSES:
            cls_dir = images_root / "val" / cls
            if not cls_dir.exists():
                continue
            for file in cls_dir.iterdir():
                if not file.is_file():
                    continue
                img = QImage(str(file))
                if img.isNull():
                    continue
                pred, _ = self.predict(img)
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

    def class_count(self) -> int:
        return len(self.centroids)

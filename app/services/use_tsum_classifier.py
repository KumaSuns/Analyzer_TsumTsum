from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from PySide6.QtGui import QImage


class UseTsumClassifier:
    """Simple classifier using saved prototypes from model files."""

    def __init__(self, models_root: Path, registry_path: Path) -> None:
        self.models_root = models_root
        self.registry_path = registry_path
        self._display_map: Dict[str, str] = {}
        self._prototypes: Dict[str, List[List[float]]] = {}
        self.reload()

    def reload(self) -> None:
        self._display_map = self._load_registry()
        self._prototypes = {}
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

    def predict(self, roi_image: QImage) -> Tuple[str, float]:
        if roi_image.isNull() or not self._prototypes:
            return ("unknown", float("inf"))
        target = self._image_to_feature(roi_image)
        if not target:
            return ("unknown", float("inf"))

        best_id = "unknown"
        best_dist = float("inf")
        for tsum_id, proto_list in self._prototypes.items():
            # nearest prototype distance
            dist = min(self._l1_distance(target, proto) for proto in proto_list)
            if dist < best_dist:
                best_dist = dist
                best_id = tsum_id
        return (self._display_map.get(best_id, best_id), best_dist)

    def _load_registry(self) -> Dict[str, str]:
        if not self.registry_path.exists():
            return {}
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    @staticmethod
    def build_models(
        images_root: Path,
        models_root: Path,
        use_tsum_rect: Tuple[float, float, float, float] | None = None,
    ) -> Dict[str, int]:
        models_root.mkdir(parents=True, exist_ok=True)
        result: Dict[str, int] = {}
        if not images_root.exists():
            return result

        for tsum_dir in sorted(p for p in images_root.iterdir() if p.is_dir()):
            feats: List[List[float]] = []
            for image_file in sorted(p for p in tsum_dir.iterdir() if p.is_file()):
                image = QImage(str(image_file))
                if image.isNull():
                    continue
                if use_tsum_rect is not None:
                    cropped = UseTsumClassifier._crop_by_normalized_rect(image, use_tsum_rect)
                    if not cropped.isNull():
                        image = cropped
                feat = UseTsumClassifier._image_to_feature(image)
                if feat:
                    feats.append(feat)
            out_dir = models_root / tsum_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "model.json"
            out_file.write_text(
                json.dumps(
                    {
                        "tsum_id": tsum_dir.name,
                        "sample_count": len(feats),
                        "prototypes": feats,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            result[tsum_dir.name] = len(feats)
        return result

    @staticmethod
    def _image_to_feature(image: QImage) -> List[float]:
        rgb = image.convertToFormat(QImage.Format.Format_RGB32)
        color_small = rgb.scaled(18, 18)
        h_bins = [0.0] * 18
        s_bins = [0.0] * 8
        v_bins = [0.0] * 8
        total = 0.0
        for y in range(color_small.height()):
            for x in range(color_small.width()):
                c = color_small.pixelColor(x, y)
                h, s, v, _a = c.getHsvF()
                if h < 0:
                    h = 0.0
                h_bins[min(17, max(0, int(h * 18)))] += 1.0
                s_bins[min(7, max(0, int(s * 8)))] += 1.0
                v_bins[min(7, max(0, int(v * 8)))] += 1.0
                total += 1.0
        if total > 0:
            inv = 1.0 / total
            h_bins = [v * inv for v in h_bins]
            s_bins = [v * inv for v in s_bins]
            v_bins = [v * inv for v in v_bins]

        gray = image.convertToFormat(QImage.Format.Format_Grayscale8)
        gray_small = gray.scaled(14, 14)
        gray_values: List[float] = []
        mean = 0.0
        for y in range(gray_small.height()):
            for x in range(gray_small.width()):
                value = gray_small.pixelColor(x, y).red() / 255.0
                gray_values.append(value)
                mean += value
        if gray_values:
            mean /= len(gray_values)
            gray_values = [v - mean for v in gray_values]

        return h_bins + s_bins + v_bins + gray_values

    @staticmethod
    def _l1_distance(a: List[float], b: List[float]) -> float:
        n = min(len(a), len(b))
        if n == 0:
            return float("inf")
        total = 0.0
        for i in range(n):
            total += abs(a[i] - b[i])
        return total / n


    @staticmethod
    def _crop_by_normalized_rect(image: QImage, rect: Tuple[float, float, float, float]) -> QImage:
        width = image.width()
        height = image.height()
        if width <= 0 or height <= 0:
            return QImage()
        nx, ny, nw, nh = rect
        x = max(0, min(int(nx * width), width - 1))
        y = max(0, min(int(ny * height), height - 1))
        w = max(1, min(int(nw * width), width - x))
        h = max(1, min(int(nh * height), height - y))
        return image.copy(x, y, w, h)

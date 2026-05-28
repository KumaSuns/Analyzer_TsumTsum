import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from app.services.scene_cnn import SceneCnnClassifier
from app.services.scene_model import SceneCentroidModel

SCENE_LABELS = ["none", "item", "ready", "go", "fever", "timeup", "bonus", "result"]


@dataclass
class AnalysisResult:
    frame_index: int
    timestamp_ms: int
    scene_label: str
    item_skill_label: str


class SceneClassifier:
    """Scene classifier: CNN (scene_cnn.pt) 優先、なければ centroid。"""

    def __init__(self) -> None:
        self.centroid = SceneCentroidModel()
        self.cnn = SceneCnnClassifier()
        self.backend = "none"
        self.loaded = False

    def uses_cnn(self) -> bool:
        return self.backend == "cnn" and self.cnn.is_loaded()

    def load_model(self, model_path: Path) -> bool:
        version_dir = model_path.parent
        cnn_path = version_dir / "scene_cnn.pt"
        self.loaded = False
        self.backend = "none"
        if self.cnn.load(cnn_path):
            self.backend = "cnn"
            self.loaded = True
            return True
        if self.centroid.load(model_path):
            self.backend = "centroid"
            self.loaded = True
            return True
        return False

    @property
    def model(self) -> SceneCentroidModel:
        """互換: window の centroid 参照用。"""
        return self.centroid

    def predict(self, frame_image) -> str:
        if frame_image is None:
            return "none"
        if self.uses_cnn():
            return self.cnn.predict_qimage(frame_image)
        label, _dist = self.centroid.predict(frame_image)
        return label

    def predict_feature(self, feat: List[float]) -> str:
        if self.uses_cnn():
            return "none"
        if not feat:
            return "none"
        return self.centroid.predict_from_feature(feat)

    def ranked_distances(self, frame_image) -> List[Tuple[str, float]]:
        if frame_image is None or getattr(frame_image, "isNull", lambda: True)():
            return [("none", 1e9)]
        if self.uses_cnn():
            return self.cnn.ranked_qimage(frame_image)
        return self.centroid.ranked_distances(frame_image)

    def ranked_from_feature(self, feat: List[float]) -> List[Tuple[str, float]]:
        if self.uses_cnn():
            return [("none", 1e9)]
        return self.centroid.ranked_from_feature(feat)


class TsumItemSkillClassifier:
    """scene=item 時にログへ載せる使用ツム表示名（dir→表示名）。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry_path = root / "registry.json"
        self.registry: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self.registry = {}
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.registry = {str(k): str(v) for k, v in data.items()}
            except Exception:
                self.registry = {}

    def display_name(self, tsum_dir: str) -> str:
        if not tsum_dir:
            return "-"
        return self.registry.get(tsum_dir, tsum_dir)


class VideoAnalyzer:
    def __init__(
        self,
        sample_every_frames: int = 5,
        model_root: Optional[Path] = None,
        use_tsum_models_root: Optional[Path] = None,
        images_root: Optional[Path] = None,
    ) -> None:
        self.sample_every_frames = max(1, sample_every_frames)
        self.scene_classifier = SceneClassifier()
        use_tsum_root = use_tsum_models_root or Path("app/models/use_tsum")
        self.item_skill_classifier = TsumItemSkillClassifier(use_tsum_root)
        self._last_sampled_frame = -1
        self.model_root = model_root or Path("app/models/main_model")
        self.images_root = images_root or Path("app/assets/images")
        self.active_model_version = "version_1"
        self.scene_model_loaded = False
        self.scene_class_count = 0
        self.scene_backend = "none"
        self.scene_model_path: Path = self.model_root / "scene_model.json"
        self.reload_model()

    def reset(self) -> None:
        self._last_sampled_frame = -1

    def reload_model(self, version: Optional[str] = None) -> None:
        if version and str(version).strip():
            self.active_model_version = str(version).strip()
        else:
            active_file = self.model_root / "ACTIVE_VERSION"
            if active_file.exists():
                self.active_model_version = active_file.read_text(encoding="utf-8").strip() or "version_1"
            else:
                self.active_model_version = "version_1"
        version_dir = self.model_root / self.active_model_version
        self.scene_model_path = version_dir / "scene_model.json"
        self.scene_model_loaded = self.scene_classifier.load_model(self.scene_model_path)
        self.scene_backend = self.scene_classifier.backend
        if self.scene_classifier.uses_cnn():
            self.scene_class_count = len(self.scene_classifier.cnn.classes)
        else:
            self.scene_class_count = self.scene_classifier.centroid.class_count()
            if self.scene_model_loaded:
                self.scene_classifier.centroid.rebuild_none_veto_exemplars(self.images_root)
        self.item_skill_classifier.reload()

    def apply_scene_centroids(
        self,
        centroids: dict,
        fever_calib: Optional[dict] = None,
    ) -> bool:
        if not centroids:
            return False
        self.scene_classifier.centroid.centroids = {
            str(k): [float(v) for v in vals] for k, vals in centroids.items()
        }
        if fever_calib:
            from app.services.scene_model import _fever_calib

            self.scene_classifier.centroid.fever_calib = _fever_calib(fever_calib)
        self.scene_classifier.backend = "centroid"
        self.scene_backend = "centroid"
        self.scene_model_loaded = True
        self.scene_class_count = self.scene_classifier.centroid.class_count()
        self.scene_classifier.centroid.rebuild_none_veto_exemplars(self.images_root)
        return True

    def active_model_meta(self) -> dict:
        meta_path = self.model_root / self.active_model_version / "model_meta.json"
        if not meta_path.exists():
            return {}
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def process_frame(
        self,
        frame_seq: int,
        position_ms: int,
        selected_tsum: str = "auto",
        frame_image=None,
        use_tsum_dir: str = "",
        scene_feature: Optional[list] = None,
    ) -> List[AnalysisResult]:
        frame_index = max(0, int(frame_seq))
        if frame_index == self._last_sampled_frame:
            return []

        if frame_index % self.sample_every_frames != 0:
            return []

        self._last_sampled_frame = frame_index
        if self.scene_classifier.uses_cnn():
            scene_label = self.scene_classifier.predict(frame_image)
        elif scene_feature:
            scene_label = self.scene_classifier.predict_feature(scene_feature)
        else:
            scene_label = self.scene_classifier.predict(frame_image)
        item_skill_label = "-"
        if scene_label == "item" and use_tsum_dir:
            item_skill_label = self.item_skill_classifier.display_name(use_tsum_dir)

        return [
            AnalysisResult(
                frame_index=frame_index,
                timestamp_ms=max(position_ms, 0),
                scene_label=scene_label,
                item_skill_label=item_skill_label,
            )
        ]

    def process_position(
        self,
        position_ms: int,
        fps: float,
        selected_tsum: str = "auto",
        frame_image=None,
        use_tsum_dir: str = "",
        scene_feature: Optional[list] = None,
    ) -> List[AnalysisResult]:
        if fps <= 0:
            fps = 30.0

        frame_index = int((max(position_ms, 0) / 1000.0) * fps)
        if frame_index == self._last_sampled_frame:
            return []

        if frame_index % self.sample_every_frames != 0:
            return []

        self._last_sampled_frame = frame_index
        if self.scene_classifier.uses_cnn():
            scene_label = self.scene_classifier.predict(frame_image)
        elif scene_feature:
            scene_label = self.scene_classifier.predict_feature(scene_feature)
        else:
            scene_label = self.scene_classifier.predict(frame_image)
        item_skill_label = "-"
        if scene_label == "item" and use_tsum_dir:
            item_skill_label = self.item_skill_classifier.display_name(use_tsum_dir)

        return [
            AnalysisResult(
                frame_index=frame_index,
                timestamp_ms=max(position_ms, 0),
                scene_label=scene_label,
                item_skill_label=item_skill_label,
            )
        ]

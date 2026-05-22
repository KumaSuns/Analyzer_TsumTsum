import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from app.services.scene_model import SceneCentroidModel

SCENE_LABELS = ["none", "item", "ready", "go", "fever", "timeup", "bonus", "result"]


@dataclass
class AnalysisResult:
    frame_index: int
    timestamp_ms: int
    scene_label: str
    item_skill_label: str


class SceneClassifier:
    """Scene classifier backed by saved training model."""

    def __init__(self) -> None:
        self.model = SceneCentroidModel()
        self.loaded = False

    def load_model(self, model_path: Path) -> bool:
        self.loaded = self.model.load(model_path)
        return self.loaded

    def predict(self, frame_image) -> str:
        if frame_image is None:
            return "none"
        label, _dist = self.model.predict(frame_image)
        return label

    def predict_feature(self, feat: List[float]) -> str:
        if not feat:
            return "none"
        return self.model.predict_from_feature(feat)


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
    def __init__(self, sample_every_frames: int = 5, model_root: Optional[Path] = None) -> None:
        self.sample_every_frames = max(1, sample_every_frames)
        self.scene_classifier = SceneClassifier()
        self.item_skill_classifier = TsumItemSkillClassifier(Path("app/models/use_tsum"))
        self._last_sampled_frame = -1
        self.model_root = model_root or Path("app/models/main_model")
        self.active_model_version = "unknown"
        self.scene_model_loaded = False
        self.scene_class_count = 0
        self.scene_model_path: Path = self.model_root / "scene_model.json"
        self.reload_model()

    def reset(self) -> None:
        self._last_sampled_frame = -1

    def reload_model(self) -> None:
        active_file = self.model_root / "ACTIVE_VERSION"
        if active_file.exists():
            self.active_model_version = active_file.read_text(encoding="utf-8").strip() or "unknown"
        self.scene_model_path = self.model_root / self.active_model_version / "scene_model.json"
        self.scene_model_loaded = self.scene_classifier.load_model(self.scene_model_path)
        self.scene_class_count = self.scene_classifier.model.class_count()
        self.item_skill_classifier.reload()

    def process_frame(
        self,
        frame_seq: int,
        position_ms: int,
        selected_tsum: str = "auto",
        frame_image=None,
        use_tsum_dir: str = "",
        scene_feature: Optional[List[float]] = None,
    ) -> List[AnalysisResult]:
        """Process using actual decoded frame sequence from video callback."""
        frame_index = max(0, int(frame_seq))
        if frame_index == self._last_sampled_frame:
            return []

        if frame_index % self.sample_every_frames != 0:
            return []

        self._last_sampled_frame = frame_index
        if scene_feature is not None:
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
        scene_feature: Optional[List[float]] = None,
    ) -> List[AnalysisResult]:
        if fps <= 0:
            fps = 30.0

        frame_index = int((max(position_ms, 0) / 1000.0) * fps)
        if frame_index == self._last_sampled_frame:
            return []

        if frame_index % self.sample_every_frames != 0:
            return []

        self._last_sampled_frame = frame_index
        if scene_feature is not None:
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

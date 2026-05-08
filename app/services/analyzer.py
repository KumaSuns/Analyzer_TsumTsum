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


class TsumItemSkillClassifier:
    """Placeholder classifier for use_tsum branch."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.registry_path = root / "registry.json"
        self.tsum_ids: list[str] = []
        self.registry: dict[str, str] = {}
        self.reload()

    def reload(self) -> None:
        self.tsum_ids = sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
        ) if self.root.exists() else []
        self.registry = {}
        if self.registry_path.exists():
            try:
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.registry = {str(k): str(v) for k, v in data.items()}
            except Exception:
                self.registry = {}

    def predict(self, frame_index: int, selected_tsum: str = "auto") -> str:
        if selected_tsum != "auto":
            return self.registry.get(selected_tsum, selected_tsum)
        if not self.tsum_ids:
            return "unknown_tsum"
        tsum_id = self.tsum_ids[(frame_index // 15) % len(self.tsum_ids)]
        return self.registry.get(tsum_id, tsum_id)


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
        self.reload_model()

    def reset(self) -> None:
        self._last_sampled_frame = -1

    def reload_model(self) -> None:
        active_file = self.model_root / "ACTIVE_VERSION"
        if active_file.exists():
            self.active_model_version = active_file.read_text(encoding="utf-8").strip() or "unknown"
        model_json = self.model_root / self.active_model_version / "scene_model.json"
        self.scene_model_loaded = self.scene_classifier.load_model(model_json)
        self.scene_class_count = self.scene_classifier.model.class_count()
        self.item_skill_classifier.reload()

    def process_frame(
        self,
        frame_seq: int,
        position_ms: int,
        selected_tsum: str = "auto",
        frame_image=None,
    ) -> List[AnalysisResult]:
        """Process using actual decoded frame sequence from video callback."""
        frame_index = max(0, int(frame_seq))
        if frame_index == self._last_sampled_frame:
            return []

        if frame_index % self.sample_every_frames != 0:
            return []

        self._last_sampled_frame = frame_index
        scene_label = self.scene_classifier.predict(frame_image)
        item_skill_label = "-"
        if scene_label == "item":
            item_skill_label = self.item_skill_classifier.predict(frame_index, selected_tsum)

        return [
            AnalysisResult(
                frame_index=frame_index,
                timestamp_ms=max(position_ms, 0),
                scene_label=scene_label,
                item_skill_label=item_skill_label,
            )
        ]

    def process_position(self, position_ms: int, fps: float, selected_tsum: str = "auto", frame_image=None) -> List[AnalysisResult]:
        if fps <= 0:
            fps = 30.0

        frame_index = int((max(position_ms, 0) / 1000.0) * fps)
        if frame_index == self._last_sampled_frame:
            return []

        if frame_index % self.sample_every_frames != 0:
            return []

        self._last_sampled_frame = frame_index
        scene_label = self.scene_classifier.predict(frame_image)
        item_skill_label = "-"
        if scene_label == "item":
            item_skill_label = self.item_skill_classifier.predict(frame_index, selected_tsum)

        return [
            AnalysisResult(
                frame_index=frame_index,
                timestamp_ms=max(position_ms, 0),
                scene_label=scene_label,
                item_skill_label=item_skill_label,
            )
        ]

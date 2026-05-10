from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.services.scene_model import SceneCentroidModel

@dataclass
class DatasetSummary:
    train_total: int
    val_total: int
    per_class_train: dict[str, int]
    per_class_val: dict[str, int]


class SimpleTrainer:
    """Lightweight trainer scaffold for UI wiring.

    This intentionally does not depend on torch yet. It validates dataset
    layout and produces a model metadata file on save.
    """

    CLASSES = ["none", "item", "ready", "go", "fever", "timeup", "bonus", "result"]

    def __init__(self, images_root: Path, model_root: Path) -> None:
        self.images_root = images_root
        self.model_root = model_root
        self.is_running = False
        self.current_epoch = 0
        self.max_epochs = 5
        self.summary: DatasetSummary | None = None
        self.scene_model = SceneCentroidModel()
        self.val_accuracy = 0.0

    def summarize_dataset(self) -> DatasetSummary:
        train_counts: dict[str, int] = {}
        val_counts: dict[str, int] = {}
        for cls in self.CLASSES:
            train_dir = self.images_root / "train" / cls
            val_dir = self.images_root / "val" / cls
            train_counts[cls] = self._count_files(train_dir)
            val_counts[cls] = self._count_files(val_dir)

        summary = DatasetSummary(
            train_total=sum(train_counts.values()),
            val_total=sum(val_counts.values()),
            per_class_train=train_counts,
            per_class_val=val_counts,
        )
        self.summary = summary
        return summary

    def start(self, log: Callable[[str], None]) -> bool:
        if self.is_running:
            log("学習はすでに実行中です。")
            return False

        summary = self.summarize_dataset()
        if summary.train_total == 0:
            log("train データが0件のため学習開始できません。")
            return False

        self.is_running = True
        self.current_epoch = 1
        per_class = self.scene_model.fit_from_dataset(self.images_root)
        correct, total = self.scene_model.evaluate_val(self.images_root)
        self.val_accuracy = (correct / total) if total > 0 else 0.0
        log(f"学習開始: train={summary.train_total}, val={summary.val_total}")
        log(f"学習完了: class_counts={per_class}")
        log(f"検証精度: {correct}/{total} ({self.val_accuracy:.3f})")
        self.is_running = False
        return True

    def step(self, log: Callable[[str], None]) -> bool:
        if self.is_running:
            log("学習中...")
        return self.is_running

    def stop(self, log: Callable[[str], None]) -> None:
        if not self.is_running:
            log("学習は停止状態です。")
            return
        self.is_running = False
        log("学習を停止しました。")

    def save(self, log: Callable[[str], None], version: str = "version_1") -> Path:
        # Safety net: if training has not been run in this session,
        # build a scene model from current dataset before saving.
        if not self.scene_model.centroids:
            self.summarize_dataset()
            per_class = self.scene_model.fit_from_dataset(self.images_root)
            correct, total = self.scene_model.evaluate_val(self.images_root)
            self.val_accuracy = (correct / total) if total > 0 else 0.0
            log(f"保存前にモデル生成: class_counts={per_class}")
            log(f"保存前評価: {correct}/{total} ({self.val_accuracy:.3f})")

        self.model_root.mkdir(parents=True, exist_ok=True)
        out_dir = self.model_root / version
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "model_meta.txt"
        out_json = out_dir / "model_meta.json"
        out_scene_model = out_dir / "scene_model.json"
        summary = self.summary or self.summarize_dataset()
        out_file.write_text(
            "\n".join(
                [
                    "Analyzer_TsumTsum model metadata",
                    f"saved_at={datetime.now().isoformat()}",
                    f"train_total={summary.train_total}",
                    f"val_total={summary.val_total}",
                    f"classes={','.join(self.CLASSES)}",
                    f"last_epoch={self.current_epoch}",
                ]
            ),
            encoding="utf-8",
        )
        out_json.write_text(
            json.dumps(
                {
                    "saved_at": datetime.now().isoformat(),
                    "classes": self.CLASSES,
                    "train_total": summary.train_total,
                    "val_total": summary.val_total,
                    "per_class_train": summary.per_class_train,
                    "per_class_val": summary.per_class_val,
                    "last_epoch": self.current_epoch,
                    "val_accuracy": self.val_accuracy,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.scene_model.save(out_scene_model)
        active_marker = self.model_root / "ACTIVE_VERSION"
        active_marker.write_text(version, encoding="utf-8")
        log(f"アクティブ版を更新: {active_marker} -> {version}")
        log(f"モデル保存: {out_file}")
        log(f"モデル保存(JSON): {out_json}")
        log(f"シーンモデル保存: {out_scene_model}")
        return out_file

    @staticmethod
    def _count_files(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for p in path.iterdir() if p.is_file())

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from app.services.scene_cnn import SceneCnnClassifier, torch_available
from app.services.scene_model import (
    SceneCentroidModel,
    describe_training_image_load_failure,
    iter_scene_dataset_images,
)

@dataclass
class DatasetSummary:
    train_total: int
    val_total: int
    per_class_train: dict[str, int]
    per_class_val: dict[str, int]


class SimpleTrainer:
    """シーン分類の学習（CNN 優先。torch 未導入時のみ centroid）。"""

    CLASSES = ["none", "item", "ready", "go", "fever", "timeup", "bonus", "result"]

    def __init__(self, images_root: Path, model_root: Path) -> None:
        self.images_root = images_root
        self.model_root = model_root
        self.is_running = False
        self.current_epoch = 0
        self.max_epochs = 25
        self.summary: DatasetSummary | None = None
        self.scene_cnn = SceneCnnClassifier()
        self.scene_model = SceneCentroidModel()
        self.val_accuracy = 0.0
        self.last_fever_metrics: dict[str, int] = {}
        self.scene_backend = "none"

    def summarize_dataset(self) -> DatasetSummary:
        train_counts: dict[str, int] = {}
        val_counts: dict[str, int] = {}
        for cls in self.CLASSES:
            train_dir = self.images_root / "train" / cls
            val_dir = self.images_root / "val" / cls
            train_counts[cls] = self._count_class_images(train_dir)
            val_counts[cls] = self._count_class_images(val_dir)

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

        if torch_available():
            log("シーン分類: CNN（深層学習）で学習します…")
            log(f"train={summary.train_total}, val={summary.val_total}")
            try:
                stats = self.scene_cnn.train_from_dataset(
                    self.images_root,
                    epochs=self.max_epochs,
                    log=log,
                )
                self.val_accuracy = float(stats.get("val_accuracy", 0.0))
                self.scene_backend = "cnn"
                per_val = stats.get("per_class_val", {})
                fever = per_val.get("fever", (0, 0))
                self.last_fever_metrics = {
                    "fever_val_total": fever[1],
                    "fever_val_strong": fever[0],
                    "fever_val_any": fever[0],
                    "false_none_go_strong": 0,
                    "false_none_go_weak": 0,
                }
                log(f"CNN 検証精度: {self.val_accuracy:.3f}")
                if fever[1]:
                    log(f"fever(val): {fever[0]}/{fever[1]}")
                weak = [
                    f"{cls}:{ok}/{total}"
                    for cls, (ok, total) in sorted(per_val.items())
                    if total and ok / total < 0.7
                ]
                if weak:
                    log(f"val で弱いクラス: {', '.join(weak)}")
            except Exception as exc:
                log(f"CNN 学習失敗: {exc}")
                self.is_running = False
                return False
        else:
            log(
                "警告: PyTorch がありません。pip install torch torchvision のあと再学習してください。"
            )
            log("暫定: 従来の centroid で学習します（精度は限定的です）。")
            per_class = self.scene_model.fit_from_dataset(self.images_root)
            correct, total = self.scene_model.evaluate_val(self.images_root)
            self.val_accuracy = (correct / total) if total > 0 else 0.0
            self.last_fever_metrics = self.scene_model.fever_gate_metrics(self.images_root)
            self.scene_backend = "centroid"
            log(f"centroid 学習完了: class_counts={per_class}")
            log(f"検証精度: {correct}/{total} ({self.val_accuracy:.3f})")
            if sum(per_class.values()) == 0 and summary.train_total > 0:
                log(describe_training_image_load_failure(self.images_root))

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
        self.summarize_dataset()
        summary = self.summary or self.summarize_dataset()

        if self.scene_cnn.is_loaded():
            self.scene_backend = "cnn"
            log(
                f"シーン CNN を保存します（学習時 val精度={self.val_accuracy:.3f}）。"
                " 再評価はスキップします。"
            )
            fever_total = int(summary.per_class_val.get("fever", 0))
            fever_ok = int(self.last_fever_metrics.get("fever_val_strong", 0))
            if fever_total and not fever_ok:
                per_val = self.scene_cnn.evaluate_val(self.images_root)
                fever = per_val.get("fever", (0, 0))
                fever_ok, fever_total = fever[0], fever[1]
                total_ok = sum(v[0] for v in per_val.values())
                total_n = sum(v[1] for v in per_val.values())
                if total_n:
                    self.val_accuracy = total_ok / total_n
            self.last_fever_metrics = {
                "fever_val_total": fever_total,
                "fever_val_strong": fever_ok,
                "fever_val_any": fever_ok,
                "false_none_go_strong": 0,
                "false_none_go_weak": 0,
            }
        elif torch_available():
            raise RuntimeError(
                "シーン CNN が未学習です。先に「学習開始」を押し、完了してから「モデル保存」を実行してください。"
            )
        else:
            per_class = self.scene_model.fit_from_dataset(self.images_root)
            correct, total = self.scene_model.evaluate_val(self.images_root)
            self.val_accuracy = (correct / total) if total > 0 else 0.0
            self.last_fever_metrics = self.scene_model.fever_gate_metrics(self.images_root)
            self.scene_backend = "centroid"
            log(f"centroid 保存用: {correct}/{total} ({self.val_accuracy:.3f})")
            if not self.scene_model.centroids:
                raise RuntimeError("シーン centroid が0件です。")

        self.model_root.mkdir(parents=True, exist_ok=True)
        out_dir = self.model_root / version
        out_dir.mkdir(parents=True, exist_ok=True)
        prev_val_acc: float | None = None
        if (out_dir / "model_meta.json").exists():
            try:
                prev_meta = json.loads((out_dir / "model_meta.json").read_text(encoding="utf-8"))
                prev_val_acc = float(prev_meta.get("val_accuracy", 0.0))
            except Exception:
                prev_val_acc = None
        if prev_val_acc is not None and self.val_accuracy + 0.005 < prev_val_acc:
            log(
                f"警告: 検証精度 {self.val_accuracy:.3f} は前回保存 ({prev_val_acc:.3f}) より低いです。"
                " 解析中の修正画像が増えすぎていないか、val 画像を見直してください。"
            )
        out_file = out_dir / "model_meta.txt"
        out_json = out_dir / "model_meta.json"
        out_scene_model = out_dir / "scene_model.json"
        out_cnn = out_dir / "scene_cnn.pt"

        meta_payload = {
            "saved_at": datetime.now().isoformat(),
            "classes": self.CLASSES,
            "train_total": summary.train_total,
            "val_total": summary.val_total,
            "per_class_train": summary.per_class_train,
            "per_class_val": summary.per_class_val,
            "last_epoch": self.current_epoch,
            "val_accuracy": self.val_accuracy,
            "scene_backend": self.scene_backend,
        }
        if self.scene_backend == "cnn" and self.scene_cnn.is_loaded():
            per_val = self.scene_cnn.evaluate_val(self.images_root)
            meta_payload["per_class_val_accuracy"] = {
                cls: (ok / total if total else 0.0)
                for cls, (ok, total) in per_val.items()
            }
        if self.scene_backend == "centroid":
            meta_payload["fever_calib"] = self.scene_model.fever_calib
            meta_payload.update(
                {
                    "fever_val_strong": self.last_fever_metrics.get("fever_val_strong", 0),
                    "fever_val_any": self.last_fever_metrics.get("fever_val_any", 0),
                    "fever_val_total": self.last_fever_metrics.get("fever_val_total", 0),
                    "fever_false_none_go_strong": self.last_fever_metrics.get(
                        "false_none_go_strong", 0
                    ),
                    "fever_false_none_go_weak": self.last_fever_metrics.get(
                        "false_none_go_weak", 0
                    ),
                }
            )
        else:
            meta_payload["fever_val_strong"] = self.last_fever_metrics.get("fever_val_strong", 0)
            meta_payload["fever_val_any"] = self.last_fever_metrics.get("fever_val_any", 0)
            meta_payload["fever_val_total"] = self.last_fever_metrics.get("fever_val_total", 0)

        out_file.write_text(
            "\n".join(
                [
                    "Analyzer_TsumTsum model metadata",
                    f"saved_at={datetime.now().isoformat()}",
                    f"scene_backend={self.scene_backend}",
                    f"train_total={summary.train_total}",
                    f"val_total={summary.val_total}",
                    f"classes={','.join(self.CLASSES)}",
                    f"val_accuracy={self.val_accuracy:.4f}",
                ]
            ),
            encoding="utf-8",
        )
        out_json.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if self.scene_cnn.is_loaded():
            log("scene_cnn.pt を書き込み中…")
            self.scene_cnn.save(out_cnn)
            log(f"CNN モデル保存: {out_cnn}")
        if self.scene_backend == "centroid":
            self.scene_model.save(out_scene_model)
            log(f"centroid 保存: {out_scene_model}")

        active_marker = self.model_root / "ACTIVE_VERSION"
        active_marker.write_text(version, encoding="utf-8")
        log(f"アクティブ版を更新: {active_marker} -> {version}")
        log(f"モデル保存: {out_json}")
        return out_file

    @staticmethod
    def _count_class_images(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for _ in iter_scene_dataset_images(path))

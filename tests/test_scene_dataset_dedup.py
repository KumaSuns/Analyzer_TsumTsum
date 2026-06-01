"""scene_dataset_dedup の単体テスト。"""
from pathlib import Path

from app.services.scene_dataset_dedup import deduplicate_scene_dataset


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def test_dedup_same_class_keeps_train(tmp_path: Path) -> None:
    root = tmp_path / "images"
    blob = b"same-image-bytes"
    _write(root / "train" / "none" / "a.png", blob)
    _write(root / "val" / "none" / "b.png", blob)
    report = deduplicate_scene_dataset(root, dry_run=False)
    assert report.files_removed == 1
    assert (root / "train" / "none" / "a.png").is_file()
    assert not (root / "val" / "none" / "b.png").exists()


def test_dedup_prefers_matching_filename(tmp_path: Path) -> None:
    root = tmp_path / "images"
    blob = b"cross-class-dup"
    _write(root / "train" / "none" / "scene_none_20260101_f1.png", blob)
    _write(root / "train" / "fever" / "scene_fever_20260101_f1.png", blob)
    report = deduplicate_scene_dataset(root, dry_run=False)
    assert report.files_removed == 1
    assert report.cross_class_groups == 1
    assert (root / "train" / "none" / "scene_none_20260101_f1.png").is_file()
    assert not (root / "train" / "fever" / "scene_fever_20260101_f1.png").exists()


def test_dry_run_does_not_delete(tmp_path: Path) -> None:
    root = tmp_path / "images"
    blob = b"dup"
    _write(root / "train" / "go" / "x.png", blob)
    _write(root / "train" / "go" / "y.png", blob)
    report = deduplicate_scene_dataset(root, dry_run=True)
    assert report.files_removed == 0
    assert len(report.removed_paths) == 1
    assert (root / "train" / "go" / "x.png").is_file()
    assert (root / "train" / "go" / "y.png").is_file()

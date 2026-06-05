"""scene_dataset_dedup の単体テスト。"""
from pathlib import Path

from PySide6.QtGui import QImage

from app.services.image_save import save_training_png
from app.services.scene_dataset_dedup import (
    deduplicate_scene_dataset,
    find_duplicate_scene_image,
    invalidate_scene_image_hash_index,
    register_saved_scene_image,
)


def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _rgb_image(w: int, h: int, rgb: tuple[int, int, int]) -> QImage:
    img = QImage(w, h, QImage.Format.Format_RGB888)
    img.fill(rgb[0] << 16 | rgb[1] << 8 | rgb[2])
    return img


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


def test_find_duplicate_scene_image_detects_existing(tmp_path: Path) -> None:
    root = tmp_path / "images"
    img = _rgb_image(64, 48, (10, 20, 30))
    out_path = root / "train" / "none" / "scene_none_20260101_f1.png"
    assert save_training_png(img, out_path)
    invalidate_scene_image_hash_index()
    dup = find_duplicate_scene_image(root, img)
    assert dup is not None
    assert dup.resolve() == out_path.resolve()


def test_find_duplicate_scene_image_after_register(tmp_path: Path) -> None:
    root = tmp_path / "images"
    img = _rgb_image(32, 32, (1, 2, 3))
    out_path = root / "train" / "ready" / "scene_ready_20260101_f1.png"
    assert save_training_png(img, out_path)
    invalidate_scene_image_hash_index()
    register_saved_scene_image(out_path)
    dup = find_duplicate_scene_image(root, img)
    assert dup is not None
    assert dup.name == out_path.name

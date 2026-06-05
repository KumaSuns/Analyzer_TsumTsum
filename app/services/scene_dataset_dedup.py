"""シーン学習データ（train/val）の完全同一画像を検出・削除する。"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from app.services.scene_model import SCENE_CLASSES, is_scene_dataset_image, iter_scene_dataset_images

LogFn = Optional[Callable[[str], None]]
_SPLITS = ("train", "val")


@dataclass
class DedupReport:
    scanned: int = 0
    unique_hashes: int = 0
    duplicate_groups: int = 0
    files_removed: int = 0
    cross_class_groups: int = 0
    train_val_groups: int = 0
    removed_paths: List[str] = field(default_factory=list)
    kept_paths: List[str] = field(default_factory=list)

    def summary_lines(self) -> List[str]:
        lines = [
            f"走査: {self.scanned}枚 / ユニーク: {self.unique_hashes}",
            f"重複グループ: {self.duplicate_groups}（クラス不一致: {self.cross_class_groups}, "
            f"train+val: {self.train_val_groups}）",
        ]
        if self.files_removed:
            lines.append(f"削除: {self.files_removed}枚")
        else:
            lines.append("削除: 0枚（重複なし、または dry_run）")
        return lines


def _split_and_class(path: Path) -> Tuple[str, str]:
    parts = path.parts
    for i, part in enumerate(parts):
        if part in _SPLITS and i + 1 < len(parts):
            return part, parts[i + 1]
    return "", ""


def _filename_matches_class(path: Path, cls: str) -> bool:
    if not cls:
        return False
    stem = path.stem.lower()
    return stem.startswith(f"scene_{cls.lower()}_") or stem == f"scene_{cls.lower()}"


def _dedup_class_priority(cls: str) -> int:
    """重複解消時のクラス優先度（小さいほど残す）。none は誤保存が多いので最下位。"""
    if cls not in SCENE_CLASSES:
        return 99
    if cls == "none":
        return len(SCENE_CLASSES)
    return SCENE_CLASSES.index(cls)


def _keep_sort_key(path: Path) -> Tuple:
    """昇順で先頭を残す（残すファイルほどキーが小さい）。"""
    split, cls = _split_and_class(path)
    cls_prio = _dedup_class_priority(cls)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    name_match = 0 if _filename_matches_class(path, cls) else 1
    split_rank = 0 if split == "train" else 1
    return (name_match, split_rank, cls_prio, mtime, len(path.name), path.as_posix())


def _collect_images(images_root: Path) -> List[Path]:
    out: List[Path] = []
    for split in _SPLITS:
        base = images_root / split
        if not base.is_dir():
            continue
        for cls in SCENE_CLASSES:
            cls_dir = base / cls
            for p in iter_scene_dataset_images(cls_dir):
                out.append(p)
    return out


def deduplicate_scene_dataset(
    images_root: Path,
    *,
    dry_run: bool = True,
    log: LogFn = None,
) -> DedupReport:
    """
    train/val 全体でバイト完全一致の重複を1枚にまとめる。

    残す優先順位:
    1. ファイル名が scene_<クラス>_… とフォルダのクラスが一致
    2. train（val より優先。val リーク防止）
    3. none 以外の SCENE_CLASSES 順（none は誤保存が多いので最下位）
    4. 保存が古い方（先に取ったフレーム）
    """
    report = DedupReport()
    images: List[Path] = _collect_images(images_root)
    report.scanned = len(images)

    by_hash: dict[str, List[Path]] = {}
    for path in images:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            if log:
                log(f"読込スキップ: {path} ({exc})")
            continue
        by_hash.setdefault(digest, []).append(path)

    report.unique_hashes = len(by_hash)

    for digest, group in by_hash.items():
        if len(group) < 2:
            continue
        report.duplicate_groups += 1
        classes = {_split_and_class(p)[1] for p in group}
        if len(classes) > 1:
            report.cross_class_groups += 1
        splits = {_split_and_class(p)[0] for p in group}
        if len(splits) > 1:
            report.train_val_groups += 1

        group_sorted = sorted(group, key=_keep_sort_key)
        keep = group_sorted[0]
        remove = group_sorted[1:]
        report.kept_paths.append(keep.as_posix())

        if log:
            log(f"重複 {len(group)}枚 hash={digest[:8]}… 残す: {keep.as_posix()}")
            for p in remove:
                sp, cl = _split_and_class(p)
                log(f"  削除{'(予定)' if dry_run else ''}: [{sp}/{cl}] {p.name}")

        for p in remove:
            report.removed_paths.append(p.as_posix())
            if dry_run:
                continue
            try:
                p.unlink()
                report.files_removed += 1
            except OSError as exc:
                if log:
                    log(f"  削除失敗: {p} ({exc})")

    invalidate_scene_image_hash_index()
    return report


SAVE_SKIP_DUPLICATE_PREFIX = "skip_duplicate:"


def file_content_hash(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def build_scene_image_hash_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in _collect_images(images_root):
        digest = file_content_hash(path)
        if digest and digest not in index:
            index[digest] = path
    return index


class _SceneImageHashIndex:
    def __init__(self) -> None:
        self._root: Path | None = None
        self._index: dict[str, Path] = {}

    def ensure(self, images_root: Path) -> None:
        root = images_root.resolve()
        if self._root != root:
            self._root = root
            self._index = build_scene_image_hash_index(root)

    def find(self, images_root: Path, digest: str) -> Path | None:
        self.ensure(images_root)
        return self._index.get(digest)

    def register(self, path: Path) -> None:
        digest = file_content_hash(path)
        if digest:
            self._index[digest] = path

    def invalidate(self) -> None:
        self._root = None
        self._index.clear()


_scene_image_hash_index = _SceneImageHashIndex()


def invalidate_scene_image_hash_index() -> None:
    _scene_image_hash_index.invalidate()


def find_duplicate_scene_image(images_root: Path, image) -> Path | None:
    """保存前のフレームが train/val に既にある完全同一 PNG ならそのパスを返す。"""
    from app.services.image_save import training_image_content_hash

    digest = training_image_content_hash(image)
    if not digest:
        return None
    return _scene_image_hash_index.find(images_root, digest)


def register_saved_scene_image(path: Path) -> None:
    _scene_image_hash_index.register(path)

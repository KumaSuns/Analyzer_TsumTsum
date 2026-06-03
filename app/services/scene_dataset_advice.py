"""シーン学習データの枚数・偏りから、追加すべき画像を案内する。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from app.services.scene_model import SCENE_CLASSES
from app.services.trainer import DatasetSummary

# 目安（実運用と val 評価の経験値）
MIN_TRAIN_WARN = 25
MIN_VAL_WARN = 8
RECOMMENDED_TRAIN = 50
RECOMMENDED_VAL = 15
# none が ready+go の何倍を超えると「偏り」とするか
NONE_READY_GO_RATIO_WARN = 4.0

CLASS_LABELS: Dict[str, str] = {
    "none": "none（通常プレイ）",
    "item": "item（アイテム選択）",
    "ready": "ready（レディ）",
    "go": "go（スタート）",
    "fever": "fever（フィーバー）",
    "timeup": "timeup（タイムアップ）",
    "bonus": "bonus（ボーナス）",
    "coin": "coin（コイン獲得）",
    "result": "result（リザルト）",
}

CLASS_HINTS: Dict[str, str] = {
    "none": "誤検知修正用に増やしすぎない。プレイ中の通常画面を少数で",
    "item": "アイテム選択画面の停止画",
    "ready": "「READY」表示の停止画",
    "go": "「GO」表示の停止画",
    "fever": "フィーバー中の画面（誤検知が多いクラス）",
    "timeup": "タイムアップ画面",
    "bonus": "ボーナス突入〜終了付近",
    "coin": "コイン獲得画面（枚数が少ないと誤判定しやすい）",
    "result": "リザルト画面",
}


@dataclass
class ClassShortage:
    cls: str
    label: str
    train: int
    val: int
    need_train: int
    need_val: int
    severity: str  # "critical" | "warn" | "ok"
    notes: List[str] = field(default_factory=list)


@dataclass
class DatasetAdviceReport:
    lines: List[str]
    shortages: List[ClassShortage]
    has_critical: bool
    ready_for_training: bool


def _label(cls: str) -> str:
    return CLASS_LABELS.get(cls, cls)


def _need_more(current: int, recommended: int) -> int:
    """目安枚数まであと何枚か（表示用）。"""
    if current >= recommended:
        return 0
    return recommended - current


def analyze_scene_dataset(
    summary: DatasetSummary,
    *,
    model_meta_path: Optional[Path] = None,
) -> DatasetAdviceReport:
    shortages: List[ClassShortage] = []
    val_acc_by_class: Dict[str, float] = {}
    if model_meta_path and model_meta_path.exists():
        try:
            meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
            raw = meta.get("per_class_val_accuracy")
            if isinstance(raw, dict):
                val_acc_by_class = {str(k): float(v) for k, v in raw.items()}
        except Exception:
            pass

    for cls in SCENE_CLASSES:
        train_n = int(summary.per_class_train.get(cls, 0))
        val_n = int(summary.per_class_val.get(cls, 0))
        need_train = _need_more(train_n, RECOMMENDED_TRAIN)
        need_val = _need_more(val_n, RECOMMENDED_VAL)
        notes: List[str] = []
        severity = "ok"

        if train_n == 0:
            severity = "critical"
            notes.append("train が0枚 → このクラスは学習に使われません")
        elif train_n < MIN_TRAIN_WARN:
            severity = "warn"
            notes.append(
                f"train が少なめ（最低ライン {MIN_TRAIN_WARN} 枚未満、目安 {RECOMMENDED_TRAIN} 枚）"
            )
        elif train_n < RECOMMENDED_TRAIN:
            severity = "warn"
        if val_n == 0 and train_n > 0:
            if severity != "critical":
                severity = "warn"
            notes.append("val が0枚 → 精度の確認ができません")
        elif val_n < MIN_VAL_WARN and train_n > 0:
            if severity == "ok":
                severity = "warn"
            notes.append(
                f"val が少なめ（最低ライン {MIN_VAL_WARN} 枚未満、目安 {RECOMMENDED_VAL} 枚）"
            )
        elif val_n < RECOMMENDED_VAL and train_n > 0:
            if severity == "ok":
                severity = "warn"

        acc = val_acc_by_class.get(cls)
        if acc is not None and acc < 0.75 and val_n >= 5:
            if severity == "ok":
                severity = "warn"
            notes.append(f"保存モデルの val 精度が低い ({acc * 100:.0f}%) → 追加推奨")

        if need_train or need_val or severity != "ok":
            shortages.append(
                ClassShortage(
                    cls=cls,
                    label=_label(cls),
                    train=train_n,
                    val=val_n,
                    need_train=need_train,
                    need_val=need_val,
                    severity=severity,
                    notes=notes,
                )
            )

    ready_n = summary.per_class_train.get("ready", 0)
    go_n = summary.per_class_train.get("go", 0)
    none_n = summary.per_class_train.get("none", 0)
    pregame = max(ready_n + go_n, 1)
    if none_n > pregame * NONE_READY_GO_RATIO_WARN and none_n >= 80:
        shortages.append(
            ClassShortage(
                cls="_balance",
                label="データの偏り",
                train=none_n,
                val=0,
                need_train=0,
                need_val=0,
                severity="warn",
                notes=[
                    f"none({none_n}) が ready+go({ready_n + go_n}) の"
                    f" {none_n / pregame:.1f} 倍です。"
                    " none の追加を控え、実プレイ画面の fever/timeup 等を増やすと誤検知が減りやすいです。",
                ],
            )
        )

    lines: List[str] = []
    lines.append(
        f"合計 train={summary.train_total} / val={summary.val_total}"
        f"（目安: 各クラス train≥{RECOMMENDED_TRAIN}, val≥{RECOMMENDED_VAL}）"
    )
    lines.append(
        "※ 下に出るのは「目安未満」または val 精度が低いクラスだけです。"
        " 出ていないクラスは枚数目安を満たしています。"
    )
    if summary.train_total == 0:
        lines.append("【要対応】train 画像がありません。動画ツール等でシーン保存してください。")
        return DatasetAdviceReport(
            lines=lines,
            shortages=shortages,
            has_critical=True,
            ready_for_training=False,
        )

    critical = [s for s in shortages if s.severity == "critical"]
    warn = [s for s in shortages if s.severity == "warn" and s.cls != "_balance"]
    balance = [s for s in shortages if s.cls == "_balance"]

    if not shortages:
        lines.append("【良好】全クラスが目安枚数を満たしています。")
    else:
        if critical:
            lines.append("【要追加】学習に必要なクラス（train=0）:")
            for s in critical:
                lines.extend(_format_shortage_lines(s))
        if warn:
            lines.append("【追加推奨】枚数または val 精度が不足:")
            for s in warn:
                lines.extend(_format_shortage_lines(s))
        if balance:
            for s in balance:
                lines.extend(_format_shortage_lines(s))

    lines.append("")
    lines.append("保存先: app/assets/images/train/<クラス>/ と val/<クラス>/")
    lines.append("動画ツールのシーン保存、または解析中の修正保存で追加できます。")

    has_critical = bool(critical) or summary.train_total == 0
    ready = summary.train_total > 0 and not critical
    return DatasetAdviceReport(
        lines=lines,
        shortages=shortages,
        has_critical=has_critical,
        ready_for_training=ready,
    )


def _format_shortage_lines(s: ClassShortage) -> List[str]:
    out = [f" ・{s.label}"]
    out.append(f"   現在 train={s.train} val={s.val}")
    parts: List[str] = []
    if s.need_train:
        parts.append(f"train +{s.need_train} 枚程度")
    if s.need_val:
        parts.append(f"val +{s.need_val} 枚程度")
    if parts:
        out.append(f"   目安まで: {', '.join(parts)}")
    hint = CLASS_HINTS.get(s.cls)
    if hint:
        out.append(f"   ヒント: {hint}")
    for note in s.notes:
        out.append(f"   ※ {note}")
    return out


def format_advice_report_text(report: DatasetAdviceReport) -> str:
    return "\n".join(report.lines)


def summarize_skill_activation_images(skills_root: Path) -> Tuple[int, List[str]]:
    """skills/<dir>/activation/ の枚数。不足ツムの案内用。"""
    if not skills_root.exists():
        return 0, []
    low: List[str] = []
    total = 0
    for tsum_dir in sorted(p for p in skills_root.iterdir() if p.is_dir()):
        act = tsum_dir / "activation"
        if not act.is_dir():
            low.append(f"{tsum_dir.name}: activation フォルダなし")
            continue
        n = sum(1 for _ in act.iterdir() if _.is_file() and not _.name.startswith("."))
        total += n
        if n < 10:
            low.append(f"{tsum_dir.name}: {n} 枚（推奨 10 枚以上）")
    return total, low

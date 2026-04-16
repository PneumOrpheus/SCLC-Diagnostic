import argparse
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "output"
DEFAULT_LOG = "swin_unetr_2026-04-15_14_logs.txt"

FLOAT_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

PHASE_EPOCH_RE = re.compile(r"\[(dapt|finetune)\]\s+Epoch\s+(\d+)/(\d+)\s+\|", re.IGNORECASE)
TRAIN_STEP_RE = re.compile(rf"\bEpoch\s+(\d+)\s+\[(\d+)/(\d+)\]\s+Loss:\s+({FLOAT_RE})")
TRAIN_SUMMARY_RE = re.compile(
    rf"Train Epoch\s+(\d+)\s+Summary\s+=>\s+Loss:\s*({FLOAT_RE}),\s*Accuracy:\s*({FLOAT_RE})(?:,\s*MacroF1:\s*({FLOAT_RE}))?"
)
PHASE_SUMMARY_RE = re.compile(
    rf"\[(dapt|finetune)\]\s+Epoch\s+(\d+)\s+Summary\s+=>\s+TrainLoss:\s*({FLOAT_RE})(?:,\s*TrainMacroF1:\s*({FLOAT_RE}),\s*ValMacroF1:\s*({FLOAT_RE})/({FLOAT_RE})\s*\(cur/roll\d+\))?",
    re.IGNORECASE,
)
VALID_RE = re.compile(
    rf"Validation Complete\s+=>\s+Loss:\s*({FLOAT_RE}),\s*Accuracy:\s*({FLOAT_RE})(?:,\s*BalancedAcc:\s*({FLOAT_RE}),\s*MacroPrecision:\s*({FLOAT_RE}),\s*MacroRecall:\s*({FLOAT_RE}),\s*MacroF1:\s*({FLOAT_RE}))?"
)


def parse_log(log_path: Path):
    records = {}
    phase_order = []

    def ensure_record(phase: str, epoch: int):
        if phase not in phase_order:
            phase_order.append(phase)
        key = (phase, epoch)
        if key not in records:
            records[key] = {"phase": phase, "epoch": epoch}
        return records[key]

    current_phase = None
    current_epoch = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()

            m_phase = PHASE_EPOCH_RE.search(line)
            if m_phase:
                current_phase = m_phase.group(1).lower()
                current_epoch = int(m_phase.group(2))
                ensure_record(current_phase, current_epoch)
                continue

            m_train_step = TRAIN_STEP_RE.search(line)
            if m_train_step:
                epoch = int(m_train_step.group(1))
                step = int(m_train_step.group(2))
                total = int(m_train_step.group(3))
                loss = float(m_train_step.group(4))
                phase = current_phase or "unknown"
                rec = ensure_record(phase, epoch)
                if step == total:
                    rec["train_loss"] = loss
                continue

            m_train_summary = TRAIN_SUMMARY_RE.search(line)
            if m_train_summary:
                epoch = int(m_train_summary.group(1))
                train_loss = float(m_train_summary.group(2))
                train_acc = float(m_train_summary.group(3))
                train_f1 = float(m_train_summary.group(4)) if m_train_summary.group(4) is not None else math.nan
                phase = current_phase or "unknown"
                rec = ensure_record(phase, epoch)
                rec["train_loss"] = train_loss
                rec["train_acc"] = train_acc
                if not math.isnan(train_f1):
                    rec["train_f1"] = train_f1
                continue

            m_phase_summary = PHASE_SUMMARY_RE.search(line)
            if m_phase_summary:
                phase = m_phase_summary.group(1).lower()
                epoch = int(m_phase_summary.group(2))
                train_loss = float(m_phase_summary.group(3))
                rec = ensure_record(phase, epoch)
                rec["train_loss"] = train_loss
                if m_phase_summary.group(4) is not None:
                    rec["train_f1"] = float(m_phase_summary.group(4))
                if m_phase_summary.group(5) is not None:
                    rec["val_f1"] = float(m_phase_summary.group(5))
                continue

            m_val = VALID_RE.search(line)
            if m_val and current_phase is not None and current_epoch is not None:
                val_loss = float(m_val.group(1))
                val_acc = float(m_val.group(2))
                rec = ensure_record(current_phase, current_epoch)
                rec["val_loss"] = val_loss
                rec["val_acc"] = val_acc
                if m_val.group(5) is not None:
                    rec["val_recall"] = float(m_val.group(5))
                if m_val.group(6) is not None:
                    rec["val_f1"] = float(m_val.group(6))

    ordered = []
    for phase in phase_order:
        epochs = sorted(epoch for (p, epoch) in records.keys() if p == phase)
        for epoch in epochs:
            ordered.append(records[(phase, epoch)])

    return ordered, phase_order


def plot_metrics(records, output_file: Path, log_path: Path, show: bool = False):
    if not records:
        raise RuntimeError("No epoch records were parsed from the log.")

    x = list(range(1, len(records) + 1))
    train_loss = [r.get("train_loss", math.nan) for r in records]
    val_loss = [r.get("val_loss", math.nan) for r in records]
    train_acc = [r.get("train_acc", math.nan) for r in records]
    val_acc = [r.get("val_acc", math.nan) for r in records]
    train_f1 = [r.get("train_f1", math.nan) for r in records]
    val_f1 = [r.get("val_f1", math.nan) for r in records]
    train_recall = [r.get("train_recall", math.nan) for r in records]
    val_recall = [r.get("val_recall", math.nan) for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    ax_loss = axes[0, 0]
    ax_acc = axes[0, 1]
    ax_f1 = axes[1, 0]
    ax_recall = axes[1, 1]

    ax_loss.plot(x, train_loss, label="Train Loss", linewidth=1.8, marker="o", markersize=3)
    ax_loss.plot(x, val_loss, label="Val Loss", linewidth=1.8, marker="o", markersize=3)
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss per Epoch")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="best")

    if any(not math.isnan(v) for v in train_acc):
        ax_acc.plot(x, train_acc, label="Train Accuracy", linewidth=1.8, marker="o", markersize=3)
    ax_acc.plot(x, val_acc, label="Val Accuracy", linewidth=1.8, marker="o", markersize=3)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy per Epoch")
    ax_acc.set_ylim(0.0, 1.0)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="best")

    if any(not math.isnan(v) for v in train_f1):
        ax_f1.plot(x, train_f1, label="Train Macro F1", linewidth=1.8, marker="o", markersize=3)
    if any(not math.isnan(v) for v in val_f1):
        ax_f1.plot(x, val_f1, label="Val Macro F1", linewidth=1.8, marker="o", markersize=3)
    ax_f1.set_ylabel("F1")
    ax_f1.set_xlabel("Global Epoch Index")
    ax_f1.set_title("Macro F1 per Epoch")
    ax_f1.set_ylim(0.0, 1.0)
    ax_f1.grid(alpha=0.3)
    ax_f1.legend(loc="best")

    if any(not math.isnan(v) for v in train_recall):
        ax_recall.plot(x, train_recall, label="Train Macro Recall", linewidth=1.8, marker="o", markersize=3)
    if any(not math.isnan(v) for v in val_recall):
        ax_recall.plot(x, val_recall, label="Val Macro Recall", linewidth=1.8, marker="o", markersize=3)
    ax_recall.set_ylabel("Recall")
    ax_recall.set_xlabel("Global Epoch Index")
    ax_recall.set_title("Macro Recall per Epoch")
    ax_recall.set_ylim(0.0, 1.0)
    ax_recall.grid(alpha=0.3)
    ax_recall.legend(loc="best")

    phase_to_indices = defaultdict(list)
    for i, rec in enumerate(records, start=1):
        phase_to_indices[rec["phase"]].append(i)

    for i in range(1, len(records)):
        if records[i]["phase"] != records[i - 1]["phase"]:
            boundary = i + 0.5
            ax_loss.axvline(boundary, color="gray", linestyle="--", alpha=0.6)
            ax_acc.axvline(boundary, color="gray", linestyle="--", alpha=0.6)
            ax_f1.axvline(boundary, color="gray", linestyle="--", alpha=0.6)
            ax_recall.axvline(boundary, color="gray", linestyle="--", alpha=0.6)

    for phase, idxs in phase_to_indices.items():
        mid = (idxs[0] + idxs[-1]) / 2.0
        ax_loss.text(
            mid,
            1.02,
            phase.upper(),
            transform=ax_loss.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
        )

    fig.suptitle(f"Training Curves from {log_path.name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=150)
    print(f"Saved plot: {output_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot loss and accuracy curves from training logs.")
    parser.add_argument(
        "--log-file",
        type=str,
        default=str(DEFAULT_LOG),
        help="Path to log file (default: latest full SwinUNETR log).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Output PNG file path. Default: same folder as log with *_loss_accuracy.png suffix.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    args = parser.parse_args()

    log_path = Path(DEFAULT_ROOT / args.log_file)
    if not log_path.is_file():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = log_path.with_name(f"{log_path.stem}_loss_accuracy.png")

    records, phase_order = parse_log(log_path)
    if not records:
        raise RuntimeError("No plottable records found in log.")

    val_count = sum(1 for r in records if "val_acc" in r)
    train_loss_count = sum(1 for r in records if "train_loss" in r)
    train_acc_count = sum(1 for r in records if "train_acc" in r)
    train_f1_count = sum(1 for r in records if "train_f1" in r)
    val_f1_count = sum(1 for r in records if "val_f1" in r)
    val_recall_count = sum(1 for r in records if "val_recall" in r)

    print(f"Parsed phases: {phase_order}")
    print(f"Epoch records: {len(records)}")
    print(f"Train loss points: {train_loss_count}")
    print(f"Train accuracy points: {train_acc_count}")
    print(f"Val accuracy points: {val_count}")
    print(f"Train macro-F1 points: {train_f1_count}")
    print(f"Val macro-F1 points: {val_f1_count}")
    print(f"Val macro-recall points: {val_recall_count}")

    plot_metrics(records, output_file=output_file, log_path=log_path, show=args.show)


if __name__ == "__main__":
    main()

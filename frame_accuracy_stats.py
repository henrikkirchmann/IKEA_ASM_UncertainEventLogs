#!/usr/bin/env python3
"""
Compute frame-wise prediction accuracy (pred_label_name vs gt_label_name) for IKEA ASM exports.

This scans:
  uncertain_event_data/ikea_asm/split=test/model=*/frames.csv

and writes a single summary CSV with:
  - accuracy including NA
  - accuracy excluding NA frames (gt_label_name != "NA")
  - helpful counts (frames, NA frames, pred NA frames, etc.)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


NA_LABEL = "NA"


@dataclass(frozen=True)
class AccuracyCounts:
    n_frames: int
    n_correct: int
    n_gt_na: int
    n_pred_na: int
    n_frames_excl_gt_na: int
    n_correct_excl_gt_na: int

    @property
    def acc_including_na(self) -> float:
        return (self.n_correct / self.n_frames) if self.n_frames else 0.0

    @property
    def acc_excluding_gt_na(self) -> float:
        return (self.n_correct_excl_gt_na / self.n_frames_excl_gt_na) if self.n_frames_excl_gt_na else 0.0


def _iter_frames_csv_paths(split_dir: Path) -> Iterable[Path]:
    yield from sorted(split_dir.glob("model=*/frames.csv"))


def _compute_counts(frames_csv: Path) -> AccuracyCounts:
    n_frames = 0
    n_correct = 0
    n_gt_na = 0
    n_pred_na = 0
    n_frames_excl_gt_na = 0
    n_correct_excl_gt_na = 0

    with frames_csv.open(newline="") as f:
        r = csv.DictReader(f)
        required = {"gt_label_name", "pred_label_name"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{frames_csv}: missing columns: {sorted(missing)}; got: {r.fieldnames}")

        for row in r:
            gt = (row.get("gt_label_name") or "").strip()
            pred = (row.get("pred_label_name") or "").strip()

            n_frames += 1
            if gt == pred:
                n_correct += 1

            if gt == NA_LABEL:
                n_gt_na += 1
            else:
                n_frames_excl_gt_na += 1
                if gt == pred:
                    n_correct_excl_gt_na += 1

            if pred == NA_LABEL:
                n_pred_na += 1

    return AccuracyCounts(
        n_frames=n_frames,
        n_correct=n_correct,
        n_gt_na=n_gt_na,
        n_pred_na=n_pred_na,
        n_frames_excl_gt_na=n_frames_excl_gt_na,
        n_correct_excl_gt_na=n_correct_excl_gt_na,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split_dir",
        type=Path,
        default=Path("uncertain_event_data/ikea_asm/split=test"),
        help="Split directory containing model=*/frames.csv (default: uncertain_event_data/ikea_asm/split=test).",
    )
    ap.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: <split_dir>/frame_accuracy__pred_vs_gt.csv",
    )
    args = ap.parse_args()

    split_dir: Path = args.split_dir
    if not split_dir.exists():
        raise SystemExit(f"split_dir does not exist: {split_dir}")

    out_csv: Path = args.out_csv or (split_dir / "frame_accuracy__pred_vs_gt.csv")

    frames_paths = list(_iter_frames_csv_paths(split_dir))
    if not frames_paths:
        raise SystemExit(f"No frames.csv found under: {split_dir} (expected model=*/frames.csv)")

    rows = []
    for frames_csv in frames_paths:
        model_dir = frames_csv.parent
        model_id = model_dir.name.removeprefix("model=")
        counts = _compute_counts(frames_csv)
        rows.append(
            {
                "model_id": model_id,
                "frames_csv": str(frames_csv),
                "n_frames": counts.n_frames,
                "n_correct": counts.n_correct,
                "acc_including_na": f"{counts.acc_including_na:.6f}",
                "n_gt_na": counts.n_gt_na,
                "n_pred_na": counts.n_pred_na,
                "n_frames_excl_gt_na": counts.n_frames_excl_gt_na,
                "n_correct_excl_gt_na": counts.n_correct_excl_gt_na,
                "acc_excluding_gt_na": f"{counts.acc_excluding_gt_na:.6f}",
            }
        )

    # Also compute a micro-average across *all* models by summing counts (useful sanity check).
    total = AccuracyCounts(
        n_frames=sum(int(r["n_frames"]) for r in rows),
        n_correct=sum(int(r["n_correct"]) for r in rows),
        n_gt_na=sum(int(r["n_gt_na"]) for r in rows),
        n_pred_na=sum(int(r["n_pred_na"]) for r in rows),
        n_frames_excl_gt_na=sum(int(r["n_frames_excl_gt_na"]) for r in rows),
        n_correct_excl_gt_na=sum(int(r["n_correct_excl_gt_na"]) for r in rows),
    )
    rows.append(
        {
            "model_id": "__MICRO_AVG_OVER_MODELS__",
            "frames_csv": "",
            "n_frames": total.n_frames,
            "n_correct": total.n_correct,
            "acc_including_na": f"{total.acc_including_na:.6f}",
            "n_gt_na": total.n_gt_na,
            "n_pred_na": total.n_pred_na,
            "n_frames_excl_gt_na": total.n_frames_excl_gt_na,
            "n_correct_excl_gt_na": total.n_correct_excl_gt_na,
            "acc_excluding_gt_na": f"{total.acc_excluding_gt_na:.6f}",
        }
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_id",
        "frames_csv",
        "n_frames",
        "n_correct",
        "acc_including_na",
        "n_gt_na",
        "n_pred_na",
        "n_frames_excl_gt_na",
        "n_correct_excl_gt_na",
        "acc_excluding_gt_na",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote: {out_csv}")


if __name__ == "__main__":
    main()





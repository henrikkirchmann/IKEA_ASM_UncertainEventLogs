"""
Aggregate per-frame predictions into segments defined by consecutive GT labels.

Input CSV (per frame) is expected to contain at least:
  - case_id (int)
  - case_name (str)
  - timestamp (int)           # frame index
  - gt_label (int)
  - probs_json (json object)  # mapping action label -> probability (or class_id -> prob)

Output CSV (per segment) will contain:
  - case_id, case_name
  - start_timestamp, end_timestamp, duration_frames
  - gt_label (+ optional gt_label_name)
  - pred_label (+ optional pred_label_name)  # argmax of averaged probabilities over the segment
  - avg_probs_json                           # averaged probability distribution (same key space as probs_json)

This is designed for process mining use: each GT-constant run becomes one event.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class SegmentAcc:
    case_id: int
    case_name: str
    gt_label: int
    gt_label_name: Optional[str]
    start_ts: int
    end_ts: int
    n: int
    # Keep probabilities in a fixed key order for speed.
    prob_keys: List[str]
    prob_sums: List[float]

    def add(self, ts: int, probs: Dict[str, float]) -> None:
        self.end_ts = ts
        self.n += 1
        for i, k in enumerate(self.prob_keys):
            self.prob_sums[i] += float(probs[k])

    def finalize(self) -> Tuple[int, str, int, int, int, int, Optional[str], int, Optional[str], Dict[str, float]]:
        avg = [v / self.n for v in self.prob_sums]
        # Pred label is argmax over averaged probs
        pred_idx = max(range(len(avg)), key=lambda i: avg[i])
        pred_label_name = self.prob_keys[pred_idx]
        avg_probs = {k: float(avg[i]) for i, k in enumerate(self.prob_keys)}
        return (
            self.case_id,
            self.case_name,
            self.start_ts,
            self.end_ts,
            self.n,
            self.gt_label,
            self.gt_label_name,
            pred_idx,
            pred_label_name,
            avg_probs,
        )


def _iter_input_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("*.csv"))
    if not files:
        raise ValueError(f"No .csv files found in {input_path}")
    return files


def _parse_probs_json(s: str) -> Dict[str, float]:
    d = json.loads(s)
    if not isinstance(d, dict) or not d:
        raise ValueError("probs_json must be a non-empty JSON object")
    return {str(k): float(v) for k, v in d.items()}


def aggregate_one(input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        required = {"case_id", "case_name", "timestamp", "gt_label", "probs_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

        # In this project/pipeline, label names are always exported; enforce to avoid silent numeric-only logs.
        has_gt_name = "gt_label_name" in (reader.fieldnames or [])
        if not has_gt_name:
            raise ValueError(f"{input_csv} is missing required column: gt_label_name")

        with open(output_csv, "w", newline="") as f_out:
            out_fields = [
                "case_id",
                "case_name",
                "start_timestamp",
                "end_timestamp",
                "duration_frames",
                "gt_label",
            ]
            out_fields.append("gt_label_name")
            out_fields += [
                "pred_label",
                "pred_label_name",
                "avg_probs_json",
            ]
            writer = csv.DictWriter(f_out, fieldnames=out_fields)
            writer.writeheader()

            acc: Optional[SegmentAcc] = None
            prob_keys: Optional[List[str]] = None

            for row_idx, row in enumerate(reader):
                case_id = int(row["case_id"])
                case_name = row["case_name"]
                ts = int(row["timestamp"])
                gt_label = int(row["gt_label"])
                gt_label_name = row.get("gt_label_name")
                probs = _parse_probs_json(row["probs_json"])

                if prob_keys is None:
                    # Establish key order from first row (exporter writes deterministic order).
                    prob_keys = list(probs.keys())
                else:
                    # Basic sanity check: same key set across rows.
                    if len(probs) != len(prob_keys) or any(k not in probs for k in prob_keys):
                        raise ValueError(
                            f"{input_csv}: probs_json keys mismatch at row {row_idx}. "
                            f"Expected keys like {prob_keys[:5]}..., got {list(probs.keys())[:5]}..."
                        )

                if acc is None:
                    acc = SegmentAcc(
                        case_id=case_id,
                        case_name=case_name,
                        gt_label=gt_label,
                        gt_label_name=gt_label_name,
                        start_ts=ts,
                        end_ts=ts,
                        n=0,
                        prob_keys=prob_keys,
                        prob_sums=[0.0 for _ in prob_keys],
                    )
                    acc.add(ts, probs)
                    continue

                # Start new segment if case changes OR gt label changes.
                if case_id != acc.case_id or gt_label != acc.gt_label:
                    (
                        out_case_id,
                        out_case_name,
                        start_ts,
                        end_ts,
                        n,
                        out_gt,
                        out_gt_name,
                        pred_label,
                        pred_label_name,
                        avg_probs,
                    ) = acc.finalize()

                    out_row = {
                        "case_id": out_case_id,
                        "case_name": out_case_name,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "duration_frames": n,
                        "gt_label": out_gt,
                        "pred_label": pred_label,
                        "pred_label_name": pred_label_name,
                        "avg_probs_json": json.dumps(avg_probs, ensure_ascii=False, separators=(",", ":")),
                    }
                    if has_gt_name:
                        out_row["gt_label_name"] = out_gt_name
                    writer.writerow(out_row)

                    acc = SegmentAcc(
                        case_id=case_id,
                        case_name=case_name,
                        gt_label=gt_label,
                        gt_label_name=gt_label_name,
                        start_ts=ts,
                        end_ts=ts,
                        n=0,
                        prob_keys=prob_keys,
                        prob_sums=[0.0 for _ in prob_keys],
                    )
                acc.add(ts, probs)

            # flush last segment
            if acc is not None:
                (
                    out_case_id,
                    out_case_name,
                    start_ts,
                    end_ts,
                    n,
                    out_gt,
                    out_gt_name,
                    pred_label,
                    pred_label_name,
                    avg_probs,
                ) = acc.finalize()
                out_row = {
                    "case_id": out_case_id,
                    "case_name": out_case_name,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                    "duration_frames": n,
                    "gt_label": out_gt,
                    "pred_label": pred_label,
                    "pred_label_name": pred_label_name,
                    "avg_probs_json": json.dumps(avg_probs, ensure_ascii=False, separators=(",", ":")),
                }
                if has_gt_name:
                    out_row["gt_label_name"] = out_gt_name
                writer.writerow(out_row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input CSV file OR directory containing CSVs (e.g. _pretrained_csv_json_caseid_num2/).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to write aggregated CSVs into.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="__gt_segments.csv",
        help="Output filename suffix appended to the input stem.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_input_files(input_path)
    for i, f in enumerate(files, start=1):
        out_name = f"{f.stem}{args.suffix}"
        out_csv = output_dir / out_name
        print(f"[{i}/{len(files)}] {f.name} -> {out_csv}")
        aggregate_one(f, out_csv)
    print("Done.")


if __name__ == "__main__":
    main()



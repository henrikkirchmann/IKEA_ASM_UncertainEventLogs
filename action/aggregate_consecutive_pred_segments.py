"""
Aggregate per-frame predictions into segments defined by consecutive PRED labels.

This is the "pred-merged" counterpart to `aggregate_consecutive_gt_segments.py`.

Input CSV (per frame) is expected to contain at least:
  - case_id (int)
  - case_name (str)
  - timestamp (int)            # frame index
  - pred_label (int)
  - pred_label_name (str)
  - probs_json (json object)   # mapping action label -> probability

Output CSV (per segment) will contain:
  - case_id, case_name
  - start_timestamp, end_timestamp, duration_frames
  - pred_label, pred_label_name                # constant inside the segment
  - avg_probs_json                              # averaged probability distribution over the segment

Important:
- This intentionally does NOT include any GT columns (gt_label / gt_label_name).
- NA ("no action") is kept like any other label; it forms segments too.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class SegmentAcc:
    case_id: int
    case_name: str
    pred_label: int
    pred_label_name: str
    start_ts: int
    end_ts: int
    n: int
    prob_keys: List[str]
    prob_sums: List[float]

    def add(self, ts: int, probs: Dict[str, float]) -> None:
        self.end_ts = ts
        self.n += 1
        for i, k in enumerate(self.prob_keys):
            self.prob_sums[i] += float(probs[k])

    def finalize(self) -> Tuple[int, str, int, int, int, int, str, Dict[str, float]]:
        avg = [v / self.n for v in self.prob_sums]
        avg_probs = {k: float(avg[i]) for i, k in enumerate(self.prob_keys)}
        return (
            self.case_id,
            self.case_name,
            self.start_ts,
            self.end_ts,
            self.n,
            self.pred_label,
            self.pred_label_name,
            avg_probs,
        )


def _parse_probs_json(s: str) -> Dict[str, float]:
    d = json.loads(s)
    if not isinstance(d, dict) or not d:
        raise ValueError("probs_json must be a non-empty JSON object")
    return {str(k): float(v) for k, v in d.items()}


def aggregate_one(input_csv: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(input_csv, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        required = {"case_id", "case_name", "timestamp", "pred_label", "pred_label_name", "probs_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

        with open(output_csv, "w", newline="") as f_out:
            out_fields = [
                "case_id",
                "case_name",
                "start_timestamp",
                "end_timestamp",
                "duration_frames",
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
                pred_label = int(row["pred_label"])
                pred_label_name = str(row["pred_label_name"])
                probs = _parse_probs_json(row["probs_json"])

                if prob_keys is None:
                    # Establish key order from first row (exporter writes deterministic order).
                    prob_keys = list(probs.keys())
                else:
                    if len(probs) != len(prob_keys) or any(k not in probs for k in prob_keys):
                        raise ValueError(
                            f"{input_csv}: probs_json keys mismatch at row {row_idx}. "
                            f"Expected keys like {prob_keys[:5]}..., got {list(probs.keys())[:5]}..."
                        )

                if acc is None:
                    acc = SegmentAcc(
                        case_id=case_id,
                        case_name=case_name,
                        pred_label=pred_label,
                        pred_label_name=pred_label_name,
                        start_ts=ts,
                        end_ts=ts,
                        n=0,
                        prob_keys=prob_keys,
                        prob_sums=[0.0 for _ in prob_keys],
                    )
                    acc.add(ts, probs)
                    continue

                # New segment if case changes OR pred label changes.
                if case_id != acc.case_id or pred_label != acc.pred_label:
                    (
                        out_case_id,
                        out_case_name,
                        start_ts,
                        end_ts,
                        n,
                        out_pred,
                        out_pred_name,
                        avg_probs,
                    ) = acc.finalize()

                    writer.writerow(
                        {
                            "case_id": out_case_id,
                            "case_name": out_case_name,
                            "start_timestamp": start_ts,
                            "end_timestamp": end_ts,
                            "duration_frames": n,
                            "pred_label": out_pred,
                            "pred_label_name": out_pred_name,
                            "avg_probs_json": json.dumps(avg_probs, ensure_ascii=False, separators=(",", ":")),
                        }
                    )

                    acc = SegmentAcc(
                        case_id=case_id,
                        case_name=case_name,
                        pred_label=pred_label,
                        pred_label_name=pred_label_name,
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
                    out_pred,
                    out_pred_name,
                    avg_probs,
                ) = acc.finalize()
                writer.writerow(
                    {
                        "case_id": out_case_id,
                        "case_name": out_case_name,
                        "start_timestamp": start_ts,
                        "end_timestamp": end_ts,
                        "duration_frames": n,
                        "pred_label": out_pred,
                        "pred_label_name": out_pred_name,
                        "avg_probs_json": json.dumps(avg_probs, ensure_ascii=False, separators=(",", ":")),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str, help="Input frames.csv")
    parser.add_argument("--output", required=True, type=str, help="Output segments_pred.csv")
    args = parser.parse_args()

    aggregate_one(Path(args.input).resolve(), Path(args.output).resolve())
    print("Done.")


if __name__ == "__main__":
    main()






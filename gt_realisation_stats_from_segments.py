#!/usr/bin/env python3
"""
Compute "ground-truth realisation" control-flow stats from GT-segment CSVs.

We define the GT realisation log by:
- taking the GT-merged segments (consecutive GT labels merged) i.e. `segments_gt.csv`
- using gt_label_name as the activity sequence
- KEEPING NA events (NA is treated as a regular label here)

This matches your requested stats:
- unique activities (including NA)
- trace variant ratio
- trace length (min/avg/max; including NA)
- num events (segments)
- num NA events (segments where gt_label_name == NA)

Input:
- Either a single `segments_gt.csv` file
- Or a root folder that contains `model=*/segments_gt.csv` (e.g. uncertain_event_data/ikea_asm/split=test)

Output:
- CSV with one row per segments_gt.csv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


@dataclass
class GtRealisationStats:
    log_name: str
    unique_activities: int
    num_traces: int
    num_trace_variants: int
    ratio_trace_variants: float
    min_trace_length: int
    avg_trace_length: float
    max_trace_length: int
    num_events: int
    num_na_events: int


def _iter_segments_files(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("model=*/segments_gt.csv"))
    if not files:
        raise ValueError(f"No segments_gt.csv found under: {input_path} (expected model=*/segments_gt.csv)")
    return files


def compute_one(segments_csv: Path, na_label: str) -> GtRealisationStats:
    # We accumulate per-trace activity sequences based on (case_id, case_name).
    traces: Dict[str, List[str]] = {}
    unique: Set[str] = set()
    num_events = 0
    num_na = 0

    with segments_csv.open(newline="") as f:
        r = csv.DictReader(f)
        required = {"case_id", "case_name", "gt_label_name"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise ValueError(f"{segments_csv} missing required columns: {sorted(missing)}")

        for row in r:
            num_events += 1
            case_id = str(row["case_id"])
            act = str(row["gt_label_name"])
            if act == na_label:
                num_na += 1
            unique.add(act)
            traces.setdefault(case_id, []).append(act)

    seqs = list(traces.values())
    num_traces = len(seqs)
    variants = set("\x1f".join(seq) for seq in seqs)
    lengths = [len(seq) for seq in seqs] if seqs else [0]

    min_len = int(min(lengths)) if lengths else 0
    max_len = int(max(lengths)) if lengths else 0
    avg_len = float(sum(lengths) / num_traces) if num_traces > 0 else 0.0
    ratio = float(len(variants) / num_traces) if num_traces > 0 else 0.0

    # Friendly name
    log_name = segments_csv.parent.name.replace("model=", "")

    return GtRealisationStats(
        log_name=log_name,
        unique_activities=len(unique),
        num_traces=num_traces,
        num_trace_variants=len(variants),
        ratio_trace_variants=ratio,
        min_trace_length=min_len,
        avg_trace_length=avg_len,
        max_trace_length=max_len,
        num_events=num_events,
        num_na_events=num_na,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str, help="segments_gt.csv OR split root containing model=*/segments_gt.csv")
    p.add_argument("--output_csv", required=True, type=str, help="Where to write CSV")
    p.add_argument("--na_label", type=str, default="NA")
    args = p.parse_args()

    in_path = Path(args.input).expanduser().resolve()
    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    files = _iter_segments_files(in_path)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "log_name",
                "unique_activities",
                "num_traces",
                "num_trace_variants",
                "ratio_trace_variants",
                "min_trace_length",
                "avg_trace_length",
                "max_trace_length",
                "num_events",
                "num_na_events",
            ]
        )
        for i, seg in enumerate(files, start=1):
            print(f"[{i}/{len(files)}] {seg}")
            s = compute_one(seg, na_label=str(args.na_label))
            w.writerow(
                [
                    s.log_name,
                    s.unique_activities,
                    s.num_traces,
                    s.num_trace_variants,
                    s.ratio_trace_variants,
                    s.min_trace_length,
                    s.avg_trace_length,
                    s.max_trace_length,
                    s.num_events,
                    s.num_na_events,
                ]
            )

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()






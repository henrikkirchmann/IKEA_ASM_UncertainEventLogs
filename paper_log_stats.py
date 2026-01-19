#!/usr/bin/env python3
"""
Compute the log statistics used in the paper for:

1) The single deterministic GT-realisation XES log (control-flow stats; optional NA-ignoring).
2) The pred-merged uncertain XES logs (control-flow + uncertainty; optional NA-ignoring).

Output is a single CSV with one row for the GT log and one row per pred-merged model log.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Optional

from gt_realisation_log_stats import compute_gt_realisation_xes_stats
from pred_merged_log_stats import compute_pred_merged_stats


def _iter_pred_merged_xes_files(root: Path, recursive: bool) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ValueError(f"pred-merged input does not exist: {root}")
    pattern = "**/*xes_uncertain_pred_merged.xes" if recursive else "*xes_uncertain_pred_merged.xes"
    files = sorted(root.glob(pattern))
    if not files:
        raise ValueError(
            f"No pred-merged .xes files found under: {root} (recursive={recursive}). "
            "Expected '*xes_uncertain_pred_merged.xes'."
        )
    return files


def main() -> None:
    p = argparse.ArgumentParser(description="Compute paper log stats (GT-realisation + pred-merged).")
    p.add_argument(
        "--pred_merged_input",
        required=True,
        type=str,
        help="Directory containing pred-merged logs (e.g., uncertain_event_data/ikea_asm/split=test) OR a single pred-merged .xes.",
    )
    p.add_argument(
        "--gt_realisation_xes",
        required=True,
        type=str,
        help="Path to the deterministic GT-realisation .xes (e.g., paper_event_logs/.../ikea_asm__test__gt_realisation__keep_na.xes).",
    )
    p.add_argument("--recursive", action="store_true", help="Search pred-merged logs recursively (recommended).")
    p.add_argument("--output_csv", type=str, default=None, help="Where to write the combined CSV.")
    p.add_argument("--na_label", type=str, default="NA")
    p.add_argument(
        "--ignore_na_in_control_flow",
        action="store_true",
        help="If set, NA events do not contribute to trace length/variants/alphabet (recommended for paper stats).",
    )
    p.add_argument("--pred_probs_key", type=str, default="probs_json")
    p.add_argument("--threshold_5pct", type=float, default=0.05)
    p.add_argument(
        "--gt_log_name",
        type=str,
        default="Certain Groundtruth",
        help="log_name used for the GT-realisation row (default: 'Certain Groundtruth').",
    )
    args = p.parse_args()

    pred_root = Path(args.pred_merged_input).expanduser().resolve()
    gt_xes = Path(args.gt_realisation_xes).expanduser().resolve()
    if not gt_xes.is_file():
        raise ValueError(f"gt_realisation_xes is not a file: {gt_xes}")

    if args.output_csv:
        out_csv = Path(args.output_csv).expanduser().resolve()
    else:
        # default: alongside gt log (paper artifact folder) if possible, else CWD
        out_csv = (gt_xes.parent / "log_stats_paper.csv") if gt_xes.parent.exists() else (Path.cwd() / "log_stats_paper.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_files = _iter_pred_merged_xes_files(pred_root, recursive=bool(args.recursive))

    # Compute GT control-flow stats.
    gt = compute_gt_realisation_xes_stats(
        gt_xes,
        na_label=str(args.na_label),
        ignore_na_in_control_flow=bool(args.ignore_na_in_control_flow),
    )

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "log_name",
                "unique_activities_excl_na",
                "num_traces",
                "num_trace_variants_excl_na",
                "ratio_trace_variants_excl_na",
                "min_trace_length_excl_na",
                "avg_trace_length_excl_na",
                "max_trace_length_excl_na",
                "num_events_total",
                "num_events_na",
                "num_events_with_probs",
                "avg_num_possibilities",
                "avg_num_possibilities_5pct",
                "avg_uncertainty",
                "avg_top1_confidence",
            ]
        )

        # GT row: fill uncertainty fields with "--" (not applicable).
        w.writerow(
            [
                str(args.gt_log_name),
                gt.unique_activities_excl_na,
                gt.num_traces,
                gt.num_trace_variants_excl_na,
                gt.ratio_trace_variants_excl_na,
                gt.min_trace_length_excl_na,
                gt.avg_trace_length_excl_na,
                gt.max_trace_length_excl_na,
                gt.num_events_total,
                gt.num_events_na,
                0,
                "--",
                "--",
                "--",
                "--",
            ]
        )

        # Pred-merged rows.
        for i, x in enumerate(pred_files, start=1):
            print(f"[pred {i}/{len(pred_files)}] {x}")
            s = compute_pred_merged_stats(
                x,
                na_label=str(args.na_label),
                probs_key=str(args.pred_probs_key),
                ignore_na_in_control_flow=bool(args.ignore_na_in_control_flow),
                threshold_5pct=float(args.threshold_5pct),
            )
            w.writerow(
                [
                    s.log_name,  # model id
                    s.unique_activities_excl_na,
                    s.num_traces,
                    s.num_trace_variants_excl_na,
                    s.ratio_trace_variants_excl_na,
                    s.min_trace_length_excl_na,
                    s.avg_trace_length_excl_na,
                    s.max_trace_length_excl_na,
                    s.num_events_total,
                    s.num_events_na,
                    s.num_events_with_probs,
                    s.avg_num_possibilities,
                    s.avg_num_possibilities_5pct,
                    s.avg_uncertainty,
                    s.avg_top1_confidence,
                ]
            )

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()


"""
GT-aligned export pipeline:
  results/ (pred.npy + action_segments.json) + gt_action.npy (+ gt_segments.json for label names)
    -> frames.csv (per frame, true GT + predicted probs)
    -> segments_gt.csv (consecutive GT labels merged; avg_probs_json + pred_label argmax over segment)
    -> XES (event concept:name is GT label; distribution stored as probs_json attribute)
    -> organized folder structure under uncertain_event_data/<dataset>/split=<split>/

Use this when you have model outputs AND true GT per frame, and want a meaningful event_accuracy downstream.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, type=str)
    p.add_argument("--split", required=True, type=str, help="e.g. all|train|val|test")
    p.add_argument("--pred_results_dir", required=True, type=str, help="Path to a `results/` dir containing pred.npy and action_segments.json")
    p.add_argument("--gt_action_npy", required=False, type=str, help="Path to gt_action.npy (required unless --use_pred_gt is set)")
    p.add_argument("--gt_segments_json", required=True, type=str, help="Path to gt_segments.json (label names / schema)")
    p.add_argument("--work_dir", required=True, type=str, help="Scratch directory for intermediates")
    p.add_argument("--out_root", required=True, type=str, help="Final root folder, e.g. uncertain_event_data/")
    p.add_argument("--na_handling", type=str, default="omit_concept_name", choices=["keep", "omit_concept_name"])
    p.add_argument(
        "--use_pred_gt",
        action="store_true",
        help="Use GT labels stored inside pred.npy (key: gt_labels). Requires inference to have saved gt_labels.",
    )
    p.add_argument(
        "--model_id",
        type=str,
        default="clip_based__i3d__depth",
        help="Stem used for output files/folders.",
    )
    p.add_argument(
        "--strict_frame_count",
        action="store_true",
        help="If set, error when GT and pred frame counts differ (default: truncate to min length).",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    work_dir = Path(args.work_dir).resolve()
    out_root = Path(args.out_root).resolve()
    pred_results_dir = Path(args.pred_results_dir).resolve()
    gt_segments_json = Path(args.gt_segments_json).resolve()
    gt_action_npy = Path(args.gt_action_npy).resolve() if args.gt_action_npy is not None else None

    # scripts
    exporter = repo_root / "action" / "export_pretrained_probs_with_gt.py"
    aggregator = repo_root / "action" / "aggregate_consecutive_gt_segments.py"
    xes_exporter = repo_root / "action" / "export_uncertain_xes_from_segments.py"
    organizer = repo_root / "action" / "organize_uncertain_event_data_exports.py"

    # scratch layout
    frames_dir = work_dir / "frames_csv"
    segments_dir = work_dir / "segments_csv"
    xes_dir = work_dir / "xes"

    if work_dir.exists():
        shutil.rmtree(work_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    xes_dir.mkdir(parents=True, exist_ok=True)

    frames_csv = frames_dir / f"{args.model_id}.csv"
    _run(
        [
            "python3",
            str(exporter),
            "--pred_dir",
            str(pred_results_dir),
            "--gt_segments_json",
            str(gt_segments_json),
            "--output_csv",
            str(frames_csv),
            "--probs_format",
            "json",
        ]
        + (["--gt_action_npy", str(gt_action_npy)] if (gt_action_npy is not None and not args.use_pred_gt) else [])
        + (["--use_pred_gt"] if args.use_pred_gt else [])
        + (["--strict_frame_count"] if args.strict_frame_count else [])
    )

    _run(
        [
            "python3",
            str(aggregator),
            "--input",
            str(frames_dir),
            "--output_dir",
            str(segments_dir),
            "--suffix",
            "__gt_segments.csv",
        ]
    )

    _run(
        [
            "python3",
            str(xes_exporter),
            "--input",
            str(segments_dir),
            "--output_dir",
            str(xes_dir),
            "--time_col",
            "start_timestamp",
            "--probs_col",
            "avg_probs_json",
            "--probs_attr_key",
            "probs_json",
            "--na_label",
            "NA",
            "--na_handling",
            args.na_handling,
        ]
    )

    _run(
        [
            "python3",
            str(organizer),
            "--dataset_name",
            args.dataset_name,
            "--split",
            args.split,
            "--out_root",
            str(out_root),
            "--frames_dir",
            str(frames_dir),
            "--segments_dir",
            str(segments_dir),
            "--xes_dir",
            str(xes_dir),
            "--gt_segments_json",
            str(gt_segments_json),
        ]
    )


if __name__ == "__main__":
    main()



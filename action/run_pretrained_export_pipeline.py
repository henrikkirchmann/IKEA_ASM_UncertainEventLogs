"""
End-to-end pipeline:
  pretrained zip + action annotation zip
    -> per-frame CSVs (probs_json)
    -> GT-segment CSVs (avg_probs_json)
    -> XES (GT as concept:name, distribution stored as a JSON string event attribute)
    -> organized folder structure under uncertain_event_data/<dataset>/split=<split>/

This script is intentionally self-contained and uses the other helper scripts in this repo.
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
    p.add_argument("--split", required=True, type=str)
    p.add_argument("--pretrained_zip", required=True, type=str)
    p.add_argument("--annotations_zip", required=True, type=str)
    p.add_argument("--work_dir", required=True, type=str, help="Scratch directory for extraction + intermediate files")
    p.add_argument("--out_root", required=True, type=str, help="Final root folder, e.g. uncertain_event_data/")
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    work_dir = Path(args.work_dir).resolve()
    out_root = Path(args.out_root).resolve()

    # scripts
    exporter = repo_root / "action" / "export_pretrained_probs_with_gt.py"
    aggregator = repo_root / "action" / "aggregate_consecutive_gt_segments.py"
    xes_exporter = repo_root / "action" / "export_uncertain_xes_from_segments.py"
    organizer = repo_root / "action" / "organize_uncertain_event_data_exports.py"

    # scratch layout
    extracted_pretrained = work_dir / "pretrained"
    extracted_ann = work_dir / "annotations"
    frames_dir = work_dir / "frames_csv"
    segments_dir = work_dir / "segments_csv"
    xes_dir = work_dir / "xes"

    # clean work dir
    if work_dir.exists():
        shutil.rmtree(work_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)
    xes_dir.mkdir(parents=True, exist_ok=True)

    # unzip pretrained + annotations
    extracted_pretrained.mkdir(parents=True, exist_ok=True)
    extracted_ann.mkdir(parents=True, exist_ok=True)
    _run(["unzip", "-o", str(Path(args.pretrained_zip).resolve()), "-d", str(extracted_pretrained)])
    _run(["unzip", "-o", str(Path(args.annotations_zip).resolve()), "-d", str(extracted_ann)])

    gt_action_npy = extracted_ann / "gt_action.npy"
    gt_segments_json = extracted_ann / "gt_segments.json"
    if not gt_action_npy.exists():
        raise FileNotFoundError(f"Missing {gt_action_npy}")
    if not gt_segments_json.exists():
        raise FileNotFoundError(f"Missing {gt_segments_json}")

    # per-frame CSV export (one CSV per model/view)
    results_dirs = sorted(extracted_pretrained.glob("action_recognition/**/results"))
    results_dirs = [d for d in results_dirs if (d / "pred.npy").exists() and (d / "action_segments.json").exists()]
    if not results_dirs:
        raise ValueError(f"No pretrained results found under {extracted_pretrained}")

    for d in results_dirs:
        # model id comes from parent directories (drop action_recognition/.../results)
        rel = d.relative_to(extracted_pretrained / "action_recognition")
        # rel like clip_based/i3d/dev3/results -> clip_based__i3d__dev3
        parts = list(rel.parts)
        if parts and parts[-1] == "results":
            parts = parts[:-1]
        model_id = "__".join(parts)
        out_csv = frames_dir / f"{model_id}.csv"
        _run(
            [
                "python3",
                str(exporter),
                "--pred_dir",
                str(d),
                "--gt_action_npy",
                str(gt_action_npy),
                "--output_csv",
                str(out_csv),
                "--probs_format",
                "json",
                "--strict_frame_count",
            ]
        )

    # aggregate into GT segments (avg probs) for all per-frame csvs
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

    # export uncertain XES for each segment CSV
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
        ]
    )

    # organize into final structure (copies)
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



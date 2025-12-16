"""
Batch-export per-frame CSVs (process-mining friendly) for ALL pretrained result bundles.

This scans a pretrained extraction root for:
  action_recognition/**/results/pred.npy
  action_recognition/**/results/action_segments.json

and for each results directory, writes one CSV to the output directory.

Tip: per-frame CSVs can become very large. Use --gzip to write `.csv.gz`.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


def _find_results_dirs(root: Path) -> List[Path]:
    pred_files = list(root.glob("action_recognition/**/results/pred.npy"))
    out: List[Path] = []
    for pred in pred_files:
        results_dir = pred.parent
        if (results_dir / "action_segments.json").exists():
            out.append(results_dir)
    # stable, readable ordering
    out.sort(key=lambda p: str(p))
    return out


def _out_name(results_dir: Path) -> str:
    # e.g. action_recognition/clip_based/i3d/dev3/results -> clip_based__i3d__dev3
    parts = list(results_dir.parts)
    # keep everything after "action_recognition"
    if "action_recognition" in parts:
        i = parts.index("action_recognition")
        rel_parts = parts[i + 1 :]
    else:
        rel_parts = parts
    # drop trailing "results"
    if rel_parts and rel_parts[-1] == "results":
        rel_parts = rel_parts[:-1]
    return "__".join(rel_parts)


def _run_one(exporter_py: Path, results_dir: Path, gt_action_npy: Path, out_csv: Path) -> None:
    cmd = [
        "python3",
        str(exporter_py),
        "--pred_dir",
        str(results_dir),
        "--gt_action_npy",
        str(gt_action_npy),
        "--output_csv",
        str(out_csv),
        "--strict_frame_count",
    ]
    subprocess.check_call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_root",
        default=None,
        type=str,
        help="Directory that contains `action_recognition/` (your unzipped pretrained models folder). "
             "Defaults to the repository root (parent of `action/`).",
    )
    parser.add_argument(
        "--gt_action_npy",
        required=True,
        type=str,
        help="Path to `gt_action.npy` from `action_annotations.zip` (extracted).",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        type=str,
        help="Directory to write per-model CSVs into. Defaults to `<repo_root>/_pretrained_csv`.",
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        default=True,
        help="Write `.csv.gz` instead of `.csv` (recommended). Enabled by default; use --no-gzip to disable.",
    )
    parser.add_argument(
        "--no-gzip",
        dest="gzip",
        action="store_false",
        help="Disable gzip output (write `.csv`).",
    )
    parser.add_argument(
        "--probs_format",
        type=str,
        default="columns",
        choices=["columns", "json"],
        help="How to store per-class probabilities in CSVs. "
             "`columns` writes prob_0..prob_{C-1}. "
             "`json` writes a single probs_json column containing a JSON object mapping action->prob.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="For quick tests/debugging: export only first N result dirs (0 = all).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    pretrained_root = Path(args.pretrained_root).resolve() if args.pretrained_root else repo_root
    gt_action_npy = Path(args.gt_action_npy).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (repo_root / "_pretrained_csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    exporter_py = (Path(__file__).resolve().parent / "export_pretrained_probs_with_gt.py").resolve()
    if not exporter_py.exists():
        raise FileNotFoundError(f"Missing exporter script: {exporter_py}")

    results_dirs = _find_results_dirs(pretrained_root)
    if not results_dirs:
        raise ValueError(f"No result dirs found under {pretrained_root}. Expected action_recognition/**/results/pred.npy")

    if args.limit and args.limit > 0:
        results_dirs = results_dirs[: args.limit]

    print(f"Pretrained root: {pretrained_root}")
    print(f"Output dir: {out_dir}")
    print(f"Found {len(results_dirs)} pretrained result dirs.")
    for i, results_dir in enumerate(results_dirs, start=1):
        name = _out_name(results_dir)
        suffix = ".csv.gz" if args.gzip else ".csv"
        out_csv = out_dir / f"{name}{suffix}"
        print(f"[{i}/{len(results_dirs)}] Exporting {name} -> {out_csv}")
        cmd = [
            "python3",
            str(exporter_py),
            "--pred_dir",
            str(results_dir),
            "--gt_action_npy",
            str(gt_action_npy),
            "--output_csv",
            str(out_csv),
            "--strict_frame_count",
            "--probs_format",
            args.probs_format,
        ]
        subprocess.check_call(cmd)

    print("Done.")


if __name__ == "__main__":
    main()



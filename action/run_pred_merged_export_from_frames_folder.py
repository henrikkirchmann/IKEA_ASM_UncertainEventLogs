"""
Build pred-merged segments + predicted-activity XES logs for a folder of model outputs.

For each model folder containing:
  - frames.csv (per-frame, includes pred_label_name and probs_json)

This script writes, next to it:
  - segments_pred.csv
  - xes_uncertain_pred_merged.xes    (concept:name := pred_label_name, no gt:* attrs)

Use case:
You want a process mining log where the activity is the model prediction (including NA),
and you do NOT want any GT columns/attributes in the derived segments/log.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# NOTE:
# This script is typically executed as `python3 action/run_pred_merged_export_from_frames_folder.py ...`
# In that mode, Python adds the script directory (`action/`) to sys.path, so we import sibling modules
# without the `action.` prefix (the repo is not necessarily installed as a package).
from aggregate_consecutive_pred_segments import aggregate_one as aggregate_pred_segments
from export_uncertain_xes_from_segments import export_one_csv_to_xes, _parse_epoch_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        required=True,
        type=str,
        help="Root folder to search for model=*/frames.csv (e.g. uncertain_event_data/ikea_asm/split=test).",
    )
    parser.add_argument(
        "--na_label",
        type=str,
        default="NA",
        help="Label name that represents 'no action' / 'no event' (default: NA).",
    )
    parser.add_argument(
        "--na_handling",
        type=str,
        default="keep",
        choices=["keep", "omit_concept_name"],
        help="How to encode NA events in concept:name. Default keep (concept:name=NA).",
    )
    parser.add_argument(
        "--epoch_iso",
        type=str,
        default="1970-01-01T00:00:00Z",
        help="Epoch for numeric timestamps (default: 1970-01-01T00:00:00Z).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    epoch = _parse_epoch_iso(args.epoch_iso)

    frames_files = sorted(root.glob("model=*/frames.csv"))
    if not frames_files:
        raise SystemExit(f"No frames.csv found under: {root} (expected model=*/frames.csv)")

    for i, frames_csv in enumerate(frames_files, start=1):
        model_dir = frames_csv.parent
        segments_csv = model_dir / "segments_pred.csv"
        out_xes = model_dir / "xes_uncertain_pred_merged.xes"

        print(f"[{i}/{len(frames_files)}] {model_dir.name}")
        print(f"  - aggregate: {frames_csv.name} -> {segments_csv.name}")
        aggregate_pred_segments(frames_csv, segments_csv)

        print(f"  - export:    {segments_csv.name} -> {out_xes.name}")
        export_one_csv_to_xes(
            input_csv=segments_csv,
            output_xes=out_xes,
            epoch=epoch,
            time_col="start_timestamp",
            probs_col="avg_probs_json",
            probs_attr_key="probs_json",
            na_label=args.na_label,
            na_handling=args.na_handling,
            activity_source="pred",
            include_gt_attrs=False,
        )

    print("Done.")


if __name__ == "__main__":
    main()



"""
Organize generated artifacts into a clean, human-readable + machine-parseable folder layout.

Target layout (no runs/, no artifacts/):

  <out_root>/<dataset_name>/split=<split>/
    labels/
      class_names.json
      label_schema.json
    model=<model_id>__<source>__<view>/
      frames.csv
      segments_gt.csv
      xes_uncertain_gt.xes
      manifest.json

This script COPIES files (does not delete originals).
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _action_list_from_gt_segments_json(gt_segments_json: Path) -> List[str]:
    js = json.loads(gt_segments_json.read_text())
    db = js.get("database", {})
    labels = set()
    for _, v in db.items():
        for ann in v.get("annotation", []):
            labels.add(ann.get("label"))
    labels.discard(None)
    if "NA" not in labels:
        raise ValueError("Expected 'NA' label in gt_segments.json")
    non_na = sorted([l for l in labels if l != "NA"])
    return ["NA"] + non_na


def _infer_source_view(model_stem: str) -> Tuple[str, str]:
    # Model stem examples:
    # - clip_based__i3d__dev1
    # - clip_based__i3d__depth
    # - frame_based__resnet50
    # - pose_based__HCN_32
    if "__depth" in model_stem:
        return "depth", "depth"
    if "__dev1" in model_stem:
        return "rgb", "dev1"
    if "__dev2" in model_stem:
        return "rgb", "dev2"
    if "__dev3" in model_stem:
        return "rgb", "dev3"
    if model_stem.startswith("pose_based__"):
        return "pose", "dev3"
    # frame_based + c3d/p3d bundles do not encode camera; in this repo defaults are dev3 RGB.
    return "rgb", "dev3"


@dataclass
class Manifest:
    dataset_name: str
    split: str
    model_id: str
    source: str
    view: str
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    notes: Optional[str] = None


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", required=True, type=str)
    p.add_argument("--split", required=True, type=str, help="train|val|test|all")
    p.add_argument("--out_root", required=True, type=str, help="Root folder, e.g. uncertain_event_data/")
    p.add_argument("--frames_dir", required=True, type=str, help="Directory with per-frame CSVs")
    p.add_argument("--segments_dir", required=True, type=str, help="Directory with segment CSVs")
    p.add_argument("--xes_dir", required=True, type=str, help="Directory with XES files")
    p.add_argument("--gt_segments_json", required=True, type=str, help="Path to gt_segments.json for label names")
    args = p.parse_args()

    out_root = Path(args.out_root).resolve()
    dataset_dir = out_root / args.dataset_name / f"split={args.split}"
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    action_list = _action_list_from_gt_segments_json(Path(args.gt_segments_json).resolve())
    (labels_dir / "class_names.json").write_text(json.dumps(action_list, ensure_ascii=False, indent=2) + "\n")
    (labels_dir / "label_schema.json").write_text(
        json.dumps(
            {
                "task": "action_recognition",
                "num_classes": len(action_list),
                "index_to_label": {str(i): action_list[i] for i in range(len(action_list))},
                "label_to_index": {action_list[i]: i for i in range(len(action_list))},
                "notes": "Index 0 is 'NA'. Remaining labels are sorted alphabetically (repo convention).",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )

    frames_dir = Path(args.frames_dir).resolve()
    seg_dir = Path(args.segments_dir).resolve()
    xes_dir = Path(args.xes_dir).resolve()

    frame_csvs = sorted([f for f in frames_dir.glob("*.csv") if not f.name.startswith("_")])
    if not frame_csvs:
        raise ValueError(f"No per-frame CSVs found in {frames_dir}")

    for frame_csv in frame_csvs:
        model_id = frame_csv.stem  # e.g. frame_based__resnet50
        source, view = _infer_source_view(model_id)
        # Avoid redundant folder names when model_id already encodes view/source (e.g. i3d__dev3, i3d__depth).
        name_parts = [f"model={model_id}", source]
        if view not in model_id:
            name_parts.append(view)
        model_dir = dataset_dir / ("__".join(name_parts))
        model_dir.mkdir(parents=True, exist_ok=True)

        seg_csv = seg_dir / f"{model_id}__gt_segments.csv"
        xes_file = xes_dir / f"{model_id}__gt_segments.xes"

        if not seg_csv.exists():
            raise FileNotFoundError(f"Missing segment CSV for {model_id}: {seg_csv}")
        if not xes_file.exists():
            raise FileNotFoundError(f"Missing XES for {model_id}: {xes_file}")

        _copy(frame_csv, model_dir / "frames.csv")
        _copy(seg_csv, model_dir / "segments_gt.csv")
        _copy(xes_file, model_dir / "xes_uncertain_gt.xes")

        # Make the manifest portable: keep "inputs" as relative paths within the exported folder,
        # and store original source locations under "source_inputs" for provenance/debugging.
        source_inputs = {
            "frames_csv": str(frame_csv),
            "segments_csv": str(seg_csv),
            "xes": str(xes_file),
            "gt_segments_json": str(Path(args.gt_segments_json).resolve()),
        }

        manifest = Manifest(
            dataset_name=args.dataset_name,
            split=args.split,
            model_id=model_id,
            source=source,
            view=view,
            inputs={
                "frames_csv": "frames.csv",
                "segments_gt_csv": "segments_gt.csv",
                "xes_uncertain_gt": "xes_uncertain_gt.xes",
                "labels_class_names": str((Path("..") / "labels" / "class_names.json").as_posix()),
            },
            outputs={
                "frames_csv": "frames.csv",
                "segments_gt_csv": "segments_gt.csv",
                "xes_uncertain_gt": "xes_uncertain_gt.xes",
                "labels_class_names": str((Path("..") / "labels" / "class_names.json").as_posix()),
            },
            notes=(
                "Files copied from generated artifacts in this repo; per-frame timestamps are frame indices. "
                "XES stores the full activity probability distribution as a JSON string event attribute "
                "(key: probs_json) for easy downstream extraction."
            ),
        )
        d = asdict(manifest)
        d["source_inputs"] = source_inputs
        (model_dir / "manifest.json").write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")

    print(f"Saved to: {dataset_dir}")
    print(f"Models exported: {len(frame_csvs)}")


if __name__ == "__main__":
    main()



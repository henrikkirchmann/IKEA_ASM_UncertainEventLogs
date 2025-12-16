"""
Export per-frame action probabilities + per-frame GT labels from the provided pretrained artifacts.

Why this exists:
- The pretrained bundles include `results/pred.npy` with per-frame softmax probabilities, but without video names.
- The corresponding `results/action_segments.json` preserves the same video order and includes video names.
- The action annotations bundle includes `gt_action.npy` with per-frame GT labels per video name.

This script merges them and can write:
- a single `.npz` (ragged arrays), and/or
- a per-frame `.csv` suitable for process mining (one row per frame).
"""

from __future__ import annotations

import argparse
import json
import os
import csv
import gzip
from typing import Dict, List, Tuple

import numpy as np


def _load_pred(pred_npy: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray] | None]:
    obj = np.load(pred_npy, allow_pickle=True)
    if not isinstance(obj, np.ndarray) or obj.dtype != object or obj.shape != ():
        raise ValueError(f"Unexpected format for pred file: {pred_npy}. Expected 0-d object array.")
    d = obj.item()
    if not isinstance(d, dict) or "logits" not in d or "pred_labels" not in d:
        raise ValueError(f"Unexpected dict keys in pred file: {pred_npy}. Found keys: {list(d.keys())}")
    logits = d["logits"]
    pred_labels = d["pred_labels"]
    gt_labels = d.get("gt_labels", None)
    if len(logits) != len(pred_labels):
        raise ValueError(f"Pred mismatch: len(logits)={len(logits)} != len(pred_labels)={len(pred_labels)}")
    if gt_labels is not None and len(gt_labels) != len(pred_labels):
        raise ValueError(f"Pred/GT mismatch: len(gt_labels)={len(gt_labels)} != len(pred_labels)={len(pred_labels)}")
    return pred_labels, logits, gt_labels


def _load_pred_video_names(pred_segments_json: str) -> List[str]:
    with open(pred_segments_json, "r") as f:
        js = json.load(f)
    results = js.get("results", None)
    if not isinstance(results, dict):
        raise ValueError(
            f"Unexpected format for {pred_segments_json}: missing or invalid 'results' dict. Keys: {list(js.keys())}"
        )
    # Python 3.7+ preserves insertion order; JSON load preserves file order.
    return list(results.keys())


def _load_gt(gt_action_npy: str) -> Tuple[List[str], List[np.ndarray]]:
    obj = np.load(gt_action_npy, allow_pickle=True)
    if not isinstance(obj, np.ndarray) or obj.dtype != object or obj.shape != ():
        raise ValueError(f"Unexpected format for GT file: {gt_action_npy}. Expected 0-d object array.")
    d = obj.item()
    if not isinstance(d, dict) or "scan_name" not in d or "gt_labels" not in d:
        raise ValueError(f"Unexpected dict keys in GT file: {gt_action_npy}. Found keys: {list(d.keys())}")
    scan_name = d["scan_name"]
    gt_labels = d["gt_labels"]
    if len(scan_name) != len(gt_labels):
        raise ValueError(f"GT mismatch: len(scan_name)={len(scan_name)} != len(gt_labels)={len(gt_labels)}")
    return scan_name, gt_labels


def _action_list_from_gt_segments_json(gt_segments_json: str) -> List[str]:
    """
    Derive the class-name list used by gt_action.npy indices.

    In this repo, label index 0 is "NA" and the remaining classes are alphabetically sorted.
    `gt_segments.json` contains the full set of label names (including "NA").
    """
    with open(gt_segments_json, "r") as f:
        js = json.load(f)
    db = js.get("database", None)
    if not isinstance(db, dict):
        raise ValueError(f"Unexpected format for {gt_segments_json}: missing/invalid 'database' dict.")
    labels = set()
    for _, v in db.items():
        for ann in v.get("annotation", []):
            labels.add(ann.get("label"))
    labels.discard(None)
    if "NA" not in labels:
        raise ValueError(f"'NA' not found in labels from {gt_segments_json}")
    non_na = sorted([l for l in labels if l != "NA"])
    action_list = ["NA"] + non_na
    return action_list


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        default=None,
        type=str,
        help="If set, reads `pred.npy` and `action_segments.json` from this directory (typically a `results/` dir). "
             "Overrides --pred_npy/--pred_segments_json.",
    )
    parser.add_argument(
        "--pred_npy",
        default=None,
        type=str,
        help="Path to pretrained `pred.npy` (contains per-frame probabilities in dict['logits']).",
    )
    parser.add_argument(
        "--pred_segments_json",
        default=None,
        type=str,
        help="Path to corresponding `action_segments.json` (used to recover video names/order).",
    )
    parser.add_argument(
        "--gt_action_npy",
        required=False,
        type=str,
        help="Path to `gt_action.npy` from `action_annotations.zip` (required unless --use_pred_gt is set).",
    )
    parser.add_argument(
        "--use_pred_gt",
        action="store_true",
        help="Use GT labels saved inside pred.npy (key: gt_labels) instead of gt_action.npy. "
             "This is the most reliable alignment if your inference script saved gt_labels.",
    )
    parser.add_argument(
        "--gt_segments_json",
        default=None,
        type=str,
        help="Optional path to `gt_segments.json` to map label ids -> human-readable label names. "
             "If omitted, will try to load `<dir_of_gt_action_npy>/gt_segments.json`.",
    )
    parser.add_argument(
        "--output_npz",
        default=None,
        type=str,
        help="Output `.npz` path (optional). Will contain ragged object arrays: video_names, probs, gt_labels.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        type=str,
        help="Output `.csv` path (optional). One row per frame: video_name, frame_idx, gt_label, pred_label, prob_* columns.",
    )
    parser.add_argument(
        "--probs_format",
        type=str,
        default="columns",
        choices=["columns", "json"],
        help="How to store per-class probabilities in CSV. "
             "`columns` writes prob_0..prob_{C-1}. "
             "`json` writes a single `probs_json` column containing a compact JSON object mapping action label -> prob "
             "(or class_id -> prob if label names are unavailable).",
    )
    parser.add_argument(
        "--strict_frame_count",
        action="store_true",
        help="If set, error when GT and pred frame counts differ for any video.",
    )
    parser.add_argument(
        "--limit_videos",
        type=int,
        default=0,
        help="For quick tests/debugging: limit exported videos to the first N (0 = no limit).",
    )
    args = parser.parse_args()

    if args.pred_dir is not None:
        pred_npy = os.path.join(args.pred_dir, "pred.npy")
        pred_segments_json = os.path.join(args.pred_dir, "action_segments.json")
    else:
        pred_npy = args.pred_npy
        pred_segments_json = args.pred_segments_json

    if pred_npy is None or pred_segments_json is None:
        raise ValueError("Provide either --pred_dir OR both --pred_npy and --pred_segments_json.")

    pred_labels, pred_probs, pred_gt_labels = _load_pred(pred_npy)
    pred_video_names = _load_pred_video_names(pred_segments_json)
    if args.use_pred_gt:
        if pred_gt_labels is None:
            raise ValueError(
                f"--use_pred_gt was set but {pred_npy} does not contain 'gt_labels'. "
                f"Re-run inference with an updated test script that saves gt_labels (e.g. action/clip_based/i3d/test_i3d.py)."
            )
        gt_scan_names = pred_video_names
        gt_labels_all = pred_gt_labels
    else:
        if args.gt_action_npy is None:
            raise ValueError("Provide --gt_action_npy, or set --use_pred_gt to use GT labels stored in pred.npy.")
        gt_scan_names, gt_labels_all = _load_gt(args.gt_action_npy)

    if len(pred_video_names) != len(pred_probs):
        raise ValueError(
            f"Video count mismatch: action_segments.json has {len(pred_video_names)} videos but pred.npy has {len(pred_probs)}."
        )

    gt_name_to_idx: Dict[str, int] = {name: i for i, name in enumerate(gt_scan_names)}

    # Optional: map label ids to label names
    gt_segments_json = args.gt_segments_json
    if gt_segments_json is None and args.gt_action_npy is not None:
        candidate = os.path.join(os.path.dirname(os.path.abspath(args.gt_action_npy)), "gt_segments.json")
        if os.path.exists(candidate):
            gt_segments_json = candidate

    action_list: List[str] | None = None
    if gt_segments_json is not None and os.path.exists(gt_segments_json):
        action_list = _action_list_from_gt_segments_json(gt_segments_json)

    out_names: List[str] = []
    out_probs: List[np.ndarray] = []
    out_gt: List[np.ndarray] = []

    missing = 0
    frame_mismatch = 0
    n_vids = len(pred_video_names)
    if args.limit_videos and args.limit_videos > 0:
        n_vids = min(n_vids, args.limit_videos)

    for i, name in enumerate(pred_video_names[:n_vids]):
        if name not in gt_name_to_idx:
            missing += 1
            continue
        gt = np.asarray(gt_labels_all[gt_name_to_idx[name]])
        probs = np.asarray(pred_probs[i])
        if gt.shape[0] != probs.shape[0]:
            frame_mismatch += 1
            if args.strict_frame_count:
                raise ValueError(
                    f"Frame count mismatch for {name}: gt={gt.shape[0]} vs pred={probs.shape[0]} "
                    f"(index {i})."
                )
            # Non-strict mode: truncate to the shared min length so downstream CSV/segment export stays consistent.
            min_len = int(min(gt.shape[0], probs.shape[0]))
            gt = gt[:min_len]
            probs = probs[:min_len]
        out_names.append(name)
        out_probs.append(probs)
        out_gt.append(gt)

    if args.output_npz is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
        np.savez_compressed(
            args.output_npz,
            video_names=np.asarray(out_names, dtype=object),
            probs=np.asarray(out_probs, dtype=object),       # each element: [T, C]
            gt_labels=np.asarray(out_gt, dtype=object),      # each element: [T]
            num_classes=int(out_probs[0].shape[1]) if out_probs else -1,
            missing_in_gt=int(missing),
            frame_mismatch=int(frame_mismatch),
            source_pred_npy=os.path.abspath(pred_npy),
            source_pred_segments_json=os.path.abspath(pred_segments_json),
            source_gt_action_npy=os.path.abspath(args.gt_action_npy),
        )
        print(f"Wrote NPZ: {args.output_npz}")

    if args.output_csv is not None:
        if not out_probs:
            raise ValueError("No videos exported; cannot write CSV.")
        num_classes = int(out_probs[0].shape[1])
        os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
        opener = gzip.open if args.output_csv.endswith(".gz") else open
        open_kwargs = {"mode": "wt", "newline": ""} if args.output_csv.endswith(".gz") else {"mode": "w", "newline": ""}
        with opener(args.output_csv, **open_kwargs) as f:
            writer = csv.writer(f)  # type: ignore[arg-type]
            # In this project/pipeline we always export label names (gt/pred), so enforce it.
            if action_list is None:
                raise ValueError(
                    "Label names are required for CSV export. Provide --gt_segments_json or place "
                    "`gt_segments.json` next to `gt_action.npy`."
                )
            # Process mining convention: first column is case id.
            # Here, we use an integer id per video (0..N-1) and keep the original string id in `video_name`.
            header = ["case_id", "case_name", "timestamp", "gt_label"]
            if len(action_list) != num_classes:
                raise ValueError(
                    f"Label-name count mismatch: got {len(action_list)} names from gt_segments.json but "
                    f"pred has {num_classes} classes."
                )
            header.append("gt_label_name")
            header.append("pred_label")
            header.append("pred_label_name")
            if args.probs_format == "columns":
                header += [f"prob_{i}" for i in range(num_classes)]
            else:
                header += ["probs_json"]
            writer.writerow(header)
            case_id_map = {name: i for i, name in enumerate(out_names)}
            for name, gt, probs in zip(out_names, out_gt, out_probs):
                # probs: [T, C]
                pred = np.argmax(probs, axis=1)
                # In non-strict mode we truncated GT/probs to the shared min length above, but keep this robust anyway.
                T = int(min(probs.shape[0], gt.shape[0]))
                for t in range(T):
                    row = [case_id_map[name], name, t, int(gt[t])]
                    row.append(action_list[int(gt[t])])
                    row.append(int(pred[t]))
                    row.append(action_list[int(pred[t])])
                    if args.probs_format == "columns":
                        row += probs[t].tolist()
                    else:
                        # Compact JSON mapping label -> prob (or id -> prob).
                        mapping = {action_list[i]: float(probs[t, i]) for i in range(num_classes)}
                        row.append(json.dumps(mapping, ensure_ascii=False, separators=(",", ":")))
                    writer.writerow(row)
        print(f"Wrote CSV: {args.output_csv}")

    if args.output_npz is None and args.output_csv is None:
        raise ValueError("Nothing to do: provide --output_npz and/or --output_csv.")

    print(f"Videos exported: {len(out_names)} / {n_vids}")
    print(f"Missing in GT: {missing}")
    print(f"Frame mismatches: {frame_mismatch}")


if __name__ == "__main__":
    main()



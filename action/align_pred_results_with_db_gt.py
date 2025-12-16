#!/usr/bin/env python3
"""
Create an "aligned" results/ folder that contains:
  - pred.npy with keys: logits, pred_labels, gt_labels, video_names
  - action_segments.json with the correct video-name order

Why:
Some pipelines rely on `results/action_segments.json` for video names/order, but the original order can be wrong
depending on how indexing_files/test_cross_env.txt was generated. This script rebuilds the mapping using the DB-driven
dataset order (the same order used by vid_idx during inference) and attaches GT labels directly from the DB.

This avoids using `gt_action.npy` (external alignment), and produces perfectly aligned GT for export and evaluation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_pred(pred_npy: Path) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    obj = np.load(str(pred_npy), allow_pickle=True)
    if not isinstance(obj, np.ndarray) or obj.dtype != object or obj.shape != ():
        raise ValueError(f"Unexpected format for pred file: {pred_npy}. Expected 0-d object array.")
    d = obj.item()
    if not isinstance(d, dict) or "logits" not in d or "pred_labels" not in d:
        raise ValueError(f"Unexpected dict keys in pred file: {pred_npy}. Found keys: {list(d.keys())}")
    logits = d["logits"]
    pred_labels = d["pred_labels"]
    if len(logits) != len(pred_labels):
        raise ValueError(f"Pred mismatch: len(logits)={len(logits)} != len(pred_labels)={len(pred_labels)}")
    return pred_labels, logits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True, type=str, help="Extracted dataset root (contains ikea_annotation_db_full + indexing_files/)")
    p.add_argument("--pred_results_dir", required=True, type=str, help="Existing results dir containing pred.npy + action_segments.json")
    p.add_argument("--out_results_dir", required=True, type=str, help="Output dir to write aligned pred.npy + action_segments.json")
    p.add_argument("--camera", type=str, default="dev3")
    p.add_argument("--frame_skip", type=int, default=1)
    p.add_argument("--frames_per_clip", type=int, default=64)
    p.add_argument("--dataset_mode", type=str, default="vid", choices=["vid", "img"])
    p.add_argument("--input_type", type=str, default="depth", choices=["rgb", "depth"])
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    # Import dataset + utils from action/
    import sys
    sys.path.append(str(repo_root / "action"))
    from IKEAActionDataset import IKEAActionVideoClipDataset as Dataset  # type: ignore
    import utils  # type: ignore

    dataset_root = Path(args.dataset_root).resolve()
    pred_results_dir = Path(args.pred_results_dir).resolve()
    out_results_dir = Path(args.out_results_dir).resolve()
    out_results_dir.mkdir(parents=True, exist_ok=True)

    pred_npy = pred_results_dir / "pred.npy"
    if not pred_npy.exists():
        raise FileNotFoundError(pred_npy)

    pred_labels, pred_probs = _load_pred(pred_npy)

    # Build dataset in the same way inference did (but we won't iterate it / load frames).
    ds = Dataset(
        str(dataset_root),
        db_filename="ikea_annotation_db_full",
        test_filename="test_cross_env.txt",
        train_filename="train_cross_env.txt",
        transform=None,
        set="test",
        camera=args.camera,
        frame_skip=int(args.frame_skip),
        frames_per_clip=int(args.frames_per_clip),
        resize=None,
        mode=args.dataset_mode,
        input_type=args.input_type,
    )

    if len(pred_probs) != len(ds.video_set):
        raise ValueError(
            f"Video count mismatch: pred has {len(pred_probs)} videos but dataset has {len(ds.video_set)} videos. "
            f"This usually means you are not using the same dataset_root/indexing_files as inference."
        )

    # Derive aligned video names + aligned GT labels per video from DB labels.
    video_names: List[str] = []
    gt_labels_per_video: List[np.ndarray] = []
    probs_out: List[np.ndarray] = []
    pred_out: List[np.ndarray] = []

    for i, (video_full_path, label_onehot, n_frames) in enumerate(ds.video_set):
        # video_full_path is absolute; parse relative name as "<furniture>/<scan>"
        rel = Path(video_full_path).resolve().relative_to(dataset_root)
        if len(rel.parts) < 2:
            raise ValueError(f"Unexpected video path (cannot parse furniture/scan): {video_full_path}")
        name = os.path.join(rel.parts[0], rel.parts[1])
        video_names.append(name)

        # label_onehot: [C, n_frames]; derive per-frame class id
        gt = np.asarray(label_onehot)
        if gt.ndim != 2:
            raise ValueError(f"Unexpected GT label shape for {name}: {gt.shape}")
        gt_ids = np.argmax(gt, axis=0).astype(np.int64)
        gt_labels_per_video.append(gt_ids)

        probs = np.asarray(pred_probs[i])
        pred = np.asarray(pred_labels[i]).astype(np.int64)

        # Best-effort length alignment (should normally match exactly for frame_skip=1).
        T = int(min(len(gt_ids), probs.shape[0], pred.shape[0]))
        probs_out.append(probs[:T])
        pred_out.append(pred[:T])
        gt_labels_per_video[-1] = gt_ids[:T]

    # Write pred.npy with extra aligned info
    out_pred_npy = out_results_dir / "pred.npy"
    np.save(
        str(out_pred_npy),
        {
            "pred_labels": [np.asarray(x) for x in pred_out],
            "logits": [np.asarray(x) for x in probs_out],  # these are already softmax probs in this repo
            "gt_labels": [np.asarray(x) for x in gt_labels_per_video],
            "video_names": np.asarray(video_names, dtype=object),
        },
    )

    # Rebuild action_segments.json with the aligned video order
    out_action_segments_json = out_results_dir / "action_segments.json"
    utils.convert_frame_logits_to_segment_json(probs_out, str(out_action_segments_json), video_names, ds.action_list)

    print(f"Wrote aligned results to: {out_results_dir}")
    print(f"- pred.npy: {out_pred_npy}")
    print(f"- action_segments.json: {out_action_segments_json}")


if __name__ == "__main__":
    main()



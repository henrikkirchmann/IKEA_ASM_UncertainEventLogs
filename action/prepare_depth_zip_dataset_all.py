"""
Prepare an extracted IKEA ASM *depth-video* dataset folder for inference on "all videos".

The depth zip typically contains:
  <dataset_root>/Lack_TV_Bench/<scan>/dev3/depth/scan_video.avi
  ... (similar for other furniture)

Inference code in this repo expects the dataset root to also contain:
  - ikea_annotation_db_full (sqlite database)
  - indexing_files/
      atomic_action_list.txt
      action_object_relation_list.txt
      train_cross_env.txt
      test_cross_env.txt

This script:
  1) copies the sqlite DB into the extracted dataset root
  2) generates class lists (atomic_action_list + action_object_relation_list) using the same logic as the repo's toolbox
  3) writes test_cross_env.txt containing *all scans that exist on disk* (depth/scan_video.avi present)
     and writes an empty train_cross_env.txt

After running this, you can run the depth I3D test script over all videos by pointing --dataset_path to the extracted root.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def _reconstruct_action_lists(db_path: Path, include_threshold: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Mirrors toolbox/run_statistics_and_generate_class_list.py logic:
      - expands compound atomic actions ("...") across objects
      - counts annotation occurrences
      - keeps classes above include_threshold
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # atomic actions (first row is blank/NA-like; dropped in toolbox)
    cur.execute("SELECT * FROM atomic_actions")
    actions = [r["atomic_action"] for r in cur.fetchall()]
    if actions:
        actions.pop(0)

    # objects (first row is blank; dropped in toolbox)
    cur.execute("SELECT * FROM objects")
    objects = [r["object"] for r in cur.fetchall()]
    if objects:
        objects.pop(0)

    # expand actions into final class list (excluding NA; model/dataset inserts NA separately)
    compound_actions_idx = []
    final_actions: List[str] = []
    relation: List[Tuple[int, int]] = []
    counter: Dict[str, int] = {}

    for i, act in enumerate(actions):
        atomic_action_id = i + 2  # sqlite table ids start at 1; toolbox used +2 after popping first row
        if "..." in act:
            compound_actions_idx.append(atomic_action_id)
            for j, obj in enumerate(objects):
                object_id = j + 2  # +2 after popping first object
                compound = act[:-4].strip() + " " + obj
                final_actions.append(compound)
                counter[compound] = 0
                relation.append((atomic_action_id, object_id))
        else:
            final_actions.append(act)
            counter[act] = 0
            relation.append((atomic_action_id, 1))  # object_id=1 means blank object row

    # count occurrences from annotations
    cur.execute("SELECT * FROM annotations")
    for row in cur.fetchall():
        if row["atomic_action_id"] in compound_actions_idx:
            compound = row["action_description"][:-4].strip() + " " + row["object_name"]
            if compound in counter:
                counter[compound] += 1
        else:
            act = row["action_description"]
            if act in counter:
                counter[act] += 1

    # threshold filter
    kept_actions: List[str] = []
    kept_relation: List[Tuple[int, int]] = []
    for i, act in enumerate(final_actions):
        if counter.get(act, 0) > include_threshold:
            kept_actions.append(act)
            kept_relation.append(relation[i])

    con.close()
    return kept_actions, kept_relation


def _scan_names_from_db(db_path: Path, camera: str) -> List[str]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT video_path, video_name FROM videos WHERE camera = ?", (camera,))
    out: List[str] = []
    for row in cur.fetchall():
        video_path = row["video_path"]
        video_name = row["video_name"] or ""
        if "special" in video_name:
            continue
        # Convert "Furniture/Scan/dev3/images" -> "Furniture/Scan"
        scan = video_path.split("dev", 1)[0].rstrip("/").rstrip("\\")
        out.append(scan)
    con.close()
    # stable order
    return sorted(set(out))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True, type=str, help="Extracted dataset root (contains furniture folders)")
    p.add_argument("--db_src", required=True, type=str, help="Path to source sqlite DB (ikea_annotation_db_full)")
    p.add_argument("--camera", type=str, default="dev3", help="Camera/view to target (default: dev3)")
    p.add_argument("--include_threshold", type=int, default=20, help="Minimum count to include an action class (default: 20)")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    db_src = Path(args.db_src).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(dataset_root)
    if not db_src.exists():
        raise FileNotFoundError(db_src)

    # Place DB at dataset root (expected by many scripts)
    db_dst = dataset_root / "ikea_annotation_db_full"
    db_dst.write_bytes(db_src.read_bytes())

    indexing_dir = dataset_root / "indexing_files"
    indexing_dir.mkdir(parents=True, exist_ok=True)

    # Create class lists
    kept_actions, kept_relation = _reconstruct_action_lists(db_dst, include_threshold=args.include_threshold)
    (indexing_dir / "atomic_action_list.txt").write_text("\n".join(kept_actions) + ("\n" if kept_actions else ""))
    (indexing_dir / "action_object_relation_list.txt").write_text(
        "\n".join([f"{a} {o}" for (a, o) in kept_relation]) + ("\n" if kept_relation else "")
    )

    # Create all-videos list for this camera + depth modality (existence check)
    scan_names = _scan_names_from_db(db_dst, camera=args.camera)
    existing: List[str] = []
    missing: List[str] = []
    for scan in scan_names:
        avi = dataset_root / scan / args.camera / "depth" / "scan_video.avi"
        if avi.exists():
            existing.append(scan)
        else:
            missing.append(scan)

    (indexing_dir / "test_cross_env.txt").write_text("\n".join(existing) + ("\n" if existing else ""))
    (indexing_dir / "train_cross_env.txt").write_text("")  # empty: we run inference in "test" mode over all

    print(f"Prepared: {dataset_root}")
    print(f"- DB: {db_dst}")
    print(f"- indexing_files/: {indexing_dir}")
    print(f"- all videos (depth): {len(existing)}")
    if missing:
        print(f"- missing depth videos (ignored): {len(missing)}")


if __name__ == "__main__":
    main()





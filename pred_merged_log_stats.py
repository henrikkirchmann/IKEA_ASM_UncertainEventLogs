#!/usr/bin/env python3
"""
Compute control-flow + uncertainty statistics for pred-merged uncertain XES logs.

This is meant for logs produced by:
  - action/aggregate_consecutive_pred_segments.py
  - action/export_uncertain_xes_from_segments.py --activity_source pred --include_gt_attrs False

Key behavior differences vs `uncertain_log_stats.py`:
- Trace/control-flow statistics ignore NA events by default because NA means "no action" and should
  not contribute to trace length / variants / alphabet.
- No accuracy metrics are computed (pred-merged logs intentionally omit GT).
- Adds "number of possibilities" metrics from probs_json:
  - avg_num_possibilities: average support size |{a : p(a) > 0}|
  - avg_num_possibilities_5pct: average count of labels with p(a) >= 0.05
- Uncertainty: average normalized Shannon entropy over events:
    H_norm(p) = -sum p(a) log p(a) / log |A|
  where A is the set of labels with p(a) > 0 (support) and p is renormalized on A.

We parse XES as XML directly (no pm4py dependency).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

XES_NS = "http://www.xes-standard.org/"


def _ns(tag: str) -> str:
    return f"{{{XES_NS}}}{tag}"


def _iter_xes_files(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path does not exist: {input_path}")
    pattern = "**/*.xes" if recursive else "*.xes"
    files = sorted(input_path.glob(pattern))
    if not files:
        raise ValueError(f"No .xes files found under: {input_path} (recursive={recursive})")
    return files


def _read_event_attributes(event_elem: ET.Element) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for child in event_elem:
        key = child.attrib.get("key")
        if not key:
            continue
        if "value" in child.attrib:
            attrs[key] = child.attrib["value"]
    return attrs


def _parse_probs_json(s: str) -> Optional[Dict[str, float]]:
    try:
        d = json.loads(s)
    except Exception:
        return None
    if not isinstance(d, dict) or not d:
        return None
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            x = float(v)
        except Exception:
            continue
        if not math.isfinite(x) or x < 0:
            continue
        out[str(k)] = x
    return out if out else None


def _support_stats(probs: Dict[str, float], threshold: float) -> Tuple[int, float, float]:
    """
    Returns:
      - support_size: count of labels with p >= threshold (after cleaning, before renorm)
      - entropy_norm: normalized Shannon entropy computed on that support (renormalized)
      - top1_conf: max probability on that support (renormalized)

    If the filtered support is empty or has zero total mass, returns (0, 0.0, 0.0).
    """
    items = [(k, float(v)) for k, v in probs.items() if float(v) >= threshold]
    if not items:
        return (0, 0.0, 0.0)
    total = sum(v for _, v in items)
    if total <= 0:
        return (0, 0.0, 0.0)
    p = [v / total for _, v in items if v > 0]
    if not p:
        return (0, 0.0, 0.0)
    k = len(p)
    top1 = max(p)
    if k <= 1:
        return (k, 0.0, float(top1))
    h = 0.0
    for pi in p:
        h -= pi * math.log(pi)
    h_max = math.log(k)
    return (k, float(h / h_max) if h_max > 0 else 0.0, float(top1))


@dataclass
class PredMergedLogStats:
    log_name: str
    # control-flow stats (computed on concept:name with NA removed)
    unique_activities_excl_na: int
    num_traces: int
    num_trace_variants_excl_na: int
    ratio_trace_variants_excl_na: float
    min_trace_length_excl_na: int
    avg_trace_length_excl_na: float
    max_trace_length_excl_na: int
    # event counts
    num_events_total: int
    num_events_na: int
    num_events_with_probs: int
    # possibilities
    avg_num_possibilities: float
    avg_num_possibilities_5pct: float
    # uncertainty
    avg_uncertainty: float
    avg_top1_confidence: float


def compute_pred_merged_stats(
    xes_path: Path,
    na_label: str,
    probs_key: str,
    ignore_na_in_control_flow: bool,
    threshold_5pct: float,
) -> PredMergedLogStats:
    unique_acts: Set[str] = set()
    trace_variants: Set[str] = set()
    trace_lengths: List[int] = []

    num_events_total = 0
    num_events_na = 0

    num_events_with_probs = 0
    sum_support = 0.0
    sum_support_5 = 0.0
    sum_uncertainty = 0.0
    sum_top1 = 0.0

    current_trace_activities: List[str] = []

    context = ET.iterparse(str(xes_path), events=("start", "end"))
    for ev, elem in context:
        if ev == "end" and elem.tag == _ns("event"):
            num_events_total += 1
            attrs = _read_event_attributes(elem)

            act = attrs.get("concept:name")
            if act == na_label:
                num_events_na += 1

            # Control-flow sequence: ignore NA (and also ignore missing activity labels)
            if act:
                if ignore_na_in_control_flow and act == na_label:
                    pass
                else:
                    unique_acts.add(act)
                    current_trace_activities.append(act)

            probs_json = attrs.get(probs_key)
            if probs_json:
                probs = _parse_probs_json(probs_json)
                if probs:
                    # Support size on p>0
                    k, h_norm, top1 = _support_stats(probs, threshold=1e-15)
                    # Support size with 5% threshold (count only)
                    k5, _, _ = _support_stats(probs, threshold=threshold_5pct)

                    if k > 0:
                        num_events_with_probs += 1
                        sum_support += float(k)
                        sum_support_5 += float(k5)
                        sum_uncertainty += float(h_norm)
                        sum_top1 += float(top1)

            elem.clear()

        elif ev == "end" and elem.tag == _ns("trace"):
            trace_len = len(current_trace_activities)
            trace_lengths.append(trace_len)
            variant_key = "\x1f".join(current_trace_activities)
            trace_variants.add(variant_key)
            current_trace_activities = []
            elem.clear()

    num_traces = len(trace_lengths)
    if num_traces == 0:
        min_len = 0
        max_len = 0
        avg_len = 0.0
    else:
        min_len = int(min(trace_lengths))
        max_len = int(max(trace_lengths))
        avg_len = float(sum(trace_lengths) / num_traces)

    num_trace_variants = len(trace_variants)
    ratio_trace_variants = float(num_trace_variants / num_traces) if num_traces > 0 else 0.0

    avg_support = float(sum_support / num_events_with_probs) if num_events_with_probs > 0 else 0.0
    avg_support_5 = float(sum_support_5 / num_events_with_probs) if num_events_with_probs > 0 else 0.0
    avg_uncertainty = float(sum_uncertainty / num_events_with_probs) if num_events_with_probs > 0 else 0.0
    avg_top1 = float(sum_top1 / num_events_with_probs) if num_events_with_probs > 0 else 0.0

    log_name = xes_path.name[:-4] if xes_path.name.endswith(".xes") else xes_path.name

    return PredMergedLogStats(
        log_name=log_name,
        unique_activities_excl_na=len(unique_acts),
        num_traces=num_traces,
        num_trace_variants_excl_na=num_trace_variants,
        ratio_trace_variants_excl_na=ratio_trace_variants,
        min_trace_length_excl_na=min_len,
        avg_trace_length_excl_na=avg_len,
        max_trace_length_excl_na=max_len,
        num_events_total=num_events_total,
        num_events_na=num_events_na,
        num_events_with_probs=num_events_with_probs,
        avg_num_possibilities=avg_support,
        avg_num_possibilities_5pct=avg_support_5,
        avg_uncertainty=avg_uncertainty,
        avg_top1_confidence=avg_top1,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compute stats for pred-merged uncertain XES logs.")
    p.add_argument("--input", required=True, type=str, help="XES file OR directory containing XES files.")
    p.add_argument("--recursive", action="store_true", help="If --input is a directory, search recursively.")
    p.add_argument("--output_csv", type=str, default=None, help="Where to write CSV (default: <input_dir>/log_stats_pred_merged.csv).")
    p.add_argument("--na_label", type=str, default="NA", help="Label treated as 'no action' (default: NA).")
    p.add_argument(
        "--ignore_na_in_control_flow",
        action="store_true",
        help="If set, NA events do not contribute to trace length/variants/alphabet (recommended).",
    )
    p.add_argument("--probs_key", type=str, default="probs_json", help="Event attribute key containing probability distribution JSON.")
    p.add_argument(
        "--threshold_5pct",
        type=float,
        default=0.05,
        help="Probability threshold used for avg_num_possibilities_5pct (default: 0.05).",
    )
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    xes_files = _iter_xes_files(input_path, recursive=bool(args.recursive))

    if args.output_csv:
        out_csv = Path(args.output_csv).expanduser().resolve()
    else:
        out_csv = (input_path / "log_stats_pred_merged.csv") if input_path.is_dir() else (Path.cwd() / "log_stats_pred_merged.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
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
        for i, x in enumerate(xes_files, start=1):
            print(f"[{i}/{len(xes_files)}] Processing {x}")
            s = compute_pred_merged_stats(
                x,
                na_label=str(args.na_label),
                probs_key=str(args.probs_key),
                ignore_na_in_control_flow=bool(args.ignore_na_in_control_flow),
                threshold_5pct=float(args.threshold_5pct),
            )
            w.writerow(
                [
                    s.log_name,
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






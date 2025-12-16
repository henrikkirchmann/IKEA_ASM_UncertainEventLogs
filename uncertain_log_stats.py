#!/usr/bin/env python3
"""
Compute summary statistics for one or more XES logs exported by this repo.

Adds:
- event-level accuracy: ratio of correctly predicted actions per event (gt:label vs pred:label)
- correctly predicted "no activity" (NA) ratio
- uncertainty score in [0,1]: normalized entropy of probs_json (1 = very uncertain, 0 = very certain)

Expected per-event attributes in XES (see: action/export_uncertain_xes_from_segments.py):
- gt:label (string)
- pred:label (string)
- probs_json (stringified JSON dict: {label: prob, ...})

Notes:
- We parse XES as XML directly (no pm4py dependency).
- If an event has no concept:name, it is excluded from control-flow stats (trace variants, alphabet).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

XES_NS = "http://www.xes-standard.org/"


def _ns(tag: str) -> str:
    return f"{{{XES_NS}}}{tag}"


@dataclass
class LogStats:
    log_name: str
    unique_activities: int
    num_traces: int
    num_trace_variants: int
    ratio_trace_variants: float
    min_trace_length: int
    avg_trace_length: float
    max_trace_length: int
    num_events_total: int
    num_events_labeled: int
    event_accuracy: float
    num_events_labeled_excl_na: int
    event_accuracy_excl_na: float
    num_events_with_probs: int
    avg_uncertainty: float
    avg_top1_confidence: float
    num_events_gt_na: int
    num_events_pred_na: int
    num_events_correct_na: int
    ratio_correct_gt_na: float
    ratio_correct_pred_na: float


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
    """
    Parse XES event attribute elements like:
      <string key="gt:label" value="..." />
      <int key="segment:start_timestamp" value="0" />
    into a {key: value_str} dict.
    """
    attrs: Dict[str, str] = {}
    for child in event_elem:
        key = child.attrib.get("key")
        if not key:
            continue
        # XES uses "value" attribute for all primitive attributes.
        if "value" in child.attrib:
            attrs[key] = child.attrib["value"]
    return attrs


def _normalized_entropy_uncertainty(probs: Dict[str, float]) -> Optional[Tuple[float, float]]:
    """
    Returns (uncertainty, top1_confidence) where:
    - uncertainty in [0,1] via normalized Shannon entropy: H(p)/log(k)
    - top1_confidence = max(p_i)
    """
    cleaned: List[float] = []
    for v in probs.values():
        try:
            x = float(v)
        except Exception:
            continue
        if not math.isfinite(x) or x < 0:
            continue
        cleaned.append(x)

    if not cleaned:
        return None

    total = sum(cleaned)
    if total <= 0:
        return None

    p = [x / total for x in cleaned]
    k = len(p)
    top1 = max(p)
    if k <= 1:
        return (0.0, float(top1))

    h = 0.0
    for pi in p:
        if pi > 0:
            h -= pi * math.log(pi)
    h_max = math.log(k)
    if h_max <= 0:
        return (0.0, float(top1))
    return (float(h / h_max), float(top1))


def compute_log_stats(
    xes_path: Path,
    na_label: str,
    include_na_in_accuracy: bool,
    probs_key: str,
) -> LogStats:
    # Streaming parse to keep memory manageable.
    # We'll collect trace-level control-flow sequence (concept:name) and event-level metrics.
    unique_acts: Set[str] = set()
    trace_variants: Set[str] = set()
    trace_lengths: List[int] = []

    num_events_total = 0
    num_events_labeled = 0
    num_events_correct = 0
    num_events_labeled_excl_na = 0
    num_events_correct_excl_na = 0
    num_events_with_probs = 0
    sum_uncertainty = 0.0
    sum_top1 = 0.0
    num_events_gt_na = 0
    num_events_pred_na = 0
    num_events_correct_na = 0

    current_trace_activities: List[str] = []

    # We iterate over end events so we can clear elements.
    context = ET.iterparse(str(xes_path), events=("start", "end"))
    for ev, elem in context:
        if ev == "end" and elem.tag == _ns("event"):
            num_events_total += 1
            attrs = _read_event_attributes(elem)

            # Control-flow activity label (may be absent for NA if exporter used omit_concept_name)
            act = attrs.get("concept:name")
            if act:
                unique_acts.add(act)
                current_trace_activities.append(act)

            gt = attrs.get("gt:label")
            pred = attrs.get("pred:label")
            if gt is not None and pred is not None:
                is_na = (gt == na_label)
                if is_na:
                    num_events_gt_na += 1
                if pred == na_label:
                    num_events_pred_na += 1
                if is_na and pred == na_label:
                    num_events_correct_na += 1
                if include_na_in_accuracy or not is_na:
                    num_events_labeled += 1
                    if gt == pred:
                        num_events_correct += 1
                if not is_na:
                    num_events_labeled_excl_na += 1
                    if gt == pred:
                        num_events_correct_excl_na += 1

            probs_json = attrs.get(probs_key)
            if probs_json:
                try:
                    d = json.loads(probs_json)
                    if isinstance(d, dict) and d:
                        ue = _normalized_entropy_uncertainty(d)  # type: ignore[arg-type]
                        if ue is not None:
                            u, top1 = ue
                            num_events_with_probs += 1
                            sum_uncertainty += u
                            sum_top1 += top1
                except Exception:
                    # Ignore malformed probability JSON
                    pass

            elem.clear()

        elif ev == "end" and elem.tag == _ns("trace"):
            # Finalize this trace.
            trace_len = len(current_trace_activities)
            trace_lengths.append(trace_len)

            # Variant key: delimiter unlikely to occur in labels.
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

    event_accuracy = float(num_events_correct / num_events_labeled) if num_events_labeled > 0 else 0.0
    event_accuracy_excl_na = (
        float(num_events_correct_excl_na / num_events_labeled_excl_na) if num_events_labeled_excl_na > 0 else 0.0
    )
    avg_uncertainty = float(sum_uncertainty / num_events_with_probs) if num_events_with_probs > 0 else 0.0
    avg_top1 = float(sum_top1 / num_events_with_probs) if num_events_with_probs > 0 else 0.0
    ratio_correct_gt_na = float(num_events_correct_na / num_events_gt_na) if num_events_gt_na > 0 else 0.0
    ratio_correct_pred_na = float(num_events_correct_na / num_events_pred_na) if num_events_pred_na > 0 else 0.0

    # Derive a friendly log name.
    log_name = xes_path.name
    if log_name.endswith(".xes"):
        log_name = log_name[:-4]

    return LogStats(
        log_name=log_name,
        unique_activities=len(unique_acts),
        num_traces=num_traces,
        num_trace_variants=num_trace_variants,
        ratio_trace_variants=ratio_trace_variants,
        min_trace_length=min_len,
        avg_trace_length=avg_len,
        max_trace_length=max_len,
        num_events_total=num_events_total,
        num_events_labeled=num_events_labeled,
        event_accuracy=event_accuracy,
        num_events_labeled_excl_na=num_events_labeled_excl_na,
        event_accuracy_excl_na=event_accuracy_excl_na,
        num_events_with_probs=num_events_with_probs,
        avg_uncertainty=avg_uncertainty,
        avg_top1_confidence=avg_top1,
        num_events_gt_na=num_events_gt_na,
        num_events_pred_na=num_events_pred_na,
        num_events_correct_na=num_events_correct_na,
        ratio_correct_gt_na=ratio_correct_gt_na,
        ratio_correct_pred_na=ratio_correct_pred_na,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compute stats for XES logs (including event accuracy + uncertainty).")
    p.add_argument(
        "--input",
        required=True,
        type=str,
        help="XES file OR directory containing XES files.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="If --input is a directory, search for .xes files recursively.",
    )
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to write CSV (default: <input_dir>/log_stats.csv or ./log_stats.csv if input is a file).",
    )
    p.add_argument(
        "--na_label",
        type=str,
        default="NA",
        help="Label treated as 'no action' (default: NA).",
    )
    p.add_argument(
        "--include_na_in_accuracy",
        action="store_true",
        help="If set, include NA events in event_accuracy (default: excluded).",
    )
    p.add_argument(
        "--probs_key",
        type=str,
        default="probs_json",
        help="Event attribute key containing probability distribution JSON (default: probs_json).",
    )
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    xes_files = _iter_xes_files(input_path, recursive=bool(args.recursive))

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
    else:
        if input_path.is_dir():
            output_csv = input_path / "log_stats.csv"
        else:
            output_csv = Path.cwd() / "log_stats.csv"

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "log_name",
                "unique_activities",
                "num_traces",
                "num_trace_variants",
                "ratio_trace_variants",
                "min_trace_length",
                "avg_trace_length",
                "max_trace_length",
                "num_events_total",
                "num_events_labeled",
                "event_accuracy",
                "num_events_labeled_excl_na",
                "event_accuracy_excl_na",
                "num_events_with_probs",
                "avg_uncertainty",  # 0=very certain, 1=very uncertain (normalized entropy)
                "avg_top1_confidence",  # 0..1 (higher means more confident)
                "num_events_gt_na",
                "num_events_pred_na",
                "num_events_correct_na",
                "ratio_correct_gt_na",  # P(pred=NA | gt=NA)
                "ratio_correct_pred_na",  # P(gt=NA | pred=NA)
            ]
        )

        for i, xes_path in enumerate(xes_files, start=1):
            print(f"[{i}/{len(xes_files)}] Processing {xes_path}")
            s = compute_log_stats(
                xes_path,
                na_label=str(args.na_label),
                include_na_in_accuracy=bool(args.include_na_in_accuracy),
                probs_key=str(args.probs_key),
            )
            writer.writerow(
                [
                    s.log_name,
                    s.unique_activities,
                    s.num_traces,
                    s.num_trace_variants,
                    s.ratio_trace_variants,
                    s.min_trace_length,
                    s.avg_trace_length,
                    s.max_trace_length,
                    s.num_events_total,
                    s.num_events_labeled,
                    s.event_accuracy,
                    s.num_events_labeled_excl_na,
                    s.event_accuracy_excl_na,
                    s.num_events_with_probs,
                    s.avg_uncertainty,
                    s.avg_top1_confidence,
                    s.num_events_gt_na,
                    s.num_events_pred_na,
                    s.num_events_correct_na,
                    s.ratio_correct_gt_na,
                    s.ratio_correct_pred_na,
                ]
            )

    print(f"Log statistics saved to {output_csv}")


if __name__ == "__main__":
    main()

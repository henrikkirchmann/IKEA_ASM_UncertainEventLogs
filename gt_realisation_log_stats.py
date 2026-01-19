#!/usr/bin/env python3
"""
Compute control-flow statistics for the *deterministic* ground-truth realisation XES log.

This is intended for:
  paper_event_logs/ikea_asm/split=test/gt_realisation/ikea_asm__test__gt_realisation__keep_na.xes
and the corresponding no-NA variant.

We parse XES as XML directly (no pm4py dependency).
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set
import xml.etree.ElementTree as ET

XES_NS = "http://www.xes-standard.org/"


def _ns(tag: str) -> str:
    return f"{{{XES_NS}}}{tag}"


def _read_event_attributes(event_elem: ET.Element) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for child in event_elem:
        key = child.attrib.get("key")
        if not key:
            continue
        if "value" in child.attrib:
            attrs[key] = child.attrib["value"]
    return attrs


@dataclass
class GtRealisationXesStats:
    log_name: str
    unique_activities_excl_na: int
    num_traces: int
    num_trace_variants_excl_na: int
    ratio_trace_variants_excl_na: float
    min_trace_length_excl_na: int
    avg_trace_length_excl_na: float
    max_trace_length_excl_na: int
    num_events_total: int
    num_events_na: int


def compute_gt_realisation_xes_stats(
    xes_path: Path,
    na_label: str,
    ignore_na_in_control_flow: bool,
) -> GtRealisationXesStats:
    unique_acts: Set[str] = set()
    trace_variants: Set[str] = set()
    trace_lengths: List[int] = []

    num_events_total = 0
    num_events_na = 0
    current_trace_activities: List[str] = []

    context = ET.iterparse(str(xes_path), events=("start", "end"))
    for ev, elem in context:
        if ev == "end" and elem.tag == _ns("event"):
            num_events_total += 1
            attrs = _read_event_attributes(elem)

            act = attrs.get("concept:name")
            if act == na_label:
                num_events_na += 1

            if act:
                if ignore_na_in_control_flow and act == na_label:
                    pass
                else:
                    unique_acts.add(act)
                    current_trace_activities.append(act)

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

    return GtRealisationXesStats(
        log_name=xes_path.stem,
        unique_activities_excl_na=len(unique_acts),
        num_traces=num_traces,
        num_trace_variants_excl_na=num_trace_variants,
        ratio_trace_variants_excl_na=ratio_trace_variants,
        min_trace_length_excl_na=min_len,
        avg_trace_length_excl_na=avg_len,
        max_trace_length_excl_na=max_len,
        num_events_total=num_events_total,
        num_events_na=num_events_na,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Compute stats for the deterministic GT-realisation XES log.")
    p.add_argument("--input", required=True, type=str, help="Path to a GT-realisation .xes file.")
    p.add_argument("--output_csv", type=str, default=None, help="Where to write CSV (default: <cwd>/log_stats_gt_realisation.csv).")
    p.add_argument("--na_label", type=str, default="NA")
    p.add_argument(
        "--ignore_na_in_control_flow",
        action="store_true",
        help="If set, NA events do not contribute to trace length/variants/alphabet (recommended for paper stats).",
    )
    args = p.parse_args()

    xes_path = Path(args.input).expanduser().resolve()
    if not xes_path.is_file():
        raise ValueError(f"Input is not a file: {xes_path}")

    out_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else (Path.cwd() / "log_stats_gt_realisation.csv")
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    s = compute_gt_realisation_xes_stats(
        xes_path,
        na_label=str(args.na_label),
        ignore_na_in_control_flow=bool(args.ignore_na_in_control_flow),
    )

    with out_csv.open("w", newline="") as f:
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
            ]
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
            ]
        )

    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()


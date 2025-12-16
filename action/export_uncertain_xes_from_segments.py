"""
Export XES event logs from segment-level CSVs with a per-event probability distribution attribute.

Unlike the Pegoraro et al. (2022) uncertain XES extension style (nested containers), this exporter
stores the full activity probability distribution as a single JSON string attribute on each event.
This makes later extraction straightforward (read one attribute, parse JSON).

Input CSV is expected to have one row per GT segment, produced by `aggregate_consecutive_gt_segments.py`,
with at least:
  - case_id (int)
  - case_name (str)
  - start_timestamp (int) or timestamp (int)
  - avg_probs_json (json object mapping action label -> probability)

Optional columns:
  - gt_label_name, pred_label_name, gt_label, pred_label, duration_frames, end_timestamp

Timestamps:
XES time:timestamp expects an ISO datetime; we map integer timestamps to seconds since epoch starting at
1970-01-01T00:00:00Z by default (configurable via --epoch_iso).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET


XES_NS = "http://www.xes-standard.org/"


def _ns(tag: str) -> str:
    return f"{{{XES_NS}}}{tag}"


def _iso_from_epoch_seconds(epoch: datetime, seconds: int) -> str:
    return (epoch + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _parse_epoch_iso(s: str) -> datetime:
    # Accept "1970-01-01T00:00:00Z" or with offset.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_probs_json(s: str) -> Dict[str, float]:
    d = json.loads(s)
    if not isinstance(d, dict) or not d:
        raise ValueError("avg_probs_json must be a non-empty JSON object")
    return {str(k): float(v) for k, v in d.items()}


def _iter_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("*.csv"))
    if not files:
        raise ValueError(f"No .csv files found in {input_path}")
    return files


def _add_string(parent: ET.Element, key: str, value: str) -> None:
    ET.SubElement(parent, _ns("string"), {"key": key, "value": value})


def _add_int(parent: ET.Element, key: str, value: int) -> None:
    ET.SubElement(parent, _ns("int"), {"key": key, "value": str(int(value))})


def _add_double(parent: ET.Element, key: str, value: float) -> None:
    ET.SubElement(parent, _ns("float"), {"key": key, "value": str(float(value))})


def _add_date(parent: ET.Element, key: str, iso: str) -> None:
    ET.SubElement(parent, _ns("date"), {"key": key, "value": iso})


def _most_likely_label(probs: Dict[str, float]) -> str:
    # argmax over probability distribution
    ml_label, _ = max(probs.items(), key=lambda kv: kv[1])
    return str(ml_label)


def _stable_probs_json(probs: Dict[str, float]) -> str:
    # Deterministic JSON (stable ordering) for reproducibility across runs.
    stable = {str(k): float(v) for k, v in probs.items()}
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def export_one_csv_to_xes(
    input_csv: Path,
    output_xes: Path,
    epoch: datetime,
    time_col: str,
    probs_col: str,
    probs_attr_key: str,
    na_label: str,
    na_handling: str,
) -> None:
    output_xes.parent.mkdir(parents=True, exist_ok=True)

    ET.register_namespace("", XES_NS)
    log = ET.Element(_ns("log"), {"xes.version": "1.0"})

    # Standard extensions (optional but helps tools)
    ET.SubElement(log, _ns("extension"), {"name": "Concept", "prefix": "concept", "uri": "http://www.xes-standard.org/concept.xesext"})
    ET.SubElement(log, _ns("extension"), {"name": "Time", "prefix": "time", "uri": "http://www.xes-standard.org/time.xesext"})

    # Classifiers help tools like Disco understand which attribute defines the activity name.
    # Without this, Disco will fall back to concept:name and show an import warning.
    #
    # XES standard: https://xes-standard.org/
    ET.SubElement(log, _ns("classifier"), {"name": "Activity", "keys": "concept:name"})
    ET.SubElement(log, _ns("classifier"), {"name": "Case", "keys": "case:name"})

    # Globals (optional) declare which attributes exist on traces/events.
    # Many tools tolerate missing globals, but having them improves compatibility.
    g_trace = ET.SubElement(log, _ns("global"), {"scope": "trace"})
    _add_string(g_trace, "concept:name", "__INVALID__")
    _add_string(g_trace, "case:name", "__INVALID__")
    g_event = ET.SubElement(log, _ns("global"), {"scope": "event"})
    _add_string(g_event, "concept:name", "__INVALID__")
    _add_string(g_event, "time:timestamp", "1970-01-01T00:00:00Z")

    _add_string(log, "concept:name", input_csv.stem)

    # Build traces grouped by case_id.
    traces: Dict[str, ET.Element] = {}

    with open(input_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{input_csv} has no header")
        required = {"case_id", "case_name", probs_col, time_col}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")
        # We always expect label names to be present in this pipeline.
        if "gt_label_name" not in set(reader.fieldnames):
            raise ValueError(f"{input_csv} is missing required column: gt_label_name")

        for row_idx, row in enumerate(reader):
            case_id = str(row["case_id"])
            case_name = row.get("case_name", "")
            ts = int(row[time_col])
            probs = _parse_probs_json(row[probs_col])

            trace = traces.get(case_id)
            if trace is None:
                trace = ET.SubElement(log, _ns("trace"))
                _add_string(trace, "concept:name", case_id)
                _add_string(trace, "case:name", case_name)
                traces[case_id] = trace

            event = ET.SubElement(trace, _ns("event"))

            # set "time:timestamp"
            _add_date(event, "time:timestamp", _iso_from_epoch_seconds(epoch, ts))

            # store optional info
            if "start_timestamp" in row:
                _add_int(event, "segment:start_timestamp", int(row["start_timestamp"]))
            if "end_timestamp" in row:
                _add_int(event, "segment:end_timestamp", int(row["end_timestamp"]))
            if "duration_frames" in row:
                _add_int(event, "segment:duration_frames", int(row["duration_frames"]))
            if "gt_label" in row:
                _add_int(event, "gt:label_id", int(row["gt_label"]))
            if "gt_label_name" in row and row["gt_label_name"]:
                _add_string(event, "gt:label", row["gt_label_name"])
            if "pred_label" in row:
                _add_int(event, "pred:label_id", int(row["pred_label"]))
            if "pred_label_name" in row and row["pred_label_name"]:
                _add_string(event, "pred:label", row["pred_label_name"])

            # Probability distribution over all activities (JSON string attribute)
            _add_string(event, probs_attr_key, _stable_probs_json(probs))

            # Most likely label from distribution (argmax)
            _add_string(event, "pred:most_likely", _most_likely_label(probs))

            # Deterministic activity label:
            # - By default we use GT label as concept:name.
            # - For NA ("no action") we can optionally omit concept:name to keep the event in the log
            #   (with probs_json etc.) but avoid treating it as an activity in some DFG tooling.
            gt_name = row["gt_label_name"]
            if gt_name == na_label:
                _add_string(event, "na:is_no_event", "true")
                if na_handling == "keep":
                    _add_string(event, "concept:name", gt_name)
                elif na_handling == "omit_concept_name":
                    # Intentionally omit concept:name
                    pass
                else:
                    raise ValueError(f"Unknown na_handling: {na_handling}")
            else:
                _add_string(event, "concept:name", gt_name)

    tree = ET.ElementTree(log)
    tree.write(output_xes, encoding="utf-8", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Input CSV file OR directory containing segment-level CSVs.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Directory to write .xes files into (one per input CSV).",
    )
    parser.add_argument(
        "--epoch_iso",
        type=str,
        default="1970-01-01T00:00:00Z",
        help="ISO datetime used as epoch for numeric timestamps (default: 1970-01-01T00:00:00Z).",
    )
    parser.add_argument(
        "--time_col",
        type=str,
        default="start_timestamp",
        help="Which column to use as timestamp (e.g. start_timestamp, timestamp).",
    )
    parser.add_argument(
        "--probs_col",
        type=str,
        default="avg_probs_json",
        help="Which column contains the probability distribution JSON (default: avg_probs_json).",
    )
    parser.add_argument(
        "--probs_attr_key",
        type=str,
        default="probs_json",
        help="Event attribute key used to store the full probability distribution JSON (default: probs_json).",
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
        help=(
            "How to encode NA events. "
            "'keep' writes concept:name=NA. "
            "'omit_concept_name' keeps the event (with probs_json) but omits concept:name for NA "
            "to reduce its impact on some DFG discovery tooling."
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    epoch = _parse_epoch_iso(args.epoch_iso)

    files = _iter_inputs(input_path)
    for i, f in enumerate(files, start=1):
        out = output_dir / f"{f.stem}.xes"
        print(f"[{i}/{len(files)}] {f.name} -> {out.name}")
        export_one_csv_to_xes(
            f,
            out,
            epoch=epoch,
            time_col=args.time_col,
            probs_col=args.probs_col,
            probs_attr_key=args.probs_attr_key,
            na_label=args.na_label,
            na_handling=args.na_handling,
        )
    print("Done.")


if __name__ == "__main__":
    main()



"""
Export a deterministic *ground-truth realisation* XES log from a GT-segment CSV.

Input:
  A single segments CSV (merge consecutive GT labels), e.g. the `*_gt_segments.csv` produced by
  `action/aggregate_consecutive_gt_segments.py` / `action/run_pretrained_export_pipeline.py`.

Output:
  A single XES log where each segment becomes one event and:
    - concept:name := gt_label_name
    - no probs_json is stored (this is a *certain* log)
    - NA events are marked via na:is_no_event=true and can optionally omit concept:name

This is intended for the paper's "Certain Groundtruth" reference log.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

XES_NS = "http://www.xes-standard.org/"


def _ns(tag: str) -> str:
    return f"{{{XES_NS}}}{tag}"


def _parse_epoch_iso(s: str) -> datetime:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso_from_epoch_seconds(epoch: datetime, seconds: int) -> str:
    return (epoch + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def _add_string(parent: ET.Element, key: str, value: str) -> None:
    ET.SubElement(parent, _ns("string"), {"key": key, "value": value})


def _add_int(parent: ET.Element, key: str, value: int) -> None:
    ET.SubElement(parent, _ns("int"), {"key": key, "value": str(int(value))})


def _add_date(parent: ET.Element, key: str, iso: str) -> None:
    ET.SubElement(parent, _ns("date"), {"key": key, "value": iso})


def export_one_csv_to_xes(
    input_csv: Path,
    output_xes: Path,
    *,
    epoch: datetime,
    time_col: str,
    na_label: str,
    na_handling: str,
) -> None:
    output_xes.parent.mkdir(parents=True, exist_ok=True)

    ET.register_namespace("", XES_NS)
    log = ET.Element(_ns("log"), {"xes.version": "1.0"})

    ET.SubElement(log, _ns("extension"), {"name": "Concept", "prefix": "concept", "uri": "http://www.xes-standard.org/concept.xesext"})
    ET.SubElement(log, _ns("extension"), {"name": "Time", "prefix": "time", "uri": "http://www.xes-standard.org/time.xesext"})
    ET.SubElement(log, _ns("classifier"), {"name": "Activity", "keys": "concept:name"})
    ET.SubElement(log, _ns("classifier"), {"name": "Case", "keys": "case:name"})

    g_trace = ET.SubElement(log, _ns("global"), {"scope": "trace"})
    _add_string(g_trace, "concept:name", "__INVALID__")
    _add_string(g_trace, "case:name", "__INVALID__")
    g_event = ET.SubElement(log, _ns("global"), {"scope": "event"})
    _add_string(g_event, "concept:name", "__INVALID__")
    _add_string(g_event, "time:timestamp", "1970-01-01T00:00:00Z")

    _add_string(log, "concept:name", input_csv.stem)

    traces: dict[str, ET.Element] = {}

    with input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{input_csv} has no header")

        required = {"case_id", "case_name", "gt_label_name", time_col}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{input_csv} is missing required columns: {sorted(missing)}")

        for row in reader:
            case_id = str(row["case_id"])
            case_name = row.get("case_name", "")
            ts = int(row[time_col])
            gt_name = str(row["gt_label_name"])

            trace = traces.get(case_id)
            if trace is None:
                trace = ET.SubElement(log, _ns("trace"))
                _add_string(trace, "concept:name", case_id)
                _add_string(trace, "case:name", case_name)
                traces[case_id] = trace

            event = ET.SubElement(trace, _ns("event"))
            _add_date(event, "time:timestamp", _iso_from_epoch_seconds(epoch, ts))

            if "start_timestamp" in row and row["start_timestamp"] != "":
                _add_int(event, "segment:start_timestamp", int(row["start_timestamp"]))
            if "end_timestamp" in row and row["end_timestamp"] != "":
                _add_int(event, "segment:end_timestamp", int(row["end_timestamp"]))
            if "duration_frames" in row and row["duration_frames"] != "":
                _add_int(event, "segment:duration_frames", int(row["duration_frames"]))

            if gt_name == na_label:
                _add_string(event, "na:is_no_event", "true")
                if na_handling == "keep":
                    _add_string(event, "concept:name", gt_name)
                elif na_handling == "omit_concept_name":
                    pass
                else:
                    raise ValueError(f"Unknown na_handling: {na_handling}")
            else:
                _add_string(event, "concept:name", gt_name)

    tree = ET.ElementTree(log)
    tree.write(output_xes, encoding="utf-8", xml_declaration=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True, type=str, help="Path to a *_gt_segments.csv file.")
    p.add_argument("--output_xes", required=True, type=str, help="Path to output XES file.")
    p.add_argument("--time_col", type=str, default="start_timestamp", help="Column used for time:timestamp (default: start_timestamp).")
    p.add_argument("--epoch_iso", type=str, default="1970-01-01T00:00:00Z")
    p.add_argument("--na_label", type=str, default="NA")
    p.add_argument("--na_handling", type=str, default="keep", choices=["keep", "omit_concept_name"])
    args = p.parse_args()

    export_one_csv_to_xes(
        input_csv=Path(args.input_csv).resolve(),
        output_xes=Path(args.output_xes).resolve(),
        epoch=_parse_epoch_iso(args.epoch_iso),
        time_col=args.time_col,
        na_label=args.na_label,
        na_handling=args.na_handling,
    )


if __name__ == "__main__":
    main()


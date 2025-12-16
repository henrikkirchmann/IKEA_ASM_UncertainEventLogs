#!/usr/bin/env python3
"""
Filter XES logs exported by this repo to remove NA ("no activity") events.

This is useful when you want a process-mining log that only contains "real" activities.

How we detect NA events (any of the following):
  - string attribute key="gt:label" value == <na_label>   (default: "NA")
  - string attribute key="na:is_no_event" value == "true" (written by exporter for NA events)
  - missing concept:name AND gt:label == <na_label>       (covers na_handling=omit_concept_name)

By default, traces that become empty after filtering are removed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

XES_NS = "http://www.xes-standard.org/"


def _ns(tag: str) -> str:
    return f"{{{XES_NS}}}{tag}"


def _event_attr_map(event_elem: ET.Element) -> dict[str, str]:
    d: dict[str, str] = {}
    for child in event_elem:
        key = child.attrib.get("key")
        if not key:
            continue
        if "value" in child.attrib:
            d[key] = child.attrib["value"]
    return d


def _is_na_event(attrs: dict[str, str], na_label: str) -> bool:
    if attrs.get("na:is_no_event", "").lower() == "true":
        return True
    gt = attrs.get("gt:label")
    if gt == na_label:
        # If the exporter used omit_concept_name for NA, concept:name may be missing.
        return True
    # Fallback: concept:name explicitly set to NA (na_handling=keep)
    if attrs.get("concept:name") == na_label:
        return True
    return False


def _stable_probs_json(d: dict[str, float]) -> str:
    return json.dumps(d, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _renormalize_probs_excluding_na(
    probs: dict[str, float],
    na_label: str,
    *,
    context: str,
) -> dict[str, float]:
    """
    Remove NA from the distribution and renormalize remaining mass to sum to 1.

    Concretely:
      1) Drop the NA entry (key == na_label)
      2) Drop invalid entries (non-finite / negative)
      3) Let S = sum(remaining probs). Return p_i' = p_i / S so that sum(p_i') == 1

    If S <= 0 after filtering, this indicates the distribution is unusable for a non-NA event. We raise an error
    (instead of silently falling back) because downstream tooling expects a proper probability distribution.
    """
    cleaned: dict[str, float] = {}
    for k, v in probs.items():
        if str(k) == na_label:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if not math.isfinite(x) or x < 0:
            continue
        cleaned[str(k)] = x

    if not cleaned:
        raise ValueError(
            "Cannot renormalize probs_json after removing NA: no non-NA keys remain.\n"
            f"Context: {context}\n"
            f"Original keys (sample): {list(probs.keys())[:10]}"
        )

    s = sum(cleaned.values())
    if s > 0:
        return {k: float(v / s) for k, v in cleaned.items()}

    # If all remaining mass is zero, this is a hard error: we cannot form a valid distribution.
    na_mass = probs.get(na_label, None)
    raise ValueError(
        "Cannot renormalize probs_json after removing NA: remaining probability mass is 0.\n"
        f"Context: {context}\n"
        f"NA mass (raw): {na_mass}\n"
        f"Non-NA key count: {len(cleaned)}\n"
        f"Non-NA mass sum: {s}\n"
        f"Non-NA keys (sample): {list(cleaned.keys())[:10]}"
    )


def _get_event_attr_elem(event_elem: ET.Element, key: str) -> ET.Element | None:
    for child in event_elem:
        if child.attrib.get("key") == key:
            return child
    return None


def _set_event_string_attr(event_elem: ET.Element, key: str, value: str) -> None:
    el = _get_event_attr_elem(event_elem, key)
    if el is None:
        # Add new string attribute if missing
        ET.SubElement(event_elem, _ns("string"), {"key": key, "value": value})
        return
    el.tag = _ns("string")
    el.attrib["value"] = value


def _set_event_int_attr(event_elem: ET.Element, key: str, value: int) -> None:
    el = _get_event_attr_elem(event_elem, key)
    if el is None:
        ET.SubElement(event_elem, _ns("int"), {"key": key, "value": str(int(value))})
        return
    el.tag = _ns("int")
    el.attrib["value"] = str(int(value))


def _infer_label_to_id(root: ET.Element) -> dict[str, int]:
    """
    Infer label->id mapping from existing event attributes:
      - gt:label (string) + gt:label_id (int)
      - pred:label (string) + pred:label_id (int)
    """
    m: dict[str, int] = {}
    for ev in root.findall(f".//{_ns('event')}"):
        attrs = _event_attr_map(ev)
        for name_key, id_key in [("gt:label", "gt:label_id"), ("pred:label", "pred:label_id")]:
            name = attrs.get(name_key)
            id_s = attrs.get(id_key)
            if name is None or id_s is None:
                continue
            try:
                m[str(name)] = int(id_s)
            except Exception:
                continue
    return m


def _trace_context(trace: ET.Element) -> str:
    """
    Best-effort extraction of trace identifiers useful for error messages.
    """
    attrs = _event_attr_map(trace)
    # In our XES exporter we set these on traces:
    # - concept:name (trace id like "0")
    # - case:name (scan id like "Furniture/0001_...")
    trace_id = attrs.get("concept:name", "?")
    case_name = attrs.get("case:name", "?")
    return f"trace_id={trace_id} case_name={case_name}"


def filter_one(input_xes: Path, output_xes: Path, na_label: str, drop_empty_traces: bool) -> None:
    # Preserve the default namespace in output (avoid ns0 prefixes; some tools like Disco are picky).
    ET.register_namespace("", XES_NS)
    tree = ET.parse(str(input_xes))
    root = tree.getroot()
    label_to_id = _infer_label_to_id(root)

    removed_events = 0
    kept_events = 0
    removed_traces = 0
    updated_probs = 0
    updated_pred = 0

    # Iterate over traces and remove events in-place.
    for trace in list(root.findall(_ns("trace"))):
        trace_ctx = _trace_context(trace)
        events = list(trace.findall(_ns("event")))
        for ev in events:
            attrs = _event_attr_map(ev)
            if _is_na_event(attrs, na_label=na_label):
                trace.remove(ev)
                removed_events += 1
            else:
                kept_events += 1
                # Also remove NA from probs_json and renormalize remaining probabilities to sum to 1.
                #
                # IMPORTANT:
                # - We only do this for kept (non-NA) events.
                # - After renormalization we update pred:most_likely and also pred:label/pred:label_id so that
                #   predictions are consistent with the new distribution (and NA is never predicted in the no-NA log).
                probs_el = _get_event_attr_elem(ev, "probs_json")
                if probs_el is not None and "value" in probs_el.attrib:
                    try:
                        probs = json.loads(probs_el.attrib["value"])
                        if isinstance(probs, dict) and probs:
                            # Add useful event context for error messages.
                            ev_ctx = [
                                str(input_xes),
                                trace_ctx,
                                f"time={attrs.get('time:timestamp', '?')}",
                                f"segment_start={attrs.get('segment:start_timestamp', '?')}",
                                f"segment_end={attrs.get('segment:end_timestamp', '?')}",
                                f"gt={attrs.get('gt:label', '?')} pred={attrs.get('pred:label', '?')}",
                            ]
                            renorm = _renormalize_probs_excluding_na(  # type: ignore[arg-type]
                                probs,
                                na_label=na_label,
                                context=" | ".join(ev_ctx),
                            )
                            _set_event_string_attr(ev, "probs_json", _stable_probs_json(renorm))

                            # Update pred:most_likely to be argmax over the renormalized non-NA distribution.
                            ml = max(renorm.items(), key=lambda kv: kv[1])[0]
                            _set_event_string_attr(ev, "pred:most_likely", str(ml))

                            # Also set predicted label + id to the new argmax (never NA).
                            if ml == na_label:
                                raise ValueError(
                                    "Internal error: renormalized distribution still predicts NA.\n"
                                    f"Context: {' | '.join(ev_ctx)}"
                                )
                            _set_event_string_attr(ev, "pred:label", str(ml))
                            if ml in label_to_id:
                                _set_event_int_attr(ev, "pred:label_id", int(label_to_id[ml]))
                            else:
                                raise ValueError(
                                    "Cannot map predicted label to pred:label_id. The input XES did not contain a "
                                    "label->id mapping for this label.\n"
                                    f"Missing label: {ml}\n"
                                    f"Context: {' | '.join(ev_ctx)}"
                                )

                            updated_pred += 1
                            updated_probs += 1
                    except Exception as e:
                        # If we fail to parse/renormalize probabilities, or cannot map label->id, fail loudly.
                        # Writing a partially-updated XES is worse than stopping with a clear error.
                        raise

        if drop_empty_traces:
            # After removal, check if any events remain
            if trace.find(_ns("event")) is None:
                root.remove(trace)
                removed_traces += 1

    output_xes.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_xes), encoding="utf-8", xml_declaration=True)

    print(f"Input:  {input_xes}")
    print(f"Output: {output_xes}")
    print(f"Removed events: {removed_events}")
    print(f"Kept events:    {kept_events}")
    print(f"Updated probs_json (renormalized excl. NA): {updated_probs}")
    print(f"Updated pred:label (argmax excl. NA):       {updated_pred}")
    if drop_empty_traces:
        print(f"Removed empty traces: {removed_traces}")


def _iter_inputs(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    pattern = "**/*.xes" if recursive else "*.xes"
    files = sorted(input_path.glob(pattern))
    if not files:
        raise ValueError(f"No .xes files found under: {input_path} (recursive={recursive})")
    return files


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str, help="Input .xes file OR directory containing .xes files.")
    p.add_argument("--output", required=True, type=str, help="Output .xes file OR directory (mirrors input filenames).")
    p.add_argument("--recursive", action="store_true", help="If input is a directory, search recursively for .xes files.")
    p.add_argument("--na_label", type=str, default="NA", help="NA label name (default: NA).")
    p.add_argument(
        "--keep_empty_traces",
        action="store_true",
        help="If set, keep traces even if all their events were removed (default: remove empty traces).",
    )
    args = p.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    inputs = _iter_inputs(input_path, recursive=bool(args.recursive))
    drop_empty_traces = not bool(args.keep_empty_traces)

    if input_path.is_file():
        # output must be a file path
        filter_one(
            input_xes=input_path,
            output_xes=output_path,
            na_label=str(args.na_label),
            drop_empty_traces=drop_empty_traces,
        )
        return

    # Directory mode
    output_path.mkdir(parents=True, exist_ok=True)
    for i, xes_file in enumerate(inputs, start=1):
        out_file = output_path / xes_file.name
        print(f"[{i}/{len(inputs)}] {xes_file.name} -> {out_file.name}")
        filter_one(
            input_xes=xes_file,
            output_xes=out_file,
            na_label=str(args.na_label),
            drop_empty_traces=drop_empty_traces,
        )


if __name__ == "__main__":
    main()



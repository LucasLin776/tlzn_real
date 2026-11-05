"""Delivery-format rule: one label per recording.

The business rule is deliberately blunt: the label of a recording is the part
of it that is **not** normal, and a recording that produces too many separate
anomaly pieces is reported as anomalous as a whole.  Too many pieces usually
means the normal envelope is fragmenting, not that the recording contains many
independent events, and downstream consumers only want a short decision.

    intervals = whole recording - normal segments
    if len(intervals) > max_intervals: intervals = [whole recording]
"""
from __future__ import annotations

MAX_INTERVALS = 5


def _pair(item):
    """Accept both ``(start, end)`` tuples and ``{"start_s", "end_s"}`` dicts.

    Evaluation rows carry dicts, detection carries tuples; silently dropping
    either shape produced empty labels, so the conversion is explicit.
    """
    if isinstance(item, dict):
        return float(item["start_s"]), float(item["end_s"])
    start, end = item
    return float(start), float(end)


def _merge(intervals):
    merged: list[list[float]] = []
    pairs = sorted(p for p in (_pair(i) for i in intervals) if p[1] > p[0])
    for start, end in pairs:
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def collapse_anomalies(intervals, duration_s: float,
                       max_intervals: int = MAX_INTERVALS) -> dict:
    """Turn anomaly intervals into the single label delivered per recording."""
    merged = _merge(intervals)
    if not merged:
        return {"intervals": [], "count": 0, "rule": "none",
                "max_intervals": max_intervals}
    if max_intervals and len(merged) > max_intervals:
        return {"intervals": [[0.0, float(duration_s)]], "count": len(merged),
                "rule": "whole_file", "max_intervals": max_intervals}
    return {"intervals": [[a, b] for a, b in merged], "count": len(merged),
            "rule": "as_detected", "max_intervals": max_intervals}


def label_text(label: dict) -> str:
    """Human/CSV friendly rendering: ``1.20-2.40; 7.00-8.10``."""
    return "; ".join(f"{a:.2f}-{b:.2f}" for a, b in label.get("intervals", []))


def label_from_normal(normal_regions, duration_s: float,
                      max_intervals: int = MAX_INTERVALS) -> dict:
    """Anomaly = recording minus the normal segments, then apply the rule."""
    gaps, prev = [], 0.0
    for start, end in sorted(_pair(r) for r in normal_regions):
        if start > prev + 1e-9:
            gaps.append((prev, start))
        prev = max(prev, end)
    if duration_s > prev + 1e-9:
        gaps.append((prev, float(duration_s)))
    return collapse_anomalies(gaps, duration_s, max_intervals)

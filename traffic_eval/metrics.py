"""Streaming parsers. Units: seconds, vehicles, and kilograms of CO2."""
from __future__ import annotations

import csv
import math
from pathlib import Path
import xml.etree.ElementTree as ET

METRICS = ["avg_waiting_time_s", "avg_trip_duration_s", "avg_queue_length", "max_queue_length", "co2_kg"]


def parse_tripinfo(path: str | Path) -> dict:
    completed, unfinished = 0, 0
    waiting, duration = 0.0, 0.0
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag != "tripinfo":
            continue
        if float(elem.attrib.get("arrival", -1)) >= 0 and not elem.attrib.get("vaporized", ""):
            completed += 1
            waiting += float(elem.attrib["waitingTime"])
            duration += float(elem.attrib["duration"])
        else:
            unfinished += 1
        elem.clear()
    return {
        "completed_trips": completed, "unfinished_trips": unfinished,
        "avg_waiting_time_s": waiting / completed if completed else math.nan,
        "avg_trip_duration_s": duration / completed if completed else math.nan,
    }


def parse_emissions(path: str | Path, step_length_s: float = 1.0) -> dict:
    """SUMO emission-output CO2 is a rate in mg/s; integrate every vehicle step.

    This includes vehicles still in the network at cutoff. Do not sum tripinfo
    completed-trip emissions instead: that would systematically omit congestion.
    """
    if step_length_s <= 0:
        raise ValueError("step_length_s must be positive")
    total_mg = 0.0
    timesteps = 0
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "vehicle":
            total_mg += float(elem.attrib["CO2"]) * step_length_s
            elem.clear()
        elif elem.tag == "timestep":
            timesteps += 1
            elem.clear()
    return {"co2_kg": total_mg / 1_000_000, "emission_timesteps": timesteps}


def parse_queues(path: str | Path) -> dict:
    """Time mean and maximum halting count per *approach*, not queue metres.

    Harness writes every unique incoming edge at every step, including zeroes.
    All rows share the fixed simulation step, so the arithmetic mean is the
    time-and-approach mean. Irregular external telemetry is rejected.
    """
    by_time: dict[float, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = float(row["time"])
            key = row["tls_id"] + "/" + row["approach_id"]
            q = float(row["queue_length"])
            if q < 0 or not math.isfinite(q):
                raise ValueError("Queues must be finite nonnegative counts")
            at_time = by_time.setdefault(t, {})
            if key in at_time:
                raise ValueError(f"Duplicate queue sample {t}/{key}")
            at_time[key] = q
    if not by_time:
        raise ValueError("Queue telemetry is empty; a signalized network is required")
    times = sorted(by_time)
    keys = set(by_time[times[0]])
    if any(set(by_time[t]) != keys for t in times):
        raise ValueError("Queue telemetry must include all approaches, including zero queues")
    if len(times) > 2:
        delta = times[1] - times[0]
        if any(not math.isclose(b-a, delta, abs_tol=1e-6) for a,b in zip(times,times[1:])):
            raise ValueError("Queue telemetry must use a fixed simulation step")
    values = [q for sample in by_time.values() for q in sample.values()]
    return {"avg_queue_length": sum(values)/len(values), "max_queue_length": max(values),
            "avg_total_queue_length": sum(sum(s.values()) for s in by_time.values())/len(times),
            "queue_samples": len(times), "approaches": len(keys)}


def parse_summary(path: str | Path) -> dict:
    last = {}
    for _, elem in ET.iterparse(path, events=("end",)):
        if elem.tag == "step":
            last = dict(elem.attrib)
            elem.clear()
    return {"pending_departures": int(last.get("waiting", 0)),
            "loaded_vehicles": int(last.get("loaded", 0))}


def parse_run(run_dir: str | Path, step_length_s: float = 1.0) -> dict:
    path = Path(run_dir)
    result = parse_tripinfo(path / "tripinfo.xml")
    result.update(parse_emissions(path / "emissions.xml", step_length_s))
    result.update(parse_queues(path / "queues.csv"))
    result.update(parse_summary(path / "summary.xml"))
    if (path/"collisions.xml").is_file():
        from .collisions import collision_records
        result["collision_events"] = len(collision_records(path/"collisions.xml"))
    total = result["completed_trips"] + result["unfinished_trips"] + result["pending_departures"]
    result["completion_rate"] = result["completed_trips"]/total if total else math.nan
    return result

"""Read-only dashboard for seeded SUMO controller comparisons.

Run: python -m streamlit run dashboard.py -- --results results
No demonstration rows are synthesized when result files are absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


METRICS = {
    "avg_waiting_time_s": ("Average waiting time", "s / completed trip"),
    "avg_trip_duration_s": ("Average trip duration", "s / completed trip"),
    "avg_queue_length": ("Average approach queue", "vehicles / approach"),
    "max_queue_length": ("Maximum approach queue", "vehicles"),
    "co2_kg": ("Total CO₂ emissions", "kg"),
}
CONTROLLERS = {"fixed": "Fixed-time", "actuated": "Actuated", "dqn": "DQN"}
COLORS = {"fixed": "#63758D", "actuated": "#1B8C82", "dqn": "#5263CC"}
REQUIRED = {"scenario", "controller", "seed", *METRICS}


def validate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate raw run records before calculating anything from them."""
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    frame = frame.copy()
    if frame.empty:
        raise ValueError("The result file contains no runs.")
    for key in ("scenario", "controller", "seed"):
        if frame[key].isna().any():
            raise ValueError(f"{key} contains missing values.")
    for key in ("scenario", "controller"):
        frame[key] = frame[key].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise")
    if not np.isfinite(frame["seed"]).all() or not (frame["seed"] % 1 == 0).all() or (frame["seed"] < 0).any():
        raise ValueError("Every seed must be a finite nonnegative integer.")
    frame["seed"] = frame["seed"].astype(int)
    if frame.duplicated(["scenario", "controller", "seed"]).any():
        raise ValueError("Duplicate scenario/controller/seed rows: select one experiment at a time.")
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="raise")
        finite = frame[metric].dropna()
        if not np.isfinite(finite).all() or (finite < 0).any():
            raise ValueError(f"{metric} must contain nonnegative finite values or be missing.")
    for key in ("completed_trips", "unfinished_trips", "completion_rate"):
        if key in frame:
            frame[key] = pd.to_numeric(frame[key], errors="raise")
    return frame.sort_values(["scenario", "controller", "seed"]).reset_index(drop=True)


def summarize_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scenario, controller), group in frame.groupby(["scenario", "controller"], sort=True):
        for metric, (label, unit) in METRICS.items():
            values = group[metric].dropna()
            rows.append({"scenario": scenario, "controller": controller, "metric": metric,
                         "measure": label, "unit": unit, "n_seeds": len(values),
                         "mean": values.mean(), "std": values.std(ddof=1)})
    return pd.DataFrame(rows)


def paired_comparisons(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Pair by scenario/seed and require equal, known demand hashes.

    Percentages are per-seed reductions, then aggregated with sample SD.
    A zero baseline is undefined, never silently converted to zero improvement.
    """
    columns = ["scenario", "controller", "baseline", "metric", "n_pairs",
               "improvement_pct_mean", "improvement_pct_std"]
    if "demand_hash" not in frame:
        return pd.DataFrame(columns=columns), ["Demand hashes are absent; matched-demand percentage comparisons are unavailable."]
    records, notes = [], []
    for scenario, group in frame.groupby("scenario", sort=True):
        for baseline in ("fixed", "actuated"):
            base = group[group.controller == baseline]
            for controller in sorted(set(group.controller) - {baseline}):
                current = group[group.controller == controller]
                pairs = current.merge(base, on=["scenario", "seed"], suffixes=("_candidate", "_baseline"))
                if pairs.empty:
                    continue
                valid_hash = (pairs.demand_hash_candidate.notna()
                              & pairs.demand_hash_baseline.notna()
                              & (pairs.demand_hash_candidate.astype(str).str.strip().str.len() > 0)
                              & (pairs.demand_hash_candidate == pairs.demand_hash_baseline))
                if not valid_hash.all():
                    notes.append(f"{scenario}: {controller} vs {baseline}: {int((~valid_hash).sum())} seed pair(s) have missing or different demand hashes and were excluded.")
                pairs = pairs[valid_hash]
                for metric in METRICS:
                    ref, new = pairs[f"{metric}_baseline"], pairs[f"{metric}_candidate"]
                    valid = ref.notna() & new.notna() & (ref > 0)
                    changes = 100.0 * (ref[valid] - new[valid]) / ref[valid]
                    if len(changes):
                        records.append({"scenario": scenario, "controller": controller,
                                        "baseline": baseline, "metric": metric, "n_pairs": len(changes),
                                        "improvement_pct_mean": changes.mean(),
                                        "improvement_pct_std": changes.std(ddof=1)})
    return pd.DataFrame(records, columns=columns), list(dict.fromkeys(notes))


def csv_download(frame: pd.DataFrame, label: str, name: str) -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), file_name=name, mime="text/csv")


def controller_name(value: str) -> str:
    return CONTROLLERS.get(value, value)


def metric_chart(summary: pd.DataFrame, metric: str) -> None:
    label, unit = METRICS[metric]
    scenarios = list(summary.scenario.unique())
    fig, axes = plt.subplots(1, len(scenarios), figsize=(max(5, 4 * len(scenarios)), 3.7), squeeze=False)
    for axis, scenario in zip(axes[0], scenarios):
        part = summary[(summary.scenario == scenario) & (summary.metric == metric)]
        positions = np.arange(len(part))
        means = part["mean"].to_numpy(dtype=float)
        errors = part["std"].fillna(0).to_numpy(dtype=float)
        axis.bar(positions, means, yerr=errors, capsize=4,
                 color=[COLORS.get(c, "#63758D") for c in part.controller], width=0.62)
        axis.set_xticks(positions, [controller_name(c) for c in part.controller])
        axis.set_title(scenario.replace("_", " ").title())
        axis.set_ylabel(unit)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis="y", alpha=0.17)
    fig.suptitle(label, fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def show_quality(frame: pd.DataFrame) -> None:
    missing = {"fixed", "actuated", "dqn"} - set(frame.controller)
    if missing:
        st.warning("The three-way comparison is incomplete. Missing controller(s): " + ", ".join(controller_name(x) for x in sorted(missing)) + ".")
    counts = frame.groupby(["scenario", "controller"]).seed.nunique()
    low = counts[counts < 5]
    if not low.empty:
        st.warning("Some groups have fewer than five seeds. These runs verify the pipeline but do not satisfy the document's five-to-ten-seed evaluation requirement.")
    high = counts[counts > 10]
    if not high.empty:
        st.info("Some groups contain more than ten seeds; all selected seeds are included.")
    if "experiment_label" in frame:
        labels = ", ".join(sorted(frame.experiment_label.dropna().astype(str).unique()))
        if labels:
            st.caption("Experiment label: " + labels)
        if frame.experiment_label.fillna("").astype(str).str.contains("smoke|demo|test|provisional|mock|uncalibrated", case=False).any():
            st.warning("This selection includes demonstration, provisional or smoke-test data. It is not a final project result.")
    if frame.scenario.str.contains("smoke|demo|test|provisional|mock", case=False).any():
        st.warning("Scenario names identify this as a provisional or test experiment. Final corridor validation remains outstanding.")
    if "unfinished_trips" in frame and (frame.unfinished_trips.fillna(0) > 0).any():
        st.warning("Unfinished vehicles remain at the run horizon. Waiting time and trip duration describe completed trips only; compare completion rates before claiming an improvement.")
    if frame[list(METRICS)].isna().any().any():
        st.warning("Some metrics are missing. Means use available values; check the displayed seed count for each metric.")
    if "pending_departures" in frame and (frame.pending_departures.fillna(0) > 0).any():
        st.warning("Some vehicles could not enter the network by the end of the run. Compare pending departures and completion rates as well as completed-trip delay.")
    for column in ("teleports", "collisions"):
        if column in frame and (pd.to_numeric(frame[column], errors="coerce").fillna(0) > 0).any():
            st.warning(f"The selected runs include {column}. Investigate network and safety issues before using these results in the final report.")
    if len(frame.scenario.unique()) < 2:
        st.info("Only one scenario is selected. The final study requires both peak and off-peak demand.")
    quality_cols = [c for c in ["scenario", "controller", "seed", "completed_trips", "unfinished_trips", "pending_departures", "completion_rate", "teleports", "collisions"] if c in frame]
    with st.expander("Completion and run coverage", expanded=False):
        st.dataframe(frame[quality_cols], hide_index=True)


def run_directory(record: pd.Series, result_path: Path) -> Path | None:
    portable = result_path / "raw" / str(record.scenario) / str(record.controller) / f"seed_{int(record.seed)}"
    if portable.is_dir():
        return portable
    if "run_dir" not in record or pd.isna(record.run_dir):
        return None
    raw = Path(str(record.run_dir))
    candidates = [raw] if raw.is_absolute() else [result_path / raw, result_path.parent / raw, Path.cwd() / raw]
    return next((path for path in candidates if path.is_dir()), None)


def show_queues(frame: pd.DataFrame, result_path: Path) -> None:
    choices = {f"{row.scenario} · {controller_name(row.controller)} · seed {row.seed}": index
               for index, row in frame.iterrows()}
    selection = st.selectbox("Run", list(choices), key="queue_run")
    directory = run_directory(frame.loc[choices[selection]], result_path)
    queue_path = directory / "queues.csv" if directory else None
    if queue_path is None or not queue_path.is_file():
        st.info("Per-approach queue traces are not available for this run. Keep the raw run directory beside runs.csv to inspect them.")
        return
    try:
        queues = pd.read_csv(queue_path)
        if not {"time", "tls_id", "approach_id", "queue_length"}.issubset(queues):
            raise ValueError("Expected time, tls_id, approach_id and queue_length columns.")
        junction = st.selectbox("Intersection", sorted(queues.tls_id.astype(str).unique()))
        queues = queues[queues.tls_id.astype(str) == junction]
        approaches = st.multiselect("Approaches", sorted(queues.approach_id.astype(str).unique()),
                                   default=sorted(queues.approach_id.astype(str).unique()))
        queues = queues[queues.approach_id.astype(str).isin(approaches)]
        if queues.empty:
            st.info("Select at least one approach.")
            return
        pivot = queues.pivot_table(index="time", columns="approach_id", values="queue_length", aggfunc="mean").sort_index()
        stride = max(1, int(np.ceil(len(pivot) / 6000)))
        if stride > 1:
            st.caption(f"The chart shows every {stride}th sample; the downloadable trace keeps every sample.")
        st.line_chart(pivot.iloc[::stride], x_label="Simulation time (s)", y_label="Halting vehicles")
        csv_download(queues, "Download queue trace", "queue_trace.csv")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the queue trace: {exc}")


def show_training(default_path: str | None) -> None:
    st.caption("Training rewards require Ananyaraj's actual Stable-Baselines3 Monitor log. Evaluation metrics are not used as substitute training curves.")
    log_path = st.text_input("Monitor CSV path (optional)", default_path or "")
    uploaded = st.file_uploader("Or upload a Monitor CSV", type="csv", key="training_upload")
    if not uploaded and not log_path.strip():
        st.info("No training log supplied.")
        return
    try:
        logs = pd.read_csv(uploaded if uploaded else Path(log_path).expanduser(), comment="#")
        if "r" not in logs:
            raise ValueError("A Stable-Baselines3 Monitor CSV must contain the episode reward column r.")
        reward = pd.to_numeric(logs.r, errors="raise")
        curve = pd.DataFrame({"Episode reward": reward, "10-episode rolling mean": reward.rolling(10, min_periods=1).mean()})
        curve.index = pd.RangeIndex(1, len(curve) + 1, name="Episode")
        st.line_chart(curve, x_label="Training episode", y_label="Reward")
        st.caption("A smooth or increasing curve alone does not establish convergence or superiority over baselines.")
        csv_download(logs, "Download training data", "training_monitor.csv")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the training log: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results", default="results")
    parser.add_argument("--training-log")
    args, _ = parser.parse_known_args(sys.argv[1:])

    st.set_page_config(page_title="Traffic Control Evaluation", page_icon="🚦", layout="wide")
    st.markdown("<style>.block-container {padding-top:2.0rem; max-width:1400px;} h1 {letter-spacing:-0.035em;} [data-testid='stMetric'] {background:rgba(99,117,141,0.07);border-radius:10px;padding:14px;} </style>", unsafe_allow_html=True)
    st.caption("SOHAM · PHASE 4 · SUMO / DEEP REINFORCEMENT LEARNING")
    st.title("Traffic control evaluation")
    st.write("Compare fixed-time, actuated and learned signal control under matched traffic demand.")

    with st.sidebar:
        st.header("Experiment")
        result_text = st.text_input("Results folder or runs.csv", args.results)
        uploaded = st.file_uploader("Or upload runs.csv", type="csv", key="runs_upload")
        st.caption("Only recorded simulator outputs are shown. Filters recalculate every table and comparison.")
    location = Path(result_text).expanduser()
    csv_path = location if location.suffix.lower() == ".csv" else location / "runs.csv"
    result_path = csv_path.parent
    if not uploaded and not csv_path.is_file():
        st.info("No evaluation data found. Run the evaluation harness, then select its output folder in the sidebar.")
        st.code("python -m traffic_eval run --config configs/evaluation.example.json --output results", language="powershell")
        st.caption("The DQN checkpoint and runnable scenarios must exist before a full comparison can run.")
        show_training(args.training_log)
        return
    try:
        runs = validate_runs(pd.read_csv(uploaded if uploaded else csv_path))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        st.error(f"Could not load valid run records: {exc}")
        return

    with st.sidebar:
        scenarios = st.multiselect("Demand scenarios", sorted(runs.scenario.unique()), default=sorted(runs.scenario.unique()))
        controllers = st.multiselect("Controllers", sorted(runs.controller.unique()), default=sorted(runs.controller.unique()), format_func=controller_name)
        seeds = st.multiselect("Seeds", sorted(runs.seed.unique()), default=sorted(runs.seed.unique()))
    frame = runs[runs.scenario.isin(scenarios) & runs.controller.isin(controllers) & runs.seed.isin(seeds)]
    if frame.empty:
        st.info("Select at least one scenario, controller and seed with recorded data.")
        return
    columns = st.columns(4)
    for column, label, value in zip(columns, ["Recorded runs", "Scenarios", "Controllers", "Unique seeds"],
                                    [len(frame), frame.scenario.nunique(), frame.controller.nunique(), frame.seed.nunique()]):
        column.metric(label, value)
    show_quality(frame)
    quality_path = result_path / "quality_checks.json"
    if not uploaded and quality_path.is_file():
        try:
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if not quality.get("final_evaluation_ready", False):
                with st.expander("Full experiment validation notes", expanded=False):
                    for warning in quality.get("warnings", []):
                        st.write("• " + str(warning))
        except (OSError, ValueError):
            st.warning("The experiment quality-check file could not be read.")
    summary = summarize_runs(frame)
    paired, pairing_notes = paired_comparisons(frame)
    for note in pairing_notes:
        st.warning(note)
    if not paired.empty and paired.n_pairs.lt(5).any():
        st.warning("Some paired comparisons contain fewer than five valid seed pairs. The paired table reports the actual pair count.")

    comparison_tab, paired_tab, queue_tab, training_tab, raw_tab = st.tabs(["Controller comparison", "Paired improvements", "Intersection queues", "Training", "Run records"])
    with comparison_tab:
        metric = st.selectbox("Measure", list(METRICS), format_func=lambda item: METRICS[item][0])
        metric_chart(summary, metric)
        st.caption("Bars are seed means; whiskers show sample standard deviation (ddof=1). A single seed has no estimable SD. Lower values are better for all five measures.")
        st.dataframe(summary.rename(columns={"std": "sample_sd"}), hide_index=True)
        csv_download(summary, "Download summary", "filtered_summary.csv")
    with paired_tab:
        st.write("Each percentage compares the same scenario and seed with the same recorded demand hash. Positive values mean a reduction; negative values mean an increase.")
        st.caption("For each seed: 100 × (baseline − candidate) / baseline. The table reports the mean and sample SD of these percentages. Zero-valued baselines are excluded.")
        if paired.empty:
            st.info("No valid paired percentage comparisons are available for the current selection.")
        else:
            baseline = st.selectbox("Reference controller", sorted(paired.baseline.unique()), format_func=controller_name)
            shown = paired[paired.baseline == baseline].copy()
            shown["measure"] = shown.metric.map(lambda name: METRICS[name][0])
            st.dataframe(shown.drop(columns="metric"), hide_index=True)
            csv_download(shown, "Download paired improvements", "paired_improvements.csv")
            checks = paired[(paired.controller == "actuated") & (paired.baseline == "fixed") & (paired.metric == "avg_waiting_time_s")]
            if not checks.empty:
                for _, row in checks.iterrows():
                    if row.improvement_pct_mean > 0:
                        st.success(f"{row.scenario}: actuated control reduced mean paired waiting time by {row.improvement_pct_mean:.2f}% relative to fixed-time ({int(row.n_pairs)} pairs).")
                    else:
                        st.warning(f"{row.scenario}: actuated control did not reduce waiting time relative to fixed-time. Check demand calibration and signal plans before trusting a final comparison; retain the measured result.")
    with queue_tab:
        show_queues(frame, result_path)
    with training_tab:
        show_training(args.training_log)
    with raw_tab:
        st.dataframe(frame, hide_index=True)
        csv_download(frame, "Download selected runs", "filtered_runs.csv")
        st.caption("Trip waiting and duration average over completed trips. Queue counts represent stopped vehicles per approach. CO₂ covers all simulated vehicles, including unfinished trips, when emitted by the evaluation harness.")


if __name__ == "__main__":
    main()

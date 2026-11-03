from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import METRICS


def validate_runs(frame: pd.DataFrame) -> None:
    required = {"scenario", "controller", "seed", "demand_hash", *METRICS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"runs.csv missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("runs.csv contains no runs")
    if frame.demand_hash.isna().any() or frame.demand_hash.astype(str).str.strip().eq("").any():
        raise ValueError("Every run requires a nonempty demand/config fingerprint")
    if frame.duplicated(["scenario", "controller", "seed"]).any():
        raise ValueError("Duplicate scenario/controller/seed rows; cannot pair results")
    if "collision_action" in frame and frame.collision_action.fillna("teleport").nunique()>1:
        raise ValueError("Do not mix collision-action diagnostic modes with normal evaluation; use separate output folders")
    if "timing_profile" in frame:
        for _,part in frame[frame.controller=="dqn"].groupby("scenario"):
            if part.timing_profile.fillna("v1").nunique()>1:
                raise ValueError("Do not average different DQN timing profiles in one scenario")
    for _, group in frame.groupby(["scenario", "seed"]):
        if group.demand_hash.nunique(dropna=False) != 1:
            raise ValueError("Demand/config fingerprints differ across controllers for a paired seed")
    for scenario, group in frame.groupby("scenario"):
        seed_sets = [frozenset(part.seed) for _, part in group.groupby("controller")]
        if len(set(seed_sets)) > 1:
            raise ValueError(f"Controllers have different seed sets in {scenario}; complete all paired runs first")


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_runs(frame)
    summary = frame.groupby(["scenario", "controller"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.insert(0, "n_seeds", frame.groupby(["scenario", "controller"]).seed.nunique())
    improvements = []
    for scenario, subset in frame.groupby("scenario"):
        for baseline in ("fixed", "actuated"):
            base = subset[subset.controller == baseline].set_index("seed")
            if base.empty:
                continue
            for controller in sorted(set(subset.controller) - {baseline}):
                other = subset[subset.controller == controller].set_index("seed")
                for metric in METRICS:
                    paired = base[[metric]].rename(columns={metric: metric + "_base"}).join(
                        other[[metric]].rename(columns={metric: metric + "_other"}), how="inner")
                    valid = paired[metric + "_base"].notna() & paired[metric + "_other"].notna() & (paired[metric + "_base"] != 0)
                    pct = 100 * (paired.loc[valid, metric+"_base"] - paired.loc[valid, metric+"_other"]) / paired.loc[valid, metric+"_base"]
                    improvements.append({"scenario": scenario, "controller": controller, "baseline": baseline,
                        "metric": metric, "n_pairs": len(pct), "zero_or_missing_pairs": len(paired)-len(pct),
                        "improvement_pct_mean": pct.mean(), "improvement_pct_std": pct.std(ddof=1)})
    return summary.reset_index(), pd.DataFrame(improvements)


def write_report(input_dir: str | Path) -> dict:
    root = Path(input_dir)
    runs = pd.read_csv(root / "runs.csv")
    summary, paired = summarize(runs)
    summary.to_csv(root / "summary.csv", index=False)
    paired.to_csv(root / "paired_improvements.csv", index=False)
    checks = []
    if not paired.empty:
        for row in paired[(paired.controller == "actuated") & (paired.baseline == "fixed")].to_dict("records"):
            row["status"] = "PASS" if row["n_pairs"] >= 5 and row["improvement_pct_mean"] > 0 else "WARN"
            checks.append(row)
    pd.DataFrame(checks, columns=["scenario", "controller", "baseline", "metric", "n_pairs", "zero_or_missing_pairs", "improvement_pct_mean", "improvement_pct_std", "status"]).to_csv(root/"baseline_checks.csv", index=False)
    warnings = []
    if "collision_action" in runs and runs.collision_action.eq("warn").any():
        warnings.append("Collision-action warn is diagnostic only: it changes post-collision behavior and cannot establish policy performance or collision causality.")
    if "controller_warnings" in runs:
        policy_warnings = []
        for value in runs.controller_warnings.dropna().unique():
            policy_warnings.extend(json.loads(value))
        warnings.extend(dict.fromkeys(policy_warnings))
    if runs[METRICS].isna().any().any():
        warnings.append("Some required metrics are undefined (for example no completed trips). Increase duration or repair demand before reporting performance.")
    if "experiment_label" not in runs or runs.experiment_label.fillna("").str.contains("smoke|demo|provisional|synthetic|test|uncalibrated",case=False,regex=True).any():
        warnings.append("These runs are labelled demonstration/provisional or lack project provenance; they are not final corridor evidence.")
    if set(runs.controller) != {"fixed", "actuated", "dqn"}:
        warnings.append("Three-way evaluation incomplete: a verified trained DQN and both baselines are required.")
    if runs.groupby(["scenario", "controller"]).seed.nunique().lt(5).any():
        warnings.append("At least one group has fewer than five seeds; these are smoke results, not final results.")
    if "completion_rate" in runs and runs.completion_rate.lt(.95).any():
        warnings.append("Some runs complete less than 95% of inserted/pending trips. Completed-trip means are censored; compare completion rates and increase the drain period.")
    if "teleports" in runs and runs.teleports.gt(0).any():
        warnings.append("Teleports occurred. Correct network/demand problems before using results in the report.")
    if "collisions" in runs and runs.collisions.gt(0).any():
        warnings.append("Collisions occurred. Inspect signal safety and simulation settings before trusting results.")
    wait_checks = [x for x in checks if x["metric"] == "avg_waiting_time_s"]
    if len(wait_checks) < runs.scenario.nunique() or any(x["status"] != "PASS" for x in wait_checks):
        warnings.append("Actuated has not been confirmed better than fixed on mean waiting time with five or more pairs in every scenario. Investigate rather than modifying results.")
    if set(runs.scenario) != {"peak", "offpeak"}:
        warnings.append("Final evaluation requires both peak and offpeak scenarios.")
    info = {"final_evaluation_ready": not warnings, "warnings": warnings,
            "standard_deviation": "sample standard deviation (ddof=1); undefined for n=1",
            "improvement": "mean of seed-paired 100*(baseline-controller)/baseline; positive means lower is better; zero baseline omitted",
            "queue_definition": "time-and-approach mean halting vehicles; maximum over all approach-time samples",
            "co2_definition": "all simulated vehicle CO2 rates integrated over the fixed simulation step; kg",
            "trip_definition": "completed, non-vaporized trips only; unfinished and pending departures reported separately"}
    (root/"quality_checks.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    _charts(summary, root/"charts")
    return info


def _charts(summary, chart_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    chart_dir.mkdir(parents=True, exist_ok=True)
    labels = {"avg_waiting_time_s":"Mean completed-trip waiting time (s)", "avg_trip_duration_s":"Mean completed-trip duration (s)",
              "avg_queue_length":"Mean halting vehicles per approach", "max_queue_length":"Maximum halting vehicles on an approach", "co2_kg":"Total CO2 (kg)"}
    scenarios = list(summary.scenario.unique())
    for metric in METRICS:
        fig, axes = plt.subplots(1, len(scenarios), figsize=(5*len(scenarios), 4), squeeze=False)
        for ax, scenario in zip(axes[0], scenarios):
            part = summary[summary.scenario == scenario].set_index("controller").reindex([x for x in ("fixed", "actuated", "dqn") if x in set(summary.controller)])
            ax.bar(part.index, part[metric+"_mean"], yerr=part[metric+"_std"].fillna(0), capsize=4, color=[{"fixed":"#718096","actuated":"#319795","dqn":"#805ad5"}[c] for c in part.index])
            ax.set_title(scenario)
            ax.set_ylabel(labels[metric]); ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Seed mean and sample standard deviation"); fig.tight_layout()
        fig.savefig(chart_dir/(metric+".png"), dpi=160); plt.close(fig)

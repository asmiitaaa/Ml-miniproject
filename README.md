# Dynamic Traffic Congestion Control: Soham's Evaluation Work

## Current trained agent

Ananyaraj's agent work is now implemented and trained as well. Use the new independent corridor policy and selected peak/off-peak models:

```powershell
Set-Location C:\Users\soham\Ml-miniproject
.\.venv\Scripts\python.exe scripts\quick_test.py --config configs\evaluation.corridor.json
.\.venv\Scripts\python.exe -m streamlit run dashboard.py -- --results report/training/corridor_results
# Watch the actual selected model control all mapped signals for 300 seconds
.\.venv\Scripts\python.exe scripts\preview_gui.py --config configs\evaluation.corridor.json --controllers dqn --scenario peak --seconds 300 --every 15 --delay-ms 0 --pace-ms 100
```

Read [the agent completion report](docs/ANANYARAJ_COMPLETION.md) and [CHANGES.md](CHANGES.md). Five full training runs total 500,000 DQN decisions. The corridor model uses a new 17-feature all-lane contract with an explicit, checked mapping of all six signal sites; the legacy 13-feature v2 model remains separate. Both families were evaluated in complete 30-run comparisons. All vehicles completed with no collision events or teleports in those tests. The corridor DQN reduced waiting relative to fixed-time but did not outperform the actuated baseline. Demand remains synthetic and uncalibrated.

The remainder of this README documents the historical v1 evaluation. Its tables, model and demonstration have been preserved.

Evaluation code, parsers, charts, Streamlit dashboard, genuine SUMO-GUI demonstration, literature audit, and IEEE report sources. The original network, vehicle definitions, teammate branch, and model weights are preserved.

## Run a test on this computer

Dependencies are already installed in the clone's `.venv`. Open PowerShell:

```powershell
Set-Location C:\Users\soham\Ml-miniproject
.\.venv\Scripts\python.exe scripts\quick_test.py
```

This runs fixed-time, actuated and the actual 20,000-step DQN for 180simulated seconds with the same demand and seed. Expect `QUICK TEST PASSED` and a new `results/quick_test_...` folder. A short, one-seed test intentionally displays evidence-quality warnings; it does not replace the full experiment.

```powershell
# Unit, model-integrity and dashboard tests
.\Start-Project.ps1 -Mode Test
# Watch the real road network and DQN for 300simulated seconds
.\Start-Project.ps1 -Mode GUI
# Or watch either baseline
.\Start-Project.ps1 -Mode GUI -Controller actuated
# Inspect the complete 30-run evaluation
.\Start-Project.ps1 -Mode Dashboard
# Repeat all 30one-hour jobs in a new timestamped folder
.\Start-Project.ps1 -Mode Evaluate
```

GUI replay uses actual TraCI model inference and avoids the local SUMO screenshot-renderer bug. The clock and vehicles should move; the replay closes after300simulated seconds. Use `docs/RUNNING.md` for detailed commands and troubleshooting.

## Results and important findings

`results/model_validation` contains all 30 full-hour runs: fixed/actuated/DQN x peak/offpeak x seeds 11-15. Vehicle inputs are explicit and frozen, with the specified 50/15/30/5 mix of motorcycles/auto-rickshaws/cars/buses. Every scheduled vehicle completed. Actuated beat fixed on waiting time in both scenarios.

Two peak DQN runs (seeds 11 and 15) produced collision-related teleports. These are retained and flagged; the measured speed improvements must not be submitted as a safe-performance claim. `docs/AGENT_HANDOFF.md` identifies the event times and lanes. Demand is synthetic and uncalibrated. The policy controls its trained junction; all other signals stay fixed for equal comparison scope. Training used 0.5 m lateral resolution, evaluation follows the documents' 0.4 m. Convergence and training/test seed separation are not established.

`demonstrations/three_controller_final/comparison.mp4` contains actual matched SUMO-GUI window captures for all three controllers, covering 90 simulated seconds at 15-second sampling intervals. These are native captures, not a generated traffic animation. The numerical experiment remains the source of all performance metrics.

## Setup on another computer

Python 3.10+ and internet are needed initially:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

`requirements-lock.txt` records the tested environment. `requirements-agent.txt` installs the checkpoint runtime. `traffic_eval/model_loading.py` buffers nested tensor archive streams to fix the Windows PyTorch/SB3 stream-reader incompatibility. Every loaded tensor and the original checkpoint hash have been verified; no model weights are modified.

## Reproduction and files

```powershell
.\.venv\Scripts\python.exe -m traffic_eval doctor --config configs\evaluation.model_validation.json
.\.venv\Scripts\python.exe -m traffic_eval report --input results\model_validation
.\.venv\Scripts\python.exe scripts\build_report.py --results results\model_validation
.\.venv\Scripts\python.exe scripts\prepare_model_evaluation.py --check
```

Every experiment has `runs.csv`, `summary.csv`, `paired_improvements.csv`, `baseline_checks.csv`, `quality_checks.json`, charts and raw SUMO XML. Per-run manifests retain input/output hashes, seeds, exact commands, simulator version and checkpoint hash. DQN action traces retain the 13input features, selected action and applied phase. `configs/evaluation.model_validation.json` is the runnable experiment; `configs/evaluation.example.json` remains the template for future field-calibrated corridor inputs.

The full original checkpoints and source snapshot came from `origin/ananyaraj` at 951ea33. The 20,000-step checkpoint was selected before examining evaluation scores because it was the latest available file. Older mock models are excluded.

## Metric definitions

* Waiting time and trip duration: arithmetic means of SUMO `waitingTime` and `duration` over completed, non-vaporized trips only. Unfinished trips and departure backlog are reported separately; censoring is a quality warning.
* Queue: halting vehicles, speed below SUMO's 0.1 m/s threshold, summed over incoming lanes of each signal-controlled edge. Zero samples are included. Reported average is over every approach and simulation step; maximum is the largest individual approach queue. Total-network mean queue is an additional diagnostic.
* CO2: emission-output rates in mg/s multiplied by simulation step seconds and summed across all vehicles and steps, including unfinished journeys. Divide by 1,000,000 to report kilograms. This differs deliberately from summing only completed-trip emissions.
* Summary: mean and sample standard deviation (`ddof=1`) across seeds, with the actual sample count displayed. A single-seed sample SD is undefined.
* Paired change: `100 * (baseline - controller) / baseline` for lower-is-better metrics, calculated within matched scenario/seed/input-hash pairs, then summarized across seeds. Zero baselines are undefined rather than divided by zero. These are not percentages calculated from unmatched aggregate means.

The actuated baseline is SUMO's gap-based controller, using the original phase states and clearance phases with configurable green min/max durations. Its superiority is checked empirically, never assumed. If it does not beat fixed-time waiting time, `baseline_checks.csv` and the dashboard flag the result for review. Do not tune demand or discard seeds merely to force a desired conclusion.

## Submission and demonstration

See `docs/SOHAM_CHECKLIST.md`, `docs/REFERENCE_VERIFICATION.md`, `docs/LITERATURE_SURVEY.md`, `docs/DASHBOARD.md`, `docs/CONTROLLER_CONTRACT.md`, and `docs/DEMONSTRATION.md`. The report sources and build instructions are in `report/`.

Real recordings require `sumo-gui` (included with the SUMO wheel), a desktop session, and `requirements-dev.txt`. They use SUMO's own screenshot API. The recording tool supports baseline-only validation, but a complete submission needs fixed-time/actuated/DQN recordings from identical scenarios and seeds.

Scopus/IEEE database confirmation and the Turnitin similarity check need the institution's access. No similarity score, DQN gain, convergence result, calibrated demand, or field validation is claimed without evidence.

SUMO documentation: [TripInfo](https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html), [emission output](https://sumo.dlr.de/docs/Simulation/Output/EmissionOutput.html), [traffic lights](https://sumo.dlr.de/docs/Simulation/Traffic_Lights.html). Road data are supplied in `network/map.osm`; retain OpenStreetMap attribution and applicable data licensing when distributing derived maps.

## GitHub review bundle

This branch is based on updated main, including the team's canonical `rl_agent/` sources. Local `integrations/` source archives are excluded. The selected 20k checkpoint copy and its provenance remain available for reproducible evaluation.

Small published CSVs and validation flags are in `report/results/`; local paths and wall-time columns were removed from the published run table. Full raw XML and action traces remain local under `results/model_validation`. When raw data are absent, the dashboard can read the published tables; queue traces require regeneration.

Use Python directly without changing PowerShell execution policy:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts\quick_test.py
.\.venv\Scripts\python.exe -m streamlit run dashboard.py -- --results report/results
```

See `docs/REVIEW_NOTES.md` for scope, known limitations and reviewer commands.

The small review video is committed at `report/media/comparison.mp4`, and the five static charts are in `report/figures/`. Full raw outputs and intermediate captures remain local.

## v2 correction and team handover

See [CHANGES.md](CHANGES.md) for the implemented v1/v2 timing contract, collision-event logs, completed checks and each member's remaining actions. The trainer now uses an isolated 0.4 m SUMO configuration and explicit seed 42. The original source scenario and v1 weights/results are preserved. A short v2 checkpoint used during verification is a learning smoke test, not a converged model.

`configs/evaluation.v2.example.json` requires a newly trained v2 checkpoint with embedded timing metadata. It rejects v1 weights relabelled as v2. `configs/diagnostic.v1.warn.json` is a separate forensic experiment. The historical published results remain v1 results.

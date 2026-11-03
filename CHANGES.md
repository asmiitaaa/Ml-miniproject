# V2 timing and collision-diagnostics handover

Prepared 4 November 2026 for Soham and Ananyaraj. This is the handover created with the implemented changes; it is not a previously located teammate CHANGES file.

## What changed

- `rl_agent/train.py`: incorporated Asmita's 6 s yellow / 3 s decisions / 12 s minimum green / 1,200 decisions per hour and distinct `dqn_traffic_sumo_v2` name. Added portable paths, output isolation, overwrite protection and embedded checkpoint provenance. Private training configuration now uses 0.4 m lateral resolution, explicit SUMO/model seed 42 and strict route errors. The original source inputs remain intact.
- `traffic_eval/agent_adapter.py` and `timing.py`: added explicit v1/v2 profiles. v2 requires matching embedded training metadata and the original ordered junction/lane mapping; relabelled v1 weights and conflicting timing are rejected. Training uncertainty stays visible in manifests and quality warnings.
- `traffic_eval/runner.py`, `collisions.py` and `metrics.py`: every new evaluation writes collision XML plus CSV records containing time, lane, vehicle IDs, types, speeds and SUMO's available position fields. XML/CSV integrity hashes are retained.
- `traffic_eval/reporting.py`: separates normal `collision_action: teleport` from diagnostic `warn`; prevents mixing those modes or averaging multiple DQN timing profiles within a scenario. Warn mode can never qualify as final evaluation evidence.
- GUI/demo tools: enable collision XML logging and pass the same timing/lateral-resolution information to the adapter. Recording extra arguments cannot override collision settings.
- `configs/evaluation.v2.example.json`: ready for a genuine newly trained v2 checkpoint. `configs/diagnostic.v1.warn.json`: explicitly labelled v1 forensic replay. `scripts/quick_test.py` now accepts `--config` and an optional genuine `--checkpoint` path.

## Verified

- 72 automated tests pass, including original agent environment checks, observation/timing compatibility, rejected v1-as-v2 use, collision records, diagnostic separation, input preservation and checkpoint overwrite protection. Five warnings originate from the installed Streamlit/Pandas integration.
- Real SUMO v2 learning smoke test: 1,200 decisions and 50 optimizer updates; saved checkpoint reloads with the expected 3/12/6 timing, seed 42 and 0.4 m resolution. This checkpoint is kept outside the production model directory and is not a converged policy.
- Actual neural inference: fixed, actuated and the bounded v2 model each completed a matched 180-second, seed-11 run. No collisions or teleports occurred in those short runs. Short duration leaves unfinished vehicles and cannot establish safe performance or convergence.
- Separate real SUMO scripted timing test: phase changes at 0, 12, 18, 30 and 36 s, confirming 12 s green / 6 s yellow in both directions under repeated switch requests. Scripted actions were used only for this timing check and never substituted for the DQN in measured comparisons.
- Both known v1 collision seeds reproduced with collision output enabled. Each produced one XML event in each response mode. Default teleport response produced one teleport; diagnostic warn produced none because it changes post-collision response.

| Seed | Replay duration | Event time | Incoming lane | Collider → victim | SUMO event type |
| --- | ---: | ---: | --- | --- | --- |
| 11 | 2,400 s | 2,311 s | `631316440#0_0` | `peak_0766` → `peak_0763` | `collision` |
| 15 | 1,200 s | 1,048 s | `631316440#0_0` | `peak_0339` → `peak_0346` | `collision` |

Both pairs were motorcycles. The detailed records locate incoming-lane events; they do **not** establish that yellow timing caused them. SUMO distinguishes rear collisions from junction collisions. Further controlled review must include signal traces, car-following, lateral behavior and vehicle parameters. Warn mode is a diagnostic change, not a repaired policy or a comparable safety score. See [collision output](https://sumo.dlr.de/docs/Simulation/Output/Collisions.html) and [collision actions](https://sumo.dlr.de/docs/Simulation/Safety.html).

Small portable verification records are under `report/diagnostics/`. Raw replay data stay local in the ignored `results/v1_collision_diagnostics_20261104` and `results/v1_collision_diagnostics_seed15_20261104` folders. Original `report/results/` tables and charts still describe v1; they have not been relabelled as v2 results.

## ACTION REQUIRED — Ananyaraj

1. Review the updated trainer and complete the full v2 training run. Preserve v1 weights. Do not rename or patch the v1 checkpoint into v2.
2. Supply the genuine checkpoint, training/validation reward curves, checkpoint hash and convergence/model-selection review. A 100,000-step run alone does not prove convergence. Training currently repeats seed 42; additional disjoint training/validation seeds are needed for robustness claims.
3. Review the motorcycle interactions and clearance design with Asmita. Do not mark the collision cause resolved from a v1/v2 performance comparison alone.

Run in PowerShell from this clone (no activation or execution-policy bypass needed):

```powershell
Set-Location C:\Users\soham\Ml-miniproject
.\.venv\Scripts\python.exe rl_agent\train.py --timesteps 100000
```

Default output is `rl_agent/models/dqn_traffic_sumo_v2.zip`, periodic checkpoints and `training_runs/<timestamp>/monitor.csv`. Existing v2 checkpoints are protected from overwrite; choose a fresh `--output-dir` for an additional run and configure that actual checkpoint path in the evaluation JSON. The source scenario is copied privately with the explicit seed/resolution before training. [SUMO seed behavior](https://sumo.dlr.de/docs/Simulation/Randomness.html).

## ACTION REQUIRED — Soham

The missing adapter and collision-logging implementation is completed and tested. The remaining work depends on the full trained checkpoint and team validation:

1. After Ananyaraj supplies the genuine v2 model, run the short integration check and then all 30 paired one-hour evaluations. Preserve every failure and collision log.
2. Review completion, collision/teleport records and both baseline comparisons. Regenerate the dashboard, tables and report from those new results. Published historical v1 tables remain unchanged until that review.
3. Coordinate demand calibration and corridor scope with Asmita; the available two demand levels are synthetic, and DQN controls one trained junction. Institutional reference/similarity checks still require the team's access.

```powershell
Set-Location C:\Users\soham\Ml-miniproject
.\.venv\Scripts\python.exe -m pytest tests rl_agent\test_environment.py -q
# The following commands require the full v2 checkpoint at the configured path.
.\.venv\Scripts\python.exe scripts\quick_test.py --config configs\evaluation.v2.example.json
$v2Results = 'results\v2_evaluation_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m traffic_eval run --config configs\evaluation.v2.example.json --output $v2Results
.\.venv\Scripts\python.exe -m streamlit run dashboard.py -- --results $v2Results
```

Optional forensic replay (keep separate from performance results):

```powershell
$diagnosticResults = 'results\v1_warn_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe -m traffic_eval run --config configs\diagnostic.v1.warn.json --controllers dqn --output $diagnosticResults
```

For an immediately runnable check using the existing genuine v1 model, use `.\.venv\Scripts\python.exe scripts\quick_test.py`. Its legacy timing and safety limitations remain visible. This does not test a fully trained v2 model.

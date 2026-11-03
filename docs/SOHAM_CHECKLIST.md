# Soham's completion checklist

Scope follows the uploaded work split: evaluation, analysis, dashboard, demonstrations, literature survey and report. Teammate branches and original network/vehicle files are preserved.

| Requirement | Delivered evidence | Status |
|---|---|---|
| Seeded three-controller harness | `traffic_eval`, `scripts/run_parallel_evaluation.py`, input/output hashes and per-run manifests | Implemented and executed |
| Both scenarios, 5-10 seeds | `results/model_validation`: 3 controllers x 2 scenarios x 5 seeds, 3600s per run | 30 runs complete; demand uncalibrated |
| Four-class 50/15/30/5 mix | Frozen explicit routes: 1000 peak and 500 offpeak vehicles | Implemented; all trips complete |
| Metric parsing | Waiting time, duration, mean/max approach queues, integrated CO2 plus quality fields | Tested against actual SUMO XML |
| Baseline sanity check | `baseline_checks.csv` | Actuated waiting-time check PASS in both scenarios |
| Mean/SD/paired percentages/charts | CSVs and `charts/` | Complete; all failed safety seeds retained |
| Streamlit dashboard | `dashboard.py`, tested data filters/exports/warnings | Complete |
| SUMO-GUI comparison | `demonstrations/three_controller_final/comparison.mp4`, original native frames and manifests | Complete sampled demonstration: 3 controllers, same seed/camera/clock sequence |
| Model loading and inference | Original 20k SUMO checkpoint, exact tensor comparison, compatibility loader and action traces | Complete; no model weights altered |
| Literature survey and bibliography audit | `LITERATURE_SURVEY.md`, all 22 entries in `REFERENCE_VERIFICATION.md` | Audit complete; specific primary/institutional checks still marked pending |
| Report | `report/report.pdf`, editable `.tex` and `.md` | Updated from all 30 actual measured runs |
| Turnitin <=15% | Institutional pre-check | Requires institutional access; no score fabricated |

## Evaluation findings and remaining team inputs

Two peak DQN runs produced collision-related teleports; numerical gains in that scenario cannot establish a safe policy benefit. See `AGENT_HANDOFF.md`. The current model is evaluated only at its trained junction, matching the actuated baseline's control scope; general corridor deployment remains an agent task. Field calibration, convergence/training logs, held-out training/test seed provenance and the 0.5 m versus 0.4 m resolution shift also remain team review items. These are not missing evaluation code.

All original runs, including safety failures, are retained. The report remains an implementation and measured-validation report rather than an unsupported final calibrated-corridor claim. Add official student identifiers/affiliation, complete institutional citation checks, compile IEEEtran in the required university template, and run the actual Turnitin pre-check before submission.

Historical integration evidence: 40 additional real-SUMO baseline runs are retained under `results/smoke_validated` and `results/provisional`. The full one-hour three-controller evidence is under `results/model_validation`.

Final software verification: 30 tests pass. The actual clone's QuickTest command completed all three controllers for 180 seconds each. Full raw reimport reproduces all 30 runs and safety metadata exactly. See VALIDATION.json.

The actual clone's live DQN GUI command also completed 300 simulated seconds and closed normally.

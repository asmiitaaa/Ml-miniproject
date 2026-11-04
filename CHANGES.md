# Agent software and calibration readiness

Agent implementation, full training, checkpoint selection and execution tests are available. A subsequent audit found that the existing routes cross only one of six controlled signal sites. The other five receive no traffic. The earlier description of a full corridor comparison overstated traffic coverage. Read [the agent report](docs/ANANYARAJ_COMPLETION.md) and [network/demand audit](report/network_validation/AUDIT.md).

- Five genuine full DQN runs: 500,000 decisions total. Both successful baseline-setting training curves met the empirical plateau criterion; unstable tuning runs are retained.
- New 17-feature all-lane environment and independent execution at all six mapped signal sites. Legacy v1/v2 checkpoints remain separate.
- Selected peak/offpeak checkpoints are in `models/corridor_peak_best.zip` and `models/corridor_offpeak_best.zip`, with embedded validation provenance and verified unchanged policy tensors.
- Ninety reserved-seed validation runs selected the v2 and corridor checkpoints; final test seeds were excluded from selection.
- Both 30-run comparisons completed all trips with zero observed collision events, teleports or unfinished trips, under the supplied single-active-junction demand.
- Corridor DQN reduced waiting versus fixed-time by 43.86% peak and 53.31% offpeak. Actuated remained better than DQN in both scenarios.
- Exact corridor phase traces passed the complete green, yellow and all-red sequence/duration audit.
- `rl_agent/run_policy.py` produces baseline-compatible raw output. `Start-Project.ps1` now defaults to the selected corridor models and current results.
- Current tables, training curves, hashes and reviews are under `report/training/`; original historical v1 tables and weights are preserved.

The earlier agent release passed 86 software tests; the audit adds regression checks for missing traffic exposure and invalid observation inputs. Real calibration is not complete because measured data are absent. Full-corridor demand, independent measured validation, network/site review, institutional submission checks and the final manuscript/video refresh remain. Historical training/model-selection records and raw metrics are preserved.

# Completed agent training and corridor handover

Ananyaraj's agent implementation, full training, per-scenario checkpoint selection and simulation testing are complete. Read [the completion report](docs/ANANYARAJ_COMPLETION.md) for the exact methods, results and copy-paste commands.

- Five genuine full DQN runs: 500,000 decisions total. Both successful baseline-setting training curves met the empirical plateau criterion; unstable tuning runs are retained.
- New 17-feature all-lane environment and independent execution at all six mapped signal sites. Legacy v1/v2 checkpoints remain separate.
- Selected peak/offpeak checkpoints are in `models/corridor_peak_best.zip` and `models/corridor_offpeak_best.zip`, with embedded validation provenance and verified unchanged policy tensors.
- Ninety reserved-seed validation runs selected the v2 and corridor checkpoints; final test seeds were excluded from selection.
- Both 30-run held-out comparisons completed all trips with zero observed collision events, teleports or unfinished trips.
- Corridor DQN reduced waiting versus fixed-time by 43.86% peak and 53.31% offpeak. Actuated remained better than DQN in both scenarios.
- Exact corridor phase traces passed the complete green, yellow and all-red sequence/duration audit.
- `rl_agent/run_policy.py` produces baseline-compatible raw output. `Start-Project.ps1` now defaults to the selected corridor models and current results.
- Current tables, training curves, hashes and reviews are under `report/training/`; original historical v1 tables and weights are preserved.

86 software tests pass, and both the standalone policy command and the current quick-test command completed in real SUMO. Remaining project inputs are demand/network field calibration, institutional submission checks, and refreshing the historical v1 IEEE manuscript/video for the new results. These simulation results do not certify real-world safety. The earlier implementation-only handover is retained in Git history at 5b93af8.

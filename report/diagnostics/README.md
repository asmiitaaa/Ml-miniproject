# V2 software verification and v1 forensic records

These records accompany `CHANGES.md`, dated 4 November 2026. They do not replace the historical 30-run v1 evaluation under `report/results/` and do not represent a converged v2 performance study.

- `v1_collision_events.csv`: four exact SUMO XML records exported to CSV: seed 11 at 2,311 s, seed 15 at 1,048 s, each replayed once with teleport response and once with diagnostic warn response. Replay durations were 2,400 s and 1,200 s respectively.
- `v1_diagnostic_metrics.json`: separate truncated replay metrics with mode, timing profile and warnings. Warn and teleport runs have deliberately different comparability fingerprints. Machine-specific result paths and wall time were removed. These truncated diagnostic means should not be compared to complete-hour published results.
- `v2_learning_smoke_verification.json`: successful reload of a real 1,200-decision / 50-update learning smoke checkpoint, embedded timing/provenance, and real SUMO scripted timing transitions. The checkpoint stayed in scratch storage; its filename does not make it a production model.
- `v2_scripted_timing_actions.csv`: only the separate repeated-switch timing test. These actions were scripted to exercise both yellow phases and were never used for DQN performance scores. Learned neural inference was tested separately for 180 s against both baselines.

Each v1 response mode produced one recorded collision per seed. Teleport mode produced one teleport per seed; warn mode produced none because the response changes. The records locate events between motorcycles on an incoming lane, and do not establish a yellow-timing cause or a safety repair.

72 automated checks passed. Original v1 weights and published result tables remain unchanged. A full new v2 training run, convergence review, all paired evaluations, and teammate review of the traffic model remain required.

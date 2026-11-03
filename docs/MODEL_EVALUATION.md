# Full-hour model validation fixture

This is a synthetic **uncalibrated model-validation experiment**, using the twelve approach/exit OD pairs and relative weights in the refreshed teammate branch's `demand/routes.xml`. It does not claim field-calibrated peak/off-peak conditions, trained-policy convergence, held-out demand from a recorded training seed set, or corridor-wide transfer.

`scripts/prepare_model_evaluation.py` builds frozen inputs before running any controller. The input snapshot is `scenarios/model_validation/source_routes.xml` (teammate commit `951ea33`). The script uses sumolib shortest paths with vehicle-class permissions, explicitly validates consecutive edge connections, and rejects unreachable class/OD combinations rather than enabling `ignore-route-errors`.

## Inputs

- Peak: 1,000 vehicles at requested departures 0, 3, 6, ... 2,997 seconds.
- Off-peak: 500 vehicles at requested departures 0, 6, 12, ... 2,994 seconds.
- Each scenario has exactly 50% motorcycles, 15% auto-rickshaws, 30% cars and 5% buses. Bicycles are excluded from both frozen type definitions and vehicles.
- Seed 42 freezes type order and weighted OD sampling once per scenario. Explicit route edges, vehicle type, initial lane and requested departure time are saved in XML. Simulator seeds 11-15 then vary vehicle behavior on those same files.
- Both scenarios run for 3,600 seconds with a 1-second simulation step and 600-second departure-free drain window. Requested departures are identical across controllers; congestion can still delay actual insertion, which is measured separately.
- Lateral resolution is 0.4 m, matching the uploaded methodology. The teammate training configuration used 0.5 m; this distribution shift remains a stated limitation.

## Matching controller scope

`configs/evaluation.model_validation.json` restricts **all three** controller conditions to the trained junction:

`cluster_11484393853_11484393854_11484393855_11484393856_#11more`

All other five signal programs remain fixed in all conditions. Fixed uses the original static program at the selected junction; actuated uses SUMO's gap-based program; DQN uses the selected 20k checkpoint and the compatibility adapter. This holds control coverage constant and does not claim independent learned control at every corridor signal.

The 20k checkpoint is selected before inspecting test results as the latest available SUMO-labelled checkpoint, as recorded in `models/provenance.json`. Older mock checkpoints are excluded. The adapter preserves its trained observation order and decision-boundary behavior, and records the effective training-replay yellow duration and native-network clearance mismatch. That compatibility mode does not establish safe field deployment.

## Reproduction

From the repository root, with Python dependencies installed:

```powershell
python scripts/prepare_model_evaluation.py
python scripts/prepare_model_evaluation.py --check
python -m traffic_eval run --config configs/evaluation.model_validation.json --controllers fixed actuated dqn --output results/model_validation
```

The fixture manifest records source/network/type hashes, every valid cached route, rejected class/OD combinations, exact type counts, OD counts, schedule and route hashes, generation seed, horizon and lateral-resolution choices. A second generation with the same inputs must produce identical route hashes. Input paths in all manifests should be retained with the artifacts when moving a run.

## Interpretation

Inspect completed and unfinished trips, pending departures, teleports, collisions and all quality warnings before interpreting averages. The drain window reduces cutoff bias but does not guarantee full completion. Report mean and sample SD across five paired simulator seeds, plus paired percentage improvements. A reduced metric is measured evidence for this fixture only; it is not a substitute for demand calibration or convergence logs.

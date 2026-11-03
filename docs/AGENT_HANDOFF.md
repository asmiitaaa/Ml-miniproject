# Agent and simulation handoff

Soham's evaluation layer is implemented and the full 30-run matrix is retained. The following findings belong to the agent/simulation review before a final research claim.

| Run | SUMO time | Lane | Logged finding |
|---|---:|---|---|
| peak/dqn/seed 11 |2311s|631316440#0_0|peak_0766 teleported after collision with peak_0763; gap -1.25m |
| peak/dqn/seed 15 |1048s|631316440#0_0|peak_0339 teleported after collision with peak_0346; gap -2.89m |

Each event produces two colliding-vehicle detections and one teleport start. Totals: four vehicle detections, two teleport starts, two affected runs. All other 28runs record neither collision detections nor teleports. All scheduled trips finished in every run, but teleporting changes journeys and limits the peak DQN comparisons. Original errors are in `results/model_validation/raw/peak/dqn/seed_11/sumo-errors.log` and the corresponding seed 15 folder. No failed seed was removed.

The cause is not established. Review car-following/deceleration and sublane interaction, demand and phase transitions together. The replay adapter intentionally reproduces training: nominal 3s yellow becomes 5s at five-second decision boundaries, versus 6s nominal yellow in the original network. Optional `network-clearance` mode enforces the longer interval but changes training dynamics and requires a separate experiment; it must not silently replace the measured policy.

The checkpoint observes only four ordered primary lanes at one junction. Fair comparison here restricts actuated and DQN control to that same junction and holds the other five signals fixed. Corridor-wide deployment requires compatible lane/phase mappings and independent validation. Training used 0.5 m lateral resolution; the uploaded methodology and evaluation use 0.4 m. Training curves and training/test seed provenance were not provided. The latest available 20k checkpoint was selected before inspecting test scores, not tuned using these five test seeds.

Synthetic scenarios contain the specified 50/15/30/5 four-class mix, but no field calibration is asserted. Resolve these findings in the simulation/agent phases, rerun the same evaluation commands, and regenerate the report from the new evidence. Do not change recorded results or hide warning flags.

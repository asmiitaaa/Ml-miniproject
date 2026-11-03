# SUMO checkpoint integration: v1 and v2

The factory `traffic_eval.agent_adapter:create_controller` now supports two explicit timing profiles. The original v1 checkpoint and published results are retained for legacy/forensic replay. New v2 training and evaluation must use the same contract:

| Profile | Decision interval | Minimum green | Nominal yellow | Effective yellow |
| --- | ---: | ---: | ---: | ---: |
| v1 | 5 s | 10 s | 3 s | 5 s |
| v2 | 3 s | 12 s | 6 s | 6 s |

Use `"timing_profile": "v2"` with a newly trained v2 checkpoint. The default remains v1 so old experiments reproduce their original behavior. Explicit conflicting timing options, a renamed v1 checkpoint, a wrong recorded profile, or a different recorded junction/lane ordering are rejected. v2 requires embedded metadata written by the updated trainer; editing an evaluation setting cannot convert v1 weights into a v2-trained policy.

## Features and signal ordering

The unnormalized 13-value float32 observation contains four primary-lane vehicle counts, four halting counts, four lane waiting times, and the internal green-phase index (0 or 2; yellow maps to its preceding green). Values are clipped to `[100]*8 + [3600]*4 + [3]`. The vehicle counts are not the evaluation's all-lane queue metric.

Ordered training lanes are `24480656#3_0`, `794634832#0_0`, `794634831#1_0`, `631316440#0_0`. The controlled junction is `cluster_11484393853_11484393854_11484393855_11484393856_#11more`. Other signals retain fixed timing for the same comparison scope as the actuated baseline. Corridor transfer requires a separately validated feature mapping.

Inference is deterministic. The current internal phase is applied for one decision block, then the internal state machine advances. A switch request persists until minimum green; requests during yellow are ignored. This reproduces the training environment's ordering. The adapter holds native phase duration until the next decision so SUMO cannot advance independently.

The v1 optional `signal_mode: network-clearance` changes dynamics and rounds native six-second yellow to ten seconds at v1 decision boundaries. It is a separate diagnostic, not equivalent-policy evidence. v2 requires `training-replay` and rejects a network whose nominal yellow exceeds six seconds. Matching the native duration alone does not certify traffic safety.

## Training provenance

`rl_agent/train.py` prepares a private run configuration with 0.4 m lateral resolution, seed 42, `random=false`, strict route errors, and default teleport collision response. It preserves the original source scenario, network and vehicle files. Evaluation seeds 11-15 are disjoint from this explicit training seed, but one training seed does not establish robustness.

The saved SB3 archive embeds timing, ordered lanes/junction, training input/source/configuration hashes, model/SUMO seeds, lateral resolution and training status. Output logs are isolated under the selected output directory. The loader's buffered tensor-stream compatibility fix remains in place and does not change weights.

## Run

From the repository root, after completing the new training run:

```powershell
.\.venv\Scripts\python.exe scripts\quick_test.py --config configs\evaluation.v2.example.json
.\.venv\Scripts\python.exe -m traffic_eval run --config configs\evaluation.v2.example.json --output results\v2_evaluation
```

The example uses two frozen synthetic demand levels, five seeds and all three controllers, with 0.4 m lateral resolution. It is an uncalibrated model validation, not field evidence. Read `CHANGES.md` for training commands and required follow-up.

Every evaluation run now writes `collisions.xml` and `collisions.csv`. DQN runs also write `dqn_actions.csv`. Manifests record output hashes, timing metadata, collision action, seed, input hashes and checkpoint hash. Diagnostic `collision_action: warn` results are explicitly flagged and kept separate from normal evaluation.

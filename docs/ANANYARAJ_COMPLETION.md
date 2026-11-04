# Agent implementation, training and traffic-coverage limits

Implemented and trained: Gymnasium/TraCI environment, checkpoints, hyperparameter comparison, independent signal execution and validation-based selection. **Scope correction:** the supplied peak/offpeak routes cross only one of the six controlled signal sites. The other five controllers execute without traffic. Existing comparisons establish performance only for this single-active-junction synthetic demand, not for interacting corridor traffic.

## Implementation

`rl_agent/corridor_env.py` trains on one real intersection. `rl_agent/corridor.py` applies the same trained network independently at every mapped signal. The supplied network has three multi-approach junctions and three additional one-approach signal sites; all six sites are controlled. `configs/corridor_mapping.json` explicitly records cardinal approach groups, every incoming lane, controlled link and source phase state. Geometry, lane/link membership, phase program and mapping signature are checked at loading. Unknown mappings and the legacy 13-feature checkpoints are rejected.

The new 17-value observation contains four all-lane vehicle counts, four halting queues, four summed lane waiting times, four fractions of approach links currently green, and phase age. Counts, waits and age use fixed scaling and clipping; empty cardinal groups have explicit zeros. The reward is the negative sum of waiting times over all incoming lanes at the training junction, without reward scaling or a substituted heuristic controller.

Actions are hold or request the next source phase. Decisions occur every 3 s, retaining the team's timing correction; this is an explicit change from the original document's 5 s draft. The signal state advances every 1 s, with 12 s minimum green, 60 s maximum green, at least 6 s yellow and all native all-red phases preserved. All source phase states remain in their original cyclic order, including the eight-phase program. The same 6 s yellow floor is applied to the derived networks for fixed, actuated and DQN comparisons. Original network, route and vehicle files remain intact.

The original v1 checkpoint, historical results, and the 13-feature v2 family remain available. The new feature contract has freshly trained weights and is never obtained by relabelling those older models.

## Training and checkpoint selection

Five full 100,000-decision runs completed: the corrected legacy v2 run, plus peak/offpeak trials at learning rate 0.001 with exploration fraction 0.3 and at 0.0003 with exploration fraction 0.5. That totals 500,000 decisions and 123,750 optimizer updates. Each trial includes 83 complete one-hour episodes plus the final partial episode. The replay buffer is 50,000, learning starts after 1,000 decisions, batch size is 64, and the target-network interval is 1,000 decisions.

Corridor training used SUMO seeds 42-44 for the first setting and 142-144 for the second, with model seeds 42 and 142. Validation used 101-103; final comparison used 11-15. Checkpoint selection never reads final-test rows. Five snapshots from each of four trials were evaluated: 60 one-hour validation runs. The legacy v2 selection separately used 30 validation runs.

The release rule requires zero collision detections/events and teleports, no unfinished or pending trips, and exactly 1,000 peak or 500 offpeak completions on every validation seed. Eligible candidates are ranked by mean completed-trip waiting time, with earlier training steps breaking ties. The selected models are:

| Scenario | Selected training decisions | Full source-run plateau | Policy tensor SHA-256 |
| --- | ---: | --- | --- |
| peak | 20,000 | True | `d8a479a6251ac76e857ee1919c90c600aa04c50cdcacfd28c9dba1916f6655a3` |
| offpeak | 100,000 | True | `d4d3b85e19f440613a18338dfcbd5271bbc71b76f80248aee9082745e8950b2f` |

Peak selected the 20,000-step snapshot, which tied later snapshots on validation and won the declared earlier-step tie-break. Its own 16-episode prefix is not labelled converged; the complete 100,000-step source run met the declared empirical plateau criterion. Offpeak selected the 100,000-step snapshot. The 0.001 trials met the plateau criterion; the lower-rate trials failed it and remain visible in the histories and curves. Plateau means a recorded stability heuristic, not a proof of optimality.

Selected archives include their validation review and source checkpoint hash. Reloaded policy tensors were verified bit-for-bit identical to the selected training snapshot after attaching that metadata. Full hashes and records are in [published models](https://github.com/asmiitaaa/Ml-miniproject/blob/soham/evaluation/report/training/corridor_selection/published_models.json).

Training histories, settings, source/input hashes and convergence reviews are under [report/training](https://github.com/asmiitaaa/Ml-miniproject/tree/soham/evaluation/report/training). Original source snapshots and per-episode training collision/trip logs remain in the ignored local trial folders under `results/agent_training`.

## Held-out results for the supplied single-active-junction demand

Three controllers x two scenarios x five seeds completed for each model family: 30 runs for the selected v2 policies and another 30 for the independent corridor policies. Every scheduled vehicle completed. Both matrices recorded zero collision events, zero teleports and zero unfinished trips.

The waiting-time results below are seed means +/- sample standard deviations, in seconds. Only the trained junction carries traffic:

| Scenario | Fixed waiting (s) | Actuated waiting (s) | DQN waiting (s) |
| --- | ---: | ---: | ---: |
| peak | 18.081 +/- 0.122 | 9.156 +/- 0.435 | 10.150 +/- 0.164 |
| offpeak | 17.862 +/- 0.052 | 7.857 +/- 0.193 | 8.339 +/- 0.148 |

Seed-paired waiting reductions versus fixed-time are **43.86% peak** and **53.31% offpeak** for DQN. Actuated outperformed DQN in both scenarios. This is a measured reduction relative to fixed-time, not a claim that DQN is the best controller.

Exact phase traces from all ten corridor DQN test runs covered all six signal sites. The audit checked 21,665 completed phase dwells and found zero phase skips, green-duration violations, or yellow/all-red-duration violations. The controller also checks the actual SUMO phase before every one-second update. The final, unfinished phase in each trace is treated as right-censored.

The [portable comparison tables and figures](https://github.com/asmiitaaa/Ml-miniproject/tree/soham/evaluation/report/training/corridor_results) retain all five seeds and quality warnings. Local raw XML, actions, phase traces and input/output integrity hashes are under `results/corridor_selected_evaluation`. All numerical comparisons use actual deterministic neural predictions and normal teleport collision response. Scripted timing checks and diagnostic warn-mode replays are separate.

Software verification finished with 86 tests passing. The standalone policy command and the user's short three-controller command also completed in real SUMO. The original IEEE manuscript and video remain historical v1 artifacts; the final submission should use the current results and this training review.

## Run on this computer

No environment activation or PowerShell execution-policy change is needed:

```powershell
Set-Location C:\Users\soham\Ml-miniproject
# Short test of the selected peak model and both baselines
.\.venv\Scripts\python.exe scripts\quick_test.py --config configs\evaluation.corridor.json
# Dashboard with all 30 completed corridor comparisons
.\.venv\Scripts\python.exe -m streamlit run dashboard.py -- --results report/training/corridor_results
# Actual SUMO-GUI replay, paced so you can watch it
.\.venv\Scripts\python.exe scripts\preview_gui.py --config configs\evaluation.corridor.json --scenario peak --controllers dqn --seconds 300 --every 15 --delay-ms 0 --pace-ms 100
# Standalone one-hour policy run with the same raw output schema as baselines
.\.venv\Scripts\python.exe rl_agent\run_policy.py --scenario offpeak --seed 11
```

For a fresh complete comparison:

```powershell
$agentResults = 'results\corridor_recheck_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe scripts\run_parallel_evaluation.py --config configs\evaluation.corridor.json --output $agentResults --jobs 3
.\.venv\Scripts\python.exe scripts\audit_signal_traces.py --raw ($agentResults + '\raw') --output ($agentResults + '\phase_audit.json')
```

Training can be reproduced with `rl_agent/train_corridor.py`; use fresh output folders. The search plan and saved training manifests record every trial's seeds and parameters. Existing outputs and selected weights are protected from overwrite. Institutional similarity/reference checks and Asmita's field calibration are separate remaining inputs.

## Practical limits

The supplied demand is synthetic and uncalibrated. Only one junction carries traffic; zero-demand execution at five other sites is not a traffic-transfer validation. `final_evaluation_ready` remains false. Zero observed events on these seeds does not certify real-road safety. Detailed historical v1 records located incoming-lane motorcycle collisions; these results do not establish that yellow timing caused them. New demand must exercise the corridor and be checked against measured traffic before performance can be generalized. See [the network and coverage audit](https://github.com/asmiitaaa/Ml-miniproject/blob/soham/evaluation/report/network_validation/AUDIT.md).

Method references: [SB3 evaluation guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html), [SUMO controlled signal links](https://sumo.dlr.de/docs/TraCI/Traffic_Lights_Value_Retrieval.html), [SUMO lane quantities](https://sumo.dlr.de/docs/TraCI/Lane_Value_Retrieval.html).

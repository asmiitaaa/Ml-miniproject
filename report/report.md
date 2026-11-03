# Dynamic Traffic Congestion Control and Signal Optimization Using Deep Reinforcement Learning

Asmita | Ananyaraj | Soham

3 November 2026

**IMPLEMENTATION REPORT - FINAL CALIBRATED EVALUATION PENDING**

## Abstract

This project specifies a SUMO-based experiment for adaptive traffic signal control under heterogeneous traffic. A DQN is intended to retain or advance each signal phase from queue, count and waiting-time observations, subject to minimum-green and yellow intervals. The evaluation layer compares fixed-time, actuated and learned control using identical seeded scenario inputs and records waiting time, trip duration, queue statistics and CO2. This report documents the implemented evaluation and reporting workflow and separates genuine validation runs from the final research experiment. A refreshed teammate branch supplies real-SUMO training code, a one-hour demand file and 10,000- and 20,000-step SUMO-labelled checkpoints. The 20,000-step checkpoint was evaluated in 30 one-hour runs covering three controllers, two synthetic demand levels and five paired seeds. Two peak-demand DQN runs recorded collision-related teleports, which limit interpretation of speed improvements. Independent evaluation does not establish training convergence or scenario calibration. Final conclusions require reviewed peak/off-peak demand, compatible policy deployment and complete paired experiments.

Keywords: Deep reinforcement learning; DQN; traffic signal control; SUMO; reproducible evaluation.

## I. Introduction

Signal timing allocates scarce junction capacity between competing approaches. A fixed program is easy to reproduce, but cannot change its split in response to an unusual queue. An actuated program uses observed arrivals to adjust green durations. Reinforcement learning offers a third approach: a policy can learn the long-term consequences of its decisions from repeated interactions with a traffic simulator. The practical question is whether that policy improves a fair baseline comparison on a specified demand distribution, rather than whether a reward curve rises during training.

The study targets a mixed-traffic Indian corridor. Its planned contribution is a reproducible comparison of efficiency and emissions, including the stronger actuated baseline. Current engineering deliverables support that comparison; the final empirical claim depends on the simulation and learned-policy inputs described below.

## II. Related Work

Haydari and Yilmaz [1] survey deep RL transport applications by problem formulation and algorithm. Noaeen et al. [2] review network-level traffic signal research and identify a gap between research systems and transport practice. These reviews motivate explicit scenario definitions and reproducibility rather than broad claims about universal superiority.

Liang et al. [4] represent traffic spatially and combine several value-learning refinements to adjust cycle timings. Their architecture differs from the small feedforward DQN planned here. Bouktif et al. [5] study consistency between state and reward using DDQN and prioritized replay, supporting a clear connection between the waiting-time reward and the delay measures used in evaluation.

Chu et al. [3] address large networks using decentralized advantage actor-critic agents with neighboring information. Wu et al. [7] use recurrent multi-agent deterministic policy gradients, centralized critics and decentralized execution. Zhu et al. [8] use broad networks with ridge regression and an interaction mechanism to reduce training cost. These are coordination methods; independently deploying the same DQN at several signals does not implement their communication schemes.

Mao et al. [6] provide a useful algorithm-comparison reference, but its primary article remains to be checked through institutional IEEE access. It is retained as a bibliographic lead without importing detailed outcome claims. Han et al. [9] highlight training/testing separation and simulator fidelity as deployment concerns. Tan et al. [10] address noisy traffic observations through perturbed-state training. The present evaluation does not establish robustness to sensor failure or field transfer.

## III. Research Question and Scope

The primary question is whether a traffic-trained DQN lowers completed-trip waiting time relative to fixed-time and actuated control on paired peak and off-peak scenarios. Secondary outcomes are completed-trip duration, mean and maximum approach queue, and total simulated CO2. Completion, pending insertions, teleports and collisions are quality indicators needed to interpret those outcomes.

Heterogeneity is a modeling assumption until calibrated against traffic observations. The specified vehicle proportions are 50% two-wheelers, 15% auto-rickshaws, 30% cars and 5% buses, with 0.4 m lateral resolution. The supplied vehicle file also defines bicycles; defining that type does not mean it should appear in the four-class experimental mix. The supplied network has six signal programs, whereas the planning document proposes three to five intersections. The team must document a reviewed subset or justify the expanded corridor before final evaluation.

## IV. Methodology

A. Network and demand. The road network is imported from OpenStreetMap and reviewed for lane connectivity, permitted turns, signal programs and vehicle-class access. Peak and off-peak routes are frozen once per scenario and reused without modification across all controllers. Input hashes and the simulator version are recorded with each run. RandomTrips-based demand generation and field calibration remain simulation-phase responsibilities; a provisional test route set is not calibrated field evidence.

B. Learning environment. The specified Gymnasium environment communicates through TraCI. Its observation includes per-approach counts, halting queues, cumulative waiting and phase identity. Every five simulated seconds the action retains the current phase or advances in its prescribed cycle. Minimum green and yellow transitions are enforced by the environment. Reward is the negative total approach waiting time. The refreshed teammate branch provides real TraCI reset, stepping and observation methods and a SUMO-mode training configuration. The model observes one selected incoming lane from each of four approaches at a single junction. Other deployment layouts require explicit lane-order and phase-map validation; the policy cannot be assumed to generalize to every signal in the corridor.

C. Training protocol. The planned Stable-Baselines3 DQN uses PyTorch, experience replay, a periodically updated target network and epsilon-greedy exploration. Starting settings are approximately 0.001 learning rate, 50,000 replay transitions and one-hour episodes. Hyperparameters and checkpoint selection use training/validation seeds only; held-out test seeds must not be used to select the best model. Independent per-signal deployment must match the observation encoding used at training time. The available checkpoint filenames indicate 10,000 and 20,000 training steps; training curves, convergence evidence and checkpoint-selection records are not supplied. Training uses 0.5 m lateral resolution, whereas this evaluation uses the documented 0.4 m. This is a distribution shift that must be reported and addressed before a final controlled experiment. Older mock checkpoints remain excluded from traffic claims.

D. Comparative experiment. For each demand scenario, run all three controllers on five to ten identical test seeds. Fixed and actuated programs share network geometry, vehicle inputs, run horizon, lateral resolution and simulator settings. An external DQN runner must emit the same output contract. Raw XML, queue telemetry, command logs and provenance are retained. The actuated-versus-fixed check is an empirical diagnostic: a failure triggers investigation of detection coverage and timings, not editing measurements to force the expected ranking.

## V. Metrics and Statistical Reporting

Waiting time is SUMO tripinfo waitingTime averaged over completed, non-vaporized trips [11]; duration is arrival minus actual departure as recorded by tripinfo. DepartDelay is a separate insertion measure and must not be mistaken for junction waiting. Unfinished trips and vehicles unable to insert are reported because completed-trip means can improve artificially when the longest trips have not finished.

Queue telemetry samples the number of halting vehicles, including zero counts, at each unique incoming approach at every simulator step. The mean is over approach-time samples and the maximum is over those samples. Queue counts are not queue distances. CO2 rates in emission output are integrated over the simulation step and converted from milligrams to kilograms [12], including vehicles still present at the run cutoff. Emission classes and simulator versions must be held fixed and reported; modeled emissions are not physical measurements.

For each controller and scenario the report gives the mean and sample standard deviation across seeds, not across individual vehicles. For each matched seed, percentage improvement is 100*(baseline - controller)/baseline; these paired percentages are then summarized. A zero baseline yields an undefined percentage and is counted explicitly rather than replaced with zero. Positive improvement means the lower-is-better metric decreased. Five seeds provide an exploratory comparison, not proof of statistical significance.

## VI. Implementation and Evidence Status

The evaluation package parses tripinfo and emission XML, records live approach queues through TraCI, validates paired demand fingerprints, and writes run-level, summary and paired-improvement CSV files with diagnostic flags. Matplotlib charts and a Streamlit dashboard expose scenario, controller and seed comparisons alongside completion quality. A report builder creates this editable report from the available measured summaries.

The original clone supplied road geometry, vehicle definitions and notes. Refreshing the teammate branch exposed additional real-SUMO environment code, a one-hour route file and SUMO-labelled model checkpoints alongside older mock checkpoints. The 20,000-step SUMO checkpoint was selected before inspecting evaluation scores because it was the latest available model. All twelve loaded tensors match the original stored values; buffering tensor archive streams resolves a Windows reader incompatibility without changing model weights or checkpoint bytes. Thirty full-hour three-controller runs, tables, charts, dashboard views and a matched 90-second sampled GUI comparison are supplied. Each controller acts at the trained junction; the other five signals retain fixed plans. Training convergence, calibrated field demand and transfer to every corridor junction remain unverified.

## VII. Measured Validation Results

Measured validation results only. These values are not a final calibrated corridor performance claim.

The measured dataset contains 30 real-SUMO runs across 2 demand settings and seeds 11, 12, 13, 14, 15. Controllers: actuated, dqn, fixed. Each table reports the seed mean (sample standard deviation), with seed counts 5 per controller/scenario.

Completed-trip fractions range from 100.00% to 100.00%. All requested vehicles completed within the horizon. The runs record 2 teleport starts and 4 colliding-vehicle detections. The latter counts involved vehicles, not distinct collision events.

Safety flags occur in peak/dqn/seed 11, peak/dqn/seed 15. Collision-related teleports alter journeys, so speed and emissions improvements in the affected scenario are diagnostic measurements, not evidence of a safe policy benefit. All seeds remain in the tables; no failed seed was discarded.

The recorded simulator is SUMO 1.28.0, with a 3600.0-second horizon and 1.0-second step. Demand remains synthetic and uncalibrated; all controllers share the trained-junction scope.

For offpeak, the paired actuated-versus-fixed waiting-time reduction is 58.48% (sample SD 1.87 percentage points; 5 pairs); the diagnostic status is PASS. This is a baseline integration result, not a DQN result or evidence of calibrated field demand.

For peak, the paired actuated-versus-fixed waiting-time reduction is 49.26% (sample SD 1.44 percentage points; 5 pairs); the diagnostic status is PASS. This is a baseline integration result, not a DQN result or evidence of calibrated field demand.

For offpeak, measured DQN waiting-time reduction relative to fixed is 61.76% (sample SD 0.48 percentage points; 5 pairs). These values must be interpreted together with the safety and calibration limitations, especially the flagged peak runs.

For offpeak, measured DQN waiting-time reduction relative to actuated is 7.74% (sample SD 4.72 percentage points; 5 pairs). These values must be interpreted together with the safety and calibration limitations, especially the flagged peak runs.

For peak, measured DQN waiting-time reduction relative to fixed is 55.14% (sample SD 0.78 percentage points; 5 pairs). These values must be interpreted together with the safety and calibration limitations, especially the flagged peak runs.

For peak, measured DQN waiting-time reduction relative to actuated is 11.51% (sample SD 3.38 percentage points; 5 pairs). These values must be interpreted together with the safety and calibration limitations, especially the flagged peak runs.

Training-replay yellow lasts 5s, shorter than the network's 6s nominal yellow. This reproduces checkpoint training but does not validate corridor clearance safety.

Policy controls only the trained junction's four primary lanes; other traffic signals retain the fixed plan. This is not validated corridor-wide deployment.

These runs are labelled demonstration/provisional or lack project provenance; they are not final corridor evidence.

Teleports occurred. Correct network/demand problems before using results in the report.

Collisions occurred. Inspect signal safety and simulation settings before trusting results.

### offpeak

| Metric | fixed | actuated | dqn |

|---|---|---|---|

| Waiting (s) | 17.86 (0.05) | 7.42 (0.34) | 6.83 (0.08) |

| Duration (s) | 46.86 (0.15) | 36.68 (0.56) | 36.32 (0.09) |

| Mean queue (veh) | 0.20 (0.00) | 0.09 (0.00) | 0.08 (0.00) |

| Max queue (veh) | 7.20 (0.45) | 4.40 (0.55) | 5.00 (0.00) |

| CO2 (kg) | 54.38 (0.17) | 47.07 (0.44) | 46.76 (0.12) |

Mean (sample SD). fixed: n=5; actuated: n=5; dqn: n=5

### peak

| Metric | fixed | actuated | dqn |

|---|---|---|---|

| Waiting (s) | 18.08 (0.12) | 9.17 (0.29) | 8.11 (0.14) |

| Duration (s) | 47.78 (0.17) | 39.13 (0.37) | 38.19 (0.19) |

| Mean queue (veh) | 0.41 (0.00) | 0.22 (0.01) | 0.19 (0.00) |

| Max queue (veh) | 12.20 (0.45) | 10.40 (1.67) | 8.00 (0.71) |

| CO2 (kg) | 111.42 (0.31) | 99.49 (0.62) | 98.30 (0.32) |

Mean (sample SD). fixed: n=5; actuated: n=5; dqn: n=5

## VIII. Limitations and Completion Plan

Before final submission, calibrate and review peak/off-peak inputs, resolve the observed DQN collision events, verify minimum-green and network clearance behavior, and document disjoint training and test seeds. Inspect every quality warning, explain the actuated baseline outcome and retain failed runs. Matched baseline-versus-DQN SUMO-GUI captures record scenario, seed, checkpoint and camera settings. Report simulation-only limits and avoid claims of deployment readiness.

All 22 supplied literature entries have an audit trail; corrected issue years, pages and article identifiers are recorded separately. Several primary IEEE/Elsevier checks and institutional Scopus verification remain pending. The report uses newly written prose, but the requested Turnitin similarity threshold of at most 15% requires an actual institutional pre-check. No similarity score is asserted. Add the full names, roll/registration numbers and institutional affiliation, review the final IEEEtran compilation and update the results section only from genuine outputs.

## References

1. A. Haydari and Y. Yilmaz, "Deep reinforcement learning for intelligent transportation systems: A survey," IEEE Trans. Intell. Transp. Syst., vol. 23, no. 1, pp. 11-32, 2022. https://doi.org/10.1109/TITS.2020.3008612

2. M. Noaeen et al., "Reinforcement learning in urban network traffic signal control: A systematic literature review," Expert Syst. Appl., vol. 199, Art. no. 116830, 2022. https://doi.org/10.1016/j.eswa.2022.116830

3. T. Chu, J. Wang, L. Codeca and Z. Li, "Multi-agent deep reinforcement learning for large-scale traffic signal control," IEEE Trans. Intell. Transp. Syst., vol. 21, no. 3, pp. 1086-1095, 2020. https://doi.org/10.1109/TITS.2019.2901791

4. X. Liang, X. Du, G. Wang and Z. Han, "A deep reinforcement learning network for traffic light cycle control," IEEE Trans. Veh. Technol., vol. 68, no. 2, pp. 1243-1253, 2019. https://doi.org/10.1109/TVT.2018.2890726

5. S. Bouktif, A. Cheniki, A. Ouni and H. El-Sayed, "Deep reinforcement learning for traffic signal control with consistent state and reward design approach," Knowl.-Based Syst., vol. 267, Art. no. 110440, 2023. https://doi.org/10.1016/j.knosys.2023.110440

6. F. Mao, Z. Li and L. Li, "A comparison of deep reinforcement learning models for isolated traffic signal control," IEEE Intell. Transp. Syst. Mag., vol. 15, no. 1, pp. 160-180, 2023. [Primary article check pending.] https://doi.org/10.1109/MITS.2022.3144797

7. T. Wu et al., "Multi-agent deep reinforcement learning for urban traffic light control in vehicular networks," IEEE Trans. Veh. Technol., vol. 69, no. 8, pp. 8243-8256, 2020. https://doi.org/10.1109/TVT.2020.2997896

8. R. Zhu, L. Li, S. Wu, P. Lv, Y. Li and M. Xu, "Multi-agent broad reinforcement learning for intelligent traffic light control," Inf. Sci., vol. 619, pp. 509-525, 2023. https://doi.org/10.1016/j.ins.2022.11.062

9. Y. Han, M. Wang and L. Leclercq, "Leveraging reinforcement learning for dynamic traffic control: A survey and challenges for field implementation," Commun. Transp. Res., vol. 3, Art. no. 100104, 2023. https://doi.org/10.1016/j.commtr.2023.100104

10. K. L. Tan, A. Sharma and S. Sarkar, "Robust deep reinforcement learning for traffic signal control," J. Big Data Anal. Transp., vol. 2, pp. 263-274, 2020. https://doi.org/10.1007/s42421-020-00029-6

11. Eclipse SUMO, "TripInfo," SUMO Documentation, accessed Nov. 3, 2026. https://sumo.dlr.de/docs/Simulation/Output/TripInfo.html

12. Eclipse SUMO, "Emission Output," SUMO Documentation, accessed Nov. 3, 2026. https://sumo.dlr.de/docs/Simulation/Output/EmissionOutput.html
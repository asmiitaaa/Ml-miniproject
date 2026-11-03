"""Compatibility adapter for SUMO-trained v1 and v2 DQN checkpoints.

This reproduces environment.py + signal_logic.py from origin/ananyaraj,
including the decision-boundary ordering and clipped primary-lane features.
It deliberately controls only the junction that supplied the training state.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from .timing import timing_settings, validate_checkpoint_timing

TRAINED_TLS = "cluster_11484393853_11484393854_11484393855_11484393856_#11more"
TRAINED_LANES = ["24480656#3_0", "794634832#0_0", "794634831#1_0", "631316440#0_0"]
OBS_HIGH = np.array([100.]*8 + [3600.]*4 + [3.], dtype=np.float32)


class TrainedSignalController:
    def __init__(self, connection, model, options):
        self.connection = connection
        self.model = model
        self.options = options
        self.tls_id = options.get("tls_id", TRAINED_TLS)
        self.lane_ids = list(options.get("lane_ids", TRAINED_LANES))
        # Different lane IDs require an explicit signed-off feature mapping;
        # the default adapter never guesses cardinal directions or padding.
        if self.tls_id != TRAINED_TLS or self.lane_ids != TRAINED_LANES:
            raise ValueError("This checkpoint adapter supports only its trained junction and ordered primary lanes. Corridor transfer needs a separately validated adapter and explicit approach mapping.")
        if options.get("control_tls_ids") and set(options["control_tls_ids"]) != {TRAINED_TLS}:
            raise ValueError("This adapter controls exactly its trained junction")
        self.timing_profile, timing = timing_settings(options)
        self.interval = timing["decision_interval_s"]
        self.min_green = timing["min_green_s"]
        step_length = float(options.get("_step_length_s",1))
        if not math.isclose(self.interval/step_length, round(self.interval/step_length)):
            raise ValueError("Decision interval must be an integer multiple of simulation step")
        if self.tls_id not in connection.trafficlight.getIDList():
            raise ValueError(f"Trained traffic-light ID missing: {self.tls_id}")
        controlled = set(connection.trafficlight.getControlledLanes(self.tls_id))
        if not all(lane in controlled for lane in self.lane_ids):
            raise ValueError("Training lanes must belong to the trained junction")
        logics = connection.trafficlight.getAllProgramLogics(self.tls_id)
        program = connection.trafficlight.getProgram(self.tls_id)
        logic = next((logic for logic in logics if logic.programID == program), None)
        if logic is None or len(logic.phases) != 4:
            raise ValueError("Checkpoint expects the original four-phase signal program")
        for i, phase in enumerate(logic.phases):
            state = phase.state.lower()
            if (i in (0,2) and ('g' not in state or 'y' in state)) or (i in (1,3) and ('y' not in state or 'g' in state)):
                raise ValueError("Signal phase order must be green A, yellow A, green B, yellow B")
        self.native_yellow = max(float(logic.phases[i].duration) for i in (1,3))
        self.signal_mode = options.get("signal_mode", "training-replay")
        if self.signal_mode not in ("training-replay", "network-clearance"):
            raise ValueError("signal_mode must be training-replay or network-clearance")
        if self.timing_profile == "v2" and self.signal_mode != "training-replay":
            raise ValueError("v2 already uses the trained 6-second clearance; signal_mode overrides are not allowed")
        self.yellow = timing["yellow_s"] if self.signal_mode == "training-replay" else max(timing["yellow_s"],self.native_yellow)
        effective_yellow = math.ceil(self.yellow/self.interval)*self.interval
        if self.timing_profile == "v2" and effective_yellow < self.native_yellow:
            raise ValueError("v2 yellow is shorter than this network's nominal clearance")
        warnings = []
        if getattr(model,"training_status",None)=="smoke-test-only":
            warnings.append("This checkpoint was trained only for a software smoke test; it is not a converged traffic policy.")
        if getattr(model,"training_status",None)=="convergence-not-assessed":
            warnings.append("Training convergence has not been assessed; a recorded step count alone does not establish policy quality.")
        if self.timing_profile == "v2" and getattr(model,"training_sumo_seed",None) is None:
            warnings.append("The training environment did not record an explicit SUMO seed; held-out seed separation is unverified.")
        if effective_yellow < self.native_yellow:
            warnings.append(f"Training-replay yellow lasts {effective_yellow:g}s, shorter than the network's {self.native_yellow:g}s nominal yellow. This reproduces checkpoint training but does not validate corridor clearance safety.")
        if self.signal_mode == "network-clearance":
            warnings.append("Network-clearance mode changes the checkpoint's training dynamics; evaluate separately and retrain before treating this as an equivalent policy.")
        if self.timing_profile == "v1":
            warnings.append("v1 is retained for legacy/forensic replay; it is not the corrected v2 controller.")
        trained_resolution = getattr(model,"training_lateral_resolution_m",None)
        evaluated_resolution = options.get("_lateral_resolution_m")
        if trained_resolution is not None and evaluated_resolution is not None and not math.isclose(float(trained_resolution),float(evaluated_resolution)):
            warnings.append(f"Training lateral resolution {trained_resolution}m differs from evaluation {evaluated_resolution}m; distribution shift requires review.")
        warnings.append("Policy controls only the trained junction's four primary lanes; other traffic signals retain the fixed plan. This is not validated corridor-wide deployment.")
        self.metadata = {"adapter":"ananyaraj-sumo-"+self.timing_profile, "timing_profile":self.timing_profile, "signal_mode":self.signal_mode,
            "decision_interval_s":self.interval,"min_green_s":self.min_green,
            "controlled_tls_ids":[self.tls_id], "lane_ids":self.lane_ids,
            "observation_semantics":"primary-lane vehicle count x4, halting count x4, lane waiting time x4, internal green phase index",
            "observation_clipping_high":OBS_HIGH.tolist(), "nominal_training_yellow_s":timing["yellow_s"],
            "effective_evaluation_yellow_s":effective_yellow, "network_nominal_yellow_s":self.native_yellow,
            "model_num_timesteps":int(getattr(model,"num_timesteps",0)), "warnings":warnings}
        for key in ("training_status","training_sumo_seed","training_model_seed","training_lateral_resolution_m",
                    "training_source_sha256","training_config_sha256","training_input_sha256"):
            if hasattr(model,key):self.metadata[key]=getattr(model,key)
        self._validate_model()
        self.phase = 0
        self.phase_elapsed = 0.
        self.switch_requested = False
        self.next_decision = 0.
        self.initial = True
        self._file = None
        self._writer = None
        if options.get("_run_dir"):
            self._file = (Path(options["_run_dir"])/"dqn_actions.csv").open("w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow(["time","tls_id","action","applied_phase","internal_elapsed_s",*[f"obs_{i}" for i in range(13)]])

    def _validate_model(self):
        validate_checkpoint_timing(self.model,self.timing_profile)
        if self.timing_profile == "v2" and (getattr(self.model,"training_tls_id",None)!=self.tls_id
                or getattr(self.model,"training_lane_ids",None)!=self.lane_ids):
            raise ValueError("v2 checkpoint junction or ordered training lanes do not match the adapter")
        if getattr(self.model.observation_space,"shape",None) != (13,):
            raise ValueError("Checkpoint observation shape must be (13,)")
        if not np.allclose(self.model.observation_space.low, np.zeros(13)) or not np.allclose(self.model.observation_space.high,OBS_HIGH):
            raise ValueError("Checkpoint observation bounds do not match the training environment")
        if getattr(self.model.action_space,"n",None) != 2:
            raise ValueError("Checkpoint action space must be Discrete(2)")
        if int(getattr(self.model,"num_timesteps",0)) <= 0:
            raise ValueError("Checkpoint has no recorded training timesteps")

    def observation(self):
        lane = self.connection.lane
        values = ([lane.getLastStepVehicleNumber(key) for key in self.lane_ids]
                  + [lane.getLastStepHaltingNumber(key) for key in self.lane_ids]
                  + [lane.getWaitingTime(key) for key in self.lane_ids]
                  + [self.phase - 1 if self.phase in (1,3) else self.phase])
        return np.clip(np.asarray(values,dtype=np.float32),0,OBS_HIGH)

    def _advance_internal_fsm(self):
        # Exact SignalController.step ordering, including yellow excess carry.
        self.phase_elapsed += self.interval
        if self.phase in (1,3):
            if self.phase_elapsed >= self.yellow:
                self.phase = (self.phase+1)%4
                self.phase_elapsed = max(0., self.phase_elapsed-self.yellow)
                self.switch_requested = False
        elif self.switch_requested and self.phase_elapsed >= self.min_green:
            self.phase += 1
            self.phase_elapsed = 0.
            self.switch_requested = False

    def step(self, simulation_time):
        if simulation_time+1e-8 < self.next_decision:
            return
        if not math.isclose(simulation_time,self.next_decision,abs_tol=1e-6):
            raise ValueError(f"Simulation skipped a trained {self.interval:g}-second decision boundary")
        if not self.initial:
            self._advance_internal_fsm()
        self.initial = False
        observation = self.observation()
        action, _ = self.model.predict(observation, deterministic=True)
        action = int(np.asarray(action).item())
        if action not in (0,1):
            raise ValueError("DQN action must be 0 (hold) or 1 (request switch)")
        if action == 1 and self.phase not in (1,3):
            self.switch_requested = True
        # In training the request is processed AFTER this decision block.
        # Freeze the phase so SUMO cannot advance independently in the block.
        self.connection.trafficlight.setPhase(self.tls_id,self.phase)
        self.connection.trafficlight.setPhaseDuration(self.tls_id,self.interval+1)
        if self._writer:
            self._writer.writerow([simulation_time,self.tls_id,action,self.phase,self.phase_elapsed,*observation.tolist()])
        self.next_decision += self.interval

    def close(self):
        if self._file:
            self._file.close()


def create_controller(*, connection, checkpoint, options):
    """Factory consumed by traffic_eval.runner; SB3 imported only for DQN runs."""
    if "mock" in Path(checkpoint).name.lower():
        raise ValueError("Mock-trained checkpoint cannot control the evidence pipeline")
    profile,_ = timing_settings(options)
    name = Path(checkpoint).name.lower()
    if ("_v1" in name and profile != "v1") or ("_v2" in name and profile != "v2"):
        raise ValueError("Checkpoint filename and timing_profile disagree; use the matching trained model")
    import torch
    cpu_threads=int(options.get("cpu_threads",1))
    if cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    from .model_loading import load_dqn
    model = load_dqn(checkpoint)
    model.set_random_seed(int(options.get("seed",0)))
    return TrainedSignalController(connection,model,options)

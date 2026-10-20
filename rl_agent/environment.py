"""
environment.py
==============
Custom Gymnasium environment for RL-based traffic signal control.

Supports two operating modes
-----------------------------
  "mock"  — Fully self-contained.  Generates synthetic observations so
             you can develop and test the RL agent *without* SUMO installed.
             Mock observations are plausible-looking numbers but are NOT
             a real traffic simulation.  Mock training results do NOT
             demonstrate actual traffic-control performance.

  "sumo"  — Integrates with SUMO via the TraCI Python API.  All TraCI
             calls are isolated in clearly marked helper methods with
             explicit TODO placeholders.  You must fill in real network
             values (traffic-light IDs, lane IDs, phase mappings) once
             the SUMO network is finalised by the other team.

Observation vector layout (per step)
--------------------------------------
Index range        Feature
0  … N-1           Queue length (vehicles) for each of the N approaches
N  … 2N-1          Number of halting vehicles for each approach
2N … 3N-1          Cumulative waiting time (s) for each approach
3N                 Current signal phase index (green phases only: 0 or 2)

With the default N=4:  vector length = 4*3 + 1 = 13

Action space
------------
Discrete(2)
  0 → Keep current phase (or do nothing if already in yellow)
  1 → Request switch to next phase (enforced via SignalController)

Reward
------
  reward = -sum(waiting_times)   (see reward.py for details)

Episode termination
-------------------
  terminated : bool  — always False in this skeleton; set to True when
                        SUMO reports the end of a traffic scenario (e.g.
                        all vehicles have left the network).
  truncated  : bool  — True when the episode reaches max_steps.
                        This follows the Gymnasium convention: truncated
                        means the episode was cut short by a time limit,
                        NOT that the task naturally ended.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from reward import compute_reward
from signal_logic import SignalController, Phase


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class TrafficSignalEnv(gym.Env):
    """
    Gymnasium environment for a single-intersection traffic signal controller.

    Parameters
    ----------
    mode : str
        Operating mode.  "mock" (default) or "sumo".
    n_approaches : int
        Number of road approaches at the intersection.  Default 4.
        The observation vector length is n_approaches * 3 + 1.
    max_steps : int
        Maximum number of RL steps per episode before truncation.
        Default 288 ≈ 24 hours at a 5-minute decision interval (adjust
        to match your scenario duration).
    min_green_seconds : float
        Minimum green duration enforced by SignalController.
    yellow_seconds : float
        Yellow interval duration enforced by SignalController.
    decision_interval : float
        Simulated seconds between consecutive RL decisions.
    sumo_config : dict or None
        SUMO-specific configuration.  Required when mode="sumo".
        See _build_sumo_config_template() for the expected keys.
    """

    # Gymnasium requires this for rendering
    metadata = {"render_modes": []}

    def __init__(
        self,
        mode: str               = "mock",
        n_approaches: int       = 4,
        max_steps: int          = 288,
        min_green_seconds: float = 10.0,
        yellow_seconds: float   = 3.0,
        decision_interval: float = 5.0,
        sumo_config: Optional[dict] = None,
    ) -> None:
        super().__init__()

        # ---- Validate arguments ----
        if mode not in ("mock", "sumo"):
            raise ValueError(f"mode must be 'mock' or 'sumo', got '{mode}'")

        self.mode               = mode
        self.n_approaches       = n_approaches
        self.max_steps          = max_steps
        self.decision_interval  = decision_interval

        # ---- Signal controller ----
        self.signal_ctrl = SignalController(
            min_green_seconds=min_green_seconds,
            yellow_seconds=yellow_seconds,
            decision_interval=decision_interval,
        )

        # ---- Observation space ----
        # Vector: [queue_lengths | halting_vehicles | waiting_times | phase]
        # Shape: (n_approaches * 3 + 1,)
        obs_dim = n_approaches * 3 + 1

        # Upper bounds are generous placeholders.  Tighten once you know
        # the real lane capacities from the SUMO network.
        self.observation_space = spaces.Box(
            low   = np.zeros(obs_dim, dtype=np.float32),
            high  = np.array(
                [100.0] * n_approaches      # queue length (vehicles)
                + [100.0] * n_approaches    # halting vehicles
                + [3600.0] * n_approaches   # cumulative wait (s) — 1 hour max
                + [3.0],                    # phase index (0..3)
                dtype=np.float32,
            ),
            dtype = np.float32,
        )

        # ---- Action space ----
        # 0 = keep phase, 1 = request switch
        self.action_space = spaces.Discrete(2)

        # ---- SUMO config ----
        if mode == "sumo":
            if sumo_config is None:
                warnings.warn(
                    "mode='sumo' was selected but sumo_config is None. "
                    "SUMO calls will fail until you provide a valid config.",
                    stacklevel=2,
                )
            self._sumo_cfg = sumo_config or {}
        else:
            self._sumo_cfg = {}

        # ---- Internal episode state ----
        self._step_count: int = 0
        self._sumo_running: bool = False
        self._prev_waiting_times: list[float] = [0.0] * n_approaches
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to a fresh episode.

        Parameters
        ----------
        seed : int or None
            Random seed for reproducible mock observations.
        options : dict or None
            Reserved for future use (e.g. scenario selection).

        Returns
        -------
        observation : np.ndarray, shape (obs_dim,)
        info : dict
            Diagnostic information (not used for training).
        """
        # Gymnasium requirement: seed the base RNG
        super().reset(seed=seed)
        # Also seed our own generator for mock observations
        self._rng = np.random.default_rng(seed)

        self._step_count = 0
        self._prev_waiting_times = [0.0] * self.n_approaches
        self.signal_ctrl.reset()

        if self.mode == "sumo":
            self._sumo_reset()
        # mock mode needs no special reset beyond the above

        obs = self._get_observation()
        info = self._build_info()
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one RL step (one decision interval of simulated time).

        Parameters
        ----------
        action : int
            0 → keep current phase, 1 → request phase switch.

        Returns
        -------
        observation  : np.ndarray
        reward       : float
        terminated   : bool   True when the episode ends naturally (e.g.
                               all vehicles cleared in SUMO).  In mock mode
                               this is always False because there is no
                               real end condition.
        truncated    : bool   True when max_steps is reached.
        info         : dict
        """
        # 1. Pass the agent's action to the signal controller.
        #    The controller will record a switch request if appropriate.
        self.signal_ctrl.request_switch(action)

        # 2. Advance the simulation by one decision interval and update
        #    the signal phase (including yellow handling).
        if self.mode == "sumo":
            self._sumo_step()
        # In mock mode the signal controller already manages its own time;
        # no external simulator needs to be ticked.
        self.signal_ctrl.step()

        # 3. Read the new observation from the environment.
        obs = self._get_observation()

        # 4. Extract waiting times for the reward calculation.
        waiting_times = self._extract_waiting_times(obs)
        reward = compute_reward(waiting_times)
        self._prev_waiting_times = waiting_times

        # 5. Determine episode end conditions.
        self._step_count += 1
        terminated = False   # No natural end condition in mock mode.
                             # In SUMO mode: set to True when TraCI reports
                             # simulation.getMinExpectedNumber() == 0
        if self.mode == "sumo":
            terminated = self._sumo_is_done()

        truncated = self._step_count >= self.max_steps

        info = self._build_info(
            action=action,
            waiting_times=waiting_times,
            reward=reward,
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # close
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release resources (closes SUMO if running)."""
        if self.mode == "sumo" and self._sumo_running:
            self._sumo_close()
        self._sumo_running = False

    # ==================================================================
    # Observation helpers
    # ==================================================================

    def _get_observation(self) -> np.ndarray:
        """Build and return the observation vector."""
        if self.mode == "mock":
            return self._mock_observation()
        else:
            return self._sumo_observation()

    def _mock_observation(self) -> np.ndarray:
        """
        Generate a synthetic observation for mock mode.

        The values are random but bounded within the observation space.
        They are NOT intended to reflect realistic traffic dynamics —
        they exist only so the environment API can be tested and the
        agent can learn *something* without SUMO.

        Interpretation:
            queue_lengths    ~ Uniform(0, 20) vehicles
            halting_vehicles ~ Uniform(0, queue_length)   (always <= queue)
            waiting_times    ~ Uniform(0, 200) seconds
            phase            ~ current green-phase index from SignalController
        """
        n = self.n_approaches
        queue_lengths    = self._rng.uniform(0, 20, size=n).astype(np.float32)
        halting_vehicles = (
            self._rng.uniform(0, 1, size=n) * queue_lengths
        ).astype(np.float32)
        waiting_times    = self._rng.uniform(0, 200, size=n).astype(np.float32)
        phase_obs        = np.array(
            [self.signal_ctrl.phase_index_for_obs()], dtype=np.float32
        )

        obs = np.concatenate([queue_lengths, halting_vehicles, waiting_times, phase_obs])
        # Safety clip — should already be in range but guards against float drift
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _extract_waiting_times(self, obs: np.ndarray) -> list[float]:
        """
        Pull the waiting-time slice out of the observation vector.

        Indices 2*N … 3*N-1 hold the waiting times (see layout at top).
        """
        n = self.n_approaches
        return obs[2 * n: 3 * n].tolist()

    # ==================================================================
    # Info dict
    # ==================================================================

    def _build_info(self, **kwargs) -> dict:
        """Return a diagnostic info dictionary (not used for training)."""
        return {
            "step": self._step_count,
            "phase": self.signal_ctrl.current_phase.name,
            "phase_elapsed_s": self.signal_ctrl.phase_elapsed,
            "is_yellow": self.signal_ctrl.is_yellow,
            "min_green_satisfied": self.signal_ctrl.min_green_satisfied,
            "mode": self.mode,
            **kwargs,
        }

    # ==================================================================
    # ──────────────────────────────────────────────────────────────────
    # SUMO / TraCI integration stubs
    # ──────────────────────────────────────────────────────────────────
    # Everything below this line is for SUMO mode.
    # Replace TODO placeholders with real values once the network is ready.
    # ==================================================================

    @staticmethod
    def _build_sumo_config_template() -> dict:
        """
        Return the expected structure of sumo_config.

        Use this as a reference when filling in real network details.
        Do NOT use these placeholder strings in production.
        """
        return {
            # Path to the SUMO .sumocfg file (generated by the network team)
            "sumocfg_path": "TODO: path/to/your.sumocfg",

            # SUMO traffic-light ID at this intersection
            # Find it in corridor.net.xml: <tlLogic id="...">
            "tl_id": "TODO: traffic_light_id_from_net_xml",

            # Lane IDs for each approach (must match corridor.net.xml)
            # Order must match the approach order in the observation vector.
            "lane_ids": [
                "TODO: lane_id_approach_0",   # approach 0 (e.g. North)
                "TODO: lane_id_approach_1",   # approach 1 (e.g. South)
                "TODO: lane_id_approach_2",   # approach 2 (e.g. East)
                "TODO: lane_id_approach_3",   # approach 3 (e.g. West)
            ],

            # Mapping from SignalController Phase index → SUMO phase index
            # SUMO phases are defined in <tlLogic> inside corridor.net.xml
            # Example: {0: 0, 2: 2}  meaning GREEN_A→SUMO phase 0, GREEN_B→SUMO phase 2
            "phase_map": {
                int(Phase.GREEN_A):  "TODO: sumo_phase_index_for_GREEN_A",
                int(Phase.GREEN_B):  "TODO: sumo_phase_index_for_GREEN_B",
                int(Phase.YELLOW_A): "TODO: sumo_phase_index_for_YELLOW_A",
                int(Phase.YELLOW_B): "TODO: sumo_phase_index_for_YELLOW_B",
            },

            # Use GUI version of SUMO (sumo-gui) or headless (sumo)?
            "use_gui": False,
        }

    def _sumo_reset(self) -> None:
        """
        Start (or restart) the SUMO simulation via TraCI.

        TODO: Replace placeholder values with real network details.
        """
        # ----------------------------------------------------------------
        # INTEGRATION POINT — fill in when network is ready
        # ----------------------------------------------------------------
        try:
            import traci
        except ImportError:
            raise ImportError(
                "TraCI not found. Install it with: pip install traci\n"
                "Also ensure SUMO is installed and SUMO_HOME is set."
            )

        if self._sumo_running:
            traci.close()
            self._sumo_running = False

        sumocfg = self._sumo_cfg.get("sumocfg_path", "")
        if not sumocfg or sumocfg.startswith("TODO"):
            raise ValueError(
                "sumo_config['sumocfg_path'] is not set. "
                "Provide the path to your .sumocfg file."
            )

        use_gui = self._sumo_cfg.get("use_gui", False)
        sumo_binary = "sumo-gui" if use_gui else "sumo"

        traci.start([sumo_binary, "-c", sumocfg, "--no-warnings", "true"])
        self._sumo_running = True

    def _sumo_step(self) -> None:
        """
        Advance SUMO by one decision interval and apply the current phase.

        TODO: Verify that the phase_map entries match the tlLogic in
              corridor.net.xml before enabling this code.
        """
        # ----------------------------------------------------------------
        # INTEGRATION POINT
        # ----------------------------------------------------------------
        import traci

        tl_id     = self._sumo_cfg.get("tl_id", "")
        phase_map = self._sumo_cfg.get("phase_map", {})

        # Set the SUMO traffic-light phase to match the signal controller.
        sumo_phase = phase_map.get(int(self.signal_ctrl.current_phase))
        if sumo_phase is not None and not str(sumo_phase).startswith("TODO"):
            traci.trafficlight.setPhase(tl_id, int(sumo_phase))

        # Advance the SUMO simulation by the decision interval.
        for _ in range(int(self.decision_interval)):
            traci.simulationStep()

    def _sumo_observation(self) -> np.ndarray:
        """
        Query TraCI for lane-level traffic data and build the obs vector.

        TODO: Verify lane IDs against corridor.net.xml.
              Each lane ID must appear in the SUMO network file.
        """
        # ----------------------------------------------------------------
        # INTEGRATION POINT
        # ----------------------------------------------------------------
        import traci

        lane_ids = self._sumo_cfg.get("lane_ids", [])
        if not lane_ids or any(lid.startswith("TODO") for lid in lane_ids):
            raise ValueError(
                "sumo_config['lane_ids'] contains placeholder values. "
                "Replace them with real lane IDs from corridor.net.xml."
            )

        n = self.n_approaches
        queue_lengths    = np.zeros(n, dtype=np.float32)
        halting_vehicles = np.zeros(n, dtype=np.float32)
        waiting_times    = np.zeros(n, dtype=np.float32)

        for i, lane_id in enumerate(lane_ids):
            # traci.lane.getLastStepVehicleNumber → total vehicles on lane
            queue_lengths[i]    = traci.lane.getLastStepVehicleNumber(lane_id)
            # traci.lane.getLastStepHaltingNumber → stopped vehicles on lane
            halting_vehicles[i] = traci.lane.getLastStepHaltingNumber(lane_id)
            # traci.lane.getWaitingTime → cumulative wait on lane (seconds)
            waiting_times[i]    = traci.lane.getWaitingTime(lane_id)

        phase_obs = np.array(
            [self.signal_ctrl.phase_index_for_obs()], dtype=np.float32
        )
        obs = np.concatenate([queue_lengths, halting_vehicles, waiting_times, phase_obs])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _sumo_is_done(self) -> bool:
        """
        Return True when all vehicles have left the SUMO network.

        TODO: This simple check ends the episode when the network is
              empty.  For a looped/continuous demand scenario you may
              want a different termination criterion.
        """
        # ----------------------------------------------------------------
        # INTEGRATION POINT
        # ----------------------------------------------------------------
        import traci
        return traci.simulation.getMinExpectedNumber() == 0

    def _sumo_close(self) -> None:
        """Terminate the TraCI connection and SUMO process."""
        # ----------------------------------------------------------------
        # INTEGRATION POINT
        # ----------------------------------------------------------------
        import traci
        traci.close()

"""
signal_logic.py
===============
Manages traffic signal phase transitions with safety constraints.

Responsibilities
----------------
- Track which phase is currently active and how long it has been active.
- Enforce a minimum green duration before any switch is allowed.
- Inject a mandatory yellow interval between green phases.
- Prevent the RL agent from bypassing yellow: while yellow is active the
  controller ignores any new switch requests.
- Advance to the next valid green phase after yellow completes.

Phase sequence (one possible 4-approach layout)
------------------------------------------------
Index  Name         Meaning
  0    GREEN_A      North-South through-traffic gets green
  1    YELLOW_A     Yellow clearance before switching away from phase 0
  2    GREEN_B      East-West through-traffic gets green
  3    YELLOW_B     Yellow clearance before switching away from phase 2

The sequence always cycles:  0 → 1 → 2 → 3 → 0 → ...
Only phases 0 and 2 are "green" phases; the agent can only request a
switch while a green phase is active.

NOTE: When you integrate with SUMO you will need to map these phase
indices to the actual SUMO traffic-light phase IDs in your network.
That mapping belongs in environment.py, not here.
"""

from enum import IntEnum


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

class Phase(IntEnum):
    GREEN_A  = 0   # Green for approach pair A (e.g. North-South)
    YELLOW_A = 1   # Mandatory yellow before leaving GREEN_A
    GREEN_B  = 2   # Green for approach pair B (e.g. East-West)
    YELLOW_B = 3   # Mandatory yellow before leaving GREEN_B


# Phases that the RL agent controls (green phases only).
GREEN_PHASES  = (Phase.GREEN_A, Phase.GREEN_B)

# The yellow phase that immediately follows each green phase.
NEXT_YELLOW: dict[Phase, Phase] = {
    Phase.GREEN_A: Phase.YELLOW_A,
    Phase.GREEN_B: Phase.YELLOW_B,
}

# The green phase that follows each yellow phase.
NEXT_GREEN: dict[Phase, Phase] = {
    Phase.YELLOW_A: Phase.GREEN_B,
    Phase.YELLOW_B: Phase.GREEN_A,
}


# ---------------------------------------------------------------------------
# Signal controller
# ---------------------------------------------------------------------------

class SignalController:
    """
    Finite-state machine for a single intersection's traffic signal.

    Parameters
    ----------
    min_green_seconds : float
        Minimum time (in simulation seconds) a green phase must be held
        before the agent is allowed to request a switch.
        Default: 10 s  — adjust to match real-world minimum green policy.
    yellow_seconds : float
        Duration of the yellow clearance interval (in simulation seconds).
        Default: 3 s  — typical urban value; adjust as needed.
    decision_interval : float
        How many simulation seconds pass between consecutive RL decisions.
        Default: 5 s.  The controller advances by this amount each step().
    initial_phase : Phase
        The phase to start in after reset().
    """

    def __init__(
        self,
        min_green_seconds: float = 10.0,
        yellow_seconds: float    = 3.0,
        decision_interval: float = 5.0,
        initial_phase: Phase     = Phase.GREEN_A,
    ) -> None:
        self.min_green_seconds  = min_green_seconds
        self.yellow_seconds     = yellow_seconds
        self.decision_interval  = decision_interval
        self.initial_phase      = initial_phase

        # Runtime state — set by reset()
        self._phase: Phase        = initial_phase
        self._phase_elapsed: float = 0.0   # seconds spent in current phase
        self._switch_requested: bool = False  # pending request during green

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> Phase:
        """Reset controller to the initial phase with zero elapsed time."""
        self._phase          = self.initial_phase
        self._phase_elapsed  = 0.0
        self._switch_requested = False
        return self._phase

    @property
    def current_phase(self) -> Phase:
        return self._phase

    @property
    def phase_elapsed(self) -> float:
        """Seconds spent in the current phase so far."""
        return self._phase_elapsed

    @property
    def is_yellow(self) -> bool:
        """True while a yellow clearance interval is active."""
        return self._phase in (Phase.YELLOW_A, Phase.YELLOW_B)

    @property
    def min_green_satisfied(self) -> bool:
        """True when the current green phase has met the minimum duration."""
        if self.is_yellow:
            return False  # not in a green phase
        return self._phase_elapsed >= self.min_green_seconds

    def request_switch(self, action: int) -> None:
        """
        Called by the environment when the agent picks action 1.

        The request is *recorded* here but not acted on immediately.
        The transition logic runs inside step().

        If we are currently in yellow, the request is silently ignored —
        the agent cannot interrupt a yellow interval.

        Parameters
        ----------
        action : int
            0 → keep current phase, 1 → request switch.
        """
        if action == 1 and not self.is_yellow:
            self._switch_requested = True

    def step(self) -> Phase:
        """
        Advance the controller by one decision interval and apply any
        pending phase transition.

        Returns
        -------
        Phase
            The phase that is active *after* this step.

        How yellow is handled
        ---------------------
        1. If the current phase is yellow and enough time has elapsed,
           automatically advance to the next green phase.
        2. If the current phase is green and a switch was requested AND
           the minimum green has been satisfied, transition to yellow.
        3. Otherwise, remain in the current phase.

        The agent never directly jumps from green to the next green;
        it must always pass through yellow.  This is enforced here, not
        in the environment.
        """
        dt = self.decision_interval

        if self.is_yellow:
            # Count time in yellow; automatically leave when done.
            self._phase_elapsed += dt
            if self._phase_elapsed >= self.yellow_seconds:
                # Yellow complete → enter next green phase.
                self._phase         = NEXT_GREEN[self._phase]
                self._phase_elapsed = max(0.0, self._phase_elapsed - self.yellow_seconds)
                self._switch_requested = False
        else:
            # Currently in a green phase.
            self._phase_elapsed += dt
            if self._switch_requested and self.min_green_satisfied:
                # Minimum green satisfied — start the yellow interval.
                self._phase         = NEXT_YELLOW[self._phase]
                self._phase_elapsed = 0.0
                self._switch_requested = False

        return self._phase

    def phase_index_for_obs(self) -> int:
        """
        Return a single integer suitable for inclusion in the observation
        vector.  Only green-phase indices (0, 2) are exposed to the agent;
        yellow phases are mapped to the preceding green so the agent sees a
        stable phase value while yellow is in progress.
        """
        if self._phase == Phase.YELLOW_A:
            return int(Phase.GREEN_A)
        if self._phase == Phase.YELLOW_B:
            return int(Phase.GREEN_B)
        return int(self._phase)

    def __repr__(self) -> str:
        return (
            f"SignalController(phase={self._phase.name}, "
            f"elapsed={self._phase_elapsed:.1f}s, "
            f"yellow={self.is_yellow}, "
            f"min_green_ok={self.min_green_satisfied})"
        )

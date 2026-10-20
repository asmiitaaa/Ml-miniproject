"""
reward.py
=========
Reward calculation for the traffic signal control agent.

Current reward: negative sum of per-approach waiting times.

    reward = -sum(waiting_times_per_approach)

Why negative?
-------------
Reinforcement learning agents *maximise* reward.  Since we want to
*minimise* total waiting time, we negate it.  A waiting time of zero
would give a reward of 0 (the best possible); a high total wait gives a
large negative reward.

Design notes
------------
* This is a **basic initial formulation**.  Real deployments often need:
    - Reward *scaling* (e.g. divide by number of vehicles or by a
      reference wait time) so rewards stay in a numerically stable range
      for the neural network.
    - *Incremental* (delta) penalties: reward based on the *change* in
      waiting time rather than the absolute value, to give more
      informative signal to the agent at each step.
    - Fairness penalties for approaches that starve (very long waits on
      one direction while another is always green).
* The function is kept separate so you can swap or extend it without
  touching the environment or training code.

Usage
-----
    from reward import compute_reward
    waiting_times = [12.5, 8.0, 3.2, 20.1]   # one value per approach
    r = compute_reward(waiting_times)           # → -43.8
"""

from __future__ import annotations
from typing import Sequence


def compute_reward(waiting_times: Sequence[float]) -> float:
    """
    Compute the step reward as the negative total waiting time.

    Parameters
    ----------
    waiting_times : sequence of float
        Cumulative waiting time (in seconds) for vehicles at each approach.
        Length must equal the number of approaches configured in the
        environment.  In mock mode these are synthetic values; in SUMO mode
        they come from TraCI lane queries.

    Returns
    -------
    float
        Negative sum of all waiting times.  Always <= 0.

    Examples
    --------
    >>> compute_reward([0.0, 0.0, 0.0, 0.0])
    -0.0
    >>> compute_reward([10.0, 5.0, 3.0, 2.0])
    -20.0
    """
    if not waiting_times:
        return 0.0
    return -float(sum(waiting_times))


def compute_reward_delta(
    prev_waiting_times: Sequence[float],
    curr_waiting_times: Sequence[float],
) -> float:
    """
    Alternative reward: negative *change* in total waiting time.

    A positive delta means waiting time grew (bad → negative reward).
    A negative delta means waiting time shrank (good → positive reward).

    This formulation is often more informative because it rewards the
    agent for *improving* conditions rather than just being in a low-
    wait state.

    NOTE: Not used in the current environment. Provided as a ready-made
    alternative for later experimentation.

    Parameters
    ----------
    prev_waiting_times : sequence of float
        Waiting times at the *previous* step.
    curr_waiting_times : sequence of float
        Waiting times at the *current* step.

    Returns
    -------
    float
        -(sum(curr) - sum(prev))  =  sum(prev) - sum(curr)
    """
    return float(sum(prev_waiting_times)) - float(sum(curr_waiting_times))

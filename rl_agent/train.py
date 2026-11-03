"""
train.py
========
DQN training script using Stable-Baselines3.

What this script does
---------------------
1. Instantiates TrafficSignalEnv in *mock* mode.
2. Validates the environment against the Gymnasium API using check_env.
3. Creates a DQN agent with MlpPolicy.
4. Trains the agent for a configurable number of timesteps.
5. Saves the trained model to disk.

⚠  IMPORTANT DISCLAIMER
-------------------------
Training in mock mode does NOT produce a traffic-control policy that
works on a real intersection.  Mock observations are synthetic random
numbers, not real traffic data.  The agent will learn to optimise
mock rewards, which have no relationship to actual traffic conditions.

Use this script to:
  - Verify the environment API is correct.
  - Confirm Stable-Baselines3 and Gymnasium integrate properly.
  - Prototype the training loop before SUMO is available.

To train on a real scenario, switch mode="sumo" and provide a valid
sumo_config dictionary.

DQN hyperparameters used
------------------------
Parameter             Value     Rationale
-----------           -----     ---------
policy                MlpPolicy Standard fully-connected net for flat obs.
learning_rate         1e-3      Starting point; tune down (1e-4) if unstable.
buffer_size           50_000    Replay buffer capacity.  Reduce if RAM is
                                tight; increase for longer episodes.
learning_starts       1_000     Steps of random action before learning
                                begins — ensures diverse replay buffer.
batch_size            64        Mini-batch size for gradient updates.
                                Default in SB3; increase to 128 for stability.
train_freq            4         Update the network every 4 env steps.
target_update_interval 1_000   Steps between target network hard updates.
exploration_fraction  0.2       Fraction of training spent linearly decaying
                                epsilon from 1.0 to exploration_final_eps.
exploration_final_eps 0.05      Final epsilon after decay (5% random actions).
verbose               1         Print training progress to the terminal.

Parameters NOT set here (using SB3 defaults):
  gamma (discount factor) = 0.99
  tau (soft update coeff) = 1.0  (hard update because target_update_interval)

How to run
----------
  cd rl_agent
  python train.py

Outputs
-------
  models/dqn_traffic_mock.zip   — saved Stable-Baselines3 model
  models/dqn_traffic_mock_10000_steps.zip  — checkpoint at N steps
"""

import os
import sys

# ---------------------------------------------------------------------------
# SUMO path bootstrap (eclipse-sumo pip package)
# Sets SUMO_HOME and adds the Scripts folder to PATH so TraCI can find
# the sumo binary regardless of whether the user configured it manually.
# ---------------------------------------------------------------------------
_SUMO_SCRIPTS = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Roaming", "Python", "Python313", "Scripts",
)
_SUMO_HOME = os.path.join(
    os.path.expanduser("~"),
    "AppData", "Roaming", "Python", "Python313", "site-packages", "sumo",
)
if "SUMO_HOME" not in os.environ:
    os.environ["SUMO_HOME"] = _SUMO_HOME
if _SUMO_SCRIPTS not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _SUMO_SCRIPTS + os.pathsep + os.environ.get("PATH", "")
# ---------------------------------------------------------------------------

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from environment import TrafficSignalEnv


# ---------------------------------------------------------------------------
# Configuration — edit these values as needed
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SUMO network values — extracted from network/corridor.net.xml
# ---------------------------------------------------------------------------
# Traffic-light logic ID (from <tlLogic id="..."> in corridor.net.xml):
#   cluster_11484393853_11484393854_11484393855_11484393856_#11more
#
# incLanes of the controlled junction (one primary lane per approach):
#   Approach 0 (North): 24480656#3_0
#   Approach 1 (East):  794634832#0_0
#   Approach 2 (South): 794634831#1_0
#   Approach 3 (West):  631316440#0_0
#
# Phase layout in tlLogic (corridor.net.xml lines 6880-6883):
#   SUMO phase 0 → 39 s green  (rrrrGGGgrrrrGGGg) → GREEN_A  (SignalController phase 0)
#   SUMO phase 1 →  6 s yellow (rrrryyyyrrrryyyy) → YELLOW_A (SignalController phase 1)
#   SUMO phase 2 → 39 s green  (GGGgrrrrGGGgrrrr) → GREEN_B  (SignalController phase 2)
#   SUMO phase 3 →  6 s yellow (yyyyrrrryyyyrrrr) → YELLOW_B (SignalController phase 3)
# ---------------------------------------------------------------------------

SUMO_CONFIG = {
    # Path is relative to the rl_agent/ working directory
    "sumocfg_path": "../simulation/corridor.sumocfg",

    # Traffic-light ID at the controlled intersection
    "tl_id": "cluster_11484393853_11484393854_11484393855_11484393856_#11more",

    # One incoming lane ID per approach (lane index _0 = first lane)
    "lane_ids": [
        "24480656#3_0",    # Approach 0 — North
        "794634832#0_0",   # Approach 1 — East
        "794634831#1_0",   # Approach 2 — South
        "631316440#0_0",   # Approach 3 — West
    ],

    # SignalController Phase enum value → SUMO phase index (identity map)
    # Phase.GREEN_A=0, Phase.YELLOW_A=1, Phase.GREEN_B=2, Phase.YELLOW_B=3
    "phase_map": {0: 0, 1: 1, 2: 2, 3: 3},

    "use_gui": False,  # Set True to watch in sumo-gui (slow; use for debugging only)
}

TRAINING_CONFIG = {
    "mode":               "sumo",   # ← flipped from mock to sumo
    "n_approaches":       4,
    # 720 steps × 5 s = 3600 s = 1 simulated hour (matches sumocfg end time)
    "max_steps":          720,
    "min_green_seconds":  10.0,
    "yellow_seconds":     3.0,
    "decision_interval":  5.0,
    "sumo_config":        SUMO_CONFIG,
}

DQN_CONFIG = {
    "learning_rate":            1e-3,
    "buffer_size":              50_000,
    "learning_starts":          1_000,
    "batch_size":               64,
    "train_freq":               4,
    "target_update_interval":   1_000,
    "exploration_fraction":     0.2,
    "exploration_final_eps":    0.05,
    "verbose":                  1,
}

TOTAL_TIMESTEPS = 100_000     # First real SUMO run — increase later if results look good
CHECKPOINT_FREQ = 10_000      # Save a model checkpoint every N steps
MODEL_SAVE_DIR  = "models"
MODEL_NAME      = "dqn_traffic_sumo_v1"


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    # 1. Create the environment
    print("=== Creating TrafficSignalEnv (sumo mode) ===")
    env = TrafficSignalEnv(**TRAINING_CONFIG)

    # 2. Validate the environment against the Gymnasium API.
    #    check_env will raise an error if anything is misconfigured.
    print("=== Running check_env ===")
    check_env(env, warn=True)
    print("check_env passed OK\n")

    # 3. Create the DQN agent
    print("=== Creating DQN agent ===")
    model = DQN(
        policy="MlpPolicy",
        env=env,
        **DQN_CONFIG,
    )

    # 4. Checkpoint callback — saves model weights during training
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_SAVE_DIR,
        name_prefix=MODEL_NAME,
        verbose=1,
    )

    # 5. Train
    print(f"=== Training for {TOTAL_TIMESTEPS:,} timesteps (SUMO mode) ===")
    print("Intersection: cluster_11484393853_11484393854_11484393855_11484393856_#11more")
    print("Model will be saved to:", os.path.join(MODEL_SAVE_DIR, MODEL_NAME + ".zip"), "\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_cb,
        progress_bar=False,  # Set True if tqdm is installed
    )

    # 6. Save the final model
    final_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
    model.save(final_path)
    print(f"\n=== Training complete.  Model saved to {final_path}.zip ===")

    env.close()


# ---------------------------------------------------------------------------
# Loading a saved model (reference snippet — not called by default)
# ---------------------------------------------------------------------------

def load_and_run_demo(model_path: str = None) -> None:
    """
    Load a previously saved model and run one episode to see it in action.

    Parameters
    ----------
    model_path : str
        Path to the .zip file (without extension).
        Defaults to MODEL_SAVE_DIR/MODEL_NAME.
    """
    if model_path is None:
        model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    env   = TrafficSignalEnv(**TRAINING_CONFIG)
    model = DQN.load(model_path, env=env)

    obs, info = env.reset(seed=42)
    total_reward = 0.0
    done = False

    print(f"\n=== Demo episode using model: {model_path} ===")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated

    print(f"Episode finished.  Total reward: {total_reward:.2f}")
    print("(Reward reflects mock data only — not real traffic performance)")
    env.close()


if __name__ == "__main__":
    train()

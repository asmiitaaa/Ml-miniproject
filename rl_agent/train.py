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
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

from environment import TrafficSignalEnv


# ---------------------------------------------------------------------------
# Configuration — edit these values as needed
# ---------------------------------------------------------------------------

TRAINING_CONFIG = {
    "mode":               "mock",   # Change to "sumo" when network is ready
    "n_approaches":       4,
    "max_steps":          288,      # Episode length (steps, not seconds)
    "min_green_seconds":  10.0,
    "yellow_seconds":     3.0,
    "decision_interval":  5.0,
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

TOTAL_TIMESTEPS = 50_000      # Increase for serious training runs
CHECKPOINT_FREQ = 10_000      # Save a model checkpoint every N steps
MODEL_SAVE_DIR  = "models"
MODEL_NAME      = "dqn_traffic_mock"


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    # 1. Create the environment
    print("=== Creating TrafficSignalEnv (mock mode) ===")
    env = TrafficSignalEnv(**TRAINING_CONFIG)

    # 2. Validate the environment against the Gymnasium API.
    #    check_env will raise an error if anything is misconfigured.
    print("=== Running check_env ===")
    check_env(env, warn=True)
    print("check_env passed ✓\n")

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
    print(f"=== Training for {TOTAL_TIMESTEPS:,} timesteps ===")
    print("NOTE: This is mock training only.  Results do not reflect")
    print("      real traffic-control performance.\n")
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

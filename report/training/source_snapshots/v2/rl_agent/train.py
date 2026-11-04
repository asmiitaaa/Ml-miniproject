"""Train SUMO-v2 DQN checkpoints with explicit timing and provenance metadata.

Run from the repository root: .venv/Scripts/python.exe rl_agent/train.py
Use --timesteps 24 --output-dir <scratch-folder> for a smoke test only.
The default 100,000-step run still requires convergence and safety review.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
# Resolve the installed simulator rather than another user's Python313 path.
spec = importlib.util.find_spec("sumo")
if spec and spec.submodule_search_locations:
    sumo_home = Path(next(iter(spec.submodule_search_locations)))
    os.environ["SUMO_HOME"] = str(sumo_home)
    os.environ["PATH"] = str(sumo_home/"bin") + os.pathsep + os.environ.get("PATH", "")

from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from traffic_eval.timing import TIMING_PROFILES

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
    # Resolve from this repository, regardless of the working directory.
    "sumocfg_path": str(REPO / "simulation/corridor.sumocfg"),

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
    # 1200 steps × 3 s = 3600 s = 1 simulated hour (matches sumocfg end time)
    "max_steps":          1200,
    # --- v2 SIGNAL TIMING (safety fix, 2026-11-03) ---------------------------
    # The network's tlLogic specifies a 6 s yellow at this junction.  v1 was
    # trained with yellow_seconds=3.0 and decision_interval=5.0, giving an
    # effective 5 s yellow — SHORTER than the network's nominal clearance.
    # Evaluation (soham/evaluation) detected collisions in 2/5 peak seeds,
    # but their causal relationship to clearance is not established.
    # v2 therefore uses:
    #   yellow_seconds    = 6.0  → matches the network's nominal yellow
    #   decision_interval = 3.0  → 6/3 is an integer, so the yellow lasts
    #                              EXACTLY 6 s (no ceil-to-interval stretch)
    #   min_green_seconds = 12.0 → multiple of the new interval
    # Any checkpoint trained with this block is a v2 checkpoint and must be
    # evaluated with the matching v2 adapter profile in traffic_eval.
    # -------------------------------------------------------------------------
    "min_green_seconds":  12.0,
    "yellow_seconds":     6.0,
    "decision_interval":  3.0,
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
    "seed":                     42,
}

TOTAL_TIMESTEPS = 100_000     # First real SUMO run — increase later if results look good
CHECKPOINT_FREQ = 10_000      # Save a model checkpoint every N steps
TRAINING_SUMO_SEED = 42       # Evaluation seeds 11-15 are held out from this run
LATERAL_RESOLUTION_M = 0.4    # Match the methodology and evaluation fixtures
MODEL_SAVE_DIR  = str(Path(__file__).resolve().parent / "models")
MODEL_NAME      = "dqn_traffic_sumo_v2"   # v2 = corrected 6 s yellow / 3 s interval / 12 s min-green


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def prepare_training_inputs(output):
    """Resolve inputs and isolate output logs from the team's tracked files."""
    original=Path(SUMO_CONFIG["sumocfg_path"])
    tree=ET.parse(original)
    hashes={"simulation/corridor.sumocfg":hashlib.sha256(original.read_bytes()).hexdigest()}
    for option in tree.findall("./input/*"):
        if option.tag in {"net-file","route-files","additional-files"}:
            paths=[(original.parent/name.strip()).resolve() for name in option.get("value","").split(",") if name.strip()]
            option.set("value",",".join(map(str,paths)))
            for path in paths:
                key=str(path.relative_to(REPO)).replace("\\","/")
                hashes[key]=hashlib.sha256(path.read_bytes()).hexdigest()
    # Only the private run configuration changes; original inputs stay intact.
    tree.find("./processing/lateral-resolution").set("value",str(LATERAL_RESOLUTION_M))
    tree.find("./processing/ignore-route-errors").set("value","false")
    random_section=tree.find("random_number")
    if random_section is None:random_section=ET.SubElement(tree.getroot(),"random_number")
    for tag,value in (("seed",str(TRAINING_SUMO_SEED)),("random","false")):
        option=random_section.find(tag)
        if option is None:option=ET.SubElement(random_section,tag)
        option.set("value",value)
    resolution=LATERAL_RESOLUTION_M
    for tag in ("lanedata-output","tripinfo-output","emission-output","summary-output","collision-output"):
        for option in tree.findall(".//"+tag):option.set("value",str(output/(tag+".xml")))
    if tree.find(".//collision-output") is None:
        section=tree.find("output")
        if section is None:section=ET.SubElement(tree.getroot(),"output")
        ET.SubElement(section,"collision-output",value=str(output/"collisions.xml"))
    processing=tree.find("processing")
    collision=processing.find("collision.action")
    if collision is None:collision=ET.SubElement(processing,"collision.action")
    collision.set("value","teleport")
    target=output/"training.sumocfg"
    tree.write(target,encoding="utf-8",xml_declaration=True)
    return target,hashes,resolution


def train(timesteps=TOTAL_TIMESTEPS, output_dir=MODEL_SAVE_DIR):
    import torch
    torch.set_num_threads(1)
    if timesteps<=0:raise ValueError("timesteps must be positive")
    output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    final=output/(MODEL_NAME+".zip")
    if final.exists() or list(output.glob(MODEL_NAME+"_*_steps.zip")):
        raise FileExistsError("v2 checkpoints already exist here; choose a fresh --output-dir")
    run=output/"training_runs"/datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run.mkdir(parents=True)
    configuration,inputs,resolution=prepare_training_inputs(run)
    settings=dict(TRAINING_CONFIG, sumo_config={**SUMO_CONFIG,"sumocfg_path":str(configuration)})
    env=Monitor(TrafficSignalEnv(**settings),str(run/"monitor.csv"))
    try:
        check_env(env,warn=True)
        model=DQN(policy="MlpPolicy",env=env,device="cpu",**DQN_CONFIG)
        # SB3 stores these attributes in every callback/final checkpoint.
        model.timing_profile="v2"
        model.training_mode=TRAINING_CONFIG["mode"]
        model.training_timing={"decision_interval_s":float(TRAINING_CONFIG["decision_interval"]),
                               "min_green_s":float(TRAINING_CONFIG["min_green_seconds"]),
                               "yellow_s":float(TRAINING_CONFIG["yellow_seconds"])}
        assert model.training_timing==TIMING_PROFILES["v2"]
        model.training_lane_ids=list(SUMO_CONFIG["lane_ids"])
        model.training_tls_id=SUMO_CONFIG["tl_id"]
        model.training_lateral_resolution_m=resolution
        model.training_input_sha256=inputs
        model.training_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        model.training_config_sha256=hashlib.sha256(configuration.read_bytes()).hexdigest()
        model.training_model_seed=DQN_CONFIG["seed"]
        # Seed is explicit in the private SUMO configuration, on every reset.
        model.training_sumo_seed=TRAINING_SUMO_SEED
        model.training_status="smoke-test-only" if timesteps<DQN_CONFIG["learning_starts"] else "convergence-not-assessed"
        callback=CheckpointCallback(save_freq=CHECKPOINT_FREQ,save_path=str(output),name_prefix=MODEL_NAME,verbose=1)
        print(f"Training v2 for {timesteps} decisions. Output: {output}",flush=True)
        model.learn(total_timesteps=timesteps,callback=callback,progress_bar=False)
        model.save(final)
        print(f"Saved {final}; status={model.training_status}. Training logs: {run}",flush=True)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Loading a saved model (reference snippet — not called by default)
# ---------------------------------------------------------------------------

def load_and_run_demo(model_path: str = None) -> None:
    """Replay a genuine v2 policy through the same strict evaluation adapter."""
    from traffic_eval.runner import load_config, run_one
    checkpoint=Path(model_path) if model_path else Path(MODEL_SAVE_DIR)/(MODEL_NAME+".zip")
    if checkpoint.suffix!=".zip":checkpoint=Path(str(checkpoint)+".zip")
    cfg=load_config(REPO/"configs/evaluation.v2.example.json")
    cfg["controllers"]["dqn"]["checkpoint"]=str(checkpoint.resolve())
    cfg["experiment_label"]="v2-demo-not-final-evaluation"
    output=REPO/"results"/("v2_demo_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    result=run_one(cfg,"peak","dqn",11,output)
    print(f"SUMO replay complete. Inspect metrics, warnings and collision logs: {output}")
    print(result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps",type=int,default=TOTAL_TIMESTEPS)
    parser.add_argument("--output-dir",default=MODEL_SAVE_DIR)
    args=parser.parse_args()
    train(args.timesteps,args.output_dir)

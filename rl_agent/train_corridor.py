"""Train per-scenario 17-feature models, initially on one real intersection."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import torch

from rl_agent.corridor import SCHEMA,TIMING,mapping_digest
from rl_agent.corridor_env import CorridorEnv,TRAINED_TLS
from traffic_eval.runner import load_config,input_files,sha256


def train(args):
    torch.set_num_threads(1)
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    cfg=load_config(ROOT/'configs/evaluation.model_validation.json')
    run=output/'training_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    env=Monitor(CorridorEnv(cfg,args.scenario,run,seeds=args.sumo_seeds),str(output/'monitor.csv'),
                info_keywords=('colliding_vehicle_detections','teleports','sumo_seed'))
    parameters=dict(learning_rate=args.learning_rate,buffer_size=50000,learning_starts=1000,
        batch_size=64,train_freq=4,target_update_interval=1000,exploration_fraction=args.exploration_fraction,
        exploration_final_eps=.05,seed=args.model_seed)
    record=dict(scenario=args.scenario,total_timesteps=args.timesteps,model_seed=args.model_seed,
        training_seeds=args.sumo_seeds,feature_schema=SCHEMA,timing=TIMING,hyperparameters=parameters,
        validation_seeds=[101,102,103],test_seeds=[11,12,13,14,15],training_control_tls_ids=[TRAINED_TLS],
        source_sha256={name:sha256(ROOT/name) for name in ['rl_agent/corridor.py','rl_agent/corridor_env.py','rl_agent/train_corridor.py']},
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in input_files(cfg,cfg['scenarios'][args.scenario])[2]})
    (output/'training_manifest.json').write_text(json.dumps(record,indent=2))
    for name,expected in record['source_sha256'].items():
        snapshot=output/'source_snapshot'/name;snapshot.parent.mkdir(parents=True,exist_ok=True)
        data=(ROOT/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==expected
        snapshot.write_bytes(data)
    try:
        check_env(env,warn=True)
        model=DQN('MlpPolicy',env,device='cpu',verbose=1,**parameters)
        model.feature_schema=SCHEMA;model.corridor_timing=TIMING.copy();model.mapping_sha256=mapping_digest()
        model.training_seeds=args.sumo_seeds;model.training_control_tls_ids=[TRAINED_TLS]
        model.training_status='convergence-not-assessed';model.training_manifest=record
        model.learn(total_timesteps=args.timesteps,progress_bar=False,
            callback=CheckpointCallback(save_freq=20000,save_path=str(output),name_prefix='corridor_dqn_'+args.scenario))
        target=output/('corridor_dqn_'+args.scenario+'_final.zip');model.save(target)
        record.update(status='complete',actual_timesteps=int(model.num_timesteps),optimizer_updates=int(model._n_updates),
                      final_checkpoint=target.name,final_checkpoint_sha256=sha256(target))
        print(json.dumps(record,indent=2),flush=True)
    except BaseException as exc:
        record.update(status='failed',error=str(exc));raise
    finally:
        env.close()
        (output/'training_manifest.json').write_text(json.dumps(record,indent=2))
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario',choices=['peak','offpeak'],required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--timesteps',type=int,default=100000)
    parser.add_argument('--learning-rate',type=float,default=1e-3)
    parser.add_argument('--exploration-fraction',type=float,default=.3)
    parser.add_argument('--model-seed',type=int,default=42)
    parser.add_argument('--sumo-seeds',type=int,nargs='+',default=[42,43,44])
    train(parser.parse_args())

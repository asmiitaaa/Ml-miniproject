"""Execute a selected policy with the same raw SUMO output schema as the baselines."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from traffic_eval.runner import load_config,check_inputs,run_one


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/evaluation.corridor.json'))
    parser.add_argument('--scenario',choices=['peak','offpeak'],default='peak')
    parser.add_argument('--seed',type=int,default=11);parser.add_argument('--seconds',type=int,default=3600)
    parser.add_argument('--output');parser.add_argument('--gui',action='store_true');args=parser.parse_args()
    cfg=load_config(args.config)
    cfg.update(project_root=cfg['_root'],duration_s=args.seconds,seeds=[args.seed])
    cfg['scenarios']={args.scenario:cfg['scenarios'][args.scenario]}
    check_inputs(cfg,['dqn'])
    output=Path(args.output) if args.output else ROOT/'results'/('policy_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    configuration=output/'run_config.json';configuration.write_text(json.dumps(cfg,indent=2))
    cfg=load_config(configuration)
    result=run_one(cfg,args.scenario,'dqn',args.seed,output/'raw'/args.scenario/'dqn'/f'seed_{args.seed}',args.gui)
    import pandas as pd
    pd.DataFrame([result]).to_csv(output/'runs.csv',index=False)
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2));print(f'Policy execution complete: {output}')


if __name__=='__main__':main()

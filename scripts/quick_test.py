"""Run a short real three-controller integration test; never final evidence."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from traffic_eval.runner import load_config, run_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--output')
    parser.add_argument('--config',default=str(ROOT/'configs/evaluation.model_validation.json'))
    parser.add_argument('--checkpoint',help='Optional real checkpoint path; timing profile still comes from --config')
    args=parser.parse_args()
    if args.seconds<30:parser.error('Use at least 30 seconds')
    cfg=load_config(args.config)
    cfg.update(project_root=cfg['_root'],duration_s=args.seconds,seeds=[11],experiment_label='quick-integration-test-not-final-results')
    if args.checkpoint:
        checkpoint=str(Path(args.checkpoint).resolve())
        cfg['controllers']['dqn']['checkpoint']=checkpoint
        for scenario in cfg['scenarios'].values():scenario['dqn_checkpoint']=checkpoint
    cfg['scenarios']={'peak':cfg['scenarios']['peak']}
    out=Path(args.output).resolve() if args.output else ROOT/'results'/('quick_test_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    configuration=out/'run_config.json';configuration.write_text(json.dumps(cfg,indent=2))
    run_experiment(configuration,out,['fixed','actuated','dqn'])
    print(f'QUICK TEST PASSED: all 3 controllers ran {args.seconds}s with seed 11.')
    print(f'Results: {out}')
    print(f'One-seed short tests intentionally show evidence-quality warnings. Full experiment configuration: {args.config}')


if __name__=='__main__':main()

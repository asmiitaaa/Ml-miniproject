"""Run a short real three-controller integration test; never final evidence."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from traffic_eval.runner import run_experiment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds',type=int,default=180)
    parser.add_argument('--output')
    args=parser.parse_args()
    if args.seconds<30 or args.seconds%5:parser.error('Use at least 30 seconds, in multiples of 5')
    cfg=json.loads((ROOT/'configs/evaluation.model_validation.json').read_text())
    cfg.update(project_root=str(ROOT),duration_s=args.seconds,seeds=[11],experiment_label='quick-integration-test-not-final-results')
    cfg['scenarios']={'peak':cfg['scenarios']['peak']}
    out=Path(args.output).resolve() if args.output else ROOT/'results'/('quick_test_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    configuration=out/'run_config.json';configuration.write_text(json.dumps(cfg,indent=2))
    run_experiment(configuration,out,['fixed','actuated','dqn'])
    print(f'QUICK TEST PASSED: all 3 controllers ran {args.seconds}s with seed 11.')
    print(f'Results: {out}')
    print('One-seed short tests intentionally show evidence-quality warnings. Full results remain in results/model_validation.')


if __name__=='__main__':main()

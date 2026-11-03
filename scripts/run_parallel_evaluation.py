"""Execute independent seeded SUMO jobs concurrently without altering inputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_eval.runner import check_inputs, load_config, run_one, sha256


def job(cfg, scenario, controller, seed, path):
    return run_one(cfg, scenario, controller, seed, Path(path))


def main():
    import pandas as pd
    from traffic_eval.reporting import write_report
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/evaluation.model_validation.json')
    parser.add_argument('--output',default='results/model_validation')
    parser.add_argument('--jobs',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.jobs<=8:parser.error('--jobs must be between1and8')
    cfg=load_config(args.config)
    controllers=['fixed','actuated','dqn']
    check_inputs(cfg,controllers)
    out=Path(args.output).resolve()
    if out.exists():raise FileExistsError('Choose a new output directory; existing evidence is preserved')
    out.mkdir(parents=True)
    manifest={'config_sha256':sha256(cfg['_config_path']),'jobs':args.jobs,'status':'running','failures':[],'note':'Concurrent processes affect wall time only; SUMO time, inputs, controller scope and seeds are identical across paired jobs.'}
    records=[]
    try:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures={}
            for scenario in cfg['scenarios']:
                for seed in cfg['seeds']:
                    for controller in controllers:
                        path=out/'raw'/scenario/controller/f'seed_{seed}'
                        futures[pool.submit(job,cfg,scenario,controller,seed,str(path))]=(scenario,controller,seed)
            for future in as_completed(futures):
                identity=futures[future]
                try:
                    result=future.result()
                    records.append(result)
                    pd.DataFrame(records).sort_values(['scenario','seed','controller']).to_csv(out/'runs.csv',index=False)
                    print(f'Completed {len(records)}/{len(futures)}: {identity}',flush=True)
                except Exception as exc:
                    manifest['failures'].append({'run':identity,'error':str(exc)})
                    print(f'FAILED {identity}: {exc}',flush=True)
        if manifest['failures']:raise RuntimeError('Some runs failed; inspect execution_manifest.json and raw logs before retrying')
        summary=write_report(out)
        manifest.update(status='complete',completed_runs=len(records),quality=summary)
        print(json.dumps(summary,indent=2),flush=True)
    except BaseException:
        manifest['status']='failed'
        raise
    finally:
        (out/'execution_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')


if __name__=='__main__':main()

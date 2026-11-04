"""Select checkpoints on predeclared validation seeds, never on final test rows."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from traffic_eval.runner import load_config,run_one,sha256


def eligibility(rows,expected):
    return bool(rows) and all(row.get('teleports',0)==0 and row.get('collisions',0)==0 and row.get('collision_events',0)==0
        and row.get('pending_departures',0)==0 and row.get('unfinished_trips',0)==0
        and row['completed_trips']==expected for row in rows)


def convergence(monitor,steps):
    import numpy as np
    import pandas as pd
    frame=pd.read_csv(monitor,comment='#')
    frame=frame.loc[frame.l.cumsum()<=steps]
    rewards=frame.r.to_numpy(dtype=float)
    out={'episodes_reviewed':len(rewards),'decision_limit':int(steps),'criterion':'at least 40 complete episodes; last-two-10-episode mean drift <=10%, last-20 CV <=25%, final mean improves on initial mean',
         'interpretation':'empirical training-curve plateau heuristic, not a proof of optimality or safety'}
    if len(rewards)<40:return dict(out,plateau_detected=False,reason='not enough completed episodes at this checkpoint')
    scale=max(abs(float(np.mean(rewards[-20:]))),1.)
    drift=abs(float(np.mean(rewards[-10:])-np.mean(rewards[-20:-10])))/scale
    cv=float(np.std(rewards[-20:],ddof=1))/scale
    start=float(np.mean(rewards[:10]));end=float(np.mean(rewards[-10:]))
    return dict(out,initial_10_mean=start,final_10_mean=end,last_20_cv=cv,last_two_10_relative_drift=drift,
                plateau_detected=bool(drift<=.1 and cv<=.25 and end>=start))


def job(config,candidate,scenario,seed,folder):
    cfg=load_config(config)
    cfg['scenarios']={scenario:cfg['scenarios'][scenario]}
    cfg['scenarios'][scenario]['dqn_checkpoint']=str((ROOT/candidate['checkpoint']).resolve())
    cfg['experiment_label']='checkpoint-selection-validation-not-final-test'
    row=run_one(cfg,scenario,'dqn',seed,Path(folder))
    row.update(candidate=candidate['id'],checkpoint=candidate['checkpoint'],
               checkpoint_sha256=sha256(ROOT/candidate['checkpoint']),checkpoint_steps=candidate['steps'])
    return row


def main(args):
    import pandas as pd
    plan=json.loads(Path(args.plan).read_text())
    seeds=plan['validation_seeds']
    if set(seeds)&set(plan['training_seeds']+plan['test_seeds']):raise ValueError('Seed split overlaps')
    config=(ROOT/plan['config']).resolve()
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'selection_plan.json').write_text(json.dumps(plan,indent=2))
    records=[];failures=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures={}
        for candidate in plan['candidates']:
            if not (ROOT/candidate['checkpoint']).is_file():raise FileNotFoundError(candidate['checkpoint'])
            for scenario in candidate['scenarios']:
                for seed in seeds:
                    folder=output/'raw'/candidate['id']/scenario/f'seed_{seed}'
                    futures[pool.submit(job,str(config),candidate,scenario,seed,str(folder))]=(candidate['id'],scenario,seed)
        for future in as_completed(futures):
            identity=futures[future]
            try:
                records.append(future.result())
                pd.DataFrame(records).to_csv(output/'validation_runs.csv',index=False)
                print(f'Validated {len(records)}/{len(futures)}: {identity}',flush=True)
            except Exception as exc:
                failures.append(dict(run=identity,error=str(exc)))
                print(f'FAILED {identity}: {exc}',flush=True)
    result={'selection_rule':plan['selection_rule'],'validation_seeds':seeds,'test_seeds':plan['test_seeds'],
            'failures':failures,'scenarios':{},'plan_sha256':sha256(args.plan),'config_sha256':sha256(config)}
    for scenario,expected in plan['expected_completed'].items():
        ranks=[]
        for candidate in plan['candidates']:
            if scenario not in candidate['scenarios']:continue
            rows=[r for r in records if r['candidate']==candidate['id'] and r['scenario']==scenario]
            if len(rows)!=len(seeds):continue
            review=convergence(ROOT/candidate['monitor'],candidate['steps'])
            ranks.append({'candidate':candidate['id'],'checkpoint':candidate['checkpoint'],
                'checkpoint_sha256':rows[0]['checkpoint_sha256'],'steps':candidate['steps'],
                'eligible':eligibility(rows,expected),'mean_waiting_s':sum(r['avg_waiting_time_s'] for r in rows)/len(rows),
                'collision_events':sum(r['collision_events'] for r in rows),'teleports':sum(r['teleports'] for r in rows),
                'completed_trips':[r['completed_trips'] for r in rows],'convergence':review})
        ranks.sort(key=lambda r:(not r['eligible'],r['mean_waiting_s'],r['steps'],r['candidate']))
        selected=next((r for r in ranks if r['eligible']),None)
        result['scenarios'][scenario]={'selected':selected,'ranking':ranks}
    (output/'model_selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({s:v['selected'] for s,v in result['scenarios'].items()},indent=2),flush=True)
    if failures or any(v['selected'] is None for v in result['scenarios'].values()):return 2
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--jobs',type=int,default=2)
    sys.exit(main(parser.parse_args()))

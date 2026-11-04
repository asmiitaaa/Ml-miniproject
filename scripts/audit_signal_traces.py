"""Audit every completed DQN run's exact phase sequence and dwell durations."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rl_agent.corridor import load_mapping,TIMING
from traffic_eval.runner import sha256


def audit(raw):
    layouts=load_mapping();runs=[]
    for manifest_path in Path(raw).rglob('manifest.json'):
        manifest=json.loads(manifest_path.read_text())
        if manifest.get('status')!='complete' or manifest.get('controller')!='dqn':continue
        trace=manifest_path.parent/'phase_transitions.csv'
        if not trace.is_file():raise ValueError(f'Missing exact phase trace: {trace}')
        if manifest.get('output_sha256',{}).get(trace.name)!=sha256(trace):raise ValueError('Phase trace hash mismatch')
        with trace.open(newline='') as stream:rows=list(csv.DictReader(stream))
        groups={key:[] for key in layouts}
        for row in rows:groups[row['tls_id']].append(row)
        durations=[]
        for key,events in groups.items():
            if not events or float(events[0]['time'])!=0 or int(events[0]['phase'])!=0:
                raise ValueError('Trace does not start from the declared phase at time zero')
            phases=layouts[key]['phases']
            for before,after in zip(events,events[1:]):
                index=int(before['phase']);phase=phases[index]
                duration=float(after['time'])-float(before['time'])
                if int(after['phase'])!=(index+1)%len(phases):raise ValueError('Skipped a source clearance/green phase')
                if before['state']!=phase['state']:raise ValueError('Trace state does not match mapping')
                if phase['green']:
                    if not TIMING['min_green_s']<=duration<=TIMING['max_green_s']:raise ValueError('Green dwell violation')
                elif duration!=phase['duration']:raise ValueError('Clearance/all-red dwell violation')
                durations.append(dict(tls_id=key,phase=index,duration_s=duration,green=phase['green']))
        runs.append({'scenario':manifest['scenario'],'seed':manifest['seed'],'signals':len(groups),
                     'completed_phase_dwells':len(durations),'violations':0})
    if not runs:raise ValueError('No completed corridor DQN traces')
    return {'runs':runs,'dqn_runs_audited':len(runs),'violations':0,
            'note':'Last, unfinished phase in each run is right-censored and excluded from dwell tests. The controller also verifies the actual SUMO phase before every one-second update.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--raw',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();result=audit(args.raw)
    Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

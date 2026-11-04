"""Shared local features and legal signal execution for independent corridor DQN.

The 17-feature contract is deliberately distinct from the original 13-feature
primary-lane v2 model. A different feature schema always requires new training.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='all-lanes-cardinal-served-fraction-17-v1'
TIMING={'decision_interval_s':3.,'min_green_s':12.,'max_green_s':60.,'yellow_floor_s':6.}


def mapping_file():
    return ROOT/'configs/corridor_mapping.json'


def mapping_digest():
    # Semantic JSON hash survives checkout line-ending conversion.
    return hashlib.sha256(json.dumps(load_mapping(),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def load_mapping():
    return json.loads(mapping_file().read_text(encoding='utf-8'))['signals']


class LocalSignal:
    """Hold/advance the original cyclic program without skipping any phase."""
    def __init__(self,connection,key,layout):
        self.connection=connection;self.key=key;self.layout=layout
        self.phases=layout['phases']
        self.slots=layout['slots']
        self.lanes=[lane for slot in self.slots for lane in slot]
        if len(self.lanes)!=len(set(self.lanes)):raise ValueError('Mapping repeats an incoming lane')
        tl=connection.trafficlight
        if key not in tl.getIDList():raise ValueError(f'Mapped signal missing: {key}')
        program=next(p for p in tl.getAllProgramLogics(key) if p.programID==tl.getProgram(key))
        if [p.state for p in program.phases]!=[p['state'] for p in self.phases]:
            raise ValueError(f'Phase states differ from reviewed mapping: {key}')
        if any(float(live.duration)<mapped['duration'] for live,mapped in zip(program.phases,self.phases) if not mapped['green']):
            raise ValueError(f'Live clearance is shorter than the mapped program: {key}')
        if set(tl.getControlledLanes(key))!=set(self.lanes):
            raise ValueError(f'Incoming lanes differ from reviewed mapping: {key}')
        links=tl.getControlledLinks(key)
        expected={lane:set(indexes) for lane,indexes in layout['lane_links'].items()}
        actual={lane:set() for lane in self.lanes}
        for index,group in enumerate(links):
            for link in group:actual[link[0]].add(index)
        if actual!=expected:raise ValueError(f'Signal links differ from reviewed mapping: {key}')
        for phase in self.phases:
            if not phase['green'] and phase['duration']<=0:raise ValueError('Clearance must have positive duration')
        self.phase=0;self.elapsed=0.;self.pending=False;self.now=0.
        self.on_transition=None
        self.apply()

    def request(self,action):
        if int(action) not in (0,1):raise ValueError('Action must be hold=0 or switch=1')
        if action==1 and self.phases[self.phase]['green']:self.pending=True

    def advance(self,now):
        dt=now-self.now
        if dt<0 or dt>1.000001:raise ValueError('Signal must advance at every one-second SUMO step')
        self.elapsed+=dt;self.now=now
        phase=self.phases[self.phase]
        leave=(phase['green'] and self.elapsed>=TIMING['min_green_s'] and (self.pending or self.elapsed>=TIMING['max_green_s']))
        leave=leave or (not phase['green'] and self.elapsed>=phase['duration'])
        if leave:
            self.phase=(self.phase+1)%len(self.phases);self.elapsed=0.;self.pending=False
            if self.on_transition:self.on_transition(now,self)
        self.apply()

    def apply(self):
        tl=self.connection.trafficlight
        tl.setPhase(self.key,self.phase)
        tl.setPhaseDuration(self.key,2.) # owned until the next one-second step

    def observation(self):
        lane=self.connection.lane
        counts=[sum(lane.getLastStepVehicleNumber(k) for k in slot)/100. for slot in self.slots]
        queues=[sum(lane.getLastStepHaltingNumber(k) for k in slot)/100. for slot in self.slots]
        waiting=[sum(lane.getWaitingTime(k) for k in slot)/3600. for slot in self.slots]
        state=self.phases[self.phase]['state']
        served=[]
        for slot in self.slots:
            indexes={index for k in slot for index in self.layout['lane_links'][k]}
            served.append(sum(state[i] in 'gG' for i in indexes)/len(indexes) if indexes else 0.)
        return np.clip(np.array(counts+queues+waiting+served+[self.elapsed/60.],dtype=np.float32),0,1)

    def waiting(self):
        return sum(self.connection.lane.getWaitingTime(key) for key in self.lanes)


class CorridorPolicy:
    def __init__(self,connection,model,options):
        if getattr(model,'feature_schema',None)!=SCHEMA:raise ValueError('Checkpoint is not trained for the 17-feature corridor contract')
        if getattr(model,'corridor_timing',None)!=TIMING:raise ValueError('Checkpoint corridor timing differs')
        if getattr(model,'mapping_sha256',None)!=mapping_digest():raise ValueError('Checkpoint mapping hash differs')
        if model.observation_space.shape!=(17,) or model.action_space.n!=2:raise ValueError('Wrong corridor observation/action spaces')
        if not np.allclose(model.observation_space.low,0) or not np.allclose(model.observation_space.high,1):raise ValueError('Wrong feature bounds')
        if model.num_timesteps<=1000:raise ValueError('A learned checkpoint is required')
        if float(options.get('_step_length_s',1))!=1:raise ValueError('Corridor controller requires one-second simulation steps')
        if options.get('_lateral_resolution_m',.4)!=.4:raise ValueError('Corridor controller was trained with 0.4 m lateral resolution')
        layouts=load_mapping()
        keys=list(options.get('control_tls_ids',layouts))
        if set(keys)!=set(layouts):raise ValueError('Corridor evaluation must control every mapped signal')
        self.connection=connection;self.model=model
        self.signals=[LocalSignal(connection,key,layouts[key]) for key in sorted(keys)]
        self.next_decision=0.;self.file=None;self.writer=None
        self.phase_file=None;self.phase_writer=None;self.started=False
        self.metadata={'adapter':'independent-corridor-17-v1','timing_profile':'corridor-v1',
            'feature_schema':SCHEMA,'mapping_sha256':mapping_digest(),'controlled_tls_ids':sorted(keys),
            'training_control_tls_ids':getattr(model,'training_control_tls_ids',[]),
            'training_seeds':getattr(model,'training_seeds',[]),'model_num_timesteps':int(model.num_timesteps),
            'timing':TIMING,'warnings':['Demand is synthetic and uncalibrated; this is simulation validation, not field deployment.']}
        if getattr(model,'training_status','')!='validation-reviewed':
            self.metadata['warnings'].append('Training convergence/model selection still requires review.')
        if options.get('_run_dir'):
            self.file=(Path(options['_run_dir'])/'dqn_actions.csv').open('w',newline='',encoding='utf-8')
            self.writer=csv.writer(self.file)
            self.writer.writerow(['time','tls_id','action','applied_phase','internal_elapsed_s',*[f'obs_{i}' for i in range(17)]])
            self.phase_file=(Path(options['_run_dir'])/'phase_transitions.csv').open('w',newline='',encoding='utf-8')
            self.phase_writer=csv.writer(self.phase_file)
            self.phase_writer.writerow(['time','tls_id','phase','state','green','minimum_duration_s'])
            for signal in self.signals:
                signal.on_transition=self._record_transition
                self._record_transition(0,signal)

    def _record_transition(self,now,signal):
        phase=signal.phases[signal.phase]
        self.phase_writer.writerow([now,signal.key,signal.phase,phase['state'],phase['green'],
            TIMING['min_green_s'] if phase['green'] else phase['duration']])

    def step(self,now):
        if self.started:
            for signal in self.signals:
                if self.connection.trafficlight.getPhase(signal.key)!=signal.phase:
                    raise RuntimeError('SUMO changed an owned signal phase between one-second updates')
        self.started=True
        for signal in self.signals:signal.advance(now)
        if now+1e-8<self.next_decision:return
        if not math.isclose(now,self.next_decision):raise ValueError('Skipped a corridor decision boundary')
        observations=np.stack([signal.observation() for signal in self.signals])
        actions,_=self.model.predict(observations,deterministic=True)
        for signal,action,obs in zip(self.signals,np.asarray(actions).reshape(-1),observations):
            signal.request(int(action));signal.advance(now)
            if self.writer:self.writer.writerow([now,signal.key,int(action),signal.phase,signal.elapsed,*obs.tolist()])
        self.next_decision+=TIMING['decision_interval_s']

    def close(self):
        if self.file:self.file.close()
        if self.phase_file:self.phase_file.close()


def create_controller(*,connection,checkpoint,options):
    import torch
    from traffic_eval.model_loading import load_dqn
    torch.set_num_threads(1)
    return CorridorPolicy(connection,load_dqn(checkpoint),options)

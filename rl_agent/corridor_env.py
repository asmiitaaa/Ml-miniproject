"""Gymnasium environment sharing the corridor inference signal implementation."""
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from traffic_eval.runner import input_files,prepare_network,find_sumo,_start_sumo,_stop_process
from traffic_eval.collisions import collision_args
from .corridor import LocalSignal,load_mapping,TIMING


TRAINED_TLS='cluster_11484393853_11484393854_11484393855_11484393856_#11more'


class CorridorEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,cfg,scenario,output,seeds=(42,43,44),tls_id=TRAINED_TLS):
        super().__init__()
        self.cfg=cfg;self.scenario=scenario;self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.seeds=list(seeds);self.tls_id=tls_id
        if not self.seeds or set(self.seeds)&{11,12,13,14,15,101,102,103}:
            raise ValueError('Training seeds must be disjoint from validation/test seeds')
        self.observation_space=spaces.Box(0.,1.,shape=(17,),dtype=np.float32)
        self.action_space=spaces.Discrete(2)
        self.connection=self.process=self.signal=None
        self.episode=-1;self.steps=0;self.collisions=0;self.teleports=0
        self.layout=load_mapping()[tls_id]
        self.sumocfg,network,_=input_files(cfg,cfg['scenarios'][scenario])
        self.private_net=self.output/'training.net.xml'
        prepare_network(network,self.private_net,'fixed',{'yellow_floor_s':6.})

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.close();self.episode+=1;self.steps=0;self.collisions=0;self.teleports=0
        seed_value=self.seeds[self.episode%len(self.seeds)]
        folder=self.output/f'episode_{self.episode:05d}_seed_{seed_value}'
        folder.mkdir(exist_ok=False)
        command=[find_sumo(self.cfg),'-c',str(self.sumocfg),'--net-file',str(self.private_net),
            '--seed',str(seed_value),'--random','false','--begin','0','--end','3600','--step-length','1',
            '--lateral-resolution','.4','--no-step-log','true','--duration-log.disable','true',
            '--tripinfo-output',str(folder/'tripinfo.xml'),'--tripinfo-output.write-unfinished','true',
            '--error-log',str(folder/'sumo-errors.log')]+collision_args({'collision_action':'teleport'},folder)
        import traci
        self.connection,self.process=_start_sumo(command,folder,traci)
        self.signal=LocalSignal(self.connection,self.tls_id,self.layout)
        return self.signal.observation(),{'sumo_seed':seed_value}

    def step(self,action):
        if self.connection is None:raise RuntimeError('Reset before stepping')
        self.signal.request(int(action))
        for _ in range(int(TIMING['decision_interval_s'])):
            self.signal.advance(self.connection.simulation.getTime())
            self.connection.simulationStep()
            self.collisions+=self.connection.simulation.getCollidingVehiclesNumber()
            self.teleports+=self.connection.simulation.getStartingTeleportNumber()
            self.signal.advance(self.connection.simulation.getTime())
        self.steps+=1
        reward=-float(self.signal.waiting())
        info={'colliding_vehicle_detections':self.collisions,'teleports':self.teleports,
              'sumo_seed':self.seeds[self.episode%len(self.seeds)],'waiting_penalty_s':-reward}
        return self.signal.observation(),reward,False,self.steps>=1200,info

    def close(self):
        try:
            if self.connection is not None:self.connection.close(wait=False)
        finally:
            self.connection=None
            if self.process is not None:
                try:self.process.wait(timeout=10)
                except Exception:_stop_process(self.process)
            self.process=None

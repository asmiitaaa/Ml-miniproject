"""Checkpoint feature ordering, delayed switching, and comparison scope checks."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from traffic_eval.agent_adapter import TrainedSignalController, TRAINED_TLS, TRAINED_LANES, OBS_HIGH
from traffic_eval.runner import prepare_network


class FakeTrafficLight:
    def __init__(self):
        self.applied=[]
        self.durations=[]
    def getIDList(self): return [TRAINED_TLS,'another_signal']
    def getControlledLanes(self,key): return TRAINED_LANES+['another_lane_1']
    def getProgram(self,key): return '0'
    def getAllProgramLogics(self,key):
        return [SimpleNamespace(programID='0',phases=[SimpleNamespace(state=s,duration=d) for s,d in [('Gr',39),('yr',6),('rG',39),('ry',6)]])]
    def setPhase(self,key,phase): self.applied.append((key,phase))
    def setPhaseDuration(self,key,duration): self.durations.append(duration)


class FakeLane:
    def getLastStepVehicleNumber(self,key): return [101,2,3,4][TRAINED_LANES.index(key)]
    def getLastStepHaltingNumber(self,key): return [1,0,2,0][TRAINED_LANES.index(key)]
    def getWaitingTime(self,key): return [4000,20,30,40][TRAINED_LANES.index(key)]


class FakeModel:
    def __init__(self,actions=None):
        self.observation_space=SimpleNamespace(shape=(13,),low=np.zeros(13),high=OBS_HIGH)
        self.action_space=SimpleNamespace(n=2)
        self.num_timesteps=20000
        self.seen=[]
        self.actions=iter(actions) if actions else None
    def predict(self,observation,deterministic):
        assert deterministic is True
        self.seen.append(observation.copy())
        return np.array(next(self.actions) if self.actions else 1),None


class AdapterTests(unittest.TestCase):
    def make(self,options=None,actions=None):
        connection=SimpleNamespace(trafficlight=FakeTrafficLight(),lane=FakeLane())
        model=FakeModel(actions)
        return TrainedSignalController(connection,model,options or {}),connection,model

    def test_features_match_training_vehicle_count_halting_wait_order_and_clip(self):
        adapter,_,_=self.make()
        np.testing.assert_array_equal(adapter.observation(),[100,2,3,4,1,0,2,0,3600,20,30,40,0])
        self.assertEqual(adapter.observation().dtype,np.float32)

    def test_replay_matches_training_boundary_order_min_green_yellow_and_carry(self):
        adapter,connection,model=self.make()
        for time in range(31): adapter.step(time)
        self.assertEqual([phase for _,phase in connection.trafficlight.applied],[0,0,1,2,2,3,0])
        self.assertEqual([int(obs[-1]) for obs in model.seen],[0,0,0,2,2,2,0])
        self.assertEqual(adapter.phase_elapsed,2) # 5-second block minus nominal yellow3
        self.assertTrue(all(key==TRAINED_TLS for key,_ in connection.trafficlight.applied))
        self.assertTrue(all(duration>5 for duration in connection.trafficlight.durations))

    def test_switch_request_persists_until_minimum_green(self):
        adapter,connection,_=self.make(actions=[1,0,0])
        for time in range(11): adapter.step(time)
        self.assertEqual([phase for _,phase in connection.trafficlight.applied],[0,0,1])

    def test_network_clearance_is_explicit_changed_dynamics(self):
        adapter,connection,_=self.make({'signal_mode':'network-clearance'})
        for time in range(21): adapter.step(time)
        self.assertEqual([phase for _,phase in connection.trafficlight.applied],[0,0,1,1,2])
        self.assertEqual(adapter.metadata['effective_evaluation_yellow_s'],10)
        self.assertTrue(any('changes' in text for text in adapter.metadata['warnings']))

    def test_arbitrary_lane_mapping_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'only its trained junction'):
            self.make({'lane_ids':list(reversed(TRAINED_LANES))})

    def test_changed_observation_bounds_rejected(self):
        connection=SimpleNamespace(trafficlight=FakeTrafficLight(),lane=FakeLane())
        model=FakeModel(); model.observation_space.high=np.ones(13)
        with self.assertRaisesRegex(ValueError,'bounds do not match'):
            TrainedSignalController(connection,model,{})


class ScopeTests(unittest.TestCase):
    def test_actuated_changes_only_selected_junction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'net.xml'; target=root/'controlled.xml'
            source.write_text('<net><tlLogic id="trained" type="static"><phase state="Gr" duration="39"/><phase state="yr" duration="6"/></tlLogic><tlLogic id="other" type="static"><phase state="Gr" duration="39"/></tlLogic></net>')
            prepare_network(source,target,'actuated',{'control_tls_ids':['trained']})
            logics=ET.parse(target).findall('tlLogic')
            self.assertEqual([logic.get('type') for logic in logics],['actuated','static'])
            self.assertEqual(logics[0].find('phase').get('minDur'),'10')
            self.assertIsNone(logics[1].find('phase').get('minDur'))


if __name__=='__main__': unittest.main()

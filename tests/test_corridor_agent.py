from types import SimpleNamespace
import numpy as np
import pytest

from rl_agent.corridor import LocalSignal,CorridorPolicy,SCHEMA,TIMING
from traffic_eval.runner import prepare_network
import xml.etree.ElementTree as ET


def make(states=('Gr','yr','rG','ry'),durations=(39,6,39,6)):
    phases=[dict(state=s,duration=d,green=('G' in s and 'y' not in s)) for s,d in zip(states,durations)]
    applied=[]
    light=SimpleNamespace(getIDList=lambda:['t'],getProgram=lambda key:'0',
        getAllProgramLogics=lambda key:[SimpleNamespace(programID='0',phases=[SimpleNamespace(state=p['state'],duration=p['duration']) for p in phases])],
        getControlledLanes=lambda key:['north_0','south_0'],
        getControlledLinks=lambda key:[[('north_0','out_0','via_0')],[('south_0','out_1','via_1')]],
        setPhase=lambda key,phase:applied.append(phase),setPhaseDuration=lambda key,duration:None)
    lane=SimpleNamespace(getLastStepVehicleNumber=lambda key:2,getLastStepHaltingNumber=lambda key:1,getWaitingTime=lambda key:10)
    conn=SimpleNamespace(trafficlight=light,lane=lane)
    layout=dict(phases=phases,slots=[['north_0'],[],['south_0'],[]],lane_links={'north_0':[0],'south_0':[1]})
    return LocalSignal(conn,'t',layout),applied


def test_exact_clearance_and_minimum_green_with_one_second_updates():
    signal,_=make();changes=[];previous=None
    for now in range(43):
        signal.advance(now)
        if now%3==0:signal.request(1);signal.advance(now)
        if signal.phase!=previous:changes.append((now,signal.phase));previous=signal.phase
    assert changes==[(0,0),(12,1),(18,2),(30,3),(36,0)]


def test_request_persists_and_yellow_cannot_be_interrupted():
    signal,_=make();signal.request(1)
    for now in range(1,13):signal.advance(now)
    assert signal.phase==1
    for now in range(13,18):signal.request(1);signal.advance(now);assert signal.phase==1
    signal.advance(18);assert signal.phase==2
    for now in range(19,60):signal.advance(now)
    assert signal.phase==2


def test_maximum_green_prevents_permanent_starvation():
    signal,_=make()
    for now in range(61):signal.advance(now)
    assert signal.phase==1 and signal.elapsed==0


def test_all_red_is_preserved_between_yellow_and_next_green():
    signal,_=make(('GG','yy','rr'),(79,6,5))
    signal.request(1)
    for now in range(19):signal.advance(now)
    assert signal.phase==2
    for now in range(19,23):signal.advance(now);assert signal.phase==2
    signal.advance(23);assert signal.phase==0


def test_fixed_shape_uses_all_lanes_and_zeroes_missing_approaches():
    signal,_=make();obs=signal.observation()
    assert obs.shape==(17,) and obs.dtype==np.float32
    assert np.allclose(obs[:4],[.02,0,.02,0])
    assert np.allclose(obs[4:8],[.01,0,.01,0])
    assert np.allclose(obs[12:16],[1,0,0,0])
    assert signal.waiting()==20


def test_mapping_and_clock_fail_instead_of_guessing():
    signal,_=make()
    with pytest.raises(ValueError,match='one-second'):signal.advance(3)
    with pytest.raises(ValueError,match='Action'):signal.request(2)


def test_primary_lane_legacy_checkpoint_cannot_control_corridor():
    with pytest.raises(ValueError,match='17-feature'):
        CorridorPolicy(None,SimpleNamespace(feature_schema=None),{})


def test_changed_approach_mapping_requires_new_training():
    model=SimpleNamespace(feature_schema=SCHEMA,corridor_timing=TIMING,mapping_sha256='different')
    with pytest.raises(ValueError,match='mapping hash differs'):CorridorPolicy(None,model,{})


def test_geometry_rotation_cannot_silently_change_approach_meanings():
    signal,_=make()
    signal.connection.lane.getShape=lambda key:[(0,1),(0,0)]
    with pytest.raises(ValueError,match='geometry differs'):
        LocalSignal(signal.connection,'t',signal.layout)


def test_clearance_floor_is_identical_for_all_baselines(tmp_path):
    source=tmp_path/'source.net.xml'
    source.write_text('<net><tlLogic id="t" type="static"><phase state="Gr" duration="39"/><phase state="yr" duration="3"/><phase state="rr" duration="5"/></tlLogic></net>')
    original=source.read_bytes()
    for mode in ('fixed','actuated','dqn'):
        target=tmp_path/(mode+'.xml')
        prepare_network(source,target,mode,{'yellow_floor_s':6})
        assert [float(p.get('duration')) for p in ET.parse(target).findall('.//phase')]==[39,6,5]
    assert source.read_bytes()==original

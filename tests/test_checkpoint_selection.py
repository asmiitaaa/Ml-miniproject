import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from traffic_eval.runner import fingerprint

spec=importlib.util.spec_from_file_location('checkpoint_selection',Path(__file__).resolve().parents[1]/'scripts/select_checkpoints.py')
selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)


def test_candidate_needs_complete_collision_free_validation_on_every_seed():
    assert not selection.eligibility([],1000)
    rows=[dict(completed_trips=1000,teleports=0,collisions=0,collision_events=0,pending_departures=0,unfinished_trips=0) for _ in range(3)]
    assert selection.eligibility(rows,1000)
    rows[1]['collision_events']=1
    assert not selection.eligibility(rows,1000)
    rows[1]['collision_events']=0;rows[2]['completed_trips']=999
    assert not selection.eligibility(rows,1000)


def test_curve_review_requires_enough_complete_episodes_and_retains_instability(tmp_path):
    file=tmp_path/'monitor.csv'
    pd.DataFrame({'r':[-100]*10+[-20]*30,'l':[1200]*40}).to_csv(file,index=False)
    assert selection.convergence(file,48000)['plateau_detected']
    assert not selection.convergence(file,24000)['plateau_detected']
    pd.DataFrame({'r':[-100]*20+[-1,-100]*10,'l':[1200]*40}).to_csv(file,index=False)
    assert not selection.convergence(file,48000)['plateau_detected']


def test_common_clearance_mismatch_cannot_be_paired(tmp_path):
    (tmp_path/'net.xml').write_text('<net/>')
    (tmp_path/'run.sumocfg').write_text('<configuration><input><net-file value="net.xml"/></input></configuration>')
    cfg={'_root':str(tmp_path),'controllers':{'fixed':{'yellow_floor_s':6},'dqn':{'yellow_floor_s':3}}}
    with pytest.raises(ValueError,match='same yellow floor'):fingerprint(cfg,{'sumocfg':'run.sumocfg'},11)

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from traffic_eval.collisions import collision_action, collision_records, export_collisions
from traffic_eval.runner import fingerprint, run_one
from traffic_eval.reporting import validate_runs, write_report
import pandas as pd


def input_config(tmp_path):
    (tmp_path/'net.xml').write_text('<net><tlLogic id="t" type="static"><phase state="Gr" duration="30"/></tlLogic></net>')
    file=tmp_path/'run.sumocfg'
    file.write_text('<configuration><input><net-file value="net.xml"/></input></configuration>')
    return {'_root':str(tmp_path),'_config_path':str(file),'scenarios':{'peak':{'sumocfg':'run.sumocfg'}}}


def test_events_preserve_exact_time_place_and_vehicle_ids(tmp_path):
    (tmp_path/'collisions.xml').write_text('<collisions><collision time="2311.00" type="collision" lane="lane_0" pos="4.5" collider="a" victim="b" colliderSpeed="1.0"/><collision time="2312.00" type="collision" lane="lane_0" pos="5.0" collider="a" victim="b"/></collisions>')
    assert export_collisions(tmp_path)==2 # repeated overlap records are retained
    with (tmp_path/'collisions.csv').open() as file:rows=list(csv.DictReader(file))
    assert rows[0]['time']=='2311.00' and rows[0]['collider']=='a'
    assert rows[0]['lane']=='lane_0' and rows[0]['victim']=='b'
    assert rows[1]['time']=='2312.00'


def test_zero_event_output_is_valid_and_writes_a_table_header(tmp_path):
    (tmp_path/'collisions.xml').write_text('<collisions/>')
    assert collision_records(tmp_path/'collisions.xml')==[]
    assert export_collisions(tmp_path)==0
    assert (tmp_path/'collisions.csv').read_text().startswith('time,type,lane,')


def test_every_command_logs_collisions_with_explicit_default_action(tmp_path):
    cfg=input_config(tmp_path)
    with patch.dict('sys.modules',{'traci':SimpleNamespace()}),patch('traffic_eval.runner.find_sumo',return_value='sumo'),patch('traffic_eval.runner._start_sumo',side_effect=RuntimeError('command captured')) as start:
        with pytest.raises(RuntimeError,match='command captured'):run_one(cfg,'peak','fixed',11,tmp_path/'run')
    cmd=start.call_args.args[0]
    assert cmd[cmd.index('--collision-output')+1]==str(tmp_path/'run/collisions.xml')
    assert cmd[cmd.index('--collision.action')+1]=='teleport'


def test_warn_mode_changes_fingerprint_and_is_marked_in_manifest(tmp_path):
    cfg=input_config(tmp_path);scenario=cfg['scenarios']['peak']
    before=fingerprint(cfg,scenario,11)[0];cfg['collision_action']='warn'
    assert fingerprint(cfg,scenario,11)[0]!=before
    with patch.dict('sys.modules',{'traci':SimpleNamespace()}),patch('traffic_eval.runner.find_sumo',return_value='sumo'),patch('traffic_eval.runner._start_sumo',side_effect=RuntimeError('command captured')) as start:
        with pytest.raises(RuntimeError):run_one(cfg,'peak','fixed',11,tmp_path/'warn')
    cmd=start.call_args.args[0];assert cmd[cmd.index('--collision.action')+1]=='warn'
    manifest=json.loads((tmp_path/'warn/manifest.json').read_text())
    assert manifest['diagnostic_mode'] is True


def test_modes_cannot_be_disabled_or_overridden_through_extra_args(tmp_path):
    with pytest.raises(ValueError,match='collision_action'):collision_action({'collision_action':'none'})
    cfg=input_config(tmp_path);cfg['sumo_extra_args']=['--collision.action','none']
    with patch.dict('sys.modules',{'traci':SimpleNamespace()}),patch('traffic_eval.runner.find_sumo',return_value='sumo'):
        with pytest.raises(ValueError,match='cannot override'):run_one(cfg,'peak','fixed',11,tmp_path/'out')


def test_warn_diagnostics_cannot_be_mixed_with_performance_rows(tmp_path):
    rows=[dict(scenario='peak',controller=c,seed=11,demand_hash='same',avg_waiting_time_s=1,avg_trip_duration_s=2,avg_queue_length=0,max_queue_length=0,co2_kg=1,collision_action=a) for c,a in [('fixed','teleport'),('dqn','warn')]]
    with pytest.raises(ValueError,match='Do not mix'):validate_runs(pd.DataFrame(rows))
    for row in rows:row['collision_action']='warn'
    pd.DataFrame(rows).to_csv(tmp_path/'runs.csv',index=False)
    quality=write_report(tmp_path)
    assert not quality['final_evaluation_ready']
    assert any('diagnostic only' in w for w in quality['warnings'])

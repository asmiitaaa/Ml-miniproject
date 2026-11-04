import json
from pathlib import Path

from traffic_eval.demand_coverage import route_coverage,coverage_warnings,from_config
from traffic_eval.runner import load_config


def inputs(tmp_path):
    net=tmp_path/'net.xml'
    net.write_text('<net><edge id="a"/><edge id="b"/><edge id="c"/><tlLogic id="one"/><tlLogic id="two"/><connection from="a" to="b" tl="one"/><connection from="b" to="a" tl="one"/><connection from="b" to="c" tl="two"/></net>')
    routes=tmp_path/'routes.xml'
    return net,routes


def test_repeated_crossings_are_distinct_from_vehicle_coverage(tmp_path):
    net,routes=inputs(tmp_path)
    routes.write_text('<routes><route id="r" edges="a b a b"/><vehicle id="v" route="r"/></routes>')
    report=route_coverage(net,[routes])
    assert report['signals']['one']['planned_crossings']==3
    assert report['signals']['one']['distinct_vehicles']==1
    assert report['active_selected_signal_count']==1
    assert not report['coverage_complete']
    assert 'only 1 of 2' in coverage_warnings(report)[0]


def test_unknown_routes_are_not_treated_as_zero_measured_demand(tmp_path):
    net,routes=inputs(tmp_path)
    routes.write_text('<routes><flow id="f" from="a" to="c" number="10"/><vehicle id="v" route="missing"/></routes>')
    report=route_coverage(net,[routes])
    assert len(report['unsupported_definitions'])==2
    assert not report['coverage_complete']


def test_late_departures_do_not_make_a_short_test_appear_fully_exposed(tmp_path):
    net,routes=inputs(tmp_path)
    routes.write_text('<routes><route id="short" edges="a b"/><route id="long" edges="a b c"/><vehicle id="early" route="short" depart="0"/><vehicle id="late" route="long" depart="600"/></routes>')
    report=route_coverage(net,[routes],end_s=180)
    assert report['active_selected_signal_count']==1
    assert report['excluded_departures_outside_horizon']==1
    assert route_coverage(net,[routes],end_s=3600)['coverage_complete']


def test_current_experiment_exposes_only_one_signal():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/evaluation.model_validation.json')
    report=from_config(cfg,cfg['scenarios']['peak'])
    assert report['resolved_vehicles']==1000
    assert report['active_selected_signal_count']==1
    assert report['selected_signal_count']==6
    assert not report['errors']


def test_full_seed_matrix_cannot_hide_missing_traffic_exposure(tmp_path,monkeypatch):
    import pandas as pd
    from traffic_eval.reporting import write_report
    root=Path(__file__).resolve().parents[1]
    pd.read_csv(root/'report/training/corridor_results/runs.csv').to_csv(tmp_path/'runs.csv',index=False)
    cfg=load_config(root/'configs/evaluation.corridor.json')
    coverage={'scenarios':{name:from_config(cfg,scenario) for name,scenario in cfg['scenarios'].items()}}
    (tmp_path/'demand_coverage.json').write_text(json.dumps(coverage))
    monkeypatch.setattr('traffic_eval.reporting._charts',lambda *args:None)
    result=write_report(tmp_path)
    assert not result['final_evaluation_ready']
    assert any('only 1 of 6' in warning for warning in result['warnings'])

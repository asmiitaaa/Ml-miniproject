import csv
import pytest
from traffic_eval.observations import COLUMNS,check_observations,hourly_geh


def write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=COLUMNS);writer.writeheader();writer.writerows(rows)


def row(**changes):
    result=dict(scenario='peak',partition='calibration',session_id='day1',observation_date='2026-11-01',begin_s=0,end_s=900,
        measurement='edge_count',edge_id='a',to_edge_id='',vehicle_type='car',value=10,unit='vehicles',source_kind='measured',source_reference='survey sheet 1')
    result.update(changes);return result


def test_empty_template_is_not_calibration(tmp_path):
    file=tmp_path/'observations.csv';write(file,[])
    report=check_observations(file)
    assert not report['basic_data_checks_passed'] and not report['calibrated']


def test_simulated_data_cannot_be_presented_as_field_measurements(tmp_path):
    file=tmp_path/'observations.csv';value=row();value['source_kind']='simulated';write(file,[value])
    with pytest.raises(ValueError,match='field calibration'):check_observations(file)


def test_one_field_session_cannot_be_used_for_both_fit_and_validation(tmp_path):
    file=tmp_path/'observations.csv';a=row();b=row();b['partition']='validation';write(file,[a,b])
    with pytest.raises(ValueError,match='cannot be reused'):check_observations(file)


def test_geh_uses_hourly_rates_and_zero_zero_is_defined():
    assert hourly_geh(10,20,900)==pytest.approx(hourly_geh(40,80,3600))
    assert hourly_geh(0,0,900)==0


def test_passing_data_structure_checks_does_not_claim_calibration(tmp_path):
    rows=[]
    for scenario in ['peak','offpeak']:
        for partition in ['calibration','validation']:
            common=dict(scenario=scenario,partition=partition,session_id=scenario+'_'+partition)
            rows.append(row(**common))
            rows.append(row(**common,measurement='travel_time',to_edge_id='b',unit='s',value=30))
    file=tmp_path/'observations.csv';write(file,rows)
    report=check_observations(file)
    assert report['basic_data_checks_passed']
    assert report['calibrated'] is False

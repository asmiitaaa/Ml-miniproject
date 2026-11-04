"""Validate real-data inputs without manufacturing observations or calibration."""
import csv
from datetime import date
import math

COLUMNS=['scenario','partition','session_id','observation_date','begin_s','end_s','measurement','edge_id','to_edge_id','vehicle_type','value','unit','source_kind','source_reference']


def check_observations(path):
    with open(path,newline='',encoding='utf-8-sig') as stream:
        reader=csv.DictReader(stream)
        if not set(COLUMNS).issubset(reader.fieldnames or []):raise ValueError('Observation CSV does not have the required columns')
        rows=list(reader)
    groups={};keys=set();sessions={}
    for number,row in enumerate(rows,2):
        prefix=f'Observation row {number}: '
        if row['scenario'] not in ('peak','offpeak'):raise ValueError(prefix+'scenario must be peak/offpeak')
        if row['partition'] not in ('calibration','validation'):raise ValueError(prefix+'partition must be calibration/validation')
        if row['source_kind']!='measured':raise ValueError(prefix+'synthetic, estimated or simulated data cannot be labelled field calibration')
        if not row['source_reference'].strip() or not row['session_id'].strip():raise ValueError(prefix+'measurement provenance and session ID are required')
        date.fromisoformat(row['observation_date'])
        begin,end,value=map(float,(row['begin_s'],row['end_s'],row['value']))
        if not all(map(math.isfinite,(begin,end,value))) or begin<0 or end<=begin or value<0:raise ValueError(prefix+'invalid numeric observation')
        units={'edge_count':'vehicles','turn_count':'vehicles','speed':'m/s','travel_time':'s'}
        if row['measurement'] not in units or row['unit']!=units[row['measurement']]:raise ValueError(prefix+'measurement/unit mismatch')
        if row['measurement'] in ('edge_count','turn_count') and value!=int(value):raise ValueError(prefix+'counts must be whole observed vehicles')
        if row['measurement']=='travel_time' and value<=0:raise ValueError(prefix+'travel time must be positive')
        if not row['edge_id'].strip():raise ValueError(prefix+'the observed edge/site must be mapped')
        if row['measurement'] in ('turn_count','travel_time') and not row['to_edge_id'].strip():raise ValueError(prefix+'the destination observation point is required')
        if row['vehicle_type'] not in ('motorcycle','auto','car','bus','all'):raise ValueError(prefix+'unknown vehicle category')
        session=(row['session_id'],row['observation_date'])
        if session in sessions and sessions[session]!=row['partition']:raise ValueError(prefix+'one observation session cannot be reused for calibration and validation')
        sessions[session]=row['partition']
        key=tuple(row[column] for column in COLUMNS if column not in ('value','source_reference','source_kind','partition'))
        if key in keys:raise ValueError(prefix+'duplicate measurement')
        keys.add(key)
        groups.setdefault((row['scenario'],row['partition']),set()).add(row['measurement'])
    missing=[]
    for scenario in ('peak','offpeak'):
        for partition in ('calibration','validation'):
            kinds=groups.get((scenario,partition),set())
            if not kinds.intersection({'edge_count','turn_count'}):missing.append(f'{scenario}/{partition}: measured counts missing')
            if not kinds.intersection({'speed','travel_time'}):missing.append(f'{scenario}/{partition}: independent speed/travel-time measurements missing')
    return {'row_count':len(rows),'basic_data_checks_passed':not missing,'missing_evidence':missing,
        'calibrated':False,'interpretation':'Input checks only. Site matching, measurement quality, model fitting and independent validation still require review.'}


def hourly_geh(simulated_count,observed_count,interval_seconds):
    if interval_seconds<=0 or min(simulated_count,observed_count)<0:raise ValueError('Invalid count interval')
    simulated=simulated_count*3600/interval_seconds;observed=observed_count*3600/interval_seconds
    return math.sqrt(2*(simulated-observed)**2/(simulated+observed)) if simulated+observed else 0.

"""Read-only network and demand audit; never labels assumed demand calibrated."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from traffic_eval.demand_coverage import from_config,coverage_warnings
from traffic_eval.runner import load_config,input_files,sha256


def main(args):
    import sumolib
    cfg=load_config(args.config);output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    _,network,_=input_files(cfg,next(iter(cfg['scenarios'].values())))
    net=sumolib.net.readNet(str(network))
    source=ET.parse(network).getroot();junctions={j.get('id'):j for j in source.findall('junction')}
    keys=[x.get('id') for x in source.findall('tlLogic')]
    close=[]
    for a in keys:
        for b in keys:
            if a>=b:continue
            ja,jb=junctions[a],junctions[b]
            distance=math.hypot(float(ja.get('x'))-float(jb.get('x')),float(ja.get('y'))-float(jb.get('y')))
            if distance<50:close.append({'signal_a':a,'signal_b':b,'distance_m':round(distance,2),'status':'review_only_not_proof_of_duplicate_signals'})
    coverage={'kind':'planned_explicit_routes_not_field_counts','scenarios':{}}
    legality=[]
    for scenario,definition in cfg['scenarios'].items():
        ids=cfg.get('controllers',{}).get('dqn',{}).get('control_tls_ids')
        coverage['scenarios'][scenario]=from_config(cfg,definition,ids)
        configuration=ROOT/definition['sumocfg'];tree=ET.parse(configuration)
        routefiles=[(configuration.parent/p.strip()).resolve() for p in tree.find('./input/route-files').get('value').split(',')]
        roots=[ET.parse(path).getroot() for path in routefiles]
        routes={r.get('id'):r.get('edges').split() for root in roots for r in root.findall('route')}
        classes={v.get('id'):v.get('vClass','passenger') for root in roots for v in root.findall('vType')}
        checked=set()
        for root in roots:
            for vehicle in root.findall('vehicle'):
                routeid=vehicle.get('route');vclass=classes.get(vehicle.get('type'),'passenger')
                if (routeid,vclass) in checked:continue
                checked.add((routeid,vclass));edges=[net.getEdge(key) for key in routes[routeid]]
                allowed=all(edge.allows(vclass) for edge in edges)
                connected=all(b in a.getAllowedOutgoing(vclass) for a,b in zip(edges,edges[1:]))
                path,cost=net.getShortestPath(edges[0],edges[-1],vClass=vclass)
                length=sum(edge.getLength() for edge in edges)
                legality.append({'scenario':scenario,'route_id':routeid,'vclass':vclass,'edges_allow_class':allowed,
                    'connections_allow_class':connected,'od_reachable':path is not None,'route_length_m':round(length,2),
                    'shortest_distance_m':None if cost is None else round(cost,2)})
    (output/'demand_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    (output/'route_legality.json').write_text(json.dumps(legality,indent=2)+'\n')
    spec=importlib.util.find_spec('sumo');tools=Path(next(iter(spec.submodule_search_locations)))/'tools/net/netcheck.py'
    connectivity={}
    for vehicle_class in ['passenger','motorcycle','bus']:
        target=output/(vehicle_class+'_connectivity.txt')
        command=[sys.executable,str(tools),str(network),'--vclass',vehicle_class,'--results-output',str(target)]
        result=subprocess.run(command,capture_output=True,text=True)
        (output/(vehicle_class+'_netcheck.log')).write_text(result.stdout+result.stderr)
        connectivity[vehicle_class]={'exit_code':result.returncode,'summary':target.read_text() if target.exists() else ''}
    errors=[row for row in legality if not all(row[k] for k in ['edges_allow_class','connections_allow_class','od_reachable'])]
    audit={'network_sha256':sha256(network),'network_location':source.find('location').attrib,
        'external_edges':len(net.getEdges()),'signal_sites':len(keys),'nearby_signal_pairs':close,
        'route_class_pairs_checked':len(legality),'route_legality_errors':errors,'connectivity':connectivity,
        'field_geometry_verified':False,'observed_traffic_data_available':False,'calibrated':False,
        'warnings':['Field lane counts, turn restrictions, signal coordination and speed limits still require site evidence.']}
    for name,item in coverage['scenarios'].items():audit['warnings'].extend(name+': '+message for message in coverage_warnings(item))
    (output/'network_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# Network and demand audit','',f'Network: {len(keys)} signal sites. Route/class combinations checked: {len(legality)}. Illegal combinations: {len(errors)}.',
        '', '**Calibration is not completed: no measured data has been provided.**','',
        '## Traffic exposure','', '| Scenario | Vehicles | Active controlled signals | Planned crossings |','| --- | ---: | ---: | ---: |']
    for name,item in coverage['scenarios'].items():lines.append(f"| {name} | {item['resolved_vehicles']} | {item['active_selected_signal_count']}/{item['selected_signal_count']} | {sum(v['planned_crossings'] for v in item['signals'].values())} |")
    lines+=['','The other five controllers execute without traffic. Existing delay improvements describe the single active junction, not a corridor traffic-interaction experiment.',
        '', 'Repeated visits to the same junction require route-plausibility review. Nearby signal pairs require review of whether they are distinct stop lines or parts of a coordinated junction; distance alone does not justify merging them.',
        '', '## Connectivity','']
    for name,item in connectivity.items():lines.extend(['### '+name,'','```text',item['summary'].strip(),'```',''])
    lines+=['## Remaining evidence','', 'Verify the physical lane layout, permitted turns, speed limits, stop lines and signal programs. Collect classified counts and turning movements for peak/offpeak periods, plus speed/travel-time observations. Hold out independent observation sessions for validation.',
        '', 'Technical references: [SUMO netcheck](https://sumo.dlr.de/docs/Tools/Net.html#netcheckpy), [observed-count demand generation](https://sumo.dlr.de/docs/Demand/Routes_from_Observation_Points.html), [FHWA calibration guidance](https://ops.fhwa.dot.gov/publications/fhwahop20057/ch1.htm).']
    (output/'AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'output':str(output),'route_legality_errors':len(errors),'active_signals':{k:v['active_selected_signal_count'] for k,v in coverage['scenarios'].items()},'calibrated':False},indent=2))
    return 1 if errors or any(v['exit_code'] for v in connectivity.values()) else 0


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',default='configs/evaluation.corridor.json');parser.add_argument('--output',required=True)
    sys.exit(main(parser.parse_args()))

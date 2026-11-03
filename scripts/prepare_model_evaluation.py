"""Freeze an explicitly uncalibrated, reproducible full-hour evaluation fixture.

The twelve OD pairs and relative weights come from the supplied teammate demand
snapshot. Geometry and vTypes are inherited; all route paths, types, initial
lanes and requested departure times are written explicitly before any controller
runs. SUMO seeds vary vehicle behavior, never the frozen departure schedule.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TLS = "cluster_11484393853_11484393854_11484393855_11484393856_#11more"
LANES = ["24480656#3_0", "794634832#0_0", "794634831#1_0", "631316440#0_0"]
MIX = {"motorcycle": .50, "auto": .15, "car": .30, "bus": .05}
COUNTS = {"peak": 1000, "offpeak": 500}
CLASSES = {"motorcycle": "motorcycle", "auto": "passenger", "car": "passenger", "bus": "bus"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def xml_bytes(root):
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def validate_generated(folder):
    meta = json.loads((folder/"fixture_manifest.json").read_text(encoding="utf-8"))
    types = ET.parse(folder/"vtypes.rou.xml").getroot()
    if {x.attrib['id'] for x in types.findall('vType')} != set(MIX):
        raise ValueError("Frozen type file must contain exactly the four specified classes")
    for name, expected in COUNTS.items():
        filename = folder/f"{name}.rou.xml"
        root = ET.parse(filename).getroot()
        vehicles = root.findall('vehicle')
        actual = Counter(v.attrib['type'] for v in vehicles)
        target = {kind: int(expected*share) for kind,share in MIX.items()}
        if len(vehicles) != expected or dict(actual) != target:
            raise ValueError(f"Wrong count/composition for {name}: {actual}")
        departures = [float(v.attrib['depart']) for v in vehicles]
        if departures != sorted(departures) or len(set(departures)) != expected:
            raise ValueError(f"Departure schedule must be unique and sorted: {name}")
        if departures[0] != 0 or departures[-1] >= 3000:
            raise ValueError(f"All requested departures must be in [0,3000): {name}")
        if sha(filename) != meta['scenarios'][name]['route_file_sha256']:
            raise ValueError(f"Route file changed since generation: {name}")
    if sha(folder/'vtypes.rou.xml') != meta['frozen_types_sha256']:
        raise ValueError("Frozen vehicle definitions changed since generation")
    return meta


def prepare(source, seed):
    import sumolib
    folder = ROOT/"scenarios"/"model_validation"
    folder.mkdir(parents=True, exist_ok=True)
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source demand snapshot missing: {source}")
    network = ROOT/"network"/"corridor.net.xml"
    original_types = ROOT/"demand"/"vtypes.xml"
    pairs=[]
    for flow in ET.parse(source).getroot().findall('flow'):
        if 'from' not in flow.attrib or 'to' not in flow.attrib:
            continue
        pairs.append({'id': flow.attrib['id'], 'from':flow.attrib['from'], 'to':flow.attrib['to'],
                      'weight':float(flow.attrib['vehsPerHour'])})
    if len(pairs) != 12 or len({(p['from'],p['to']) for p in pairs}) != 12:
        raise ValueError("Expected exactly the twelve distinct approach/exit pairs from the teammate demand")
    if any(p['weight'] <= 0 for p in pairs):
        raise ValueError("Pair weights must be positive")
    frozen_types = ET.Element('routes')
    raw_types = {v.attrib['id']:v for v in ET.parse(original_types).getroot().findall('vType')}
    for kind in MIX:
        if kind not in raw_types or raw_types[kind].attrib.get('vClass') != CLASSES[kind]:
            raise ValueError(f"Missing or incompatible original type {kind}")
        frozen_types.append(deepcopy(raw_types[kind]))
    write(folder/'vtypes.rou.xml', xml_bytes(frozen_types))
    net = sumolib.net.readNet(str(network))
    if TLS not in {t.getID() for t in net.getTrafficLights()}:
        raise ValueError("Trained junction does not exist in network")
    for lane_id in LANES:
        net.getLane(lane_id)  # fail if source geometry no longer matches training
    cache={}; valid={}; rejected=[]
    for vclass in sorted(set(CLASSES.values())):
        valid[vclass]=[]
        for pair in pairs:
            start, end = net.getEdge(pair['from']), net.getEdge(pair['to'])
            path, cost = net.getShortestPath(start,end,vClass=vclass)
            reason=None
            if not path or len(path) < 2:
                reason="unreachable or degenerate route"
            elif any(not edge.allows(vclass) for edge in path):
                reason="vehicle class prohibited on route edge"
            else:
                for a,b in zip(path,path[1:]):
                    connections=[c for c in a.getConnections(b)
                                 if c.getFromLane().allows(vclass) and c.getToLane().allows(vclass)]
                    if not connections:
                        reason=f"no class-compatible connection {a.getID()} -> {b.getID()}"
                        break
            if reason:
                rejected.append({'vehicle_class':vclass,'pair':pair['id'],'reason':reason})
                continue
            start_lanes=[c.getFromLane().getIndex() for c in path[0].getConnections(path[1])
                         if c.getFromLane().allows(vclass) and c.getToLane().allows(vclass)]
            cache[vclass,pair['id']]={'edges':[e.getID() for e in path], 'cost_m':cost,
                                      'depart_lane':min(start_lanes)}
            valid[vclass].append(pair)
        if not valid[vclass]:
            raise ValueError(f"No valid route pair exists for {vclass}; repair the scenario")
    meta={'fixture_kind':'uncalibrated-model-validation','generation_seed':seed,
          'source_ref':'origin/ananyaraj','source_commit':'951ea33',
          'source_demand_sha256':sha(source),'network_sha256':sha(network),
          'original_types_sha256':sha(original_types),'frozen_types_sha256':sha(folder/'vtypes.rou.xml'),
          'mode_shares':MIX, 'original_pair_weights':pairs, 'rejected_class_pair_combinations':rejected,
          'valid_pairs_by_vehicle_class':{k:[p['id'] for p in v] for k,v in valid.items()},
          'routing':'sumolib getShortestPath with vClass and explicit connection-permission checks',
          'evaluation_duration_s':3600,'departure_horizon_s':3000,'drain_window_s':600,
          'evaluation_lateral_resolution_m':.4,'training_lateral_resolution_m':.5,
          'controlled_tls_ids':[TLS], 'scenarios':{}}
    for name,count in COUNTS.items():
        rng=random.Random(seed)
        vehicle_types=[kind for kind,share in MIX.items() for _ in range(int(count*share))]
        if len(vehicle_types) != count:
            raise ValueError("Counts cannot realize exact specified composition")
        rng.shuffle(vehicle_types)
        root=ET.Element('routes')
        root.append(ET.Comment('Uncalibrated model-validation fixture; not field-calibrated peak/off-peak demand.'))
        route_ids={}
        for kind in MIX:
            for pair in valid[CLASSES[kind]]:
                rid=f"route_{kind}_{pair['id']}"
                route_ids[kind,pair['id']]=rid
                ET.SubElement(root,'route',id=rid,edges=' '.join(cache[CLASSES[kind],pair['id']]['edges']))
        spacing=3000/count
        type_counts=Counter(); pair_counts=Counter(); schedule=[]
        for i,kind in enumerate(vehicle_types):
            choices=valid[CLASSES[kind]]
            pair=rng.choices(choices,weights=[p['weight'] for p in choices],k=1)[0]
            route=cache[CLASSES[kind],pair['id']]
            attrs={'id':f'{name}_{i:04d}','type':kind,'route':route_ids[kind,pair['id']],
                   'depart':f'{i*spacing:.3f}','departLane':str(route['depart_lane']),
                   'departPos':'0','departSpeed':'0'}
            ET.SubElement(root,'vehicle',attrs)
            type_counts[kind]+=1;pair_counts[pair['id']]+=1;schedule.append(attrs)
        file=folder/f'{name}.rou.xml';write(file,xml_bytes(root))
        cfg=ET.Element('configuration'); inputs=ET.SubElement(cfg,'input')
        ET.SubElement(inputs,'net-file',value='../../network/corridor.net.xml')
        ET.SubElement(inputs,'route-files',value=f'vtypes.rou.xml,{name}.rou.xml')
        timing=ET.SubElement(cfg,'time')
        for key,val in [('begin','0'),('end','3600'),('step-length','1')]: ET.SubElement(timing,key,value=val)
        processing=ET.SubElement(cfg,'processing')
        ET.SubElement(processing,'lateral-resolution',value='0.4')
        ET.SubElement(processing,'ignore-route-errors',value='false')
        write(folder/f'{name}.sumocfg',xml_bytes(cfg))
        meta['scenarios'][name]={'vehicle_count':count,'type_counts':dict(type_counts),
             'pair_counts':dict(pair_counts),'departure_spacing_s':spacing,'last_requested_departure_s':(count-1)*spacing,
             'route_file_sha256':sha(file),
             'schedule_sha256':hashlib.sha256(json.dumps(schedule,sort_keys=True).encode()).hexdigest()}
    meta['cached_routes']=[{'vehicle_class':vc,'pair':pair,**route} for (vc,pair),route in sorted(cache.items())]
    write(folder/'fixture_manifest.json',(json.dumps(meta,indent=2,sort_keys=True)+'\n').encode())
    config={'experiment_label':'uncalibrated-model-validation','project_root':'..',
            'duration_s':3600,'step_length_s':1,'lateral_resolution_m':.4,'seeds':[11,12,13,14,15],
            'scenarios':{name:{'sumocfg':f'scenarios/model_validation/{name}.sumocfg',
                               'demand_files':['scenarios/model_validation/fixture_manifest.json']}
                         for name in COUNTS},
            'controllers':{'fixed':{'control_tls_ids':[TLS]},
                 'actuated':{'control_tls_ids':[TLS],'min_green_s':10,'max_green_s':60,'max_gap_s':3},
                 'dqn':{'control_tls_ids':[TLS],'checkpoint':'models/dqn_traffic_sumo_v1_20000_steps.zip',
                        'factory':'traffic_eval.agent_adapter:create_controller','tls_id':TLS,'lane_ids':LANES,
                        'decision_interval_s':5,'min_green_s':10,'signal_mode':'training-replay'}}}
    write(ROOT/'configs'/'evaluation.model_validation.json',(json.dumps(config,indent=2)+'\n').encode())
    validate_generated(folder)
    return meta


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-demand',type=Path,default=ROOT/'scenarios'/'model_validation'/'source_routes.xml')
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--check',action='store_true',help='verify existing fixture hashes/counts without rewriting')
    args=ap.parse_args()
    meta=validate_generated(ROOT/'scenarios'/'model_validation') if args.check else prepare(args.source_demand,args.seed)
    for name,data in meta['scenarios'].items():
        print(f"{name}: {data['vehicle_count']} vehicles, types={data['type_counts']}, sha256={data['route_file_sha256']}")
    print('Config: configs/evaluation.model_validation.json; full paired matrix: 30 runs')
    print('Uncalibrated demand; checkpoint convergence, corridor transfer and training/evaluation resolution shift remain limitations.')


if __name__=='__main__': main()

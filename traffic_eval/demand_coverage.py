"""Audit planned route exposure; this is not a measurement or calibration."""
from collections import Counter,defaultdict
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET


def route_coverage(network,route_files,selected_ids=None,end_s=None):
    net=ET.parse(network).getroot()
    all_ids=[logic.get('id') for logic in net.findall('tlLogic')]
    selected=list(selected_ids or all_ids)
    unknown=set(selected)-set(all_ids)
    if unknown:raise ValueError(f'Unknown selected signals: {sorted(unknown)}')
    edges={edge.get('id') for edge in net.findall('edge')}
    pairs=defaultdict(set);connections=set()
    for link in net.findall('connection'):
        pair=(link.get('from'),link.get('to'));connections.add(pair)
        if link.get('tl'):pairs[pair].add(link.get('tl'))
    routes={};vehicles=[];unsupported=[];hashes={};errors=[]
    for file in route_files:
        file=Path(file);hashes[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
        root=ET.parse(file).getroot()
        for route in root.findall('route'):
            if route.get('id'):routes[route.get('id')]=route.get('edges','').split()
        vehicles.extend(root.findall('vehicle'))
        unsupported.extend(node.tag+':'+node.get('id','') for node in root if node.tag in ('flow','trip','routeDistribution'))
    crossed=Counter();seen=defaultdict(set);repeat=Counter();resolved=0;excluded=0
    for number,vehicle in enumerate(vehicles):
        key=vehicle.get('id',str(number));inline=vehicle.find('route')
        try:depart=float(vehicle.get('depart','0'))
        except ValueError:
            unsupported.append('non-numeric departure: '+key);continue
        if end_s is not None and not 0<=depart<float(end_s):
            excluded+=1;continue
        route=inline.get('edges','').split() if inline is not None else routes.get(vehicle.get('route'))
        if not route:
            unsupported.append('vehicle:'+key);continue
        resolved+=1;visits=Counter()
        if any(edge not in edges for edge in route):errors.append('unknown route edge: '+key)
        for pair in zip(route,route[1:]):
            if pair not in connections:errors.append('missing edge connection: '+key+' '+str(pair))
            for signal in pairs[pair]:visits[signal]+=1
        for signal,count in visits.items():
            crossed[signal]+=count;seen[signal].add(key)
            if count>1:repeat[signal]+=1
    signals={key:{'planned_crossings':crossed[key],'distinct_vehicles':len(seen[key]),
                  'vehicles_crossing_more_than_once':repeat[key]} for key in all_ids}
    active=[key for key in selected if crossed[key]>0]
    return {'kind':'planned_explicit_routes_not_observed_traffic','vehicle_definitions':len(vehicles),
        'resolved_vehicles':resolved,'excluded_departures_outside_horizon':excluded,'scheduled_departure_end_s':end_s,
        'selected_tls_ids':selected,'selected_signal_count':len(selected),
        'active_selected_tls_ids':active,'active_selected_signal_count':len(active),'signals':signals,
        'unsupported_definitions':unsupported,'errors':sorted(set(errors)),'route_file_sha256':hashes,
        'coverage_complete':not unsupported and not errors and len(active)==len(selected)}


def coverage_warnings(coverage):
    messages=[]
    if coverage['unsupported_definitions']:
        messages.append('Demand coverage is unresolved for some flow/trip/distribution definitions; route them before claiming full signal exposure.')
    if coverage['errors']:messages.append('The route-coverage audit found invalid edges/connections.')
    active=coverage['active_selected_signal_count'];selected=coverage['selected_signal_count']
    if active<selected and not coverage['unsupported_definitions']:
        messages.append(f'Planned demand crosses only {active} of {selected} controlled signal sites. These results do not establish multi-intersection traffic performance.')
    return messages


def from_config(cfg,scenario,selected_ids=None):
    root=Path(cfg['_root']);configuration=root/scenario['sumocfg']
    doc=ET.parse(configuration)
    net=doc.find('./input/net-file')
    option=doc.find('./input/route-files')
    if net is None:raise ValueError('SUMO configuration has no network')
    files=[(configuration.parent/name.strip()).resolve() for name in option.get('value','').split(',') if name.strip()] if option is not None else []
    return route_coverage((configuration.parent/net.get('value')).resolve(),files,selected_ids,end_s=cfg.get('duration_s',3600))

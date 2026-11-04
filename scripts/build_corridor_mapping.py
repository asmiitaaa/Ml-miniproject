"""Create an explicit, reviewable cardinal approach/phase map from source XML."""
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def build(source):
    root=ET.parse(source).getroot()
    lanes={lane.get('id'):lane for edge in root.findall('edge') for lane in edge.findall('lane')}
    signals={}
    for logic in root.findall('tlLogic'):
        key=logic.get('id');links={}
        for link in root.findall('connection'):
            if link.get('tl')==key:
                lane=link.get('from')+'_'+link.get('fromLane')
                links.setdefault(lane,[]).append(int(link.get('linkIndex')))
        slots=[[] for _ in range(4)] # source side N, E, S, W
        for key_lane in sorted(links):
            points=[tuple(map(float,p.split(','))) for p in lanes[key_lane].get('shape').split()]
            x,y=points[-2];endx,endy=points[-1]
            bearing=math.atan2(x-endx,y-endy)
            index=int(math.floor((bearing+math.pi/4)/(math.pi/2)))%4
            slots[index].append(key_lane)
        phases=[]
        for phase in logic.findall('phase'):
            state=phase.get('state');duration=float(phase.get('duration'))
            green=any(c in 'gG' for c in state) and not any(c in 'yY' for c in state)
            if any(c in 'yY' for c in state):duration=max(6.,duration)
            phases.append(dict(state=state,duration=duration,green=green))
        signals[key]={'slots':slots,'lane_links':{k:sorted(set(v)) for k,v in sorted(links.items())},'phases':phases}
    return {'source_network_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        'slot_order':['north','east','south','west'],'yellow_floor_s':6.,'signals':signals}


if __name__=='__main__':
    target=ROOT/'configs/corridor_mapping.json'
    target.write_text(json.dumps(build(ROOT/'network/corridor.net.xml'),indent=2)+'\n',encoding='utf-8')
    print(target)

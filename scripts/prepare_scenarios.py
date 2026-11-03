"""Build reproducible integration fixtures; these are NOT calibrated research demand."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import random
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from traffic_eval.runner import find_sumo


def write_xml(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def make_config(name, network, route_files, end):
    path = ROOT / "scenarios" / f"{name}.sumocfg"
    cfg = ET.Element("configuration")
    inp = ET.SubElement(cfg, "input")
    import os
    ET.SubElement(inp, "net-file", value=os.path.relpath(network, path.parent).replace("\\", "/"))
    ET.SubElement(inp, "route-files", value=",".join(os.path.relpath(x, path.parent).replace("\\", "/") for x in route_files))
    timing = ET.SubElement(cfg, "time")
    ET.SubElement(timing, "begin", value="0")
    ET.SubElement(timing, "end", value=str(end))
    processing = ET.SubElement(cfg, "processing")
    ET.SubElement(processing, "lateral-resolution", value="0.4")
    write_xml(cfg, path)
    return {"sumocfg": str(path.relative_to(ROOT)).replace("\\", "/")}


def build_smoke():
    folder = ROOT / "scenarios" / "smoke"
    folder.mkdir(parents=True, exist_ok=True)
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="J", x="0", y="0", type="traffic_light")
    edges = ET.Element("edges")
    for name, x, y in [("W", -250, 0), ("E", 250, 0), ("N", 0, 250), ("S", 0, -250)]:
        ET.SubElement(nodes, "node", id=name, x=str(x), y=str(y), type="priority")
        ET.SubElement(edges, "edge", {"id":name+"_in", "from":name, "to":"J", "numLanes":"1", "speed":"13.9"})
        ET.SubElement(edges, "edge", {"id":name+"_out", "from":"J", "to":name, "numLanes":"1", "speed":"13.9"})
    write_xml(nodes, folder/"nodes.nod.xml")
    write_xml(edges, folder/"edges.edg.xml")
    binary = Path(find_sumo({"_root":str(ROOT)})).with_name("netconvert.exe" if sys.platform=="win32" else "netconvert")
    network = folder / "junction.net.xml"
    subprocess.run([str(binary), "--node-files", str(folder/"nodes.nod.xml"), "--edge-files", str(folder/"edges.edg.xml"), "--output-file", str(network), "--tls.default-type", "static", "--no-turnarounds", "true"], check=True)
    result = {}
    for scenario, period in [("peak", 2), ("offpeak", 4)]:
        rng = random.Random(42)
        routes = ET.Element("routes")
        # Exactly the requested 50/15/30/5 mix in every block of 20 vehicles.
        types = ["motorcycle"]*10 + ["auto"]*3 + ["car"]*6 + ["bus"]
        rng.shuffle(types)
        for i, depart in enumerate(range(0, 120, period)):
            direction = rng.choices(["WE", "EW", "NS", "SN"], [.4,.4,.1,.1])[0]
            v = ET.SubElement(routes, "vehicle", id=f"{scenario}_{i}", type=types[i%20], depart=str(depart), departLane="best", departSpeed="max")
            ET.SubElement(v, "route", edges=f"{direction[0]}_in {direction[1]}_out")
        route_file = folder/f"{scenario}.rou.xml"
        write_xml(routes, route_file)
        result[f"smoke_{scenario}"] = make_config(f"smoke_{scenario}",network,[ROOT/"demand/vtypes.xml",route_file],180)
    return result, 180


def build_corridor():
    folder = ROOT / "scenarios" / "provisional"
    folder.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.find_spec("sumo")
    if not spec or not spec.submodule_search_locations:
        raise RuntimeError("Install eclipse-sumo to obtain randomTrips.py, or supply calibrated scenarios.")
    random_trips = Path(next(iter(spec.submodule_search_locations))) / "tools" / "randomTrips.py"
    network = ROOT / "network/corridor.net.xml"
    result = {}
    # Chosen only to exercise data collection. No field traffic counts were supplied.
    for scenario, base_period in [("peak",2.0),("offpeak",5.0)]:
        route_files = [ROOT/"demand/vtypes.xml"]
        for i,(kind, vclass, share) in enumerate([("motorcycle","motorcycle",.5),("auto","passenger",.15),("car","passenger",.3),("bus","bus",.05)]):
            route_file = folder/f"{scenario}_{kind}.rou.xml"
            command = [sys.executable,str(random_trips),"-n",str(network),"-r",str(route_file),"-o",str(folder/f"{scenario}_{kind}.trips.xml"),"-b","0","-e","480","-p",str(base_period/share),"--seed",str(42+i),"--prefix",f"{scenario}_{kind}_","--edge-permission",vclass,"--trip-attributes",f'type="{kind}" departLane="best"',"--additional-file",str(ROOT/"demand/vtypes.xml"),"--min-distance","500","--validate"]
            subprocess.run(command,check=True)
            # duarouter embeds referenced vehicle types. Keep the single shared
            # definition file so loading all four route files cannot redefine IDs.
            routed = ET.parse(route_file)
            for definition in routed.getroot().findall("vType"):
                routed.getroot().remove(definition)
            routed.write(route_file, encoding="utf-8", xml_declaration=True)
            route_files.append(route_file)
        result[f"provisional_{scenario}"] = make_config(f"provisional_{scenario}",network,route_files,600)
    return result,600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind",choices=["smoke","corridor"],default="smoke")
    args=parser.parse_args()
    (ROOT/"scenarios").mkdir(exist_ok=True)
    scenarios,duration=build_smoke() if args.kind=="smoke" else build_corridor()
    label="smoke" if args.kind=="smoke" else "provisional"
    cfg={"experiment_label":label+" - integration evidence only, not calibrated research results", "project_root":"..", "duration_s":duration,"step_length_s":1,"lateral_resolution_m":.4,"seeds":[11,12,13,14,15],"scenarios":scenarios,"controllers":{"fixed":{},"actuated":{"min_green_s":10,"max_green_s":60,"max_gap_s":3},"dqn":{"checkpoint":"rl_agent/models/REAL_SUMO_CHECKPOINT_REQUIRED.zip","factory":"agent_adapter:create_policy"}}}
    path=ROOT/"configs"/f"evaluation.{label}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg,indent=2),encoding="utf-8")
    print(f"Prepared {path}. These scenarios are test fixtures, not calibrated research scenarios.")


if __name__=="__main__":main()

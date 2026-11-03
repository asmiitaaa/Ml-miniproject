"""Keep original SUMO collision records and export an inspectable event table."""
import csv
from pathlib import Path
import xml.etree.ElementTree as ET


def collision_action(cfg):
    action = cfg.get("collision_action", "teleport")
    if action not in ("teleport", "warn"):
        raise ValueError("collision_action must be teleport (normal evaluation) or warn (diagnostic only)")
    return action


def collision_args(cfg, run_dir):
    return ["--collision-output",str(Path(run_dir)/"collisions.xml"),"--collision.action",collision_action(cfg)]


def collision_records(path):
    records=[]
    for _,elem in ET.iterparse(path,events=("end",)):
        if elem.tag == "collision":
            records.append(dict(elem.attrib))
        elem.clear()
    return records


def export_collisions(run_dir):
    path=Path(run_dir)
    records=collision_records(path/"collisions.xml")
    preferred=["time","type","lane","pos","collider","victim","colliderType","victimType","colliderSpeed","victimSpeed"]
    fields=preferred+sorted(set().union(*(r.keys() for r in records))-set(preferred))
    with (path/"collisions.csv").open("w",newline="",encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(records)
    return len(records)

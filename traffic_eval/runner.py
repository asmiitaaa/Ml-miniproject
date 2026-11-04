"""Run all controllers on immutable common inputs and collect TraCI telemetry."""
from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

from .metrics import parse_run
from .collisions import collision_action, collision_args, export_collisions
from .demand_coverage import from_config,coverage_warnings


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    cfg["_config_path"] = str(path)
    cfg["_root"] = str((path.parent / cfg.get("project_root", "..")).resolve())
    seeds = cfg.get("seeds", [11,22,33,44,55])
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be nonempty distinct nonnegative integers")
    cfg["seeds"] = seeds
    if float(cfg.get("duration_s", 3600)) <= 0 or float(cfg.get("step_length_s", 1)) <= 0:
        raise ValueError("duration_s and step_length_s must be positive")
    if not cfg.get("scenarios"):
        raise ValueError("At least one scenario is required")
    for name in cfg["scenarios"]:
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("Scenario names may contain only letters, digits, underscores, hyphens")
    return cfg


def resolve(cfg, path):
    return (Path(cfg["_root"]) / path).resolve()


def input_files(cfg, scenario):
    sumocfg = resolve(cfg, scenario["sumocfg"])
    if not sumocfg.is_file():
        raise FileNotFoundError(f"Missing SUMO scenario: {sumocfg}. Obtain Asmita's runnable scenarios.")
    doc = ET.parse(sumocfg)
    files = {sumocfg}
    net = None
    for option in doc.findall("./input/*"):
        if option.tag in {"net-file", "route-files", "additional-files", "weight-files", "state-files"}:
            for name in option.attrib.get("value", "").split(","):
                if not name.strip():
                    continue
                file = (sumocfg.parent/name.strip()).resolve()
                if not file.is_file():
                    raise FileNotFoundError(f"SUMO input missing: {file}")
                files.add(file)
                if option.tag == "net-file": net = file
    files.update(resolve(cfg, name) for name in scenario.get("demand_files", []))
    if net is None:
        raise ValueError(f"Scenario {sumocfg} has no net-file")
    return sumocfg, net, sorted(files)


def fingerprint(cfg, scenario, seed):
    _, _, files = input_files(cfg, scenario)
    hashes = {str(f):sha256(f) for f in files}
    shared = {k:cfg.get(k, d) for k,d in (("duration_s",3600),("step_length_s",1),("lateral_resolution_m",.4))}
    shared.update(seed=seed, files=hashes, sumo_extra_args=cfg.get("sumo_extra_args",[]),collision_action=collision_action(cfg))
    floors=[options.get('yellow_floor_s') for options in cfg.get('controllers',{}).values()]
    if any(value is not None for value in floors):
        if any(value is None for value in floors) or len(set(map(float,floors)))!=1:
            raise ValueError('All compared controllers must use the same yellow floor')
        shared['common_yellow_floor_s']=float(floors[0])
    return hashlib.sha256(json.dumps(shared, sort_keys=True).encode()).hexdigest(), hashes


def prepare_network(source, destination, controller, options):
    tree = ET.parse(source)
    logics = tree.findall("tlLogic")
    if not logics:
        raise ValueError("Network contains no traffic lights; evaluation needs signalized approaches")
    all_ids = {logic.attrib["id"] for logic in logics}
    selected = set(options.get("control_tls_ids",all_ids))
    if not selected or not selected.issubset(all_ids):
        raise ValueError(f"control_tls_ids must select existing traffic lights; unknown IDs: {sorted(selected-all_ids)}")
    for logic in logics:
        controlled = controller == "actuated" and logic.attrib["id"] in selected
        logic.set("type", "actuated" if controlled else "static")
        # The corridor protocol uses this common clearance floor for ALL
        # controllers. Source XML and historical v1/v2 configurations stay intact.
        if "yellow_floor_s" in options:
            floor=float(options["yellow_floor_s"])
            if floor<=0:raise ValueError("yellow_floor_s must be positive")
            for phase in logic.findall("phase"):
                if any(c in phase.attrib["state"] for c in "yY"):
                    phase.set("duration",str(max(floor,float(phase.attrib["duration"]))))
        if controlled:
            for phase in logic.findall("phase"):
                state = phase.attrib["state"]
                if any(c in state for c in "gG") and not any(c in state for c in "yY"):
                    phase.set("minDur", str(options.get("min_green_s", 10)))
                    phase.set("maxDur", str(options.get("max_green_s", 60)))
            ET.SubElement(logic, "param", key="max-gap", value=str(options.get("max_gap_s", 3)))
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def find_sumo(cfg, gui=False):
    requested = cfg.get("sumo_gui_binary" if gui else "sumo_binary")
    name = "sumo-gui" if gui else "sumo"
    if requested:
        location = shutil.which(requested) or str(resolve(cfg,requested))
        if Path(location).is_file(): return location
        raise FileNotFoundError(f"SUMO executable not found: {location}")
    location = shutil.which(name)
    if location: return location
    if os.environ.get("SUMO_HOME"):
        candidate = Path(os.environ["SUMO_HOME"])/"bin"/(name+('.exe' if os.name=='nt' else ''))
        if candidate.is_file(): return str(candidate)
    # Pip eclipse-sumo exposes binaries under site-packages/sumo/bin.
    spec = importlib.util.find_spec("sumo")
    if spec and spec.submodule_search_locations:
        candidate = Path(next(iter(spec.submodule_search_locations)))/"bin"/(name+('.exe' if os.name=='nt' else ''))
        if candidate.is_file(): return str(candidate)
    raise FileNotFoundError("SUMO not installed. Run: python -m pip install eclipse-sumo traci sumolib")


def check_inputs(cfg, controllers):
    collision_action(cfg)
    for scenario in cfg["scenarios"].values(): input_files(cfg,scenario)
    for controller in controllers:
        if controller not in {"fixed", "actuated", "dqn"}: raise ValueError(f"Unsupported controller: {controller}")
        if controller == "dqn":
            options = cfg.get("controllers",{}).get("dqn",{})
            for scenario in cfg["scenarios"].values():
                checkpoint = scenario.get("dqn_checkpoint", options.get("checkpoint"))
                if not checkpoint or not resolve(cfg,checkpoint).is_file():
                    raise FileNotFoundError("A real trained DQN checkpoint is required; configure controllers.dqn.checkpoint or scenarios.<name>.dqn_checkpoint. No substitute policy is used.")
                if "mock" in Path(checkpoint).name.lower() or options.get("mode") == "mock":
                    raise ValueError("Mock-trained checkpoints cannot be used as traffic-controller evaluation evidence")
            if not options.get("factory") or ":" not in options["factory"]:
                raise ValueError("Set controllers.dqn.factory to module:function; see docs/CONTROLLER_CONTRACT.md")
            importlib.import_module(options["factory"].split(":",1)[0])
    find_sumo(cfg)


def _stop_process(process):
    """A failed TraCI handshake must never leave a SUMO server behind."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def _start_sumo(command, run_dir, traci):
    """Own the subprocess explicitly; traci.start can orphan failed servers.

    Five retries allow network loading but avoid the default minute-long retry
    cycle when Windows denies localhost sockets. A fresh, dedicated connection
    also avoids changing TraCI's global connection used by another tool.
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as port_socket:
            port_socket.bind(("127.0.0.1", 0))
            port = port_socket.getsockname()[1]
    except OSError as exc:
        raise RuntimeError("Cannot open a localhost TraCI socket. If Windows reports WinError 10013, run this command in a normal PowerShell terminal or grant this process localhost access.") from exc
    process = None
    try:
        with (run_dir/"sumo-console.log").open("w", encoding="utf-8") as console:
            process = subprocess.Popen(command+["--remote-port",str(port)], stdout=console,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0) if os.name=="nt" else 0)
        connection = traci.connect(port=port, host="127.0.0.1",numRetries=5,proc=process,
                                   waitBetweenRetries=1,label="eval-"+uuid.uuid4().hex)
        return connection, process
    except Exception as exc:
        _stop_process(process)
        raise RuntimeError(f"SUMO/TraCI startup failed: {exc}. Inspect {run_dir/'sumo-console.log'}. If localhost is denied (Windows error 10013), run in a normal PowerShell terminal or grant localhost access.") from exc


def run_one(cfg, scenario_name, controller, seed, run_dir, gui=False):
    import traci
    scenario = cfg["scenarios"][scenario_name]
    sumocfg, net, files = input_files(cfg, scenario)
    demand_hash, hashes = fingerprint(cfg, scenario, seed)
    options = dict(cfg.get("controllers",{}).get(controller,{}))
    if controller == "dqn" and scenario.get("dqn_checkpoint"):
        options["checkpoint"] = scenario["dqn_checkpoint"]
    run_dir.mkdir(parents=True, exist_ok=False)
    network_copy = run_dir/"controlled.net.xml"
    prepare_network(net, network_copy, controller, options)
    step = float(cfg.get("step_length_s",1))
    duration = float(cfg.get("duration_s",3600))
    command = [find_sumo(cfg,gui), "-c", str(sumocfg), "--net-file", str(network_copy),
               "--seed", str(seed), "--random", "false", "--begin", "0", "--end", str(duration), "--step-length", str(step),
               "--lateral-resolution", str(cfg.get("lateral_resolution_m",.4)),
               "--tripinfo-output", str(run_dir/"tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
               "--emission-output", str(run_dir/"emissions.xml"), "--emission-output.step-scaled", "false", "--summary-output", str(run_dir/"summary.xml"),
               "--no-step-log", "true", "--duration-log.disable", "true",
               "--log", str(run_dir/"sumo.log"), "--error-log", str(run_dir/"sumo-errors.log")]
    command += collision_args(cfg,run_dir)
    if gui: command += ["--start", "true", "--quit-on-end", "true"]
    extra = cfg.get("sumo_extra_args", [])
    # Output/input/time/seed changes here would invalidate comparability.
    blocked = ("--random", "--seed", "--net", "-n", "--route", "-r", "--additional", "-a", "--begin", "-b", "--end", "-e", "--step", "--emission", "--tripinfo", "--summary", "--configuration", "-c", "--lateral-resolution", "--collision")
    if any(any(str(arg).startswith(b) for b in blocked) for arg in extra):
        raise ValueError("sumo_extra_args cannot override common inputs, outputs, seed, or timing")
    command += extra
    manifest = {"scenario":scenario_name,"controller":controller,"seed":seed,"demand_hash":demand_hash,
                "input_sha256":hashes,"config_sha256":sha256(cfg["_config_path"]),"command":command,
                "controller_options":options,"duration_s":duration,"step_length_s":step,
                "controller_network_sha256":sha256(network_copy), "python_version":sys.version,
                "experiment_label":cfg.get("experiment_label","project"),"status":"running"}
    manifest.update(collision_action=collision_action(cfg),diagnostic_mode=collision_action(cfg)=="warn")
    coverage=from_config(cfg,scenario,options.get('control_tls_ids'))
    manifest['planned_demand_coverage']=coverage
    if controller == "dqn":
        checkpoint = resolve(cfg, options["checkpoint"])
        manifest["checkpoint_sha256"] = sha256(checkpoint)
    (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    connection = None; policy = None; process = None
    started = time.monotonic(); teleports=0; collisions=0
    try:
        connection, process = _start_sumo(command,run_dir,traci)
        manifest["sumo_version"] = list(connection.getVersion())
        approaches = {}
        for tls in connection.trafficlight.getIDList():
            groups = {}
            for lane in sorted(set(connection.trafficlight.getControlledLanes(tls))):
                edge = connection.lane.getEdgeID(lane)
                if not edge.startswith(":"): groups.setdefault(edge,[]).append(lane)
            approaches[tls] = groups
        if not any(approaches.values()): raise ValueError("No signal-controlled incoming approaches found")
        # Subscriptions return the same last-step values in one SUMO response,
        # avoiding a socket request per lane at every simulation step.
        subscribed=hasattr(connection.lane,'subscribe')
        if subscribed:
            for lanes in approaches.values():
                for group in lanes.values():
                    for lane in group:connection.lane.subscribe(lane,[0x10,0x14,0x7a])
        if controller == "dqn":
            module, function = options["factory"].split(":",1)
            policy = getattr(importlib.import_module(module),function)(connection=connection,checkpoint=str(checkpoint),options={**options,"seed":seed,"_run_dir":str(run_dir),"_step_length_s":step,"_lateral_resolution_m":cfg.get("lateral_resolution_m",.4)})
            manifest["controller_module_sha256"] = sha256(importlib.import_module(module).__file__)
            manifest["controller_metadata"] = getattr(policy,"metadata",{})
            metadata=manifest['controller_metadata']
            metadata['warnings']=list(dict.fromkeys(metadata.get('warnings',[])+coverage_warnings(coverage)))
        with (run_dir/"queues.csv").open("w",newline="",encoding="utf-8") as handle:
            writer=csv.writer(handle); writer.writerow(["time","tls_id","approach_id","queue_length"])
            while connection.simulation.getTime() < duration-1e-8:
                now=connection.simulation.getTime()
                if policy is not None: policy.step(now)
                connection.simulationStep()
                teleports += connection.simulation.getStartingTeleportNumber()
                collisions += connection.simulation.getCollidingVehiclesNumber()
                for tls, groups in approaches.items():
                    for edge, lanes in groups.items():
                        queue = sum(connection.lane.getSubscriptionResults(lane)[0x14] if subscribed
                                    else connection.lane.getLastStepHaltingNumber(lane) for lane in lanes)
                        writer.writerow([connection.simulation.getTime(),tls,edge,queue])
        connection.close(wait=False); connection=None
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _stop_process(process)
        # Ensure common inputs were not changed mid-experiment.
        if fingerprint(cfg,scenario,seed)[0] != demand_hash:
            raise RuntimeError("Common scenario inputs changed during evaluation")
        export_collisions(run_dir)
        result=parse_run(run_dir,step)
        result.update(scenario=scenario_name,controller=controller,seed=seed,demand_hash=demand_hash,
                      run_dir=str(run_dir),experiment_label=manifest["experiment_label"],teleports=teleports,collisions=collisions,
                      wall_time_s=time.monotonic()-started)
        result.update(collision_action=manifest["collision_action"],diagnostic_mode=manifest["diagnostic_mode"])
        result['active_controlled_tls_count']=coverage['active_selected_signal_count'] if not coverage['unsupported_definitions'] else None
        result['controlled_tls_count']=coverage['selected_signal_count']
        result['demand_coverage_kind']=coverage['kind']
        if policy is not None:
            metadata = getattr(policy,"metadata",{})
            result["controller_warnings"] = json.dumps(metadata.get("warnings",[]))
            result["model_num_timesteps"] = metadata.get("model_num_timesteps",0)
            result["controlled_tls_count"] = len(metadata.get("controlled_tls_ids",[]))
            if "timing_profile" in metadata:result["timing_profile"]=metadata["timing_profile"]
            if hasattr(policy,"close"): policy.close()
        manifest.update(status="complete",metrics=result,
                        output_sha256={name:sha256(run_dir/name) for name in ("tripinfo.xml","emissions.xml","queues.csv","summary.xml","dqn_actions.csv","phase_transitions.csv","collisions.xml","collisions.csv") if (run_dir/name).is_file()})
        return result
    except Exception as exc:
        manifest.update(status="failed",error=str(exc))
        raise
    finally:
        try:
            if connection is not None: connection.close(wait=False)
        finally:
            _stop_process(process)
            try:
                if policy is not None and hasattr(policy,"close"): policy.close()
            finally:
                (run_dir/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")


def run_experiment(config, output, controllers, gui=False):
    import pandas as pd
    from .reporting import write_report
    cfg=load_config(config); output=Path(output).resolve()
    check_inputs(cfg,controllers)
    if (output/"runs.csv").exists():
        raise FileExistsError(f"Results already exist at {output}; choose a new --output directory to preserve evidence")
    output.mkdir(parents=True,exist_ok=True)
    results=[]
    for scenario in cfg["scenarios"]:
        for seed in cfg["seeds"]:
            for controller in controllers:
                print(f"Running {scenario}/{controller}/seed={seed}",flush=True)
                result=run_one(cfg,scenario,controller,seed,output/"raw"/scenario/controller/f"seed_{seed}",gui)
                results.append(result)
                pd.DataFrame(results).to_csv(output/"runs.csv",index=False)
    info=write_report(output)
    print(json.dumps(info,indent=2))
    return info


def import_results(raw_dir, output):
    import pandas as pd
    from .reporting import write_report
    records=[]
    for manifest_path in Path(raw_dir).rglob("manifest.json"):
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete": continue
        if not manifest.get("output_sha256"):
            raise ValueError(f"Missing output integrity hashes: {manifest_path}")
        for name, expected in manifest["output_sha256"].items():
            if Path(name).name != name or sha256(manifest_path.parent/name) != expected:
                raise ValueError(f"Raw output integrity check failed: {manifest_path.parent/name}")
        row=parse_run(manifest_path.parent,manifest.get("step_length_s",1))
        for key in ("scenario","controller","seed","demand_hash","experiment_label"):
            row[key]=manifest[key]
        row["run_dir"]=str(manifest_path.parent.resolve())
        row.update({k:manifest.get("metrics",{}).get(k,0) for k in ("teleports","collisions")})
        for key in ("controller_warnings","model_num_timesteps","controlled_tls_count","active_controlled_tls_count","demand_coverage_kind","timing_profile","collision_action","diagnostic_mode"):
            if key in manifest.get("metrics",{}):row[key]=manifest["metrics"][key]
        records.append(row)
    if not records: raise ValueError("No completed run manifests found")
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(records).to_csv(output/"runs.csv",index=False)
    return write_report(output)


def doctor(config=None):
    checks={"python":sys.version,"packages":{name:bool(importlib.util.find_spec(name)) for name in ("numpy","pandas","matplotlib","traci","sumolib","streamlit")}}
    cfg=load_config(config) if config else {"_root":str(Path.cwd())}
    try: checks["sumo_binary"]=find_sumo(cfg)
    except (FileNotFoundError,ModuleNotFoundError) as exc: checks["sumo_error"]=str(exc)
    if config:
        try: check_inputs(cfg,["fixed","actuated","dqn"]); checks["full_evaluation_inputs"]="ready"
        except Exception as exc: checks["full_evaluation_inputs"]=str(exc)
    print(json.dumps(checks,indent=2))
    return checks

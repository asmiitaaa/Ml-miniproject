"""Capture genuine SUMO-GUI frames and compose matched side-by-side videos."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_eval.runner import (check_inputs, find_sumo, fingerprint, input_files,
                                 load_config, prepare_network, resolve, sha256)


def free_port():
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return candidate.getsockname()[1]


def wait_for_file(path, process, timeout=5):
    from PIL import Image
    started = time.monotonic()
    previous = None
    while time.monotonic() - started < timeout:
        if path.is_file() and path.stat().st_size > 0:
            try:
                data = path.read_bytes()
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                digest = hashlib.sha256(data).hexdigest()
                if digest == previous:
                    return digest
                previous = digest
            except (OSError, ValueError, SyntaxError):
                previous = None
        if process.poll() is not None:
            break
        time.sleep(.05)
    raise RuntimeError(f"SUMO-GUI did not save {path.name}. Inspect GUI logs; retry --visible if hidden-window OpenGL capture is unsupported.")


def capture(cfg, scenario_name, controller, seed, output, args, expected_hash):
    import traci
    # A paused/unresponsive GUI must fail with logs instead of hanging forever.
    socket.setdefaulttimeout(45)
    directory = output / controller
    directory.mkdir()
    frames = directory / "frames"
    frames.mkdir()
    scenario = cfg["scenarios"][scenario_name]
    sumocfg, source_network, _ = input_files(cfg, scenario)
    options = dict(cfg.get("controllers", {}).get(controller, {}))
    if controller == "dqn" and scenario.get("dqn_checkpoint"):
        options["checkpoint"] = scenario["dqn_checkpoint"]
    network = directory / "controlled.net.xml"
    prepare_network(source_network, network, controller, options)
    step = float(cfg.get("step_length_s", 1))
    snapshots = directory / "snapshots.view.xml"
    if args.capture_method == "scheduled":
        settings = ET.Element("viewsettings")
        for index in range(int(args.seconds / args.every)):
            target = (index + 1) * args.every - step
            ET.SubElement(settings, "snapshot", file=str(frames / f"frame_{index:05d}.png"), time=str(target))
        ET.ElementTree(settings).write(snapshots, encoding="utf-8", xml_declaration=True)
    port = free_port()
    command = [find_sumo(cfg, gui=True), "-c", str(sumocfg), "--net-file", str(network),
               "--seed", str(seed), "--random", "false", "--begin", "0", "--end", str(args.seconds + step),
               "--step-length", str(step), "--lateral-resolution", str(cfg.get("lateral_resolution_m", .4)),
               "--remote-port", str(port), "--start", "true", "--quit-on-end", "true", "--delay", "0",
               "--breakpoints", str(args.seconds + 2 * step),
               "--window-size", f"{args.width + 40},{args.height + 160}",
               "--tripinfo-output", str(directory / "tripinfo.xml"), "--tripinfo-output.write-unfinished", "true",
               "--emission-output", str(directory / "emissions.xml"), "--emission-output.step-scaled", "false",
               "--summary-output", str(directory / "summary.xml"),
               "--no-step-log", "true", "--duration-log.disable", "true",
               "--log", str(directory / "sumo.log"), "--error-log", str(directory / "sumo-errors.log")]
    if args.capture_method == "scheduled":
        command += ["--gui-settings-file", str(snapshots)]
    extra = cfg.get("sumo_extra_args", [])
    blocked = ("--random", "--seed", "--net", "-n", "--route", "-r", "--additional", "-a", "--begin", "-b", "--end", "-e", "--step", "--emission", "--tripinfo", "--summary", "--configuration", "-c", "--lateral-resolution", "--remote-port", "--start", "--quit", "--window", "--gui", "--delay", "--breakpoints")
    if any(any(str(value).startswith(key) for key in blocked) for value in extra):
        raise ValueError("sumo_extra_args overrides recording inputs or camera/process settings")
    command += extra
    startup = None
    if os.name == "nt" and not args.visible:
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
    manifest = {"scenario": scenario_name, "controller": controller, "seed": seed,
                "demand_hash": expected_hash, "command": command, "status": "capturing",
                "capture_interval_s": args.every, "capture_end_s": args.seconds,
                "capture_method": args.capture_method,
                "controller_options": options, "frames": []}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    connection = policy = process = None
    try:
        with (directory / "process.log").open("w", encoding="utf-8") as process_log:
            process = subprocess.Popen(command, stdout=process_log, stderr=subprocess.STDOUT, startupinfo=startup)
            connection = traci.connect(port=port, host="127.0.0.1", numRetries=30, proc=process)
            manifest["sumo_version"] = list(connection.getVersion())
            views = connection.gui.getIDList()
            if not views:
                raise RuntimeError("SUMO-GUI has no view; recording requires a desktop/OpenGL session")
            view = views[0]
            boundary = args.boundary or [coordinate for corner in connection.simulation.getNetBoundary() for coordinate in corner]
            connection.gui.setBoundary(view, *boundary)
            connection.gui.setAngle(view, 0)
            manifest["camera_boundary"] = boundary
            if controller == "dqn":
                module, factory = options["factory"].split(":", 1)
                checkpoint = resolve(cfg, options["checkpoint"])
                policy = getattr(importlib.import_module(module), factory)(connection=connection, checkpoint=str(checkpoint), options={**options, "seed": seed})
                manifest["checkpoint_sha256"] = sha256(checkpoint)
                manifest["controller_module_sha256"] = sha256(importlib.import_module(module).__file__)
            count = 0
            for tick in range(1, int(round(args.seconds / step)) + 1):
                now = connection.simulation.getTime()
                if args.trace_steps:
                    print(f"TRACE {controller}: before policy at {now:g}s", flush=True)
                if policy is not None:
                    policy.step(now)
                capture_step = tick % int(round(args.every / step)) == 0
                frame = frames / f"frame_{count:05d}.png" if capture_step else None
                if frame is not None and args.capture_method == "traci":
                    # Request before the step: SUMO fulfills the snapshot on GUI render.
                    connection.gui.screenshot(view, str(frame), width=args.width, height=args.height)
                if args.trace_steps:
                    print(f"TRACE {controller}: before simulationStep tick {tick}", flush=True)
                connection.simulationStep()
                if args.trace_steps:
                    print(f"TRACE {controller}: after simulationStep tick {tick}", flush=True)
                if frame is not None:
                    frame_digest = wait_for_file(frame, process)
                    timestamp_key = "capture_requested_time_s" if args.capture_method == "traci" else "scheduled_snapshot_time_s"
                    manifest["frames"].append({"file": str(frame.relative_to(output)),
                                               timestamp_key: now,
                                               "simulation_clock_s": connection.simulation.getTime(),
                                               "sha256": frame_digest})
                    print(f"  {controller}: frame {count + 1}, clock {connection.simulation.getTime():g}s", flush=True)
                    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                    count += 1
            if fingerprint(cfg, scenario, seed)[0] != expected_hash:
                raise RuntimeError("Common inputs changed during recording")
            manifest["status"] = "complete"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), process_running_at_error=process is not None and process.poll() is None)
        raise
    finally:
        if policy is not None and hasattr(policy, "close"):
            try:
                policy.close()
            except Exception as exc:
                manifest["policy_cleanup_error"] = str(exc)
        if connection is not None:
            try:
                # TraCI's default wait=True blocks indefinitely if SUMO-GUI
                # keeps its window open after CMD_CLOSE. We own bounded wait below.
                connection.close(wait=False)
            except Exception as exc:
                manifest["connection_cleanup_error"] = str(exc)
        if process is not None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10)
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def compose(output, records, args, label):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 20)
        small = ImageFont.truetype("arial.ttf", 15)
    except OSError:
        font = small = ImageFont.load_default()
    sequences = [record["frames"] for record in records]
    counts = {len(items) for items in sequences}
    if len(counts) != 1 or not next(iter(counts)):
        raise ValueError("Controllers produced different or empty frame sequences")
    folder = output / "side_by_side_frames"
    folder.mkdir()
    width, height = args.width * len(records), args.height + 110
    paths = []
    names = {"fixed": "Fixed-time", "actuated": "Actuated", "dqn": "DQN"}
    for index in range(next(iter(counts))):
        clocks = {sequence[index]["simulation_clock_s"] for sequence in sequences}
        if len(clocks) != 1:
            raise ValueError("Capture clock mismatch between controllers")
        canvas = Image.new("RGB", (width, height), "#122135")
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 10), label, font=font, fill="white")
        requested = sequences[0][index].get("capture_requested_time_s")
        scheduled = sequences[0][index].get("scheduled_snapshot_time_s")
        timing = f"Step clock: {next(iter(clocks)):g} s"
        if requested is not None:
            timing += f" (snapshot requested at {requested:g} s)"
        elif scheduled is not None:
            timing += f" (snapshot scheduled at {scheduled:g} s)"
        draw.text((16, 40), f"Scenario: {args.scenario}    Seed: {args.seed}    {timing}", font=small, fill="#D0D8E4")
        for panel, (record, sequence) in enumerate(zip(records, sequences)):
            x = panel * args.width
            draw.text((x + 16, 70), names.get(record["controller"], record["controller"]), font=font, fill="white")
            with Image.open(output / sequence[index]["file"]) as screenshot:
                canvas.paste(screenshot.convert("RGB").resize((args.width, args.height)), (x, 110))
        path = folder / f"frame_{index:05d}.png"
        canvas.save(path)
        paths.append(path)
    if args.no_video:
        return None, paths
    destination = output / ("comparison." + args.format)
    if args.format == "gif":
        images = [Image.open(path).convert("RGB") for path in paths]
        try:
            images[0].save(destination, save_all=True, append_images=images[1:], duration=int(1000 / args.fps), loop=0)
        finally:
            for image in images:
                image.close()
    else:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(str(destination), (width, height), fps=args.fps,
                                             codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
                                             output_params=["-movflags", "+faststart"], macro_block_size=2)
        writer.send(None)
        try:
            for path in paths:
                with Image.open(path) as frame:
                    writer.send(frame.convert("RGB").tobytes())
        finally:
            writer.close()
    return destination, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--controllers", nargs="+", default=["fixed", "actuated", "dqn"], choices=["fixed", "actuated", "dqn"])
    parser.add_argument("--output", default="demonstrations/comparison")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--every", type=float, default=5)
    parser.add_argument("--fps", type=float, default=6)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--boundary", type=float, nargs=4, metavar=("XMIN", "YMIN", "XMAX", "YMAX"))
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--no-video", action="store_true", help="Keep genuine individual and composed PNG frames only")
    parser.add_argument("--visible", action="store_true", help="Show SUMO-GUI if hidden-window OpenGL capture is unsupported")
    parser.add_argument("--trace-steps", action="store_true", help="Print each TraCI step for GUI troubleshooting")
    parser.add_argument("--capture-method", choices=["scheduled", "traci"], default="traci", help="Select runtime TraCI screenshots or a native GUI snapshot schedule")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.scenario not in cfg["scenarios"]:
        parser.error("Scenario is not defined in the configuration")
    step = float(cfg.get("step_length_s", 1))
    if args.seed < 0 or min(args.seconds, args.every, args.fps, args.width, args.height) <= 0 or args.every > args.seconds:
        parser.error("Seed must be nonnegative; dimensions, timing and fps must be positive; interval must not exceed duration")
    if len(set(args.controllers)) != len(args.controllers):
        parser.error("Controller names must be unique")
    if any(not math.isclose(value / step, round(value / step), abs_tol=1e-8) for value in [args.seconds, args.every]):
        parser.error("Duration and capture interval must be integer multiples of simulation step_length_s")
    if args.width % 2 or args.height % 2:
        parser.error("Use even frame dimensions for MP4 encoding")
    if args.boundary and not (args.boundary[0] < args.boundary[2] and args.boundary[1] < args.boundary[3]):
        parser.error("Boundary must have xmin<xmax and ymin<ymax")
    # Validate selected scenario before creating anything; never substitute a missing DQN.
    selected_cfg = dict(cfg, scenarios={args.scenario: cfg["scenarios"][args.scenario]})
    check_inputs(selected_cfg, args.controllers)
    find_sumo(cfg, gui=True)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory to preserve previous recordings")
    output.mkdir(parents=True)
    digest, hashes = fingerprint(cfg, cfg["scenarios"][args.scenario], args.seed)
    status = "Three-controller comparison" if set(args.controllers) == {"fixed", "actuated", "dqn"} else "Baseline-only integration demonstration"
    experiment = str(cfg.get("experiment_label", "project"))
    if any(word in experiment.lower() for word in ["mock", "smoke", "test", "provisional", "demo"]):
        status += " | PROVISIONAL / NOT FINAL RESULTS"
    records = []
    for controller in args.controllers:
        print(f"Capturing {controller}: {args.scenario}, seed {args.seed}", flush=True)
        records.append(capture(cfg, args.scenario, controller, args.seed, output, args, digest))
    video, frames = compose(output, records, args, status)
    manifest = {"status": "complete", "label": status, "experiment_label": experiment,
                "scenario": args.scenario, "seed": args.seed, "controllers": args.controllers,
                "demand_hash": digest, "input_sha256": hashes, "config_sha256": sha256(cfg["_config_path"]),
                "capture_interval_s": args.every, "fps": args.fps,
                "playback_simulated_seconds_per_video_second": args.every * args.fps,
                "frames_per_controller": len(frames), "video": video.name if video else None,
                "video_sha256": sha256(video) if video else None,
                "note": "Genuine SUMO-GUI snapshots, not desktop screenshots. Capture-step clocks are identical across panels. Numerical evaluation is separate."}
    (output / "recording_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

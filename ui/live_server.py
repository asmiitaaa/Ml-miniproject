"""
Live corridor viewer.

Runs SUMO headless via TraCI and serves a browser page that draws every
vehicle moving on the real HSR corridor map, with live signal states and
live metrics. Pure standard library + sumolib/traci — no extra installs.

Usage (from the project root):
    python ui/live_server.py                     # peak scenario, fixed-time net
    python ui/live_server.py --scenario offpeak
    python ui/live_server.py --speed 4           # 4x faster playback

Then open  http://localhost:8008  in a browser.
Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sumolib
import traci

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).resolve().parent

VEHICLE_COLORS = {
    "motorcycle": "#eb6834",   # orange — dominant class
    "auto":       "#eda100",   # yellow
    "car":        "#2a78d6",   # blue
    "bus":        "#008300",   # green
    "bicycle":    "#1baf7a",   # aqua
}

state_lock = threading.Lock()
shared = {"time": 0.0, "vehicles": [], "tls": [], "running": 0,
          "waiting_now": 0.0, "arrived": 0, "done": False}


class MapProjector:
    """SUMO network coordinates -> SVG pixel coordinates."""

    def __init__(self, net, width=960):
        xs, ys = [], []
        for e in net.getEdges():
            for x, y in e.getShape():
                xs.append(x); ys.append(y)
        self.minx, self.maxx = min(xs), max(xs)
        self.miny, self.maxy = min(ys), max(ys)
        self.w = width
        self.h = round(width * (self.maxy - self.miny) / (self.maxx - self.minx))

    def px(self, x, y):
        tx = (x - self.minx) / (self.maxx - self.minx) * self.w
        ty = self.h - (y - self.miny) / (self.maxy - self.miny) * self.h
        return round(tx, 1), round(ty, 1)


def build_map_svg(net, proj) -> str:
    """Static road map as SVG (drawn once; vehicles are overlaid client-side)."""
    major, minor = [], []
    for e in net.getEdges():
        pts = " ".join(f"{proj.px(x, y)[0]},{proj.px(x, y)[1]}" for x, y in e.getShape())
        (major if (e.getSpeed() >= 12 or e.getLaneNumber() >= 2) else minor).append(pts)
    parts = [f'<svg id="map" viewBox="0 0 {proj.w} {proj.h}" '
             'style="width:100%;height:auto;display:block">']
    parts.append('<g stroke="#d8d6cf" stroke-width="1" fill="none">')
    parts += [f'<polyline points="{p}"/>' for p in minor]
    parts.append('</g><g stroke="#aaa8a0" stroke-width="2.5" fill="none" stroke-linecap="round">')
    parts += [f'<polyline points="{p}"/>' for p in major]
    parts.append('</g><g id="tls-layer"></g><g id="veh-layer"></g></svg>')
    return "".join(parts)


def simulation_loop(args, net, proj):
    sumo_cmd = ["sumo", "-c", str(ROOT / "scenarios/model_validation" / f"{args.scenario}.sumocfg"),
                "--no-step-log", "true", "--seed", str(args.seed)]
    traci.start(sumo_cmd)
    tls_ids = traci.trafficlight.getIDList()
    tls_px = {}
    for tid in tls_ids:
        lanes = traci.trafficlight.getControlledLanes(tid)
        if not lanes:
            continue
        shape = traci.lane.getShape(lanes[0])
        x, y = shape[-1]
        tls_px[tid] = proj.px(x, y)
    step_pause = 1.0 / max(args.speed, 0.25) / 10.0  # 10 sim-steps/s at speed 1
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            vehicles = []
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                px, py = proj.px(x, y)
                vtype = traci.vehicle.getTypeID(vid).split("@")[0]
                vehicles.append([px, py, VEHICLE_COLORS.get(vtype, "#888")])
            tls = []
            for tid, (px, py) in tls_px.items():
                s = traci.trafficlight.getRedYellowGreenState(tid)
                color = "#22a06b" if "G" in s else ("#eda100" if "y" in s else "#d93025")
                tls.append([px, py, color])
            with state_lock:
                shared["time"] = traci.simulation.getTime()
                shared["vehicles"] = vehicles
                shared["tls"] = tls
                shared["running"] = len(vehicles)
                shared["arrived"] = shared["arrived"] + traci.simulation.getArrivedNumber()
                shared["waiting_now"] = round(sum(
                    traci.lane.getWaitingTime(l) for ls in
                    (traci.trafficlight.getControlledLanes(t) for t in tls_ids)
                    for l in set(ls)), 1)
            time.sleep(step_pause)
    finally:
        with state_lock:
            shared["done"] = True
        traci.close()


class Handler(BaseHTTPRequestHandler):
    map_svg = ""

    def log_message(self, *a):  # quiet
        pass

    def _send(self, body, ctype="text/html; charset=utf-8"):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send((UI_DIR / "live.html").read_text(encoding="utf-8"))
        elif self.path == "/map.svg":
            self._send(self.map_svg, "image/svg+xml")
        elif self.path == "/state":
            with state_lock:
                body = json.dumps(shared)
            self._send(body, "application/json")
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["peak", "offpeak"], default="peak")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--speed", type=float, default=2.0,
                    help="playback speed multiplier (sim seconds per wall second / 10)")
    ap.add_argument("--port", type=int, default=8008)
    args = ap.parse_args()

    net = sumolib.net.readNet(str(ROOT / "network/corridor.net.xml"))
    proj = MapProjector(net)
    Handler.map_svg = build_map_svg(net, proj)

    sim = threading.Thread(target=simulation_loop, args=(args, net, proj), daemon=True)
    sim.start()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Live corridor viewer: http://localhost:{args.port}  "
          f"(scenario={args.scenario}, seed={args.seed})  —  Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

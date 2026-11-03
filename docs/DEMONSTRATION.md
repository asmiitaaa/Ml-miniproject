# Matched SUMO-GUI demonstration recordings

The recorder captures real simulator renderings through TraCI's `gui.screenshot` API, then combines identically timed frames into side-by-side MP4 or GIF. It does not capture your desktop or manufacture vehicle animations. Both controllers consume the same scenario, vehicle demand, random seed, simulation step, lateral resolution and camera boundary. Their trajectories can differ because their signal policies differ.

From the project folder:

```powershell
python scripts/record_demo.py --config configs/evaluation.example.json --scenario peak --seed 11 --seconds 120 --every 5 --output demonstrations/peak_seed 11
```

The default requests all three controllers and fails if Ananyaraj's actual DQN checkpoint or adapter is missing. Once the full experiment is ready, record both peak and off-peak examples with a seed included in the evaluation study. The per-scenario `dqn_checkpoint` override is honored.

For a **baseline-only integration test**, use a runnable test configuration:

```powershell
python scripts/prepare_scenarios.py --kind smoke
python scripts/record_demo.py --config configs/evaluation.smoke.json --scenario smoke_peak --seed 11 --controllers fixed actuated --seconds 120 --every 5 --output demonstrations/baseline_smoke
```

Use the scenario names actually present in the chosen configuration. The preparation script creates `evaluation.smoke.json` and its synthetic `smoke_peak` / `smoke_offpeak` test scenarios. A baseline-only recording is labeled as an integration demonstration and does not stand in for the required three-controller demonstration. Smoke and provisional experiment labels appear on the composed frames.

## Output and playback

- `comparison.mp4` is a side-by-side H.264 video with controller, scenario, seed and capture-step clock labels. Default playback is six captured frames per second; with a five-second capture interval, that is 30 simulated seconds per video second. Change `--fps` or `--every` to adjust playback.
- `<controller>/frames/` retains the original simulator PNGs. Each controller manifest records its command, SUMO version, camera boundary, frame times and hashes; DQN runs also record checkpoint and adapter hashes.
- `side_by_side_frames/` retains the composed PNG frames. `recording_manifest.json` records shared input fingerprints, playback speed and the final video hash.
- `sumo.log`, `sumo-errors.log` and `process.log` retain diagnostic output per controller.

Use a new `--output` path for every recording; existing directories are preserved. `--format gif` uses Pillow and produces a larger but portable animated image. `--no-video` keeps original/composed frames and manifests without requiring FFmpeg. MP4 requires `imageio-ffmpeg` and Pillow. The intended GUI renderer must be installed (`sumo-gui`), not only headless SUMO.

On Windows, the recorder initially requests a hidden SUMO-GUI window to avoid interrupting other work. Some graphics drivers do not render hidden OpenGL windows. If no screenshots are produced, inspect the log and retry with `--visible` and a new output directory. That fallback opens the simulator because the graphics driver requires a drawable window.

The recording command explicitly uses zero GUI delay and replaces any saved simulation breakpoints with one beyond the capture horizon, so previous interactive GUI preferences cannot pause the recording midway.

If rendering stalls, the recorder times out instead of waiting indefinitely and retains its failed-run manifest, completed PNG frames and logs. `--trace-steps` identifies the last successful simulator step. `--capture-method scheduled` preloads native GUI snapshot times as an alternative to runtime TraCI screenshot requests. Neither mode can guarantee compatibility with every local graphics driver. A short preview composed from completed frames must retain a partial-preview label and must not be presented as the full demonstration.

All panels use the complete network boundary by default. A large network may make individual cars hard to see. Supply a single shared `--boundary XMIN YMIN XMAX YMAX` in SUMO network coordinates to focus on the corridor or intersection; the same bounds apply to every controller. Optional `--width 800 --height 600` raises the resolution. Dimensions must be even for the MP4 encoder.

Capture requests are fulfilled on SUMO's GUI render during simulation stepping. The manifest stores both the time when a snapshot was requested and the clock after its simulation step; captions retain both so the render boundary is explicit. Corresponding panels use identical request and step clocks. The video is a qualitative demonstration; scientific waiting time, queue and emission comparisons come from the separately logged evaluation outputs and their full simulation horizons.

Reference: [SUMO GUI state changes and screenshot capture](https://sumo.dlr.de/docs/TraCI/Change_GUI_State.html).

## Completed native comparison

`demonstrations/three_controller_final/comparison.mp4` is the completed three-panel comparison for peak, seed 11. It uses the actual 20k DQN, fixed and actuated control at the same trained junction. Each panel contains native Windows captures of the genuine SUMO-GUI window at TraCI sampling clocks 15, 30, 45, 60, 75, 90 seconds. Six frames play at 1 fps, so the video is a sampled-time demonstration rather than continuous real-time recording. Camera bounds and input fingerprints match across controllers. The GUI time display reflects its latest rendered step; numerical timestamps and metrics are read through TraCI.

The local SUMO OpenGL snapshot path stalls on later requests. Native Windows capture through the Computer Use helper avoids that renderer path. `scripts/preview_gui.py --external-capture` coordinates paused sampling points; the helper saves each real window frame and acknowledges its hash. Normal `Start-Project.ps1 -Mode GUI` needs no capture helper and replays the actual model live. The original SUMO snapshot recorder remains available for other graphics environments.

All 18 source frames and their hashes are retained. The recording illustrates control behavior and does not override collision warnings from the separate full-hour experiment.

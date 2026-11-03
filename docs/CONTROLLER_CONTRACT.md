# DQN evaluation adapter contract

Soham's harness owns the SUMO process, inputs, seed, simulation clock, queue logging and output files. Ananyaraj's trained controller supplies only the signal-control policy through a small factory. This keeps fixed-time, actuated and DQN evaluation on the same traffic demand and simulation horizon.

## Configuration

Configure the `dqn` controller in an evaluation JSON file:

```json
{
  "controllers": {
    "dqn": {
      "factory": "agent.eval_adapter:create_controller",
      "checkpoint": "models/peak_best.zip",
      "decision_interval_s": 5,
      "min_green_s": 10,
      "yellow_s": 3
    }
  }
}
```

The factory is a Python `module:function` import. Its module must be importable from the project folder or installed environment. The checkpoint path is resolved relative to `project_root` in the evaluation configuration. Select the actual best checkpoint for the demand scenario; a path alone is insufficient evidence of a trained DQN. Set `scenarios.peak.dqn_checkpoint` and `scenarios.offpeak.dqn_checkpoint` to override the shared checkpoint for each demand scenario. Do not evaluate the off-peak checkpoint under peak demand by accident.

The factory is invoked once per run:

```python
controller = create_controller(
    connection=existing_traci_connection,
    checkpoint="C:/absolute/path/to/models/peak_best.zip",
    options={"seed": 11, "decision_interval_s": 5, "min_green_s": 10,
             "yellow_s": 3, "factory": "agent.eval_adapter:create_controller",
             "checkpoint": "models/peak_best.zip"},
)
```

It returns an object with:

```python
class EvaluationController:
    def step(self, simulation_time: float) -> None:
        """Observe through the supplied connection and update signal commands."""

    def close(self) -> None:
        """Optional local cleanup; do not close the shared TraCI connection."""
```

The harness calls `step(simulation_time)` **before every simulation step**, beginning at time zero. The adapter implements its checkpoint's decision cadence: five seconds for legacy v1, three seconds for corrected v2. It must preserve the matching yellow/minimum-green semantics. Calls may use a configurable fractional simulation step. Do not count Python function calls as seconds. The concrete profiles and required embedded v2 metadata are documented in `ANANYARAJ_ADAPTER.md` and `CHANGES.md`.

## Required behavior

1. Load a genuine trained DQN checkpoint with the training library, normally `stable_baselines3.DQN.load`. Reject text placeholders, mocks, randomly initialized substitutes and incompatible models. An invalid checkpoint must raise an error and fail the run.
2. Reconstruct the training observation exactly: ordered approaches, feature order, padding, scales, phase encoding and any normalization statistics. Validate the model's observation and action spaces. Preserve normalization at evaluation time without updating its running statistics.
3. Use deterministic prediction, normally `model.predict(observation, deterministic=True)`, to disable epsilon-greedy exploration during evaluation. Apply the checkpoint independently at each signal, as prescribed by the methodology. If intersection dimensions differ, use the same explicitly documented representation as training; never silently reshape observations.
4. Respect the documented binary action: retain the current green phase or advance to the next legal green phase. Enforce the same minimum-green and mandatory yellow intervals as training. Preserve all required all-red/intergreen phases from the signal plan. Never jump straight from one conflicting green to another.
5. Use the supplied connection object for all TraCI calls. Do not start SUMO, switch global connections, restart episodes, change seeds, create or remove vehicles, modify routes or change simulation time. The harness alone advances the clock with `simulationStep`.
6. Leave SUMO emission, tripinfo, summary and queue logging under harness control. Do not overwrite output paths or alter vehicle classes, demand composition, lateral resolution or other simulation physics to benefit one controller.
7. Validate all controlled junction IDs and phase definitions on construction. Raise an informative exception on an unsupported layout, missing model, invalid action or violated invariant; do not silently fall back to fixed-time, random or maximum-pressure control.
8. Store source code, checkpoint provenance, training seed, training scenario, observation specification, normalization files and reward/training curves with the experiment. Manifests record checkpoint and adapter module hashes; any auxiliary model files must also be retained with their hashes in the submitted experiment record.

## Integration and verification

Start by running a short episode on the same intersection used for training. Verify observation shape, legal action decoding, deterministic repeated predictions and signal transition timing. Independently inspect phase traces to ensure every transition includes yellow and every green respects its minimum duration. Then run the corridor with identical demand for all three controllers and compare completion rates, pending departures, collisions and teleports before interpreting delay improvements.

The harness supplies a fixed-time network for DQN control so that the adapter has a consistent initial program. The adapter must take explicit ownership of phase durations and transitions rather than assume a green will remain indefinitely under SUMO's original fixed-time schedule.

Use the harness's diagnostic command before the full run:

```powershell
python -m traffic_eval doctor --config configs/evaluation.example.json
python -m traffic_eval run --config configs/evaluation.example.json --output results/final
```

A full final experiment needs both demand scenarios and five to ten seeds for each controller. A successful diagnostic import proves availability, not model quality or protocol compliance. Missing checkpoints and teammate inputs remain visible blockers; the harness does not create a trained controller on their behalf.

See [SUMO TraCI](https://sumo.dlr.de/docs/TraCI.html) for connection semantics and [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) for checkpoint loading and deterministic prediction.

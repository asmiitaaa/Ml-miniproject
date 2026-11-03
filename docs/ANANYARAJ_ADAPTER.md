# SUMO-trained checkpoint integration

The `traffic_eval.agent_adapter:create_controller` factory supports the SUMO-v1
checkpoint family from Ananyaraj's branch. The source `train.py` sets `mode="sumo"`;
the available 10,000- and 20,000-step snapshots are distinct from the older mock
checkpoints. Snapshot existence and the training source do not establish training
convergence or selection of a best policy. Keep the source commit, checkpoint hash,
training curve and model-selection evidence with the final experiment.

## Feature compatibility

The model expects an unnormalized 13-value float32 vector:

1. Vehicle counts on the four primary lanes, in the original training order.
2. Halting vehicle counts on the same lanes.
3. `lane.getWaitingTime` on the same lanes.
4. The internal green-phase index, 0 or 2. Yellow maps to its preceding green.

Bounds are `[100]*8 + [3600]*4 + [3]`, with every observation clipped to those
bounds. The first four values are total lane vehicle counts: despite the source's
`queue_lengths` variable name they are **not** halting queues. This adapter preserves
the actual implementation. The evaluation's queue metric independently sums all
incoming lanes by approach, rather than reusing the model's primary-lane feature.

The ordered lane IDs are `24480656#3_0`, `794634832#0_0`, `794634831#1_0`, and
`631316440#0_0`. Only junction
`cluster_11484393853_11484393854_11484393855_11484393856_#11more` is controlled.
Other signals retain their fixed plans. Unmapped corridor transfer is rejected;
adding another junction needs an explicitly validated, separate mapping adapter.

## Decision and clearance semantics

Inference is deterministic and occurs every five simulated seconds. The adapter
reproduces the training environment's ordering: the current internal phase is
applied for the next five-second block, then the signal state machine processes a
pending switch request. A request persists until the 10-second minimum green is
satisfied. New requests during yellow are ignored. Yellow excess time is carried
into the internal green timer, exactly as `SignalController.step` does.

The native signal is frozen until the next decision to prevent SUMO from advancing
its phase independently. For the original 39/6/39/6-second program this preserves
the training behaviour, because each native phase originally lasted longer than
the five-second block.

**Clearance discrepancy:** the training state machine requests three-second yellow
but operates in five-second blocks, yielding five seconds of actual yellow. The
network's native yellow is six seconds. `signal_mode="training-replay"` reproduces
the five-second interval and emits a prominent provenance/safety warning. It does
not establish that the corridor clearance design is safe.

`signal_mode="network-clearance"` conservatively honours the longer native yellow
at decision boundaries (six seconds rounds up to ten). This changes the checkpoint's
training dynamics and must be reported as a distinct experiment. Retraining with
consistent clearance is required before final policy claims.

The teammate's training SUMO configuration uses lateral resolution **0.5 m**,
whereas the methodology specifies **0.4 m**. The model-validation template uses
0.5 m for training compatibility. A later methodology-compliant 0.4 m evaluation
must disclose that change, and calibration remains Asmita's deliverable.

## Run and evidence

After placing a sanitized teammate scenario at `scenarios/teammate_model.sumocfg`
and the genuine checkpoint at the configured integration path:

```powershell
python -m traffic_eval run --config configs/evaluation.model-validation.json --output results/model_validation
```

This template evaluates one training-demand scenario with five paired seeds and
all three controllers. It is a model integration test, not the final two-scenario
calibrated corridor experiment. The selected snapshot is the 20,000-step checkpoint;
there is no claim that it is the best converged model.

Each DQN run writes `dqn_actions.csv` with the complete decision observation,
action, applied phase and internal timer. Its manifest records checkpoint hash,
adapter source hash, model timestep count, lane mapping, clearance semantics and
warnings. The parser preserves all-controller fairness and flags the limitations
in `quality_checks.json`. No mock checkpoint or synthetic policy is substituted.

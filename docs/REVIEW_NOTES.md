# Review scope

Soham's branch adds the evaluation harness, metrics, paired statistics, charts, dashboard, inference compatibility adapter, tests, report and run scripts. It does not change the trained model weights, original network/vehicle definitions, or Ananyaraj's canonical agent sources. The branch starts from updated main.

Published result tables under `report/results/` cover 30 one-hour runs: three controllers, two synthetic demand levels, five seeds. Both baseline waiting-time checks pass, and all vehicles completed. Two peak DQN runs (seeds 11 and 15) have collision-related teleports. Those findings remain visible; this is not a safety-approved final policy. Read `AGENT_HANDOFF.md` before merging or presenting performance claims. Calibration, training/test seed provenance, convergence evidence, network clearance and corridor mapping still need team review.

The local raw output directories, virtual environment, logs, demonstration intermediates and archived copies of teammate sources are ignored. Published run CSVs omit machine-specific paths. Original source and checkpoints are already on updated main under `rl_agent/`; the evaluation's selected 20k model copy is retained with its hash to keep the saved experiment reproducible.

## Verify without PowerShell bypass

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest rl_agent/test_environment.py -q
.\.venv\Scripts\python.exe scripts/prepare_model_evaluation.py --check
.\.venv\Scripts\python.exe scripts/quick_test.py
.\.venv\Scripts\python.exe -m streamlit run dashboard.py -- --results report/results
```

The agent's original unit suite uses mock mode. Passing it does not establish that the real DQN is safe. The evaluation's checkpoint loader uses buffered tensor archive streams because direct ZipExtFile loading fails in this Windows environment; this documented compatibility workaround has not been represented as removed.

The frozen source_routes.xml is a verbatim source snapshot, including its original final blank line. Hashed scenario inputs are marked -text to prevent checkout line-ending conversion. PDF/model/media files are binary to prevent corruption on Windows.

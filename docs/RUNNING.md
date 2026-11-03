# Running and testing

The project is C:\Users\soham\Ml-miniproject. The virtual environment and SUMO are installed. Open a normal PowerShell window and run:

```powershell
cd C:\Users\soham\Ml-miniproject
powershell -ExecutionPolicy Bypass -File .\Start-Project.ps1 -Mode QuickTest
```

Success means three controllers execute without exceptions and `QUICK TEST PASSED` is printed. Results go into a new timestamped directory. One seed and a short cutoff intentionally produce quality warnings; inspect complete study results under `results/model_validation`.

For a visual check:

```powershell
powershell -ExecutionPolicy Bypass -File .\Start-Project.ps1 -Mode GUI
```

SUMO-GUI opens the actual road network at the trained intersection. The time display advances, vehicles move, and DQN inference runs every 5simulated seconds. It closes after 300simulated seconds. The terminal prints `GUI REPLAY COMPLETED dqn`. This does not use SUMO's problematic built-in snapshot function. You can select `-Controller fixed` or `-Controller actuated`.

For tables, plots, quality flags and raw records:

```powershell
powershell -ExecutionPolicy Bypass -File .\Start-Project.ps1 -Mode Dashboard
```

Open the localhost URL printed by Streamlit. If the existing server is still running, http://localhost:8501 already serves the complete model validation. Stop a server with Ctrl+C in the terminal that started it. Filters affect summary charts, paired percentages, queues and CSV exports. Keep all five seeds when assessing the full study. Red/amber flags are real findings, not software failures.

For software checks and full reproduction:

```powershell
.\Start-Project.ps1 -Mode Test
.\Start-Project.ps1 -Mode Evaluate
```

Full evaluation runs 30jobs in three separate processes, one simulated hour each, and creates a new output folder. Timing varies by computer. It retains all seeds, including safety failures. Do not overwrite the saved evidence to make results look better.

If Python dependencies are missing, run `Setup-Project.ps1`. If the session blocks PowerShell scripts, the `powershell -ExecutionPolicy Bypass -File ...` form above applies only to that invocation. If SUMO cannot connect to TraCI, check that local localhost communication is permitted for Python and SUMO. No internet connection is needed for an already-installed simulation.

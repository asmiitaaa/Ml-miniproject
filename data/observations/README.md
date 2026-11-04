# Measured traffic inputs — currently absent

`observations.csv` is a header-only collection template. It contains no invented counts, zeros standing for missing measurements, or simulated data presented as field evidence.

Collect observations at the actual mapped sites. Record the road layout, lane counts, permitted turns, speed limits, stop-line positions, fixed signal plan and any construction/closures. The network audit's signal catalogue maps IDs to incoming edges, cardinal slots and road names.

For both peak and offpeak, record classified edge/turning counts in clearly timed intervals and speed or travel-time samples between defined points. Use categories `motorcycle`, `auto`, `car`, `bus`, or `all`. Counts are raw `vehicles`, speeds `m/s` and travel times `s`. A genuine observed zero is valid; a missing measurement stays absent.

Keep separate `calibration` and `validation` observation sessions, preferably on independent representative days. Record `session_id`, ISO date, observation window in seconds from a documented session origin and `source_reference` (dataset URL/table, survey sheet, or video/timestamp/measurement-point reference). Use `source_kind=measured` only for genuine observations. The input checker cannot independently verify the truth of a provenance declaration.

The checklist must cover the multi-intersection network, not only the existing trained junction. Existing source routes cross just one of the six signal sites; other controllers execute with no traffic. Do not scale these routes by an arbitrary factor and call that calibration.

Run:

```powershell
.\.venv\Scripts\python.exe scripts\check_observations.py --input data\observations\observations.csv
```

Exit code 2 is expected for the empty template. Passing input checks means data are ready for review, not that the model has been calibrated.

Once data are available: verify site matching and observation quality; fit demand routes/counts; compare queues, speeds and travel times on held-out observations; document the chosen accuracy criteria and all residuals. Parameters should not be changed merely to make DQN outperform a baseline. Changed network/demand/vehicle physics require new model compatibility checks, retraining where appropriate and fresh seeded controller comparisons.

References: [SUMO demand from observed counts](https://sumo.dlr.de/docs/Demand/Routes_from_Observation_Points.html), [FHWA calibration requires observations](https://ops.fhwa.dot.gov/publications/fhwahop20057/ch1.htm).

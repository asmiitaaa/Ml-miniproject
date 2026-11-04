# Calibration status and next inputs

**Real calibration is not complete. No measured traffic data have been supplied.**

The technical audit and input-preparation tools are implemented. The previous public code/models were verified pushed at f923f6c on soham/evaluation; this audit is a follow-up on the same branch.

## What the audit established

- All 92 scheduled route/vehicle-class combinations have permitted edges, permitted connections and a reachable origin/destination.
- For passenger cars, motorcycles and buses, SUMO netcheck finds one weak component covering 580 of 584 accessible edges (99.32%). Four isolated edges are not used by the supplied routes. Weak connectivity does not prove that every directed origin/destination pair is reachable.
- Peak has 1,000 vehicles and 1,331 planned visits to the trained signal. Offpeak has 500 vehicles and 664 planned visits. Some vehicles revisit that junction, which warrants route-plausibility review.
- Only 1 of the 6 controlled signal sites receives planned traffic. The other five controllers execute without traffic. Earlier delay improvements do not establish multi-intersection traffic performance.
- Nearby signal sites, stop lines, physical lane counts, turn restrictions, speeds and signal coordination remain subject to site verification. The audit does not merge signals or remove roads.

[Audit and route evidence](https://github.com/asmiitaaa/Ml-miniproject/tree/soham/evaluation/report/network_validation)

Historical simulation metrics and model weights are unchanged. The report and dashboard now display the traffic-exposure limitation. New runs record planned exposure and warn when selected signals receive no scheduled routes. Routes departing after a short test's horizon are excluded from its coverage calculation.

## What is still needed

Collect site-matched classified approach/turn counts and speed or travel-time observations for peak and offpeak. Keep separate observation sessions for calibration and validation. Verify the road layout and signal plan for the observation dates. Additional observed vehicle categories and unusual conditions must be retained and reviewed.

The modelled approach catalogue identifies edge IDs, road names and lane IDs for the 14th Main Road / 17th Cross Road / 19th Main Road network. Its lane counts and speeds are model values, not field measurements. Data must exercise the corridor rather than only the existing trained junction.

[Observation template and collection instructions](https://github.com/asmiitaaa/Ml-miniproject/tree/soham/evaluation/data/observations)

The public-source search did not obtain an admissible matched dataset. Historical aggregate HSR 27th Main counts were not substituted for this project's unverified site-specific turning counts. The search record is included with the template.

## Time estimate

The technical audit/setup is available now. Once clean, site-matched measurements exist, allow roughly 2-4 hours for an initial fitting and validation pass; more iterations may be needed. Data collection, physical network corrections and any necessary DQN retraining/fresh controller comparisons are additional. Genuine calibration cannot be completed while observations are missing.

If measurements cannot be obtained, describe the work as a synthetic SUMO experiment with these limitations rather than a calibrated field model.

## Commands

```powershell
Set-Location C:\Users\soham\Ml-miniproject
# Reproduce the technical audit into a fresh output folder
$auditFolder = 'results\network_audit_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
.\.venv\Scripts\python.exe scripts\audit_network_demand.py --config configs\evaluation.corridor.json --output $auditFolder
# Exit code 2 is expected for the empty observation template
.\.venv\Scripts\python.exe scripts\check_observations.py --input data\observations\observations.csv
```

Passing CSV checks means inputs are ready for review, not that a calibrated model exists. [FHWA guidance](https://ops.fhwa.dot.gov/publications/fhwahop20057/ch1.htm) defines calibration against observations; [SUMO](https://sumo.dlr.de/docs/Demand/Routes_from_Observation_Points.html) supports demand generation from measured counts.

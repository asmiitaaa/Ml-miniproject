# Network and demand audit

Network: 6 signal sites. Route/class combinations checked: 92. Illegal combinations: 0.

**Calibration is not completed: no measured data has been provided.**

## Traffic exposure

| Scenario | Vehicles | Active controlled signals | Planned crossings |
| --- | ---: | ---: | ---: |
| peak | 1000 | 1/6 | 1331 |
| offpeak | 500 | 1/6 | 664 |

The other five controllers execute without traffic. Existing delay improvements describe the single active junction, not a corridor traffic-interaction experiment.

Repeated visits to the same junction require route-plausibility review. Nearby signal pairs require review of whether they are distinct stop lines or parts of a coordinated junction; distance alone does not justify merging them.

## Connectivity

### passenger

```text
Total Edges: 584
Largest Component: #1 Edge Count: 580 Coverage: 99.32%
Edges	Incidence
1	4
580	1
```

### motorcycle

```text
Total Edges: 584
Largest Component: #1 Edge Count: 580 Coverage: 99.32%
Edges	Incidence
1	4
580	1
```

### bus

```text
Total Edges: 584
Largest Component: #1 Edge Count: 580 Coverage: 99.32%
Edges	Incidence
1	4
580	1
```

## Remaining evidence

Verify the physical lane layout, permitted turns, speed limits, stop lines and signal programs. Collect classified counts and turning movements for peak/offpeak periods, plus speed/travel-time observations. Hold out independent observation sessions for validation.

Technical references: [SUMO netcheck](https://sumo.dlr.de/docs/Tools/Net.html#netcheckpy), [observed-count demand generation](https://sumo.dlr.de/docs/Demand/Routes_from_Observation_Points.html), [FHWA calibration guidance](https://ops.fhwa.dot.gov/publications/fhwahop20057/ch1.htm).

Existing mean-queue metrics average all 13 network approaches, including quiet sites; they are not an active-junction-only queue mean. Raw metric definitions and values have been preserved.

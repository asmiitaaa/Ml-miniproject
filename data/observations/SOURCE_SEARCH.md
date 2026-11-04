# Public measurement-source review

Reviewed for this project on 4 November 2026. No admissible corridor-matched observation dataset was obtained in this search. This is a limited search result, not a claim that no such data exist.

The supplied OSM/network inputs identify approaches on 14th Main Road, 17th Cross Road and 19th Main Road in the Bengaluru area. The modelled approach catalogue is in `report/network_validation/signal_catalog.csv`; these model values still require site verification.

Public material reviewed included Karnataka urban-transport/municipal/traffic-police searches and the [Bengaluru Comprehensive Mobility Plan](https://staging.opencity.in/dataset/474ad80b-ef63-4d39-add4-7803f247e317/resource/22889607-501d-4dc2-9c09-6988ada67752/download/538f7e60-4546-4573-aafb-0a1da8787293.pdf). The plan contains historical aggregate figures for HSR Layout 27th Main Junction. They were **not adopted**: the measurement sites, dates, observation windows, turning flows and vehicle classifications were not verified as matching this network. Citywide congestion percentages, bus schedules and estimated/simulated GitHub datasets were also not treated as measured turning counts.

Useful next sources are the original project survey, a corridor-specific DULT/Bengaluru Traffic Police survey with site metadata, or locally recorded counts and travel times. A secondary public report may be useful only after verifying its original measurement provenance and site/time compatibility.

Calibration must reproduce observations of the existing system; changing assumed volumes alone does not demonstrate calibration. [FHWA calibration guidance](https://ops.fhwa.dot.gov/publications/fhwahop20057/ch1.htm). SUMO supports demand generation from [edge, turning and origin-destination counts](https://sumo.dlr.de/docs/Demand/Routes_from_Observation_Points.html).

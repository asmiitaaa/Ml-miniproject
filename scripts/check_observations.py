"""Check measured-data readiness; never equates successful CSV parsing with calibration."""
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from traffic_eval.observations import check_observations

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='data/observations/observations.csv');parser.add_argument('--output')
    args=parser.parse_args();report=check_observations(args.input)
    if args.output:Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));sys.exit(0 if report['basic_data_checks_passed'] else 2)

"""Export small portable result tables without rewriting historical v1 outputs."""
import argparse
import json
from pathlib import Path
import shutil


def main(args):
    import pandas as pd
    source=Path(args.source);target=Path(args.target);target.mkdir(parents=True,exist_ok=False)
    for name in ['runs.csv','summary.csv','paired_improvements.csv','baseline_checks.csv']:
        frame=pd.read_csv(source/name)
        frame=frame.drop(columns=['run_dir','wall_time_s'],errors='ignore')
        frame.to_csv(target/name,index=False)
    for name in ['quality_checks.json','phase_audit.json','execution_manifest.json']:
        if (source/name).is_file():shutil.copy2(source/name,target/name)
    if (source/'charts').is_dir():shutil.copytree(source/'charts',target/'charts')
    print(target.resolve())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',required=True);parser.add_argument('--target',required=True)
    main(parser.parse_args())

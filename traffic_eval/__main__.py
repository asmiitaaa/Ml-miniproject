import argparse
import sys

from .runner import doctor, import_results, run_experiment


def main():
    parser=argparse.ArgumentParser(description="Soham's reproducible SUMO controller evaluation")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("doctor",help="Check dependencies and configured teammate deliverables")
    p.add_argument("--config")
    p=sub.add_parser("run",help="Evaluate with common seeded inputs; never substitutes a fake DQN")
    p.add_argument("--config",required=True); p.add_argument("--output",default="results")
    p.add_argument("--controllers",nargs="+",choices=["fixed","actuated","dqn"],default=["fixed","actuated","dqn"])
    p.add_argument("--gui",action="store_true")
    p=sub.add_parser("import",help="Reparse completed raw runs with provenance manifests")
    p.add_argument("--raw",required=True); p.add_argument("--output",required=True)
    p=sub.add_parser("report",help="Regenerate seed summaries, paired comparisons, charts")
    p.add_argument("--input",default="results")
    args=parser.parse_args()
    try:
        if args.command=="doctor": doctor(args.config)
        elif args.command=="run": run_experiment(args.config,args.output,args.controllers,args.gui)
        elif args.command=="import": import_results(args.raw,args.output)
        else:
            from .reporting import write_report
            write_report(args.input)
    except Exception as exc:
        print(f"Evaluation failed: {exc}",file=sys.stderr)
        return 1
    return 0


if __name__=="__main__": sys.exit(main())

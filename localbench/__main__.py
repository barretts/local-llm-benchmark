import argparse
import json
from pathlib import Path
import sys
from .config import load
from .state import State, gpu_lock


def main():
    parser = argparse.ArgumentParser(description="Resumable local coding-agent benchmark")
    parser.add_argument("--config")
    subs = parser.add_subparsers(dest="command", required=True)
    doctor = subs.add_parser("doctor")
    doctor.add_argument("--skip-weight-hashes", action="store_true", help="inventory only; not sufficient to freeze model identity")
    for name in ("init-fixtures", "verify-fixtures", "sweep", "status", "resume", "report", "self-test", "prepare-graders"):
        subs.add_parser(name)
    probe = subs.add_parser("probe")
    probe.add_argument("--engine",required=True)
    probe.add_argument("--model",required=True)
    stop = subs.add_parser("stop")
    stop.add_argument("--after-current", action="store_true", required=True)
    demo = subs.add_parser("agent-demo")
    demo.add_argument("--configuration",required=True)
    endpoint=subs.add_parser('agent-endpoint')
    endpoint.add_argument('--plan',required=True)
    args = parser.parse_args()
    config = load(args.config)
    for name in ("logs", "state", "artifacts"):
        Path(config["paths"][name]).mkdir(parents=True, exist_ok=True)
    state = State(config)
    if args.command == "status":
        # Durable local state only. No engine/model/tool discovery.
        result = state.snapshot()
    elif args.command == "stop":
        state.stop()
        result = {"stop_after_current":True}
    elif args.command == "doctor":
        from .doctor import doctor
        result = doctor(config, not args.skip_weight_hashes)
        state.control("doctor_verified",result["disk_headroom_ok"])
    elif args.command == "self-test":
        import subprocess
        import time
        result_path = Path(config["paths"]["logs"]) / ("harness-tests-"+str(time.time_ns())+".log")
        with result_path.open("wb") as f:
            p = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],stdout=f,stderr=subprocess.STDOUT)
        state.control("harness_verified",p.returncode==0)
        result = {"passed":p.returncode==0,"log":str(result_path)}
        if p.returncode:
            print(json.dumps(result, indent=2))
            return p.returncode
    else:
        # Incrementally implemented commands fail explicitly until integrated.
        from .runner import dispatch
        result = dispatch(config, state, args)
    state.snapshot()
    print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
    state.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(json.dumps({"status":"unavailable","reason":str(error)}),file=sys.stderr)
        raise SystemExit(2)

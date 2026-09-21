from pathlib import Path
import json
import os
import time
from .state import gpu_lock


def dispatch(config, state, args):
    if args.command == "prepare-graders":
        from .grading import prepare_graders
        return prepare_graders(config)
    if args.command == "init-fixtures":
        from .fixtures import init_fixtures
        return init_fixtures(config,state)
    if args.command == "verify-fixtures":
        from .fixtures import verify_fixtures
        with gpu_lock(config):
            return verify_fixtures(config,state)
    if args.command == "report":
        from .report import report
        return report(config,state)
    if args.command in ("probe", "sweep", "resume", "agent-demo", "agent-endpoint"):
        if not state.get_control("doctor_verified") or not state.get_control("harness_verified"):
            raise RuntimeError("doctor/harness gates not verified")
        if not state.get_control("fixtures_verified"):
            raise RuntimeError("fixtures are not verified; no coding qualification permitted")
        with gpu_lock(config):
            if args.command == "agent-endpoint":
                from .serving import run_endpoint
                return run_endpoint(config,state,args.plan)
            if args.command == "resume":
                state.recover()
            from .scheduler import run
            return run(config,state,args)
    raise RuntimeError("unknown operation")

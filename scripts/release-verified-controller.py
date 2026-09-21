"""Release the waiting private-key helper; this file handles no credential."""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localbench.config import load, atomic_json, file_hash
from localbench.state import State


def main():
    config=load();state=State(config)
    try:
        if state.run['id']!='localbench-90f326f86086':raise RuntimeError('original_run_required')
        for gate in ('doctor_verified','fixtures_verified','harness_verified'):
            if state.get_control(gate) is not True:raise RuntimeError('verification_gate_failed:'+gate)
        if state.run['stop_requested']:raise RuntimeError('stop_request_must_be_resolved_before_release')
        state.check_budget(1800)
        sources={p.relative_to(Path(config['paths']['project'])).as_posix():file_hash(p)
            for p in sorted((Path(config['paths']['project'])/'localbench').rglob('*.py'))}
        receipt={'run_id':state.run['id'],'ready':True,'verified_at':time.time(),
            'fixture_version':state.get_control('fixture_version'),
            'fixture_verification_signature':state.get_control('fixture_verification_signature'),
            'controller_sources':sources,'deadline':state.run['deadline']}
        atomic_json(Path(config['paths']['artifacts'])/'credential-controller-ready.json',receipt)
        state.control('verified_controller_release',receipt)
        state.snapshot()
        print('Verified controller released; original run, clock and budget retained.')
    finally:state.close()


if __name__=='__main__':main()

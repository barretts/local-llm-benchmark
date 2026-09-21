"""Apply an authorized execution destination without changing benchmark contracts."""
from pathlib import Path
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localbench.config import load, atomic_json
from localbench.state import State, gpu_lock


def main():
    config=load();state=State(config)
    try:
        root=Path(config['paths']['new_model_root'])
        if root!=Path(r'E:\modelmadness') or root.is_symlink() or root.is_junction():
            raise RuntimeError('authorized ordinary download directory required')
        root.mkdir(exist_ok=True)
        with gpu_lock(config):
            before=state.run
            with state.db:state.db.execute('UPDATE runs SET stop_requested=0 WHERE id=?',(before['id'],))
            if any(state.run[k]!=before[k] for k in ('id','spec_hash','created','started','deadline')):
                raise RuntimeError('run identity or clock changed')
            receipt={'confirmed_at':time.time(),'run_id':before['id'],'base_spec_hash':config['_spec_hash'],
                'execution_hash':config['_execution_hash'],'overrides':config['_execution_overrides'],
                'original_started':before['started'],'original_deadline':before['deadline'],
                'fixture_version':state.get_control('fixture_version'),
                'new_weight_bytes_reserved':state.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0]}
            state.control('execution_override',receipt)
            atomic_json(Path(config['paths']['artifacts'])/'execution-override-receipt.json',receipt)
            state.snapshot();print(json.dumps(receipt,indent=2))
    finally:state.close()


if __name__=='__main__':main()

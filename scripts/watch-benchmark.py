"""Print bounded, nonsecret operational state without touching runtime workloads."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localbench.config import load
from localbench.state import State


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--since-attempt',type=int,default=0)
    args=parser.parse_args();config=load();state=State(config)
    try:
        rows=state.db.execute('SELECT id,status,reason,result,started,finished FROM attempts WHERE id>? ORDER BY id DESC LIMIT 12',(args.since_attempt,)).fetchall()
        recent=[]
        for row in reversed(rows):
            result=json.loads(row['result'] or '{}');metrics=result.get('metrics') or {}
            recent.append({k:row[k] for k in ('id','status','reason','started','finished')} | {
                'kind':result.get('kind'),'configuration_id':result.get('configuration_id'),
                'fixture':result.get('fixture'),'seed':result.get('seed'),'passed':result.get('passed'),
                'first_action_seconds':metrics.get('first_action_seconds'),'decode_tokens_per_second':metrics.get('decode_tokens_per_second')})
        receipt={'run':state.run,'remaining_hours':(state.run['deadline']-time.time())/3600 if state.run['deadline'] else None,
            'jobs':{row['status']:row['n'] for row in state.db.execute('SELECT status,COUNT(*) AS n FROM jobs GROUP BY status')},
            'active_job':state.get_control('active_job'),'controller':None,
            'new_weight_bytes_reserved':state.db.execute('SELECT COALESCE(SUM(bytes),0) FROM weights').fetchone()[0],
            'search_covered':state.get_control('search_covered'),'waiting_for_idle':state.get_control('waiting_for_idle'),
            'recent_attempts':recent}
        row=state.db.execute("SELECT data FROM entities WHERE kind='controller' AND id=?",(state.run['id'],)).fetchone()
        if row:
            controller=json.loads(row['data']);receipt['controller']={k:controller.get(k) for k in ('pid','command','started')}
        print(json.dumps(receipt,indent=2))
    finally:state.close()


if __name__=='__main__':main()

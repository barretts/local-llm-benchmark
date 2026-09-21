"""Acquire exact isolated source and expose only nonsecret review identities."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from localbench.config import load
from localbench.state import State,gpu_lock
from localbench.runtime_build import prepare_pinned_source,detect_windows_toolchain
from localbench.tabby_setup import prepare_tabby_source


def main():
    config=load();state=State(config)
    try:
        with gpu_lock(config):
            state.recover()
            for engine,prepare in (
                ('turboquant-cuda',lambda:prepare_pinned_source(config,state,'turboquant-cuda')),
                ('exllamav3-tabby',lambda:prepare_tabby_source(config,state))):
                prepared=prepare()
                state.control('pending_source_review:'+engine,prepared)
                print(json.dumps({'engine':engine,'bundle':prepared['review_bundle'],'fingerprint':prepared['review_fingerprint']}),flush=True)
            print(json.dumps({'toolchain':detect_windows_toolchain(config)}),flush=True)
    finally:state.close()


if __name__=='__main__':main()

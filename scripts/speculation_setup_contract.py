"""Query-only maintenance ownership and exact reviewed setup contracts."""
from pathlib import Path
import subprocess

from localbench.config import file_hash

RUN=('localbench-90f326f86086',1789753599.011325,1790358399.011325)
IDS=('05c146133e492ac2e0df5f3637ecee19c2b9d797d579431a2321d54c5f91e46f',
     '79de48fe4c1c209671f80683dd4458e2325547181c883d43b4f099dd423c6267')
# Original Windows kernel creation times, verified using query-only handles.
PROCESS_PINS=[{'pid':26500,'created_filetime':134343106498708666},
              {'pid':29624,'created_filetime':134343106497933726},
              {'pid':28852,'created_filetime':134343106870971306}]

def checkpoint_ready(run,pause,active):
    return bool(run and tuple(run[:3])==RUN and run[3]==1 and active==0
        and isinstance(pause,dict) and pause.get('run_id')==RUN[0]
        and pause.get('original_started')==RUN[1] and pause.get('original_deadline')==RUN[2]
        and pause.get('reason')=='user_authorized_speculation_integration'
        and isinstance(pause.get('checkpoints'),dict) and set(pause['checkpoints'])==set(IDS)
        and all(isinstance(x,dict) and x.get('complete') is True for x in pause['checkpoints'].values()))

def original_handles(query_handle,pins,paused,absent):
    if pins!=PROCESS_PINS:
        raise RuntimeError('original_process_creation_pins_required')
    handles=[]
    try:
        for pin in pins:
            try:watch=query_handle(pin['pid'])
            except RuntimeError:
                if not paused() or not absent(pin['pid']):
                    raise RuntimeError('original_process_exit_unverifiable') from None
                continue
            if watch.created!=pin['created_filetime']:
                watch.close()
                if not paused():raise RuntimeError('original_process_creation_changed_before_checkpoint')
                # A different creation time proves the original process exited.
                # Leave the unrelated replacement untouched.
                continue
            handles.append(watch)
        return handles
    except Exception:
        for watch in handles:watch.close()
        raise

def process_absent(pwsh,pid):
    if type(pid) is not int or pid<=0:raise ValueError('invalid_original_pid')
    script="$p=Get-CimInstance Win32_Process -Filter 'ProcessId="+str(pid)+"';if($null-eq$p){'absent'}else{'present'}"
    done=subprocess.run([pwsh,'-NoProfile','-NonInteractive','-Command',script],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=15,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return done.returncode==0 and done.stdout.decode('utf-8-sig').strip()=='absent'

def setup_files(root):
    root=Path(root)
    paths=set((root/'scripts').glob('*speculation*.py'))
    paths.update(root/'scripts'/name for name in ('unattended-supervisor.py',
        'credential-controller.ps1','release-verified-controller.py'))
    return {p.name:file_hash(p) for p in sorted(paths)}

def test_files(root):
    root=Path(root)
    return {p.relative_to(root).as_posix():file_hash(p) for p in sorted((root/'tests').rglob('*.py'))}

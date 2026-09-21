"""Read-only diagnostic for an existing native engine's CUDA DLL loader."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys

binary=Path(sys.argv[1]).resolve()
cuda=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin')
result={'binary':str(binary),'cuda_bin':str(cuda),'attempts':[]}
for use_cuda in (False,True):
    handles=[os.add_dll_directory(str(binary.parent))]
    if use_cuda and cuda.is_dir():handles.append(os.add_dll_directory(str(cuda)))
    try:
        library=ctypes.WinDLL(str(binary.parent/'ggml-cuda.dll'))
        result['attempts'].append({'cuda_search':use_cuda,'loaded':True})
    except OSError as error:
        result['attempts'].append({'cuda_search':use_cuda,'loaded':False,'winerror':error.winerror,'error':str(error)})
    finally:
        for handle in handles:handle.close()
env=os.environ.copy()
if cuda.is_dir():env['PATH']=str(cuda)+os.pathsep+env.get('PATH','')
probe=subprocess.run([str(binary),'--list-devices','--verbose'],env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30)
result['process_local_cuda_search']={'returncode':probe.returncode,'stdout':probe.stdout,'stderr':probe.stderr}
print(json.dumps(result,indent=2))

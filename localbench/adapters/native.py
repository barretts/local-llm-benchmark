from pathlib import Path
import ctypes
import json
import os
import re
import subprocess
import time
from ..config import atomic_json, digest, file_hash
from ..doctor import command
from ..ownership import check_port
from .base import EngineAdapter, LocalHTTP


def creation_time(pid):
    if os.name != "nt":
        return None
    kernel=ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetProcessTimes.argtypes=[ctypes.c_void_p]+[ctypes.c_void_p]*4
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,pid)
    if not handle:
        return None
    try:
        values=[ctypes.c_ulonglong() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*(ctypes.byref(x) for x in values)):
            return None
        return values[0].value
    finally:
        kernel.CloseHandle(handle)


class NativeAdapter(EngineAdapter):
    def __init__(self, config, state, engine, model, binary=None, settings=None):
        self.config,self.state,self.engine,self.model=config,state,engine,model
        self.binary=Path(binary or config["installed_tools"]["bundled_llama_server"])
        self.requested={"context":65536,"slots":1,"gpu_layers":99,"flash_attention":"on","cache_k":"f16","cache_v":"f16","batch":2048,"ubatch":512,"threads":16,"threads_batch":16,"jinja":True,"context_shift":False,"speculation":"off","reasoning":"default"}
        self.requested.update(settings or {})
        portkey={"bundled-llama":"native","upstream-llama-nightly":"native","ik-llama":"ik","turboquant-cuda":"turboquant"}[engine]
        self.port=config["private_ports"][portkey]
        self.http=LocalHTTP(f"http://127.0.0.1:{self.port}",config["limits"]["per_64k_request_timeout_seconds"])
        self.process=None;self.handles=[];self.effective={};self._metadata={}
        self.launch_count=0
        # The installed Unsloth build imports CUDA13 cuBLAS. Reuse existing
        # Ollama libraries only in this owned process; no app files are edited.
        cuda13=Path(self.config["installed_tools"]["ollama"]).parent/"lib"/"ollama"/"cuda_v13"
        self.dependency_directory=cuda13 if engine=="bundled-llama" and (cuda13/"cublas64_13.dll").is_file() else None
        if self.dependency_directory:self.requested["cuda_dependency_directory"]=str(self.dependency_directory)

    def process_environment(self):
        env=os.environ.copy()
        env.pop(self.config["safety"]["credential_env"],None)
        if self.requested.get('speculation','off') != 'off':
            from ..native_speculation import clean_environment
            env=clean_environment(env)
        if self.dependency_directory:
            env["PATH"]=str(self.dependency_directory)+os.pathsep+env.get("PATH","")
        return env

    def discover_capabilities(self):
        if not self.binary.is_file():
            raise RuntimeError("missing_native_binary:"+str(self.binary))
        self.state.start_execution("runtime_probe:"+self.engine)
        if self.state.get_control('runtime_first_probe:'+self.engine) is None:self.state.control('runtime_first_probe:'+self.engine,self.state.clock())
        help_result=command(self.config,self.engine+"-help",[str(self.binary),"--help"],timeout=30,environment=self.process_environment())
        if help_result["exit_code"]!=0:
            raise RuntimeError("native_help_failed:"+help_result["log"])
        self.help=help_result["output"]
        version=command(self.config,self.engine+"-version",[str(self.binary),"--version"],timeout=30,environment=self.process_environment())
        self._metadata={"engine_id":self.engine,"binary":str(self.binary),"binary_sha256":file_hash(self.binary),"adjacent_dlls":{p.name:file_hash(p) for p in sorted(self.binary.parent.glob("*.dll"))},"version_output":version["output"],"help_log":help_result["log"],"version_log":version["log"],"model_path":self.model["path"],"model_sha256":self.model.get("sha256") or file_hash(self.model["path"]),"requested_settings":self.requested}
        self.state.entity("engine",digest(self._metadata),self._metadata)
        self._metadata["runtime_dependencies"]={p.name:file_hash(p) for p in sorted(self.dependency_directory.glob("cublas*64_13.dll"))} if self.dependency_directory else {}
        devices=command(self.config,self.engine+"-devices",[str(self.binary),"--list-devices"],timeout=30,environment=self.process_environment())
        self._metadata["device_probe"]=devices
        if "CUDA0" not in devices["output"]:
            raise RuntimeError("native_cuda_backend_unavailable:"+devices["log"])
        return {"help_log":help_result["log"],"cache_reset":"owned_restart_required","tokenization":"apply-template+tokenize to be verified","supported_flags":sorted(set(re.findall(r"--[a-z0-9-]+",self.help)))}

    def minimum_context_tokens(self):
        # The primary benchmark keeps its 64K gate. Explicitly labelled
        # experiments may override this hook without weakening that gate.
        return 65536

    def launch(self, config=None):
        if not hasattr(self,"help"):
            self.discover_capabilities()
        self.state.check_budget(estimated_seconds=self.config["limits"]["server_start_timeout_seconds"])
        check_port(self.port)
        if os.name=='nt' and ctypes.windll.shell32.IsUserAnAdmin():raise RuntimeError('administrator_runtime_forbidden')
        s=self.requested
        from ..native_speculation import arguments as speculative_arguments
        draft_argv=speculative_arguments(self.config,self.state,self.model,s,self.help)
        if s['speculation']!='off':
            from ..native_speculation import verify_baseline_engine
            verify_baseline_engine(self.config,self.state,s,self._metadata)
        if s["slots"]!=1 or s["context"]<self.minimum_context_tokens() or s["ubatch"]>s["batch"] or s["context_shift"]:
            raise ValueError("invalid primary native configuration")
        flag_values=[("--model",self.model["path"]),("--ctx-size",s["context"]),("--parallel",s["slots"]),("--n-gpu-layers",s["gpu_layers"]),("--flash-attn",s["flash_attention"]),("--cache-type-k",s["cache_k"]),("--cache-type-v",s["cache_v"]),("--batch-size",s["batch"]),("--ubatch-size",s["ubatch"]),("--threads",s["threads"]),("--threads-batch",s["threads_batch"]),("--host","127.0.0.1"),("--port",self.port)]
        argv=[str(self.binary)]
        for flag,value in flag_values:
            if flag not in self.help:
                raise RuntimeError("unsupported_native_flag:"+flag)
            argv.extend([flag,str(value)])
        for flag in ("--jinja","--no-context-shift"):
            if flag not in self.help:
                raise RuntimeError("unsupported_native_flag:"+flag)
            argv.append(flag)
        if "--device" in self.help:argv += ["--device","CUDA0"]
        if "--verbosity" in self.help:argv += ["--verbosity","4"]
        if "--cors-origins" in self.help:argv += ["--cors-origins","http://127.0.0.1:"+str(self.port)]
        if s['speculation']=='off':
            if "--spec-type" in self.help:argv += ["--spec-type","none"]
        else:
            if '--metrics' not in self.help:raise RuntimeError('real_speculative_metrics_unavailable')
            argv += draft_argv
        if "--metrics" in self.help:
            argv.append("--metrics")
        if "--slots" in self.help:
            argv.append("--slots")
        if s["reasoning"]!="default":
            if "--reasoning-budget" not in self.help:
                raise RuntimeError("unsupported_reasoning_control")
            argv += ["--reasoning-budget","0" if s["reasoning"]=="disabled" else "256"]
        self.launch_count+=1
        prefix=Path(self.config["paths"]["logs"])/(self.engine+"-"+self.model["id"]+"-"+str(time.time_ns()))
        stdout=Path(str(prefix)+"-stdout.log");stderr=Path(str(prefix)+"-stderr.log")
        self.handles=[stdout.open("wb"),stderr.open("wb")]
        env=self.process_environment()
        began=time.perf_counter()
        self.process=subprocess.Popen(argv,stdout=self.handles[0],stderr=self.handles[1],stdin=subprocess.DEVNULL,cwd=self.binary.parent,env=env,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        if hasattr(self,"collector"):self.collector.mark_owned_pid(self.process.pid)
        saved={"owner":"localbench","run_id":self.state.run["id"],"pid":self.process.pid,"created":creation_time(self.process.pid),"command_hash":digest(argv),"argv":argv,"endpoint":f"http://127.0.0.1:{self.port}","stdout":str(stdout),"stderr":str(stderr)}
        atomic_json(Path(self.config["paths"]["state"])/("owned-"+str(self.port)+".json"),saved)
        self.saved=saved
        deadline=time.monotonic()+self.config["limits"]["server_start_timeout_seconds"]
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                self.close_logs()
                raw=stderr.read_text(encoding="utf-8",errors="replace")
                code="model_oom" if any(x in raw.lower() for x in ("out of memory","cuda error 2","failed to allocate")) else "native_runtime_crash"
                raise RuntimeError(code+":"+str(stderr))
            if self.health():
                break
            time.sleep(.25)
        else:
            self.unload_owned()
            raise RuntimeError("server_start_timeout:"+str(stderr))
        self.startup_seconds=time.perf_counter()-began
        props=self.http.json("/props")
        atomic_json(str(prefix)+"-props.json",props)
        raw=stderr.read_text(encoding="utf-8",errors="replace")+stdout.read_text(encoding="utf-8",errors="replace")
        ctx=re.findall(r"n_ctx_per_seq\s*=\s*(\d+)",raw)
        slots=re.findall(r"n_seq_max\s*=\s*(\d+)",raw)
        offload=re.findall(r"offloaded (\d+)/(\d+) layers",raw)
        default=props.get("default_generation_settings",{})
        props_context=default.get("n_ctx") or props.get("n_ctx")
        effective_context=int(ctx[-1]) if ctx else props_context
        effective_slots=int(slots[-1]) if slots else props.get("total_slots")
        self.effective={**s,"effective_context":effective_context,"effective_slots":effective_slots,"offload_layers":list(map(int,offload[-1])) if offload else None,"props_path":str(prefix)+"-props.json","startup_seconds":self.startup_seconds,"startup_logs":[str(stdout),str(stderr)],"launch_argv":argv,"context_shift_disabled":"--no-context-shift" in argv,"template_controls":{"jinja":True},"sampler_effective":"unverified until real request counters/config"}
        if effective_context is None or effective_context<self.minimum_context_tokens() or effective_slots!=1:
            self.unload_owned()
            raise RuntimeError("unverified_or_insufficient_per_slot_context")
        if s['speculation']!='off':
            from ..native_speculation import startup_evidence
            self.effective['speculation_startup']=startup_evidence(raw,s)
            self.effective['offload_layers']=list(map(int,offload[0])) if offload else None
            self.effective['draft_offload_layers']=list(map(int,offload[-1])) if len(offload)>1 else None
            self._metadata['draft_model']=dict(s['draft_model'])
            self._metadata['speculation_contract_sha256']=s['speculation_contract_sha256']
        # Stable configuration registration belongs to the scheduler. Startup
        # timestamps and log paths must never consume another configuration.
        return {"endpoint":saved["endpoint"],"handle":saved,"effective":self.effective}

    def health(self):
        try:
            return LocalHTTP(f"http://127.0.0.1:{self.port}",2).json("/health").get("status")=="ok"
        except (OSError,RuntimeError,ValueError):
            return False

    def tokenize(self, messages, tools):
        rendered=self.http.json("/apply-template",{"messages":messages,"tools":tools,"add_generation_prompt":True})
        prompt=rendered.get("prompt")
        if not isinstance(prompt,str) or (tools and not any(t["function"]["name"] in prompt for t in tools)):
            raise RuntimeError("template_endpoint_omits_tools_or_prompt")
        tokenized=self.http.json("/tokenize",{"content":prompt,"add_special":True,"parse_special":True})
        tokens=tokenized.get("tokens")
        if not isinstance(tokens,list) or any(type(t) is not int for t in tokens):
            raise RuntimeError("invalid_exact_tokenization_response")
        return {"count":len(tokens),"provenance":"owned-server /apply-template + /tokenize add_special=true parse_special=true","template_sha256":digest(prompt),"tokenizer_engine_sha256":self._metadata["binary_sha256"]}

    def tokenize_text(self,text):
        return self.http.json("/tokenize",{"content":text,"add_special":False,"parse_special":False})["tokens"]

    def stream_chat(self, request, **kwargs):
        if self.requested.get('speculation','off')!='off':
            from ..native_speculation import measured_stream
            return measured_stream(self,request,kwargs)
        return self.http.measure(request,**kwargs)

    def reset_cache(self):
        return {"status":"owned_server_restart_required","verified":False}

    def fresh_restart(self):
        self.unload_owned()
        return self.launch()

    def close_logs(self):
        for f in self.handles:
            if not f.closed:f.close()
        self.handles=[]

    def unload_owned(self):
        if self.process is not None and self.process.poll() is None:
            if creation_time(self.process.pid)!=self.saved["created"]:
                raise RuntimeError("stale_pid_identity: refusing termination")
            # Popen owns the original Windows process handle; no name/global kill.
            self.process.terminate()
            try:self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=15)
        self.close_logs()
        if self.process is not None and hasattr(self,"collector"):
            self.collector.mark_owned_pid(self.process.pid,owned=False)
        if self.process is not None:
            exit_code=self.process.poll()
            if exit_code is None:raise RuntimeError('original_native_stop_unverified;identity_retained')
            atomic_json(Path(self.config['paths']['state'])/('owned-'+str(self.port)+'.json'),
                {**self.saved,'stopped':True,'stopped_at':time.time(),'exit_code':exit_code,'stopped_proof':'original retained Popen handle exited'})
        self.process=None

    def metadata(self):
        return {**self._metadata,"effective_settings":self.effective}

"""Strict cleanup of saved benchmark orphans; no runtime probes or app kills.

The controller calls these functions explicitly under its sequential lease.
Process cleanup uses one retained kernel handle after creation-time and exact
Windows argv checks. CIM command lines and credentials never enter logs.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from .adapters.base import LocalHTTP
from .config import atomic_json, digest, file_hash


def _record_path(config, path):
    path = Path(path)
    root = Path(config["paths"]["state"]).resolve()
    if path.parent.resolve() != root or not path.resolve().is_relative_to(root):
        raise ValueError("recovery requires a dedicated state handle")
    current = path
    while current != current.parent:
        if current.is_symlink() or (hasattr(current, "is_junction") and current.is_junction()):
            raise ValueError("recovery handle links are forbidden")
        current = current.parent
    return path


def _read_handle(config, path):
    path = _record_path(config, path)
    if not path.exists():
        return path, None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("invalid recovery handle")
    return path, value


def _split_windows_command_line(command_line):
    if os.name != "nt":
        raise RuntimeError("Windows argv parsing unavailable")
    shell = ctypes.WinDLL("shell32", use_last_error=True)
    shell.CommandLineToArgvW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_int)]
    shell.CommandLineToArgvW.restype = ctypes.POINTER(ctypes.c_wchar_p)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    count = ctypes.c_int()
    array = shell.CommandLineToArgvW(command_line, ctypes.byref(count))
    if not array:
        raise RuntimeError("Windows argv parsing failed")
    try:
        return [array[index] for index in range(count.value)]
    finally:
        kernel.LocalFree(array)


def _cim_argv(pid):
    """Fixed read-only query with an integer PID; never save its raw output."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid process identity")
    executable = shutil.which("pwsh") or str(Path(os.environ.get("SystemRoot", "C:/Windows"))/"System32"/"WindowsPowerShell"/"v1.0"/"powershell.exe")
    script = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);$entry=Get-CimInstance Win32_Process -Filter 'ProcessId="+str(pid)+"';if($null -ne $entry){$entry|Select-Object ProcessId,CommandLine|ConvertTo-Json -Compress}"
    environment = {key: value for key, value in os.environ.items()
                   if not any(term in key.lower() for term in ("token", "secret", "password", "api_key", "apikey"))}
    try:
        result = subprocess.run([executable, "-NoProfile", "-NonInteractive", "-Command", script],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=15, env=environment, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("process command identity unavailable") from None
    if result.returncode != 0:
        raise RuntimeError("process command identity unavailable")
    if not result.stdout.strip():
        return None
    value = json.loads(result.stdout.decode("utf-8-sig"))
    if not isinstance(value, dict) or value.get("ProcessId") != pid or not isinstance(value.get("CommandLine"), str) or not value["CommandLine"]:
        raise RuntimeError("process command identity unavailable")
    return _split_windows_command_line(value["CommandLine"])


class WindowsProcess:
    """The original kernel handle stays open across inspection and termination."""
    def __init__(self, pid):
        if os.name != "nt":
            raise RuntimeError("Windows process recovery unavailable")
        from ctypes import wintypes
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE]+[ctypes.c_void_p]*4
        self.kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.pid = pid
        self.handle = self.kernel.OpenProcess(0x1 | 0x1000 | 0x100000, False, pid)
        if not self.handle and ctypes.get_last_error() not in (87, 1168):
            raise RuntimeError("original process handle unavailable")

    def alive(self):
        if not self.handle:
            return False
        from ctypes import wintypes
        code = wintypes.DWORD()
        if not self.kernel.GetExitCodeProcess(self.handle, ctypes.byref(code)):
            raise RuntimeError("original process state unavailable")
        return code.value == 259

    def created(self):
        if not self.handle:
            return None
        values = [ctypes.c_ulonglong() for _ in range(4)]
        if not self.kernel.GetProcessTimes(self.handle, *(ctypes.byref(value) for value in values)):
            raise RuntimeError("original process creation unavailable")
        return values[0].value

    def argv(self):
        return _cim_argv(self.pid)

    def terminate(self, timeout=15):
        if not self.alive():
            return True
        if not self.kernel.TerminateProcess(self.handle, 1):
            if not self.alive():
                return True
            raise RuntimeError("original process termination failed")
        return self.kernel.WaitForSingleObject(self.handle, int(timeout*1000)) == 0

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _validate_process_handle(config, state, path, saved):
    if saved.get("owner") != "localbench" or saved.get("run_id") != state.run["id"] or type(saved.get("pid")) is not int or saved["pid"] <= 0 or type(saved.get("created")) is not int or saved["created"] <= 0:
        raise RuntimeError("unverified_saved_process_identity")
    argv = saved.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(value, str) for value in argv) or saved.get("command_hash") != digest(argv):
        raise RuntimeError("unverified_saved_process_command")
    match = re.fullmatch(r"owned-(\d+)\.json", path.name)
    tokenizer = re.fullmatch(r"owned-tokenizer-(\d+)\.json", path.name)
    port = int(match[1]) if match else None
    if tokenizer:port=int(tokenizer[1])
    if (not tokenizer and port not in config["private_ports"].values()) or (tokenizer and not 1024<=port<=65535) or saved.get("endpoint") != "http://127.0.0.1:"+str(port):
        raise RuntimeError("unverified_saved_process_endpoint")
    binary = Path(argv[0]).resolve()
    permitted = [Path(config["installed_tools"][key]).resolve() for key in
                 ("bundled_llama_server", "ollama") if config["installed_tools"].get(key)]
    runtime = Path(config["paths"]["runtime_root"]).resolve()
    if binary not in permitted and (binary == runtime or not binary.is_relative_to(runtime)):
        raise RuntimeError("unverified_saved_process_binary")
    if tokenizer:
        from .tokenizer_helper import cpu_launch_plan
        if not binary.is_relative_to(runtime) or binary.name.lower()!='llama-server.exe' or argv.count('--model')!=1:
            raise RuntimeError('unverified_tokenizer_process_binary')
        try:
            model=argv[argv.index('--model')+1]
            expected=cpu_launch_plan(binary,model,port,' '.join(argv[1:]))
        except (IndexError,ValueError,RuntimeError):
            raise RuntimeError('unverified_tokenizer_cpu_controls') from None
        if argv!=expected:raise RuntimeError('unverified_tokenizer_cpu_controls')
    if "config_path" in saved:
        configuration = Path(saved["config_path"])
        if not configuration.resolve().is_relative_to(runtime) or configuration.is_symlink() or not configuration.is_file() or file_hash(configuration) != saved.get("config_sha256"):
            raise RuntimeError("unverified_saved_process_configuration")


def recover_process(config, state, handle_path, process_factory=WindowsProcess):
    """Stop one exact original benchmark process; never use a name/tree kill."""
    path, saved = _read_handle(config, handle_path)
    if saved is None or saved.get("stopped") is True:
        return {"status": "no_owned_orphan", "record": str(path)}
    _validate_process_handle(config, state, path, saved)
    if saved.get("children"):
        return {"status": "pending_descendants", "reason": "recorded_children_lack_original_command_fingerprints", "record": str(path)}
    process = process_factory(saved["pid"])
    try:
        if not process.alive():
            status = "already_exited"
        else:
            observed_creation=process.created()
            if observed_creation != saved["created"]:
                # A stable different creation time proves the original PID
                # instance is gone. The current instance is never terminated.
                if type(observed_creation) is not int or observed_creation<=0 or process.created()!=observed_creation:
                    raise RuntimeError("stale_process_creation:refusing_termination")
                receipt={"original_handle":saved,"observed_creation":observed_creation,
                    "proof":"Windows PID reused by a different process instance","retired_at":time.time(),"termination_performed":False}
                evidence=path.with_name('retired-'+path.stem+'-'+str(time.time_ns())+'.json')
                atomic_json(evidence,receipt)
                atomic_json(path,{**saved,'stopped':True,'recovered':False,'retirement_evidence':str(evidence),
                    'stopped_proof':'original instance exited, established by stable different PID creation time'})
                return {'status':'original_process_replaced','pid':saved['pid'],'record':str(path),'evidence':str(evidence),'termination_performed':False}
            argv = process.argv()
            if argv is None or digest(argv) != saved["command_hash"] or argv != saved["argv"]:
                raise RuntimeError("stale_process_command:refusing_termination")
            if process.created() != saved["created"]:
                raise RuntimeError("stale_process_creation:refusing_termination")
            if not process.terminate(timeout=15):
                raise RuntimeError("original_process_stop_timeout;identity_retained")
            status = "recovered"
    finally:
        process.close()
    atomic_json(path, {**saved, "stopped": True, "recovered": True, "recovered_at": time.time()})
    return {"status": status, "pid": saved["pid"], "record": str(path)}


def _lm_instances(inventory):
    if not isinstance(inventory, dict) or not isinstance(inventory.get("models"), list):
        raise RuntimeError("invalid_managed_model_inventory")
    instances = {}
    for model in inventory["models"]:
        if not isinstance(model, dict) or not isinstance(model.get("key"), str):
            raise RuntimeError("invalid_managed_model_inventory")
        for instance in model.get("loaded_instances", []):
            if not isinstance(instance, dict) or not isinstance(instance.get("id"), str) or not isinstance(instance.get("config", {}), dict) or instance["id"] in instances:
                raise RuntimeError("invalid_managed_instance_inventory")
            instances[instance["id"]] = {"model_key": model["key"], "config": instance.get("config", {})}
    return instances


def recover_lm_studio(config, state, http=None):
    """Unload only the unique saved instance using a fresh environment key."""
    path, saved = _read_handle(config, Path(config["paths"]["state"])/"owned-lm-studio-instance.json")
    if saved is None or saved.get("unloaded") is True:
        return {"status": "no_owned_orphan", "record": str(path)}
    identifier = saved.get("instance_id")
    argv = saved.get("argv")
    original_ids = saved.get("original_loaded_instance_ids", [])
    if saved.get("owner") != "localbench" or saved.get("run_id") != state.run["id"] or saved.get("endpoint") != "http://127.0.0.1:1234" or not isinstance(identifier, str) or not re.fullmatch(r"localbench-[a-f0-9]{32}", identifier) or not isinstance(saved.get("model_key"), str) or not isinstance(saved.get("load_config"), dict) or not isinstance(argv, list) or len(argv)<5 or not all(isinstance(item, str) for item in argv) or argv[:2] != [str(config["installed_tools"]["lms"]), "load"] or argv.count("--identifier") != 1 or argv.index("--identifier")+1>=len(argv) or argv[argv.index("--identifier")+1] != identifier or not isinstance(original_ids, list) or not all(isinstance(item, str) for item in original_ids) or identifier in original_ids:
        raise RuntimeError("unverified_managed_instance_handle:refusing_unload")
    if not os.environ.get("LM_BENCH_TOKEN"):
        return {"status": "pending_auth", "reason": "fresh_environment_credential_unavailable", "record": str(path)}
    client = http or LocalHTTP("http://127.0.0.1:1234", 15, credential_env="LM_BENCH_TOKEN")
    expected = {"model_key": saved["model_key"], "config": saved["load_config"]}
    try:
        first = _lm_instances(client.json("/api/v1/models")).get(identifier)
        pending=saved.get('phase')=='pending_instance_config'
        if pending and first is not None:
            if first['model_key']!=saved['model_key'] or _lm_instances(client.json('/api/v1/models')).get(identifier)!=first:
                raise RuntimeError('stale_pending_managed_instance_identity:refusing_unload')
            saved={**saved,'load_config':first['config'],'phase':'observed_owned_instance'}
            atomic_json(path,saved);expected=first
        elif pending and first is None and not saved.get('load_completed') and not saved.get('load_not_started'):
            return {'status':'pending_load','reason':'owned_load_may_still_complete','record':str(path)}
        if first is not None:
            if first != expected or _lm_instances(client.json("/api/v1/models")).get(identifier) != expected:
                raise RuntimeError("stale_managed_instance_identity:refusing_unload")
            response = client.json("/api/v1/models/unload", {"instance_id": identifier})
            if not isinstance(response, dict) or response.get("instance_id") != identifier or identifier in _lm_instances(client.json("/api/v1/models")):
                raise RuntimeError("owned_managed_instance_unload_unverified")
        elif _lm_instances(client.json('/api/v1/models')).get(identifier) is not None:
            raise RuntimeError('owned_managed_instance_absence_unverified')
    except RuntimeError as error:
        if "401" in str(error) or "403" in str(error):
            return {"status": "pending_auth", "reason": "fresh_environment_credential_rejected", "record": str(path)}
        raise
    atomic_json(path, {**saved, "unloaded": True, "recovered": True, "recovered_at": time.time()})
    return {"status": "recovered" if first is not None else "already_unloaded", "instance_id": identifier, "record": str(path)}


def recover_processes(config, state, process_factory=WindowsProcess, include_lm_studio=True):
    """Known local handle paths only; the caller records refusal/pending states."""
    results = []
    root = Path(config["paths"]["state"])
    for port in sorted(set(config["private_ports"].values())):
        path = root/("owned-"+str(port)+".json")
        if path.exists():
            try:
                results.append(recover_process(config, state, path, process_factory))
            except (OSError, ValueError, RuntimeError) as error:
                results.append({"status": "refused", "reason": str(error), "record": str(path)})
    for path in sorted(root.glob('owned-tokenizer-*.json')):
        try:
            results.append(recover_process(config,state,path,process_factory))
        except (OSError,ValueError,RuntimeError) as error:
            results.append({'status':'refused','reason':str(error),'record':str(path)})
    if include_lm_studio:
        try:
            results.append(recover_lm_studio(config, state))
        except (OSError, ValueError, RuntimeError) as error:
            results.append({"status": "refused", "reason": str(error), "record": str(root/"owned-lm-studio-instance.json")})
    return results

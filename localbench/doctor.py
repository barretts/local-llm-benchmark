from pathlib import Path
import csv
import datetime
import io
import json
import re
import shutil
import struct
import subprocess
import time
import urllib.error
import urllib.request
from .config import atomic_json, file_hash


def sanitize(text):
    text = re.sub(r'(?im)(Authorization\s*[:=]\s*)([^\r\n]+)', r'\1[REDACTED]', str(text))
    return re.sub(r'(?i)((?:api[_-]?key|access[_-]?token|password|LM_BENCH_TOKEN)\s*[:=]\s*)([^\s,]+)', r'\1[REDACTED]', text)


def command(config, label, argv, timeout=30, environment=None):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = Path(config["paths"]["logs"]) / ("doctor-" + stamp + "-" + label + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    began = time.time()
    try:
        p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, stdin=subprocess.DEVNULL, env=environment)
        raw = p.stdout
        encoding = "utf-16-le" if raw.count(b"\0") > max(1, len(raw) // 6) else "utf-8"
        output, code = raw.decode(encoding, errors="replace"), p.returncode
    except (OSError, subprocess.TimeoutExpired) as e:
        output, code = str(e), None
    # Persist complete sanitized output before any parsing.
    log.write_text(sanitize(output), encoding="utf-8")
    return {"argv": argv, "exit_code": code, "seconds": time.time() - began, "log": str(log), "output": output}


def powershell(config, label, script, timeout=30):
    return command(config, label, ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script], timeout)


def gguf_metadata(path):
    types = {0:"B",1:"b",2:"H",3:"h",4:"I",5:"i",6:"f",7:"?",10:"Q",11:"q",12:"d"}
    with Path(path).open("rb") as f:
        def read(n):
            b = f.read(n)
            if len(b) != n:
                raise ValueError("incomplete GGUF")
            return b
        def unpack(t):
            return struct.unpack("<" + t, read(struct.calcsize(t)))[0]
        def string():
            length = unpack("Q")
            if length > 32 * 1024 * 1024:
                raise ValueError("invalid GGUF string length")
            return read(length).decode("utf-8", errors="replace")
        def value(t):
            if t in types:
                return unpack(types[t])
            if t == 8:
                return string()
            if t == 9:
                subtype, n = unpack("I"), unpack("Q")
                if n > 2_000_000:
                    raise ValueError("invalid GGUF array")
                return [value(subtype) for _ in range(n)]
            raise ValueError("unsupported GGUF metadata type")
        if read(4) != b"GGUF":
            raise ValueError("not GGUF")
        version, tensors, count = unpack("I"), unpack("Q"), unpack("Q")
        if version not in (2,3) or count > 20000:
            raise ValueError("unsupported GGUF header")
        metadata = {"gguf_version": version, "tensor_count": tensors}
        for _ in range(count):
            key, t = string(), unpack("I")
            v = value(t)
            if key in ("general.architecture", "general.name", "general.file_type", "tokenizer.chat_template", "tokenizer.ggml.model", "tokenizer.ggml.pre") or any(term in key for term in ("context_length", "block_count", "head_count", "key_length", "value_length", "sliding_window")):
                metadata[key] = v
        return metadata


def http_inventory(config, label, url):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log = Path(config["paths"]["logs"]) / ("doctor-" + stamp + "-" + label + ".log")
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            raw = response.read().decode("utf-8")
            code = response.status
    except (urllib.error.URLError, OSError) as e:
        raw, code = str(e), None
    log.write_text(sanitize(raw), encoding="utf-8")
    try:
        data = json.loads(raw)
    except ValueError:
        data = None
    return {"url": url, "http_status": code, "log": str(log), "data": data}


def doctor(config, hash_weights=True):
    tools = config["installed_tools"]
    artifacts = Path(config["paths"]["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    records = {}
    query = "name,uuid,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu,compute_cap"
    records["nvidia"] = command(config, "nvidia", ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"])
    if records["nvidia"]["exit_code"] != 0:
        records["nvidia_fallback"] = command(config, "nvidia-fallback", ["nvidia-smi"])
    records["gpu_processes"] = command(config, "gpu-processes", ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"])
    records["gpu_full"] = command(config, "gpu-full", ["nvidia-smi"])
    records["windows"] = powershell(config, "windows", "$cpu=Get-CimInstance Win32_Processor; $os=Get-CimInstance Win32_OperatingSystem; $disk=Get-CimInstance Win32_LogicalDisk; [pscustomobject]@{cpu=$cpu; os=$os; disks=$disk} | ConvertTo-Json -Depth 4")
    records["listeners"] = powershell(config, "listeners", "Get-NetTCPConnection -State Listen | Select-Object LocalAddress,LocalPort,OwningProcess | ConvertTo-Json")
    records["process_names"] = powershell(config, "process-names", "Get-Process | Select-Object Id,ProcessName,StartTime | ConvertTo-Json", 40)
    records["wsl"] = command(config, "wsl-state", [tools["wsl"], "--list", "--verbose"])
    records["wsl_version"] = command(config, "wsl-version", [tools["wsl"], "--version"])
    records["docker"] = command(config, "docker-info", [tools["docker"], "info", "--format", "{{json .}}"])
    records["containers"] = command(config, "docker-containers", [tools["docker"], "ps", "-a", "--format", '{{json .ID}} {{json .Names}} {{json .Image}} {{json .Status}} {{json .Ports}} {{json .Labels}}'])
    records["docker_images"] = command(config, "docker-images", [tools["docker"], "image", "ls", "--digests", "--no-trunc"])
    version_args = {"python":["--version"], "node":["--version"], "npm":["--version"], "lms":["--version"], "ollama":["--version"], "docker":["--version"]}
    for name, args in version_args.items():
        argv = [tools[name]] + args
        if name == "npm":
            argv = ["cmd.exe", "/d", "/c", tools[name]] + args
        records["version_"+name] = command(config, "version-"+name, argv)
    for name, args in {"git":["--version"],"cmake":["--version"],"nvcc":["--version"],"cl":[]}.items():
        records["version_"+name] = command(config, "version-"+name, [name] + args)
    records["lms_loaded"] = command(config, "lms-loaded", [tools["lms"], "ps", "--json"])
    records["lms_models"] = command(config, "lms-models", [tools["lms"], "ls", "--json"])
    app = http_inventory(config, "lms-api-models", "http://127.0.0.1:1234/v1/models")
    wslconfig = Path.home() / ".wslconfig"
    wsl_text = wslconfig.read_text(encoding="utf-8-sig") if wslconfig.exists() else None
    if wsl_text is not None:
        (Path(config["paths"]["logs"]) / "doctor-wslconfig.log").write_text(wsl_text, encoding="utf-8")
    paths = {name: {"configured": path, "exists": Path(path).exists()} for name,path in tools.items() if isinstance(path,str) and (":\\" in path)}
    disks = {}
    from pathlib import PureWindowsPath
    download_volume=PureWindowsPath(config['paths']['new_model_root']).drive.rstrip(':').upper()
    for letter in dict.fromkeys(['C','F']+([download_volume] if download_volume else [])):
        try:
            use = shutil.disk_usage(letter + ":\\")
            disks[letter] = {"total":use.total,"used":use.used,"free":use.free}
        except OSError as e:
            disks[letter] = {"error":str(e)}
    host = {"observed_at": time.time(), "commands":records,"wslconfig":wsl_text,"disk":disks,"tools":paths,"lms_api":app,"gpu_runtime_access":"not probed until harness verification"}
    atomic_json(artifacts / "host.json", host)
    models = []
    candidates = {str(Path(m["path"])):m for m in config["installed_model_candidates"]}
    root = Path(config["paths"]["installed_model_root"])
    for path in sorted(root.rglob("*.gguf")) if root.exists() else []:
        stat = path.stat()
        item = {"path":str(path),"bytes":stat.st_size,"mtime_ns":stat.st_mtime_ns,"reuse_new_bytes":0,"candidate":candidates.get(str(path)),"is_projector":"mmproj" in path.name.lower()}
        try:
            item["metadata"] = gguf_metadata(path)
            if hash_weights:
                item["sha256"] = file_hash(path)
        except (OSError,ValueError) as e:
            item["error"] = str(e)
        models.append(item)
        atomic_json(artifacts / "installed-weight-ledger.json", models)
    caches = []
    for cache_root in (Path.home()/".cache"/"huggingface"/"hub", root/"huggingface"/"hub", root/"hf-cache"/"hub"):
        if cache_root.exists():
            for snapshot in sorted(cache_root.glob("models--*/snapshots/*")):
                files = [{"path":str(p),"bytes":p.stat().st_size,"resolved":str(p.resolve())} for p in snapshot.rglob("*") if p.is_file()]
                caches.append({"snapshot":str(snapshot),"files":files,"has_config":(snapshot/"config.json").exists(),"has_index":(snapshot/"model.safetensors.index.json").exists()})
    inventory = {"models":models,"hf_snapshots":caches,"commands":records,"lms_api":app,"missing_configured_models":[m for m in config["installed_model_candidates"] if not Path(m["path"]).is_file()]}
    atomic_json(artifacts / "inventory.json", inventory)
    headroom = disks.get("C",{}).get("free",0)>=config["limits"]["minimum_free_c_bytes"] and disks.get("F",{}).get("free",0)>=config["limits"]["minimum_free_f_bytes"]
    if download_volume and download_volume not in ('C','F'):
        headroom=headroom and disks.get(download_volume,{}).get('free',0)>=config['limits'].get('minimum_free_'+download_volume.lower()+'_bytes',config['limits']['minimum_free_f_bytes'])
    return {"host_path":str(artifacts/"host.json"),"inventory_path":str(artifacts/"inventory.json"),"models":len(models),"disk_headroom_ok":headroom,"nvidia":records["nvidia"]["output"],"docker_exit_code":records["docker"]["exit_code"]}

import json
from pathlib import Path
from .config import load

c=load();root=Path(c["paths"]["artifacts"])
h=json.loads((root/"host.json").read_text(encoding="utf-8"));i=json.loads((root/"inventory.json").read_text(encoding="utf-8"))
print(json.dumps({"disks":h["disk"],"wsl":h["commands"]["wsl"]["output"],"docker":{k:json.loads(h["commands"]["docker"]["output"]).get(k) for k in ("DockerRootDir","NCPU","MemTotal","ServerVersion")},"versions":{k:{"exit_code":v["exit_code"],"output":v["output"][:1500]} for k,v in h["commands"].items() if k.startswith("version_") or k in ("lms_loaded",)},"configured_models":[{"id":m["id"],"path":m["path"],"file":next(({"bytes":n["bytes"],"metadata":{k:v for k,v in n.get("metadata",{}).items() if k!="tokenizer.chat_template"}} for n in i["models"] if n["path"]==m["path"]),None)} for m in c["installed_model_candidates"]],"hf_snapshots":[{"snapshot":s["snapshot"],"files":len(s["files"]),"weight_bytes":sum(f["bytes"] for f in s["files"] if f["path"].endswith((".safetensors",".gguf")))} for s in i["hf_snapshots"]]},indent=2))

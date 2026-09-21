from pathlib import Path
import hashlib
import json


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    import uuid
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        f.flush()
        import os
        os.fsync(f.fileno())
    tmp.replace(path)


def load(path=None):
    path = Path(path or Path(__file__).resolve().parents[1] / "benchmark-spec.json")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported benchmark schema")
    for key in ("paths", "limits", "measurement", "grading", "runtime_candidates", "installed_tools", "default_sampling"):
        if key not in config:
            raise ValueError("missing required config: " + key)
    lim, m, g = config["limits"], config["measurement"], config["grading"]
    for key in ("benchmark_elapsed_seconds", "new_model_weight_bytes", "concurrent_gpu_jobs", "maximum_unique_runtime_configurations"):
        if type(lim[key]) is not int or lim[key] <= 0:
            raise ValueError("invalid limit: " + key)
    if lim["concurrent_gpu_jobs"] != 1:
        raise ValueError("this benchmark requires sequential GPU execution")
    if m["full_prompt_tokens"] + m["reserved_output_tokens"] != m["context_window_tokens"]:
        raise ValueError("prompt/output context allocation mismatch")
    if len(g["regular_fixture_ids"]) * len(g["seeds"]) != g["regular_total"] or len(g["long_fixture_ids"]) * len(g["seeds"]) != g["long_total"]:
        raise ValueError("grading denominator mismatch")
    if config["safety"]["bind_host"] != "127.0.0.1":
        raise ValueError("local-only API required")
    ports = list(config["private_ports"].values())
    if len(set(ports)) != len(ports) or any(type(p) is not int or not 1024 <= p <= 65535 or p == 1234 for p in ports):
        raise ValueError("invalid private ports")
    config["_spec_path"] = str(path.resolve())
    config["_spec_hash"] = file_hash(path)
    overrides_path=path.parent/'execution-overrides.json'
    overrides=None
    if overrides_path.exists():
        overrides=json.loads(overrides_path.read_text(encoding='utf-8'))
        if set(overrides)!={'schema_version','user_instruction','paths','limits'} or overrides['schema_version']!=1:
            raise ValueError('unsupported execution override')
        if set(overrides['paths'])!={'new_model_root'} or set(overrides['limits'])!={'minimum_free_e_bytes'}:
            raise ValueError('execution overrides may only redirect new model downloads and destination headroom')
        from pathlib import PureWindowsPath
        target=PureWindowsPath(overrides['paths']['new_model_root'])
        if target!=PureWindowsPath('E:/modelmadness'):
            raise ValueError('download root must match the user-authorized E:/modelmadness')
        floor=overrides['limits']['minimum_free_e_bytes']
        if type(floor) is not int or floor<config['limits']['minimum_free_f_bytes']:
            raise ValueError('download volume headroom may not reduce the existing floor')
        config['paths'].update(overrides['paths']);config['limits'].update(overrides['limits'])
    config['_execution_overrides']=overrides
    config['_execution_hash']=digest({'spec_hash':config['_spec_hash'],'overrides':overrides})
    return config

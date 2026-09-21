"""Pinned Windows Tabby/EXL3 setup, confined to the benchmark runtime root.

No action occurs on import. The controller reads the source review bundle
before install_tabby accepts its fingerprint. Package resolution is frozen to
public wheel URLs, exact versions and SHA256 before installing the local venv.
"""
from pathlib import Path
import json
import re
import tomllib
import urllib.parse

from .acquisition import download, _disk_headroom, _runtime_headroom
from .config import atomic_json, digest, file_hash
from .runtime_build import run_owned

TABBY_COMMIT = "53da7919d4e45c63f4acbcbbc00cbe0f60a1ce65"
EXL3_COMMIT = "0740edc2da569fb99174023c1d2988b1e98cb41e"
WHEEL_SHA = "bd95f6492236e4e060b824c845d0ad800e6573407b2e4b8d72c8399f64bbd59e"
WHEEL_NAME = "exllamav3-1.5.0+cu128.torch2.9.0-cp313-cp313-win_amd64.whl"


def plan_tabby_setup(config):
    discovery = json.loads((Path(config["paths"]["artifacts"]) / "engine-discovery.json").read_text(encoding="utf-8-sig"))
    record = next(item for item in discovery["engines"] if item["id"] == "exllamav3-tabby")
    pin = record["pin"]
    native = next(item for item in record["paths"] if item["kind"] == "native_windows_local_venv")
    if pin.get("tabby_commit") != TABBY_COMMIT or pin.get("exllamav3_commit") != EXL3_COMMIT or native.get("sha256") != WHEEL_SHA or native.get("wheel") != WHEEL_NAME:
        raise ValueError("Tabby/EXL3 discovery pin changed; review before setup")
    if not config["installed_tools"].get("python_observed_version", "").startswith("3.13."):
        raise RuntimeError("tabby_native_requires_existing_python313")
    root = Path(config["paths"]["runtime_root"]) / ("tabby-" + TABBY_COMMIT)
    return {"id": "exllamav3-tabby", "root": str(root), "source": str(root / "source"),
            "venv": str(root / "venv"), "python": str(root / "venv" / "Scripts" / "python.exe"),
            "entrypoint": str(root / "source" / "main.py"), "tabby_commit": TABBY_COMMIT,
            "exllamav3_commit": EXL3_COMMIT, "wheel": {"url": native["url"], "sha256": WHEEL_SHA,
                                                         "bytes": native["bytes"], "filename": WHEEL_NAME},
            "base_python": config["installed_tools"]["python"]}


def _setup(config, state):
    if not state.get_control("harness_verified"):
        raise RuntimeError("Tabby setup requires verified harness")
    state.start_execution("runtime_setup:exllamav3-tabby")
    key = "runtime_setup:exllamav3-tabby@" + TABBY_COMMIT
    setup = state.get_control(key)
    if setup is None:
        candidate = next(item for item in config["runtime_candidates"] if item["id"] == "exllamav3-tabby")
        now = state.clock()
        setup = {"started": now, "deadline": now + candidate["probe_minutes"] * 60, "phase": "source"}
        state.control(key, setup)
    state.check_budget()
    if state.clock() >= setup["deadline"]:
        raise RuntimeError("engine_setup_budget_exhausted")
    return key, setup


def _step(config, state, setup, runner, label, argv, cwd=None):
    remaining = setup["deadline"] - state.clock()
    if remaining <= 0:
        raise RuntimeError("engine_setup_budget_exhausted")
    result = runner(config, state, "tabby-" + label, argv, cwd=cwd, timeout=remaining, setup_deadline=setup["deadline"])
    if result["exit_code"] != 0:
        raise RuntimeError("tabby_setup_failed:" + label + ":" + result.get("log", ""))
    return result


def _review(source):
    files = [p for p in source.rglob("*") if p.is_file() and ".git" not in p.relative_to(source).parts and
             (p.name in ("pyproject.toml", "setup.py", "setup.cfg", "Dockerfile") or p.suffix.lower() in (".py", ".bat", ".sh", ".ps1"))]
    if sum(p.stat().st_size for p in files) > 16 * 1024 * 1024:
        raise RuntimeError("tabby_review_size_limit")
    hashes = {str(p.relative_to(source)).replace("\\", "/"): file_hash(p) for p in sorted(files)}
    if not {"main.py", "pyproject.toml"}.issubset(hashes):
        raise RuntimeError("tabby_source_incomplete")
    return hashes


def prepare_tabby_source(config, state, *, runner=run_owned):
    plan = plan_tabby_setup(config)
    key, setup = _setup(config, state)
    _disk_headroom(config, 4 * 1024 ** 3)
    _runtime_headroom(config, 4 * 1024 ** 3)
    root, source = Path(plan["root"]), Path(plan["source"])
    marker = root / "localbench-owned.json"
    owner = {"owner": "localbench", "run_id": state.run["id"], "tabby_commit": TABBY_COMMIT}
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise RuntimeError("unowned_tabby_runtime_directory")
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) != owner:
        raise RuntimeError("tabby_runtime_ownership_mismatch")
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(marker, owner)
    hooks = root / "disabled-hooks"
    hooks.mkdir(exist_ok=True)
    if any(hooks.iterdir()):
        raise RuntimeError("tabby_hooks_directory_not_empty")
    git = config["installed_tools"].get("git", "git")
    prefix = [git, "-c", "core.hooksPath=" + str(hooks), "-c", "core.fsmonitor=false", "-c", "credential.helper=",
              "-c", "protocol.file.allow=never", "-c", "submodule.recurse=false"]
    commands = []
    repo = "https://github.com/theroyallab/tabbyAPI"
    if not (source / ".git").exists():
        if source.exists() and any(source.iterdir()):
            raise RuntimeError("unowned_tabby_checkout")
        commands.append(_step(config, state, setup, runner, "git-init", prefix + ["init", str(source)]))
        commands.append(_step(config, state, setup, runner, "git-remote", prefix + ["-C", str(source), "remote", "add", "origin", repo]))
    else:
        remote = _step(config, state, setup, runner, "git-remote-check", prefix + ["-C", str(source), "remote", "get-url", "origin"])
        if remote["output"].strip() != repo:
            raise RuntimeError("tabby_source_remote_mismatch")
    objects = _step(config, state, setup, runner, "git-objects", prefix + ["-C", str(source), "rev-parse", "--all"])
    if TABBY_COMMIT not in objects["output"].split():
        commands.append(_step(config, state, setup, runner, "git-fetch", prefix + ["-C", str(source), "fetch", "--depth", "1", "origin", TABBY_COMMIT]))
    commands.append(_step(config, state, setup, runner, "git-checkout", prefix + ["-C", str(source), "checkout", "--detach", TABBY_COMMIT]))
    sha = _step(config, state, setup, runner, "git-sha", prefix + ["-C", str(source), "rev-parse", "HEAD"])
    clean = _step(config, state, setup, runner, "git-clean", prefix + ["-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"])
    if sha["output"].strip() != TABBY_COMMIT or clean["output"].strip():
        raise RuntimeError("tabby_checkout_not_exact_clean_pin")
    hashes = _review(source)
    review = Path(config["paths"]["artifacts"]) / ("source-review-tabby-" + TABBY_COMMIT + ".txt")
    with review.open("w", encoding="utf-8") as output:
        for name, file_sha in hashes.items():
            output.write("\nFILE " + name + " SHA256 " + file_sha + "\n")
            output.write((source / name).read_text(encoding="utf-8", errors="replace") + "\n")
    prepared = {**plan, "script_hashes": hashes, "review_bundle": str(review),
                "review_fingerprint": digest({"commit": TABBY_COMMIT, "scripts": hashes}),
                "source_commands": commands, "setup_key": key, "setup_deadline": setup["deadline"]}
    atomic_json(root / "source-manifest.json", prepared)
    return prepared


PIP_HELPER = '''import json, os, pathlib, sys, urllib.parse
root = pathlib.Path(__file__).resolve().parent
os.environ["PIP_CONFIG_FILE"] = os.devnull
os.environ["PIP_CACHE_DIR"] = str(root / "pip-cache")
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
os.environ["PIP_NO_INPUT"] = "1"
os.environ["TEMP"] = os.environ["TMP"] = str(root / "tmp")
from pip._internal.models.installation_report import InstallationReport
original = InstallationReport.to_dict
def public_report(self):
    raw = original(self)
    records = []
    for item in raw["install"]:
        info = item["download_info"]
        url = info["url"]
        parsed = urllib.parse.urlsplit(url)
        allowed = parsed.scheme == "https" and parsed.hostname in ("files.pythonhosted.org", "pypi.org", "download.pytorch.org", "github.com")
        local = parsed.scheme == "file" and pathlib.Path(urllib.parse.unquote(parsed.path.lstrip("/") if os.name == "nt" else parsed.path)).resolve().is_relative_to(root)
        if not (allowed or local) or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeError("package source must be a public pinned wheel or owned local wheel")
        metadata = item["metadata"]
        records.append({"download_info": info, "metadata": {"name": metadata["name"], "version": metadata["version"]}})
    return {"version": raw["version"], "install": records}
InstallationReport.to_dict = public_report
from pip._internal.cli.main import main
raise SystemExit(main(sys.argv[1:]))
'''


def _freeze_report(output, root, expected_wheel):
    start = output.find('{')
    if start < 0:
        raise RuntimeError("tabby_dependency_report_missing")
    report, _ = json.JSONDecoder().raw_decode(output[start:])
    records, lines = [], []
    for item in report["install"]:
        name, version = item["metadata"]["name"], item["metadata"]["version"]
        url = item["download_info"]["url"]
        hashes = item["download_info"].get("archive_info", {}).get("hashes", {})
        sha = hashes.get("sha256")
        parsed = urllib.parse.urlsplit(url)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not re.fullmatch(r"[A-Za-z0-9_.+!-]+", version) or not re.fullmatch(r"[a-f0-9]{64}", sha or ""):
            raise RuntimeError("tabby_dependency_identity_or_hash_invalid")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise RuntimeError("tabby_dependency_url_must_be_secret_free")
        if parsed.scheme == "file":
            path = Path(urllib.parse.unquote(parsed.path.lstrip("/") if __import__("os").name == "nt" else parsed.path))
            if path.resolve() != expected_wheel.resolve() or sha != WHEEL_SHA:
                raise RuntimeError("tabby_unpinned_local_package")
        elif parsed.scheme != "https" or parsed.hostname not in {"files.pythonhosted.org", "pypi.org", "download.pytorch.org", "github.com"}:
            raise RuntimeError("tabby_dependency_source_not_public")
        records.append({"name": name, "version": version, "url": url, "sha256": sha})
        lines.append(name + " @ " + url + " --hash=sha256:" + sha)
    versions = {item["name"].lower().replace("_", "-"): item["version"] for item in records}
    if not versions.get("exllamav3", "").startswith("1.5.0") or versions.get("torch") != "2.9.0+cu128" or "triton-windows" not in versions:
        raise RuntimeError("tabby_dependency_stack_not_expected_cuda128")
    return records, "\n".join(lines) + "\n"


def install_tabby(config, state, prepared, review_fingerprint, *, runner=run_owned, downloader=download):
    """Resolve/freeze/install wheels in F: after the exact source review."""
    plan = plan_tabby_setup(config)
    root, source = Path(plan["root"]), Path(plan["source"])
    if prepared["root"] != str(root) or prepared["source"] != str(source):
        raise ValueError("Tabby setup paths do not match the owned pin")
    current = digest({"commit": TABBY_COMMIT, "scripts": _review(source)})
    if current != prepared["review_fingerprint"] or current != review_fingerprint:
        raise RuntimeError("tabby_source_review_required_or_changed")
    git = config["installed_tools"].get("git", "git")
    prefix = [git, "-c", "core.fsmonitor=false", "-c", "credential.helper=", "-C", str(source)]
    sha = runner(config, state, "tabby-git-sha", prefix + ["rev-parse", "HEAD"], timeout=30)
    clean = runner(config, state, "tabby-git-clean", prefix + ["status", "--porcelain=v1", "--untracked-files=all"], timeout=30)
    if sha["exit_code"] != 0 or clean["exit_code"] != 0 or sha["output"].strip() != TABBY_COMMIT or clean["output"].strip():
        raise RuntimeError("tabby_checkout_not_exact_clean_pin")
    marker = json.loads((root / "localbench-owned.json").read_text(encoding="utf-8"))
    if marker != {"owner": "localbench", "run_id": state.run["id"], "tabby_commit": TABBY_COMMIT}:
        raise RuntimeError("tabby_runtime_ownership_mismatch")
    manifest_path = root / "engine-manifest.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        same = (previous.get("tabby_commit") == TABBY_COMMIT and previous.get("review_fingerprint") == current
                and Path(plan["python"]).is_file() and Path(previous["dependency_lock"]).is_file()
                and file_hash(previous["dependency_lock"]) == previous["dependency_lock_sha256"]
                and file_hash(previous["requirements_lock"]) == previous["requirements_lock_sha256"])
        if same:
            state.check_budget()
            script = "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
            observed = runner(config, state, "tabby-reuse-versions", [plan["python"], "-c", script], timeout=30)
            if observed["exit_code"] == 0 and json.loads(observed["output"]) == previous["packages"]:
                return previous
    key, setup = _setup(config, state)
    _disk_headroom(config, 12 * 1024 ** 3)
    _runtime_headroom(config, 12 * 1024 ** 3)
    commands = []
    for directory in ("tmp", "pip-cache", "packages", "triton-cache"):
        (root / directory).mkdir(exist_ok=True)
    if not Path(plan["python"]).is_file():
        commands.append(_step(config, state, setup, runner, "venv", [plan["base_python"], "-m", "venv", plan["venv"]]))
    helper = root / "run-pip.py"
    helper.write_text(PIP_HELPER, encoding="utf-8")
    native_wheel = root / "packages" / WHEEL_NAME
    downloader(config, state, plan["wheel"], native_wheel)
    lock_path, requirements = root / "dependency-lock.json", root / "requirements.lock.txt"
    if not lock_path.is_file() or not requirements.is_file():
        project = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
        base = project["project"]["dependencies"]
        cu12 = project["project"]["optional-dependencies"]["cu12"]
        selected = [item for item in base + cu12 if not item.lower().startswith("exllamav3")]
        selected.append(str(native_wheel))
        requested = root / "requirements.requested.txt"
        requested.write_text("\n".join(selected) + "\n", encoding="utf-8")
        report = _step(config, state, setup, runner, "resolve", [plan["python"], str(helper), "install", "--quiet", "--dry-run", "--ignore-installed",
                       "--only-binary=:all:", "--index-url", "https://pypi.org/simple", "--report", "-", "-r", str(requested)])
        records, lock = _freeze_report(report["output"], root, native_wheel)
        atomic_json(lock_path, {"tabby_commit": TABBY_COMMIT, "exllamav3_wheel_sha256": WHEEL_SHA, "packages": records})
        requirements.write_text(lock, encoding="utf-8")
        commands.append(report)
    frozen = json.loads(lock_path.read_text(encoding="utf-8"))
    if frozen.get("tabby_commit") != TABBY_COMMIT or frozen.get("exllamav3_wheel_sha256") != WHEEL_SHA:
        raise RuntimeError("tabby_saved_dependency_lock_pin_mismatch")
    reconstructed = {"install": [{"metadata": {"name": item["name"], "version": item["version"]},
                                "download_info": {"url": item["url"], "archive_info": {"hashes": {"sha256": item["sha256"]}}}}
                               for item in frozen["packages"]]}
    _, expected_lock = _freeze_report(json.dumps(reconstructed), root, native_wheel)
    if requirements.read_text(encoding="utf-8") != expected_lock:
        raise RuntimeError("tabby_requirements_file_differs_from_frozen_lock")
    setup["phase"] = "install"
    state.control(key, setup)
    saved = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else None
    if not saved or saved.get("dependency_lock_sha256") != file_hash(lock_path) or saved.get("review_fingerprint") != current:
        commands.append(_step(config, state, setup, runner, "install", [plan["python"], str(helper), "install", "--only-binary=:all:", "--no-index", "--no-deps", "--require-hashes", "-r", str(requirements)]))
    versions_script = "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:d.version for d in m.distributions()},sort_keys=True))"
    versions = _step(config, state, setup, runner, "versions", [plan["python"], "-c", versions_script])
    packages = json.loads(versions["output"])
    _disk_headroom(config,0)
    _runtime_headroom(config,0)
    normalized = {name.lower().replace("_", "-"): version for name, version in packages.items()}
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if any(normalized.get(item["name"].lower().replace("_", "-")) != item["version"] for item in lock["packages"]):
        raise RuntimeError("tabby_installed_dependency_versions_differ_from_lock")
    manifest = {**plan, "review_fingerprint": current, "review_bundle": prepared["review_bundle"],
                "dependency_lock": str(lock_path), "dependency_lock_sha256": file_hash(lock_path),
                "requirements_lock": str(requirements), "requirements_lock_sha256": file_hash(requirements),
                "packages": packages, "source_script_hashes": prepared["script_hashes"],
                "setup_started": setup["started"], "setup_finished": state.clock(), "setup_deadline": setup["deadline"],
                "commands": commands, "executed_model": False, "manifest_path": str(manifest_path)}
    atomic_json(manifest_path, manifest)
    setup["phase"] = "complete"
    state.control(key, setup)
    state.entity("engine", "exllamav3-tabby", manifest)
    return manifest

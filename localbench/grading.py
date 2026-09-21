from pathlib import Path
import json
import os
import re
import subprocess
import time
import uuid
import urllib.request
from .config import atomic_json, canonical, file_hash
from .doctor import command
from .ownership import validate_container


def prepare_graders(config):
    docker = config["installed_tools"]["docker"]
    artifacts = Path(config["paths"]["artifacts"])
    images = {}
    for name, tag in (("python", "python:3.13.15-slim-bookworm"), ("node", "node:24-bookworm-slim")):
        pull = command(config, "grader-pull-"+name, [docker,"pull",tag], timeout=600)
        if pull["exit_code"] != 0:
            raise RuntimeError("grader_image_acquisition_failed:"+pull["log"])
        inspect = command(config, "grader-inspect-"+name,[docker,"image","inspect",tag],timeout=30)
        info = json.loads(inspect["output"])[0]
        pins = info["RepoDigests"]
        if not pins:
            raise RuntimeError("grader_image_missing_digest")
        images[name] = {"tag":tag,"digest":pins[0],"image_id":info["Id"],"bytes":info["Size"],"pull_log":pull["log"]}
    with urllib.request.urlopen("https://registry.npmjs.org/typescript/latest",timeout=30) as response:
        registry = json.loads(response.read())
    atomic_json(artifacts / "typescript-registry.json",registry)
    version = registry["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+",version):
        raise RuntimeError("unexpected_typescript_version")
    build = Path(config["paths"]["project"])/".grader-build"
    build.mkdir(exist_ok=True)
    # Package metadata is pinned and npm creates a dependency lock in the image.
    (build/"Dockerfile").write_text(f'FROM {images["node"]["digest"]}\nRUN mkdir -p /opt/localbench-ts && cd /opt/localbench-ts && npm init -y && npm install --save-exact --ignore-scripts typescript@{version} && ln -s /opt/localbench-ts/node_modules/.bin/tsc /usr/local/bin/tsc\n',encoding="utf-8")
    label = "localbench-ts:" + version
    result = command(config,"grader-build-typescript",[docker,"build","--label","localbench.owner=localbench","-t",label,str(build)],timeout=600)
    if result["exit_code"] != 0:
        raise RuntimeError("typescript_grader_build_failed:"+result["log"])
    info=json.loads(command(config,"grader-inspect-typescript",[docker,"image","inspect",label])["output"])[0]
    images["typescript"]={"tag":label,"image_id":info["Id"],"bytes":info["Size"],"base_digest":images["node"]["digest"],"typescript_version":version,"typescript_integrity":registry["dist"]["integrity"],"registry_revision":registry.get("gitHead"),"dockerfile_sha256":file_hash(build/"Dockerfile"),"build_log":result["log"]}
    lock_log=command(config,"grader-ts-lock",[docker,"run","--rm","--network","none","--read-only","--user","65534:65534","--cap-drop","ALL","--security-opt","no-new-privileges",info["Id"],"cat","/opt/localbench-ts/package-lock.json"])
    if lock_log["exit_code"]==0:
        atomic_json(artifacts/"typescript-package-lock.json",json.loads(lock_log["output"]))
    atomic_json(artifacts/"grader-images.json",images)
    return images


class DockerGrader:
    def __init__(self, config, run_id):
        self.config, self.run_id = config, run_id
        self.docker = config["installed_tools"]["docker"]
        pins = Path(config["paths"]["artifacts"])/"grader-images.json"
        if not pins.exists():
            raise RuntimeError("pinned_grader_images_not_prepared")
        self.images = json.loads(pins.read_text(encoding="utf-8"))

    def grade(self, workspace, fixture, phase="public", log_id=None):
        if phase not in ("public","hidden"):
            raise ValueError("invalid test phase")
        workspace = Path(workspace).resolve()
        owned_root=Path(self.config["paths"]["agent_workspaces"]).resolve()
        if not workspace.is_relative_to(owned_root):
            raise ValueError("only owned fixture workspaces may be mounted")
        test_dir=Path(fixture["grader_path"])/phase
        if not test_dir.is_dir():
            raise RuntimeError("missing_controller_tests")
        driver=Path(self.config["paths"]["project"])/"scripts"
        job_id = log_id or uuid.uuid4().hex
        name="localbench-grade-"+uuid.uuid4().hex[:16]
        argv=[self.docker,"run","--name",name,"--label","localbench.owner=localbench","--label","localbench.run="+self.run_id,"--label","localbench.job="+job_id,"--network","none","--read-only","--user","65534:65534","--cap-drop","ALL","--security-opt","no-new-privileges","--cpus","2","--memory","2g","--pids-limit","128","--tmpfs","/tmp:rw,noexec,nosuid,size=128m","--mount",f"type=bind,source={workspace},target=/workspace","--mount",f"type=bind,source={test_dir.resolve()},target=/tests,readonly","--mount",f"type=bind,source={driver.resolve()},target=/driver,readonly","--workdir","/workspace","--env","PYTHONDONTWRITEBYTECODE=1","--env","PYTHONPATH=/workspace","--env","NODE_DISABLE_COLORS=1","--env","DO_NOT_TRACK=1"]
        for relative in fixture.get("readonly", []):
            dependency = (workspace / relative).resolve()
            if not dependency.is_relative_to(workspace) or not dependency.is_file():
                raise ValueError("invalid readonly fixture dependency")
            argv += ["--mount", f"type=bind,source={dependency},target=/workspace/{relative},readonly"]
        if fixture["language"]=="python":
            argv += [self.images["python"]["digest"],"python","/driver/grade_python.py"]
        else:
            argv += [self.images["typescript"]["image_id"],"sh","/driver/grade_typescript.sh"]
        began=time.time()
        log=Path(self.config["paths"]["logs"])/(job_id+"-"+phase+".log")
        code=None;timed_out=False
        with log.open("wb") as f:
            try:
                p=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,timeout=60)
                code=p.returncode
            except subprocess.TimeoutExpired:
                timed_out=True
            finally:
                # Verify labels before any stop/remove, including timeout cleanup.
                inspected=subprocess.run([self.docker,"inspect",name],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
                if inspected.returncode==0:
                    labels=json.loads(inspected.stdout)[0]["Config"].get("Labels") or {}
                    if validate_container(labels,self.run_id,job_id):
                        subprocess.run([self.docker,"rm","-f",name],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
        raw=log.read_text(encoding="utf-8",errors="replace")
        records=[line[len("LOCALBENCH_TESTS_JSON:"):] for line in raw.splitlines() if line.startswith("LOCALBENCH_TESTS_JSON:")]
        parsed=None
        if len(records)==1:
            try:parsed=json.loads(records[0])
            except ValueError:pass
        expected=fixture["public_count"] if phase=="public" else fixture["hidden_count"]
        complete=parsed is not None and parsed.get("tests_run")==expected and len(parsed.get("records",[]))==expected and len({r.get("id") for r in parsed["records"]})==expected
        passed=complete and code==0 and not timed_out and parsed.get("successful") is True and all(r.get("status")=="passed" for r in parsed["records"])
        reason="passed" if passed else "test_timeout" if timed_out else "incomplete_test_execution" if not complete else "test_failure"
        return {"passed":passed,"complete":complete,"expected_tests":expected,"exit_code":code,"seconds":time.time()-began,"reason":reason,"records":parsed.get("records",[]) if parsed else [],"log":str(log),"feedback":raw[-16000:] if phase=="public" else {"passed":passed,"tests_run":parsed.get("tests_run") if parsed else None},"launch_argv":argv}

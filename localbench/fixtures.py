"""Frozen deterministic fixtures and resumable, container-only acceptance.

Source and test text are inert data in this controller. Only DockerGrader may
execute emitted code. Gold, hidden tests and oracles never enter source tools.
"""

from __future__ import annotations

import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import random
import re
import uuid

from .config import atomic_json, canonical, digest, file_hash
from .fixture_python import build as build_python
from .fixture_typescript import build as build_typescript
from .grading import DockerGrader
from .paths import safe_source
from .schema import ValidationError, validate_tool


class FixtureFreezeError(ValueError):
    """Changing a frozen generator/grader requires an explicit new version."""


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _roots(config):
    project = Path(config["paths"]["project"]).resolve()
    roots = {name: Path(config["paths"][key]) for name, key in (
        ("source", "fixture_source"), ("grader", "held_out_tests"),
        ("workspaces", "agent_workspaces"), ("artifacts", "artifacts"),
    )}
    for root in roots.values():
        if not root.resolve().is_relative_to(project) or root.resolve() == project:
            raise ValueError("fixture roots must be dedicated directories in the project")
        _no_links(project, root)
    roots["project"] = project
    return roots


def _no_links(root, path):
    root, path = Path(root).resolve(), Path(path)
    if not path.resolve().is_relative_to(root):
        raise ValueError("fixture path escapes owned root")
    cursor = path
    while cursor != root:
        if cursor.is_symlink() or (hasattr(cursor, "is_junction") and cursor.is_junction()):
            raise ValueError("fixture symlinks and junctions are forbidden")
        if cursor == cursor.parent:
            raise ValueError("fixture path does not reach owned root")
        cursor = cursor.parent


def _write_frozen(root, relative, text):
    if not isinstance(relative, str) or "\\" in relative or ":" in relative or relative.startswith("/"):
        raise ValueError("unsafe frozen file path")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("unsafe frozen file path")
    path = root.joinpath(*parts)
    _no_links(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or file_hash(path) != _sha(text):
            raise FixtureFreezeError("frozen fixture file changed: " + relative)
    else:
        path.write_text(text, encoding="utf-8", newline="\n")


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def _document(identifier, region, text):
    fractions = {"early": 0.05, "middle": 0.50, "late": 0.95}
    return {"id": identifier, "region": region, "target_fraction": fractions[region],
            "authority": "active", "text": text}


def _answer_fixture(fixture_id, seed):
    rng = random.Random(seed)
    tenant, revision = f"tenant-{seed}-{rng.randrange(1000,9999)}", rng.randint(2,99)
    evidence = [f"{fixture_id}-{seed}-{region}-{rng.randrange(100000,999999)}"
                for region in ("early", "middle", "late")]
    if fixture_id == "long01":
        answer = {"route":f"/v{revision}/{rng.choice(['invoices','orders','payments'])}-{rng.randrange(100,999)}",
                  "timeout_ms":rng.randrange(20,301)*50, "retry_limit":rng.randint(0,9)}
        documents = [
            _document(evidence[0],"early",f"ACTIVE SIGNED ROUTE {evidence[0]}\nTenant {tenant}; revision {revision}; active route {answer['route']}."),
            _document(evidence[1],"middle",f"ACTIVE SIGNED TIMEOUT {evidence[1]}\nTenant {tenant}; revision {revision}; timeout_ms {answer['timeout_ms']}."),
            _document(evidence[2],"late",f"ACTIVE SIGNED RETRY {evidence[2]}\nTenant {tenant}; active revision confirmed {revision}; retry_limit {answer['retry_limit']}."),
        ]
        properties = {"route":{"type":"string"},"timeout_ms":{"type":"integer"},"retry_limit":{"type":"integer"}}
        cases = {"tenant":tenant,"revision":revision,"active":answer}
        task = f"Return the active route, timeout_ms and retry_limit for tenant {tenant}, EXACT revision {revision}."
    elif fixture_id == "long02":
        currency = rng.choice(["USD","EUR","GBP","CAD"])
        discounts = {"NONE":"0","PROMO":rng.choice(["0.05","0.10","0.25"])}
        lines = [{"unit_price":"0.005","quantity":1,"discount_code":"NONE"},
                 {"unit_price":"0.005","quantity":1,"discount_code":"NONE"}]
        for _ in range(3):
            lines.append({"unit_price":f"{rng.randint(1,100)}.{rng.randint(0,999):03}",
                          "quantity":rng.randint(1,5),"discount_code":rng.choice(list(discounts))})
        tax_rate = rng.choice(["0.05","0.075","0.10","0.125"])
        cent = Decimal("0.01")
        rounded = [(Decimal(line["unit_price"])*line["quantity"]*(1-Decimal(discounts[line["discount_code"]]))).quantize(cent,rounding=ROUND_HALF_UP)
                   for line in lines]
        subtotal = sum(rounded,Decimal("0.00"))
        tax = (subtotal*Decimal(tax_rate)).quantize(cent,rounding=ROUND_HALF_UP)
        answer = {"total":format(subtotal+tax,".2f"),"currency":currency,"schema_revision":revision}
        documents = [
            _document(evidence[0],"early",f"ACTIVE SIGNED MANIFEST {evidence[0]}\nTenant {tenant}; schema_revision {revision}; currency {currency}; discount table {canonical(discounts)}."),
            _document(evidence[1],"middle",f"ACTIVE SIGNED INVOICE {evidence[1]}\nTenant {tenant}; schema_revision {revision}; lines {canonical(lines)}."),
            _document(evidence[2],"late",f"ACTIVE SIGNED MONEY POLICY {evidence[2]}\nTenant {tenant}; schema_revision {revision}; tax_rate {tax_rate}. For each line calculate unit_price*quantity*(1-discount), round separately to cents using HALF_UP, sum those rounded lines, then round subtotal*tax_rate separately to cents using HALF_UP. Total=subtotal+rounded tax. Empty=0.00. Never binary float or bankers' rounding."),
        ]
        properties = {"total":{"type":"string"},"currency":{"type":"string"},"schema_revision":{"type":"integer"}}
        cases = {"tenant":tenant,"revision":revision,"discounts":discounts,"lines":lines,
                 "tax_rate":tax_rate,"rounded_lines":[str(line) for line in rounded],
                 "subtotal":str(subtotal),"tax":str(tax)}
        task = f"Calculate this generated invoice for tenant {tenant}, exact active schema revision {revision}; return total as a two-decimal string, currency and schema_revision."
    else:
        raise ValueError("unknown answer fixture")
    schema = {"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}
    contract = (task + " Active matching tenant and exact revision outrank all explicitly OBSOLETE, EXAMPLE or UNRELATED material. "
                "The three active authoritative documents are " + ", ".join(evidence) + ". "
                "Submit exactly one submit_answer with object-valued answer matching the supplied schema and all three evidence_ids in source appearance order (early, middle, late). No other tools or actions. No conflicting active fact exists.")
    distractors = [{"id":f"obsolete-{fixture_id}-{seed}-{i}","authority":"obsolete",
                    "text":f"OBSOLETE EXAMPLE UNRELATED tenant old-{tenant}; revision {revision+i+1}; route /legacy/{i}; timeout_ms {100+i}; retry_limit {30+i}; currency OLD; aggregate bankers rounding. Ignore for the active query."}
                   for i in range(12)]
    return {"language":"answer", "source":{"src/context.py":"# Answer-only task; authoritative documents are embedded in the initial prompt.\n"},
            "gold":{},"public_tests":{},"hidden_tests":{},"public_count":0,"hidden_count":0,
            "contract":contract,"cases":cases,"writable":[],"documents":documents,
            "distractors":distractors,"answer_schema":schema,
            "oracle":{"answer":answer,"evidence_ids":evidence},
            "allowed_tools":["submit_answer"],"permitted_final_action":"submit_answer"}


def _build(fixture_id, seed):
    if fixture_id in ("long01","long02"):
        return _answer_fixture(fixture_id,seed)
    fixture = build_python(fixture_id,seed) if fixture_id.startswith("py") or fixture_id=="long03" else build_typescript(fixture_id,seed)
    fixture["writable"] = [path for path in fixture["source"] if path != "src/errors.ts"]
    fixture["readonly"] = [path for path in fixture["source"] if path not in fixture["writable"]]
    fixture["oracle"] = {"cases":fixture["cases"],"public_count":fixture["public_count"],"hidden_count":fixture["hidden_count"]}
    if fixture_id.startswith("long"):
        fixture["permitted_final_action"] = "write_file"
        fixture["allowed_tools"] = ["write_file"]
        rng = random.Random(seed)
        ids = [f"{fixture_id}-{seed}-{region}-{rng.randrange(100000,999999)}" for region in ("early","middle","late")]
        if fixture_id=="long03":
            early = "ACTIVE SIGNED CONTRACT: settle(lines,tax_rate)->Decimal. Each line is (Decimal unit_price,int quantity,Decimal discount). Prices/rates finite; price>=0; integer quantity>=0; discount/tax in [0,1]. Invalid values raise ValueError. Formula unit_price*quantity*(1-discount). Preserve callable signature."
            late = "ACTIVE SIGNED POLICY: ROUND_HALF_UP each line to cents, sum rounded lines, ROUND_HALF_UP subtotal*tax_rate separately to cents, then sum. Empty=Decimal('0.00'). Never floats, never input mutation."
        else:
            early = "ACTIVE SIGNED CONTRACT: requestWithRetry(transport,url,options)->Promise<Response>. maxAttempts positive integer; baseDelayMs finite>=0. Preserve full exported types below. Validate before transport."
            late = "ACTIVE SIGNED POLICY: retry only HTTP 429/503 or TimeoutError; other thrown errors propagate immediately. Delay baseDelayMs*2**failure_index before next attempt, never after last. Final HTTP response unchanged including 429/503; final TimeoutError object propagates. Fresh request/header per attempt, same URL/method, no options mutation."
        middle = "ACTIVE SIGNED CALLER AND SOURCE:\n" + "\n".join(f"FILE {path}\n{text}" for path,text in fixture["source"].items())
        if fixture_id=="long04":
            middle += "\nACTIVE UNSAFE-METHOD POLICY: retry GET, or POST with a nonempty idempotencyKey. Whitespace-only keys are empty. Unsafe POST uses ONE attempt. A keyed POST sends Idempotency-Key on EVERY attempt as originally supplied, without trimming. src/errors.ts is READ-ONLY."
        fixture["documents"] = [_document(ids[0],"early",early),_document(ids[1],"middle",middle),_document(ids[2],"late",late)]
        fixture["distractors"] = [{"id":f"obsolete-{fixture_id}-{seed}","authority":"obsolete","text":"OBSOLETE EXAMPLE unrelated revision: floats, aggregate bankers rounding, retry every failure and every unsafe POST. Never use this example for the active task."}]
        fixture["contract"] += "\nThe three ACTIVE SIGNED documents outrank all explicitly obsolete, example and unrelated material. Exactly one final write_file action, no read/test turns, at most 4096 generated tokens."
    else:
        fixture["allowed_tools"] = ["list_files","read_file","write_file","apply_patch","run_tests"]
    return fixture


def _generator_hashes(config):
    here = Path(__file__).parent
    result = {name:file_hash(here/name) for name in ("fixtures.py","fixture_python.py","fixture_typescript.py","grading.py")}
    project = Path(config["paths"]["project"])
    for relative in ("FIXTURES.md","scripts/grade_python.py","scripts/grade_typescript.sh","scripts/node_reporter.cjs"):
        if (project/relative).exists():
            result[relative] = file_hash(project/relative)
    return result


def _entry(fixture_id, seed, bundle):
    return {"id":fixture_id,"seed":seed,"bundle_hash":digest(bundle),
            "language":bundle["language"],"public_count":bundle["public_count"],"hidden_count":bundle["hidden_count"],
            "hashes":{name:{path:_sha(text) for path,text in bundle[name].items()}
                      for name in ("source","gold","public_tests","hidden_tests")},
            "contract_hash":_sha(bundle["contract"]),"cases_hash":digest(bundle["cases"]),
            "oracle_hash":digest(bundle["oracle"]),"documents_hash":digest(bundle.get("documents",[])),
            "writable":bundle["writable"]}


def init_fixtures(config, state):
    """Build/freeze all 48 seed jobs without executing any emitted text."""
    roots = _roots(config)
    ids = config["grading"]["regular_fixture_ids"] + config["grading"]["long_fixture_ids"]
    bundles = [(fixture_id,seed,_build(fixture_id,seed)) for fixture_id in ids for seed in config["grading"]["seeds"]]
    entries = [_entry(fixture_id,seed,bundle) for fixture_id,seed,bundle in bundles]
    payload = {"format":"localbench.fixtures.v1","spec_hash":config["_spec_hash"],
               "generator_hashes":_generator_hashes(config),"fixtures":entries}
    version = digest(payload)
    path = roots["artifacts"]/"fixture-manifest.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("version") != version:
            state.control("fixtures_verified",False)
            raise FixtureFreezeError("incompatible existing fixture freeze; explicitly reconcile a new fixture version, retaining earlier results")
        try:
            for fixture_id,seed,_ in bundles:
                load_fixture(config,fixture_id,seed)
        except (FixtureFreezeError,OSError,ValueError):
            state.control("fixtures_verified",False)
            raise
        return previous
    state.control("fixtures_verified",False)
    for fixture_id,seed,bundle in bundles:
        prefix = f"{fixture_id}/{seed}"
        for relative,text in bundle["source"].items():
            safe_source(roots["source"]/prefix,relative)
            _write_frozen(roots["source"],prefix+"/"+relative,text)
        _write_frozen(roots["source"],prefix+"/contract.txt",bundle["contract"])
        for phase in ("public","hidden"):
            for relative,text in bundle[phase+"_tests"].items():
                _write_frozen(roots["grader"],prefix+"/"+phase+"/"+relative,text)
        for relative,text in bundle["gold"].items():
            _write_frozen(roots["grader"],prefix+"/gold/"+relative,text)
        _write_frozen(roots["grader"],prefix+"/oracle.json",_json_text(bundle["oracle"]))
        _write_frozen(roots["grader"],prefix+"/fixture.json",_json_text(bundle))
    manifest = {**payload,"version":version,"frozen_at":state.clock(),
                "test_mount_policy":"public/hidden controller directories are Docker readonly binds; gold and oracles are never mounted"}
    atomic_json(path,manifest)
    state.control("fixture_version",version)
    return manifest


def load_fixture(config, fixture_id, seed):
    if not re.fullmatch(r"(?:py|ts)0[1-6]|long0[1-4]",str(fixture_id)) or type(seed) is not int:
        raise ValueError("invalid fixture identity")
    roots = _roots(config)
    manifest = json.loads((roots["artifacts"]/"fixture-manifest.json").read_text(encoding="utf-8"))
    payload = {key:manifest[key] for key in ("format","spec_hash","generator_hashes","fixtures")}
    if digest(payload)!=manifest.get("version") or manifest.get("spec_hash")!=config["_spec_hash"]:
        raise FixtureFreezeError("frozen manifest changed")
    if manifest.get("generator_hashes") != _generator_hashes(config):
        raise FixtureFreezeError("frozen generator/grader hashes changed; prior acceptance is invalid")
    entry = next((item for item in manifest["fixtures"] if item["id"]==fixture_id and item["seed"]==seed),None)
    if entry is None:
        raise ValueError("fixture seed is outside the frozen manifest")
    prefix = f"{fixture_id}/{seed}"
    grader = roots["grader"]/prefix
    _no_links(roots["grader"],grader/"fixture.json")
    bundle = json.loads((grader/"fixture.json").read_text(encoding="utf-8"))
    if digest(bundle) != entry["bundle_hash"]:
        raise FixtureFreezeError("controller fixture bundle changed")
    for name,hashes in entry["hashes"].items():
        directory = roots["source"]/prefix if name=="source" else grader/"gold" if name=="gold" else grader/name.removesuffix("_tests")
        for relative,expected in hashes.items():
            path = directory/relative
            _no_links(roots["project"],path)
            if not path.is_file() or file_hash(path)!=expected:
                raise FixtureFreezeError("frozen fixture source/test/gold changed")
    _no_links(roots["source"],roots["source"]/prefix/"contract.txt")
    if file_hash(roots["source"]/prefix/"contract.txt") != entry["contract_hash"]:
        raise FixtureFreezeError("frozen contract changed")
    _no_links(roots["grader"],grader/"oracle.json")
    oracle = json.loads((grader/"oracle.json").read_text(encoding="utf-8"))
    if digest(oracle)!=entry["oracle_hash"]:
        raise FixtureFreezeError("frozen oracle changed")
    return {**bundle,"id":fixture_id,"seed":seed,"version":manifest["version"],
            "fixture_hash":entry["bundle_hash"],"grader_path":str(grader.resolve()),
            "source_path":str((roots["source"]/prefix).resolve())}


def create_workspace(config, fixture, job_id, gold=False):
    """Create only synthetic source files; reuse a matching owned workspace."""
    if not isinstance(job_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}",job_id):
        raise ValueError("invalid workspace job id")
    roots = _roots(config)
    workspace = roots["workspaces"]/job_id
    _no_links(roots["workspaces"],workspace)
    marker = workspace/".localbench-workspace.json"
    metadata = {"job_id":job_id,"fixture":fixture["id"],"seed":fixture["seed"],
                "version":fixture["version"],"fixture_hash":fixture["fixture_hash"],"gold":bool(gold)}
    if workspace.exists():
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8"))!=metadata:
            raise ValueError("workspace exists without matching benchmark ownership")
        for relative in fixture["source"]:
            target = safe_source(workspace,relative)
            if not target.is_file():
                raise ValueError("resumable workspace is missing source")
        return workspace
    workspace.mkdir(parents=True)
    source = fixture["gold"] if gold else fixture["source"]
    for relative,text in source.items():
        target = safe_source(workspace,relative)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text,encoding="utf-8",newline="\n")
    atomic_json(marker,metadata)
    return workspace


def score_answer(fixture, submission):
    """Exact, type-checked answer/evidence scoring without model or code judge."""
    try:
        validate_tool("submit_answer",submission,fixture["answer_schema"])
    except (ValidationError,KeyError,TypeError,ValueError):
        return {"passed":False,"reason":"answer_schema_failure"}
    oracle = fixture["oracle"]
    if canonical(submission["answer"]) != canonical(oracle["answer"]):
        return {"passed":False,"reason":"answer_mismatch"}
    if submission["evidence_ids"] != oracle["evidence_ids"]:
        return {"passed":False,"reason":"evidence_mismatch"}
    return {"passed":True,"reason":"passed"}


def _answer_controls(fixture):
    gold = copy.deepcopy(fixture["oracle"])
    wrong = copy.deepcopy(gold)
    field = next(iter(wrong["answer"]))
    wrong["answer"][field] = str(wrong["answer"][field])+"-wrong"
    evidence = copy.deepcopy(gold); evidence["evidence_ids"][0]+="-wrong"
    schema = copy.deepcopy(gold); del schema["answer"][field]
    random_answer = {"answer":{},"evidence_ids":["random-evidence"]}
    truncated = copy.deepcopy(gold)
    del truncated["answer"][next(reversed(truncated["answer"]))]
    truncated["evidence_ids"] = truncated["evidence_ids"][:-1]
    reordered = copy.deepcopy(gold); reordered["evidence_ids"].reverse()
    candidates = {"gold":gold,"wrong_answer":wrong,"wrong_evidence":evidence,
                  "wrong_schema":schema,"random_answer":random_answer,
                  "synthetic_missing_late_document":truncated,"reordered_evidence":reordered}
    results = {name:score_answer(fixture,value) for name,value in candidates.items()}
    return {"accepted":results["gold"]["passed"] and all(not result["passed"] for name,result in results.items() if name!="gold"),
            "controls":results,"truncation_control_scope":"synthetic omitted late-region facts/evidence; actual model capacity probes remain separate"}


def _assertion_failures(result, language):
    failed = [record for record in result.get("records",[]) if record.get("status")=="failed"]
    if language=="python":
        return len(failed)
    classified = sum(record.get("failure_kind")=="assertion" for record in failed)
    if classified:
        return classified
    log = Path(result.get("log",""))
    if not log.is_file():
        return 0
    raw = log.read_text(encoding="utf-8",errors="replace")
    return sum(("ERR_ASSERTION" in line or '"name":"AssertionError"' in line)
               for line in raw.splitlines() if line.startswith("FAILURE:"))


def _workspace_intact(workspace, fixture, gold):
    expected = fixture["gold"] if gold else fixture["source"]
    for relative,text in expected.items():
        target = safe_source(workspace,relative)
        if not target.is_file() or file_hash(target)!=_sha(text):
            return False
    actual = {path.relative_to(workspace).as_posix() for path in (workspace/"src").rglob("*") if path.is_file() and path.suffix in (".py",".ts")}
    return actual == set(expected)


def _reusable(record, signature):
    if not record or not record.get("accepted") or record.get("signature")!=signature:
        return False
    if "grader_result" in record:
        log = Path(record["grader_result"].get("log",""))
        return log.is_file() and file_hash(log)==record.get("log_sha256")
    return True


def verify_fixtures(config, state):
    """Checkpoint each variant/phase; never start the execution budget clock."""
    manifest = init_fixtures(config,state)
    roots = _roots(config)
    images_path = roots["artifacts"]/"grader-images.json"
    pins = json.loads(images_path.read_text(encoding="utf-8")) if images_path.exists() else None
    signature = digest({"fixture_version":manifest["version"],"grader_images":pins})
    checkpoint = roots["artifacts"]/"fixture-verification.json"
    previous = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    records = previous.get("records",{}) if previous.get("signature")==signature else {}
    report = {"format":"localbench.fixture-verification.v1","fixture_version":manifest["version"],
              "signature":signature,"verified":False,"records":records,"execution_clock_started":state.run["started"]}
    state.control("fixtures_verified",False)
    grader = None
    if pins is not None:
        grader = DockerGrader(config,state.run["id"])
    expected_keys = []
    for entry in manifest["fixtures"]:
        fixture = load_fixture(config,entry["id"],entry["seed"])
        prefix = f"{entry['id']}-{entry['seed']}"
        if fixture["language"]=="answer":
            key = prefix+"-answers"; expected_keys.append(key)
            if not _reusable(records.get(key),signature):
                records[key] = {"signature":signature,"checked_at":state.clock(),**_answer_controls(fixture)}
                atomic_json(checkpoint,report)
            continue
        for variant in ("bug","gold"):
            for phase in ("public","hidden"):
                key = f"{prefix}-{variant}-{phase}"; expected_keys.append(key)
                if _reusable(records.get(key),signature):
                    continue
                if grader is None:
                    records[key] = {"signature":signature,"accepted":False,"reason":"pinned_grader_images_not_prepared"}
                    atomic_json(checkpoint,report)
                    continue
                workspace = create_workspace(config,fixture,"verify-"+digest([signature,key])[:48],gold=variant=="gold")
                if not _workspace_intact(workspace,fixture,variant=="gold"):
                    records[key] = {"signature":signature,"accepted":False,"reason":"verification_workspace_source_changed"}
                    atomic_json(checkpoint,report)
                    continue
                try:
                    result = grader.grade(workspace,fixture,phase=phase,log_id="fixture-verify-"+key+"-"+uuid.uuid4().hex[:10])
                except (OSError,RuntimeError,ValueError) as error:
                    records[key] = {"signature":signature,"accepted":False,"reason":"grader_infrastructure_error","error_type":type(error).__name__,"checked_at":state.clock()}
                    atomic_json(checkpoint,report)
                    continue
                assertions = _assertion_failures(result,fixture["language"])
                source_intact = _workspace_intact(workspace,fixture,variant=="gold")
                accepted = (result["passed"] if variant=="gold" else
                            result["complete"] and not result["passed"] and assertions>=1 and
                            not any(record.get("status") in ("error","skipped","expected_failure","unexpected_success") for record in result.get("records",[])))
                accepted = bool(accepted and source_intact)
                log = Path(result["log"])
                records[key] = {"signature":signature,"accepted":accepted,"variant":variant,"phase":phase,
                                "assertion_failures":assertions,"source_intact":source_intact,"checked_at":state.clock(),
                                "grader_result":result,"log_sha256":file_hash(log) if log.is_file() else None}
                atomic_json(checkpoint,report)
    report["verified"] = all(records.get(key,{}).get("accepted") for key in expected_keys)
    report["expected_combinations"] = len(expected_keys)
    report["accepted_combinations"] = sum(bool(records.get(key,{}).get("accepted")) for key in expected_keys)
    report["checked_at"] = state.clock()
    atomic_json(checkpoint,report)
    state.control("fixtures_verified",report["verified"])
    state.control("fixture_verification_signature",signature if report["verified"] else None)
    return report

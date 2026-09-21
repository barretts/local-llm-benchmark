"""Fixture integration tests inspect emitted data and fake grading records only."""

import copy
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from localbench.config import atomic_json, load
from localbench.fixtures import (
    FixtureFreezeError, create_workspace, init_fixtures, load_fixture,
    score_answer, verify_fixtures,
)
from localbench.state import State


class FixtureIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="localbench-fixtures-")
        self.root = Path(self.temporary.name)
        self.config = load()
        for key,relative in {"project":".","fixture_source":"fixtures","held_out_tests":"grader",
                             "agent_workspaces":"workspaces","artifacts":"artifacts","state":"state","logs":"logs"}.items():
            self.config["paths"][key] = str(self.root/relative)
        (self.root/"logs").mkdir()
        self.state = State(self.config,clock=lambda:1000.0)

    def tearDown(self):
        self.state.close()
        self.temporary.cleanup()

    def test_freeze_contains_all_48_jobs_and_is_idempotent(self):
        manifest = init_fixtures(self.config,self.state)
        self.assertEqual(len(manifest["fixtures"]),48)
        self.assertEqual(manifest,init_fixtures(self.config,self.state))
        self.assertIsNone(self.state.run["started"])
        self.assertFalse(self.state.get_control("fixtures_verified"))
        self.assertEqual(self.state.get_control("fixture_version"),manifest["version"])
        self.assertIn("grading.py",manifest["generator_hashes"])
        for entry in manifest["fixtures"]:
            fixture = load_fixture(self.config,entry["id"],entry["seed"])
            self.assertEqual(fixture["fixture_hash"],entry["bundle_hash"])
            self.assertEqual(fixture["version"],manifest["version"])
            self.assertEqual(fixture["cases"],json.loads(json.dumps(fixture["cases"])))

    def test_gold_hidden_tests_and_oracles_stay_outside_agent_workspace(self):
        init_fixtures(self.config,self.state)
        fixture = load_fixture(self.config,"ts06",42)
        workspace = create_workspace(self.config,fixture,"task-42")
        actual = {path.relative_to(workspace).as_posix() for path in workspace.rglob("*") if path.is_file()}
        self.assertEqual(actual,set(fixture["source"])|{".localbench-workspace.json"})
        self.assertTrue(Path(fixture["grader_path"]).is_relative_to(self.root/"grader"))
        for relative,text in fixture["source"].items():
            self.assertEqual((workspace/relative).read_text(encoding="utf-8"),text)
        self.assertEqual(workspace,create_workspace(self.config,fixture,"task-42"))

    def test_readonly_retry_dependency_and_single_action_metadata(self):
        init_fixtures(self.config,self.state)
        fixture = load_fixture(self.config,"long04",314159)
        self.assertEqual(fixture["writable"],["src/request.ts"])
        self.assertEqual(fixture["readonly"],["src/errors.ts"])
        self.assertEqual(fixture["allowed_tools"],["write_file"])
        self.assertEqual(fixture["permitted_final_action"],"write_file")
        self.assertEqual([doc["target_fraction"] for doc in fixture["documents"]],[0.05,0.5,0.95])
        self.assertIn("Idempotency-Key",fixture["documents"][1]["text"])
        self.assertIn("TimeoutError",fixture["documents"][2]["text"])

    def test_existing_freeze_rejects_source_tamper_and_invalidates_gate(self):
        init_fixtures(self.config,self.state)
        self.state.control("fixtures_verified",True)
        source = self.root/"fixtures"/"py01"/"42"/"src"/"cache.py"
        original = source.read_text(encoding="utf-8")
        source.write_text(original+"\n# altered\n",encoding="utf-8")
        with self.assertRaises(FixtureFreezeError):
            init_fixtures(self.config,self.state)
        self.assertFalse(self.state.get_control("fixtures_verified"))
        self.assertTrue(source.read_text(encoding="utf-8").endswith("# altered\n"))

    def test_existing_freeze_rejects_incompatible_seed_set(self):
        init_fixtures(self.config,self.state)
        changed = copy.deepcopy(self.config)
        changed["grading"]["seeds"] = [42]
        with self.assertRaises(FixtureFreezeError):
            init_fixtures(changed,self.state)

    def test_workspace_ids_outside_roots_and_unowned_paths_are_rejected(self):
        init_fixtures(self.config,self.state)
        fixture = load_fixture(self.config,"py01",42)
        for job_id in ("../personal","C:\\data","/tmp/job","a/b","","a"*129):
            with self.subTest(job=job_id),self.assertRaises(ValueError):
                create_workspace(self.config,fixture,job_id)
        unowned = self.root/"workspaces"/"existing"
        unowned.mkdir(parents=True)
        with self.assertRaises(ValueError):
            create_workspace(self.config,fixture,"existing")
        changed = copy.deepcopy(self.config)
        changed["paths"]["fixture_source"] = str(self.root.parent/"outside")
        with self.assertRaises(ValueError):
            init_fixtures(changed,self.state)

    def test_answer_fixtures_require_exact_values_types_and_ordered_evidence(self):
        init_fixtures(self.config,self.state)
        all_ids = set()
        for fixture_id in ("long01","long02"):
            for seed in self.config["grading"]["seeds"]:
                fixture = load_fixture(self.config,fixture_id,seed)
                gold = copy.deepcopy(fixture["oracle"])
                self.assertTrue(score_answer(fixture,gold)["passed"])
                self.assertEqual(fixture["writable"],[])
                for doc in fixture["documents"]:
                    self.assertNotIn(doc["id"],all_ids)
                    all_ids.add(doc["id"])
                reversed_ids = copy.deepcopy(gold); reversed_ids["evidence_ids"].reverse()
                self.assertFalse(score_answer(fixture,reversed_ids)["passed"])
                missing = copy.deepcopy(gold); missing["evidence_ids"].pop()
                self.assertFalse(score_answer(fixture,missing)["passed"])
                extra = copy.deepcopy(gold); extra["answer"]["unexpected"] = 1
                self.assertFalse(score_answer(fixture,extra)["passed"])
                numeric_key = "timeout_ms" if fixture_id=="long01" else "schema_revision"
                boolean = copy.deepcopy(gold); boolean["answer"][numeric_key]=True
                self.assertFalse(score_answer(fixture,boolean)["passed"])

    def test_invoice_oracle_has_half_cent_lines_and_rounds_tax_separately(self):
        init_fixtures(self.config,self.state)
        cent = Decimal("0.01")
        for seed in self.config["grading"]["seeds"]:
            fixture = load_fixture(self.config,"long02",seed)
            cases = fixture["cases"]
            self.assertEqual(cases["rounded_lines"][:2],["0.01","0.01"])
            subtotal = sum(((Decimal(line["unit_price"])*line["quantity"]*(1-Decimal(cases["discounts"][line["discount_code"]]))).quantize(cent,rounding=ROUND_HALF_UP)
                            for line in cases["lines"]),Decimal("0.00"))
            tax=(subtotal*Decimal(cases["tax_rate"])).quantize(cent,rounding=ROUND_HALF_UP)
            self.assertEqual(fixture["oracle"]["answer"]["total"],format(subtotal+tax,".2f"))
            self.assertIn("HALF_UP",fixture["documents"][2]["text"])

    def test_fake_grading_checkpoint_resume_and_raw_log_integrity(self):
        atomic_json(self.root/"artifacts"/"grader-images.json",{"python":{"digest":"fake"},"typescript":{"image_id":"fake"}})
        calls=[]
        config=self.config
        class FakeGrader:
            def __init__(self,config,run_id):
                pass
            def grade(self,workspace,fixture,phase,log_id):
                calls.append((fixture["id"],fixture["seed"],phase))
                is_gold=json.loads((workspace/".localbench-workspace.json").read_text())["gold"]
                count=fixture[phase+"_count"]
                records=[{"id":f"assertion-{i}","status":"passed" if is_gold else "failed","failure_kind":"assertion"} for i in range(count)]
                log=Path(config["paths"]["logs"])/(log_id+".log")
                log.write_text("FAKE GRADER: emitted source was not executed.\n")
                return {"passed":is_gold,"complete":True,"reason":"passed" if is_gold else "test_failure",
                        "records":records,"feedback":{},"log":str(log),"launch_argv":["fake"]}
        with patch("localbench.fixtures.DockerGrader",FakeGrader):
            first=verify_fixtures(self.config,self.state)
            self.assertTrue(first["verified"])
            self.assertEqual(first["expected_combinations"],174)
            self.assertEqual(len(calls),168)
            self.assertIsNone(self.state.run["started"])
            self.assertTrue(self.state.get_control("fixtures_verified"))
            again=verify_fixtures(self.config,self.state)
            self.assertTrue(again["verified"])
            self.assertEqual(len(calls),168)
            record=next(record for record in first["records"].values() if "grader_result" in record)
            Path(record["grader_result"]["log"]).write_text("tampered raw log\n")
            repaired=verify_fixtures(self.config,self.state)
            self.assertTrue(repaired["verified"])
            self.assertEqual(len(calls),169)


if __name__ == "__main__":
    unittest.main()

"""CPU-only ownership/collision tests for isolated existing-DLL packaging."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from localbench.config import atomic_json, file_hash
from localbench.state import State

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "pin-turboquant-runtime-deps.py"
SPEC = importlib.util.spec_from_file_location("bench_runtime_dependency_helper", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


class RuntimeDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = {"_spec_hash": "dependency-safety-tests", "paths": {
            "state": str(self.root / "state"), "runtime_root": str(self.root / "runtimes"),
            "logs": str(self.root / "logs"), "artifacts": str(self.root / "artifacts")}}
        self.state = State(self.config)
        self.runtime = self.root / "runtimes" / ("turboquant-cuda-" + helper.PIN)
        self.bin = self.runtime / "build-sm89-windows" / "bin" / "Release"
        self.bin.mkdir(parents=True)
        self.binary = self.bin / "llama-server.exe"
        self.binary.write_bytes(b"fake PE; never executed")
        self.vs = self.root / "existing-vs"
        self.vs.mkdir()
        self.cl = self.vs / "cl.exe"
        self.cl.write_bytes(b"mock installed compiler")
        self.dumpbin = self.vs / "dumpbin.exe"
        self.dumpbin.write_bytes(b"mock metadata tool; never executed")
        self.ssl = self.root / "existing-openssl"
        self.ssl.mkdir()
        (self.ssl / "libssl-3-x64.dll").write_bytes(b"existing ssl")
        (self.ssl / "libcrypto-3-x64.dll").write_bytes(b"existing crypto")
        self.setup = {"started": time.time(), "deadline": time.time() + 60, "phase": "complete"}
        self.state.control("runtime_setup:test", self.setup)
        self.state.control("source_review:turboquant-cuda", {"fingerprint": "reviewed"})
        atomic_json(self.runtime / "localbench-owned.json", {"owner": "localbench", "run_id": self.state.run["id"],
                    "engine": "turboquant-cuda", "commit": helper.PIN})
        atomic_json(self.runtime / "source-manifest.json", {"setup_key": "runtime_setup:test", "setup_deadline": self.setup["deadline"],
                    "review_fingerprint": "reviewed"})
        self.manifest = {"commit": helper.PIN, "review_fingerprint": "reviewed", "binary_path": str(self.binary),
            "files": [{"path": str(self.binary), "sha256": file_hash(self.binary), "bytes": self.binary.stat().st_size}],
            "build_plan": {"build": str(self.runtime / "build-sm89-windows"),
                           "toolchain": {"cl": str(self.cl), "visual_studio": str(self.vs)}}}
        self.save_manifest()
        self.calls = []
        self.imports = {"llama-server.exe": ["libssl-3-x64.dll", "unapproved_dependency.dll"],
                        "libssl-3-x64.dll": ["libcrypto-3-x64.dll"], "libcrypto-3-x64.dll": ["KERNEL32.dll"]}

    def save_manifest(self):
        atomic_json(self.runtime / "engine-manifest.json", self.manifest)

    def tearDown(self):
        self.state.close()
        self.temp.cleanup()

    def fake_inspection(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        self.assertEqual(argv[:3], [str(self.dumpbin), "/nologo", "/dependents"])
        self.assertEqual(kwargs["cwd"], str(self.runtime))
        imports = self.imports.get(Path(argv[-1]).name, [])
        return SimpleNamespace(returncode=0, stdout="\n".join("    " + name for name in imports), stderr="")

    def run_helper(self):
        with patch.object(helper, "load", return_value=self.config), patch.object(helper, "State", return_value=self.state), \
             patch.object(self.state, "close"), patch.object(helper, "OPENSSL_ROOT", self.ssl), \
             patch.object(helper.subprocess, "run", side_effect=self.fake_inspection), contextlib.redirect_stdout(io.StringIO()):
            helper.main()

    def test_closes_only_allowed_imports_and_pins_copies_without_executing_pe(self):
        before = self.state.run
        self.run_helper()
        result = json.loads((self.runtime / "engine-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual({Path(item["path"]).name for item in result["additional_existing_dependencies"]}, helper.ALLOWED)
        self.assertEqual(file_hash(self.bin / "libcrypto-3-x64.dll"), file_hash(self.ssl / "libcrypto-3-x64.dll"))
        self.assertFalse((self.bin / "unapproved_dependency.dll").exists())
        self.assertFalse(result["pe_dependency_inspection"]["selected_binaries_executed"])
        self.assertEqual(self.state.run, before)
        self.assertEqual(self.state.get_control("runtime_setup:test"), self.setup)
        self.assertEqual(len(self.calls), 3)

    def test_does_not_copy_openssl_when_imports_do_not_require_it(self):
        self.imports = {"llama-server.exe": ["KERNEL32.dll"]}
        self.run_helper()
        self.assertEqual(list(self.bin.glob("*.dll")), [])
        result = json.loads((self.runtime / "engine-manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(result["additional_existing_dependencies"], [])

    def test_existing_different_dll_collision_is_preserved(self):
        target = self.bin / "libcrypto-3-x64.dll"
        target.write_bytes(b"preserve this unrelated payload")
        original = (self.runtime / "engine-manifest.json").read_bytes()
        with self.assertRaisesRegex(RuntimeError, "dependency collision"):
            self.run_helper()
        self.assertEqual(target.read_bytes(), b"preserve this unrelated payload")
        self.assertEqual((self.runtime / "engine-manifest.json").read_bytes(), original)

    def test_source_directory_is_rejected_as_runtime_binary_destination(self):
        source = self.runtime / "source"
        source.mkdir()
        bad = source / "llama-server.exe"
        bad.write_bytes(b"do not package into immutable source")
        self.manifest["binary_path"] = str(bad)
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "binary directory escapes"):
            self.run_helper()
        self.assertEqual(self.calls, [])
        self.assertEqual(list(source.glob("*.dll")), [])

    def test_mismatched_run_ownership_refuses_metadata_tool_and_copies(self):
        owned = json.loads((self.runtime / "localbench-owned.json").read_text())
        owned["run_id"] = "unrelated-run"
        atomic_json(self.runtime / "localbench-owned.json", owned)
        with self.assertRaisesRegex(RuntimeError, "ownership mismatch"):
            self.run_helper()
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.bin.glob("*.dll")), [])

    def test_expired_original_deadline_is_not_reset(self):
        self.setup["deadline"] = time.time() - 1
        self.state.control("runtime_setup:test", self.setup)
        source = json.loads((self.runtime / "source-manifest.json").read_text())
        source["setup_deadline"] = self.setup["deadline"]
        atomic_json(self.runtime / "source-manifest.json", source)
        with self.assertRaisesRegex(RuntimeError, "budget_exhausted"):
            self.run_helper()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.state.get_control("runtime_setup:test"), self.setup)


if __name__ == "__main__":
    unittest.main()

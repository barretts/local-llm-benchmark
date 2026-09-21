"""Private model storage, cumulative copies and read-only serving reuse; no runtimes."""
import copy
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from localbench.adapters.ollama import OllamaAdapter
from localbench.config import file_hash, load
from localbench.recovery import _validate_process_handle
from localbench.state import State


class OllamaStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load()
        for name in ("project", "state", "logs", "artifacts", "runtime_root", "new_model_root"):
            self.config["paths"][name] = str(self.root / name)
        self.source = self.root / "existing-installed.gguf"
        self.payload = b"unexecuted fake weight fixture" * 13
        self.source.write_bytes(self.payload)
        self.binary = self.root / "unexecuted-cli.exe"
        self.binary.write_bytes(b"mock binary, never execute")
        self.config["installed_tools"]["ollama"] = str(self.binary)
        self.model = {"id": "mock-installed", "path": str(self.source), "sha256": file_hash(self.source)}
        self.state = State(self.config)
        self.readonly_databases = []
        self.headroom = patch("localbench.adapters.ollama._disk_headroom")
        self.disk_check = self.headroom.start()
        self.addCleanup(self.headroom.stop)

    def tearDown(self):
        for database in self.readonly_databases:
            database.close()
        self.state.close()
        self.temp.cleanup()

    def adapter(self, state=None):
        adapter = OllamaAdapter(self.config, state or self.state, self.model)
        adapter._metadata = {"engine_id": "ollama", "model_sha256": self.model["sha256"]}
        return adapter

    def blob(self, adapter):
        return adapter.models_root / "blobs" / ("sha256-" + self.model["sha256"])

    def copy_blob(self):
        adapter = self.adapter()
        with patch("localbench.adapters.ollama.os.link", side_effect=OSError("mock cross-volume")):
            target = adapter._prepare_blob()
        return adapter, target

    def readonly_serving(self):
        database = Path(self.config["paths"]["state"]) / "benchmark.sqlite3"
        db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        self.readonly_databases.append(db)
        facade = SimpleNamespace(db=db, run=copy.deepcopy(self.state.run),
            reserve_weight=MagicMock(side_effect=RuntimeError("serving_acquisition_forbidden")),
            snapshot=MagicMock(side_effect=AssertionError("serving snapshot forbidden")),
            check_budget=MagicMock(), entity=MagicMock())
        return facade

    def test_effective_e_root_scopes_models_environment_and_runtime_stays_separate(self):
        actual = load()
        self.assertEqual(actual["paths"]["new_model_root"], "E:\\modelmadness")
        adapter = OllamaAdapter(actual, self.state, self.model)
        expected = Path("E:\\modelmadness") / "ollama" / self.state.run["id"] / "models"
        self.assertEqual(adapter.models_root, expected)
        self.assertEqual(adapter._environment()["OLLAMA_MODELS"], str(expected))
        self.assertTrue(adapter.root.is_relative_to(Path(actual["paths"]["runtime_root"])))
        self.assertFalse(expected.is_relative_to(adapter.root))
        self.assertIsNone(self.state.run["started"])

    def test_cross_volume_copy_and_partial_use_model_root_and_full_cumulative_reservation(self):
        adapter = self.adapter()
        legacy = adapter.root / "models" / "blobs" / "existing-owned-legacy"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"preserve old owned paths")
        target = self.blob(adapter)
        target.parent.mkdir(parents=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(self.payload[:19])
        with patch("localbench.adapters.ollama.os.link", side_effect=OSError("mock cross-volume")):
            self.assertEqual(adapter._prepare_blob(), target)
        row = self.state.db.execute("SELECT * FROM weights").fetchone()
        self.assertEqual((row["bytes"], row["path"], row["acquired"]), (len(self.payload), str(target), 1))
        self.assertEqual(row["purpose"], "Ollama physical import copy")
        self.assertTrue(target.is_relative_to(Path(self.config["paths"]["new_model_root"])))
        self.disk_check.assert_called_once_with(self.config, len(self.payload) - 19, unittest.mock.ANY, destination=target)
        self.assertFalse(partial.exists())
        self.assertEqual(self.source.read_bytes(), self.payload)
        self.assertEqual(legacy.read_bytes(), b"preserve old owned paths")
        adapter._prepare_blob()
        self.assertEqual(self.state.db.execute("SELECT SUM(bytes) FROM weights").fetchone()[0], len(self.payload))
        self.assertEqual(adapter._metadata["weight_import"]["additional_physical_weight_bytes"], 0)
        self.assertEqual(adapter._metadata["weight_import"]["cumulative_reserved_weight_bytes"], len(self.payload))

    def test_copy_cap_is_enforced_before_writing_and_deletion_does_not_refund(self):
        self.config["limits"]["new_model_weight_bytes"] = len(self.payload) - 1
        adapter = self.adapter()
        with patch("localbench.adapters.ollama.os.link", side_effect=OSError("mock cross-volume")):
            with self.assertRaisesRegex(RuntimeError, "weight_budget_exhausted"):
                adapter._prepare_blob()
        self.assertFalse(self.blob(adapter).exists())
        self.assertFalse(self.blob(adapter).with_name(self.blob(adapter).name + ".part").exists())
        self.config["limits"]["new_model_weight_bytes"] = len(self.payload)
        adapter, target = self.copy_blob()
        target.unlink()
        self.assertEqual(self.state.db.execute("SELECT SUM(bytes) FROM weights").fetchone()[0], len(self.payload))

    def test_destination_headroom_failure_preserves_full_reservation_and_writes_no_partial(self):
        self.disk_check.side_effect = RuntimeError("disk_headroom_e")
        adapter = self.adapter()
        with patch("localbench.adapters.ollama.os.link", side_effect=OSError("mock cross-volume")):
            with self.assertRaisesRegex(RuntimeError, "disk_headroom_e"):
                adapter._prepare_blob()
        target = self.blob(adapter)
        self.assertEqual(self.disk_check.call_args.kwargs["destination"], target)
        self.assertEqual(self.state.db.execute("SELECT SUM(bytes) FROM weights").fetchone()[0], len(self.payload))
        self.assertFalse(target.exists())
        self.assertFalse(target.with_name(target.name + ".part").exists())

    def test_exact_acquired_copy_reused_through_readonly_database_without_reservation_or_update(self):
        _, target = self.copy_blob()
        facade = self.readonly_serving()
        adapter = self.adapter(facade)
        self.assertEqual(adapter._prepare_blob(), target)
        facade.reserve_weight.assert_not_called()
        facade.snapshot.assert_not_called()
        item = facade.entity.call_args.args[2]
        self.assertEqual(item["additional_physical_weight_bytes"], 0)
        self.assertEqual(item["cumulative_reserved_weight_bytes"], len(self.payload))
        self.assertEqual(self.source.read_bytes(), self.payload)

    def test_missing_or_unacquired_copy_receipt_cannot_authorize_readonly_serving(self):
        _, target = self.copy_blob()
        reservation = "ollama-import-copy-" + self.model["sha256"]
        for missing in (False, True):
            with self.subTest(missing=missing):
                with self.state.db:
                    if missing:
                        self.state.db.execute("DELETE FROM weights WHERE id=?", (reservation,))
                    else:
                        self.state.db.execute("UPDATE weights SET acquired=0 WHERE id=?", (reservation,))
                facade = self.readonly_serving()
                with self.assertRaisesRegex(RuntimeError, "serving_acquisition_forbidden"):
                    self.adapter(facade)._prepare_blob()
                self.assertEqual(target.read_bytes(), self.payload)
                facade.snapshot.assert_not_called()

    def test_mismatched_copy_ledger_or_corrupt_blob_cannot_authorize_serving(self):
        _, target = self.copy_blob()
        reservation = "ollama-import-copy-" + self.model["sha256"]
        for column, value in (("bytes", len(self.payload) + 1), ("purpose", "wrong-purpose"), ("path", "wrong-owned-path")):
            with self.subTest(column=column):
                original = dict(self.state.db.execute("SELECT * FROM weights WHERE id=?", (reservation,)).fetchone())
                with self.state.db:
                    self.state.db.execute("UPDATE weights SET " + column + "=? WHERE id=?", (value, reservation))
                facade = self.readonly_serving()
                with self.assertRaisesRegex(RuntimeError, "ollama_import_copy_ledger_mismatch"):
                    self.adapter(facade)._prepare_blob()
                facade.reserve_weight.assert_not_called()
                with self.state.db:
                    self.state.db.execute("UPDATE weights SET " + column + "=? WHERE id=?", (original[column], reservation))
        target.write_bytes(b"x" * len(self.payload))
        with self.assertRaisesRegex(RuntimeError, "private_ollama_blob_corrupt"):
            self.adapter(self.readonly_serving())._prepare_blob()

    def test_saved_legacy_owned_handle_keeps_literal_f_models_path_for_recovery(self):
        from localbench.config import digest
        old_models = str(Path(self.config["paths"]["runtime_root"]) / "ollama" / self.state.run["id"] / "models")
        argv = [str(self.binary), "serve"]
        saved = {"owner": "localbench", "run_id": self.state.run["id"], "pid": 4321, "created": 12345,
            "argv": argv, "command_hash": digest(argv), "endpoint": "http://127.0.0.1:38207",
            "environment_overrides": {"OLLAMA_MODELS": old_models}}
        before = json.dumps(saved, sort_keys=True)
        _validate_process_handle(self.config, self.state, Path(self.config["paths"]["state"]) / "owned-38207.json", saved)
        self.assertEqual(json.dumps(saved, sort_keys=True), before)

    def test_nonregular_blob_is_rejected_without_new_weight_authority(self):
        adapter = self.adapter()
        target = self.blob(adapter)
        target.mkdir(parents=True)
        with self.assertRaisesRegex(RuntimeError, "blob_is_not_a_regular_file"):
            adapter._prepare_blob()
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM weights").fetchone()[0], 0)

    def test_redirected_owned_weight_path_is_rejected_before_link_or_copy(self):
        adapter = self.adapter()
        target = self.blob(adapter)
        with patch.object(Path, "is_junction", side_effect=lambda: True):
            with self.assertRaisesRegex(RuntimeError, "ordinary_owned_model_storage"):
                adapter._prepare_blob()
        self.assertFalse(target.exists())
        self.assertEqual(self.state.db.execute("SELECT COUNT(*) FROM weights").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()

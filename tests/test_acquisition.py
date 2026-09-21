from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import http.client
import http.server
import io
import json
import tempfile
import threading
import unittest
import zipfile

from localbench.acquisition import (acquire_gguf, acquire_upstream, acquire_hf_subset, download,
                                    plan_hf_subset, safe_extract)
from localbench.config import atomic_json
from localbench.state import State


class FakeHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        server = self.server
        server.requests.append(dict(self.headers))
        offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
        status = 206 if offset else 200
        if server.mode == "ignore_range":
            offset, status = 0, 200
        body = server.body[offset:]
        self.send_response(status)
        if status == 206:
            first = offset + 1 if server.mode == "bad_range" else offset
            self.send_header("Content-Range", "bytes " + str(first) + "-" + str(len(server.body) - 1) + "/" + str(len(server.body)))
        length = len(body) + (1 if server.mode == "bad_length" else 0)
        self.send_header("Content-Length", str(length))
        self.end_headers()
        if server.mode == "interrupt_once" and len(server.requests) == 1:
            self.wfile.write(body[:11])
            self.wfile.flush()
            self.close_connection = True
        else:
            self.wfile.write(body)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = 1000
        self.config = {
            "_spec_hash": "temporary-acquisition-test",
            "paths": {"state": str(self.root / "state"), "artifacts": str(self.root / "artifacts"),
                      "runtime_root": str(self.root / "runtime"), "new_model_root": str(self.root / "models"),
                      "installed_model_root": str(self.root / "installed")},
            "limits": {"benchmark_elapsed_seconds": 604800, "new_model_weight_bytes": 300000000000,
                       "minimum_free_c_bytes": 50, "minimum_free_f_bytes": 64,
                       "runtime_and_build_soft_cap_bytes": 1000000000, "retry_transient_attempts": 2},
            "acquisition_disk_roots": {"C": "test-c", "F": "test-f"},
        }
        self.state = State(self.config, clock=lambda: self.now)
        self.state.control("harness_verified", True)
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHandler)
        self.server.body = b"deterministic fake model bytes" * 4096
        self.server.mode = "normal"
        self.server.requests = []
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.path = self.root / "models" / "fake.gguf"
        self.artifact = {"artifact_id": "test-weight", "url": "https://huggingface.co/testing/model/resolve/" + "a" * 40 + "/fake.gguf",
                         "bytes": len(self.server.body), "sha256": hashlib.sha256(self.server.body).hexdigest()}

    def tearDown(self):
        self.state.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.tmp.cleanup()

    def disk(self, root):
        return SimpleNamespace(free=1000000000)

    def opener(self, request, timeout):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=timeout)
        connection.request("GET", "/fake", headers=dict(request.header_items()))
        response = connection.getresponse()
        original = response.close
        def close():
            original()
            connection.close()
        response.close = close
        return response

    def transfer(self, **options):
        return download(self.config, self.state, self.artifact, self.path,
                        weight=True, opener=self.opener, disk_usage=self.disk, **options)

    def partial(self, data):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.with_name(self.path.name + ".part").write_bytes(data)
        atomic_json(self.path.with_name(self.path.name + ".part.json"),
                    {k: self.artifact.get(k) for k in ("bytes", "sha256", "git_blob_sha1", "url")})

    def test_interrupted_http_download_resumes_and_reserves_once(self):
        self.server.mode = "interrupt_once"
        self.assertEqual(self.transfer().read_bytes(), self.server.body)
        self.assertEqual(len(self.server.requests), 2)
        self.assertEqual(self.server.requests[1]["Range"], "bytes=11-")
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], self.artifact["bytes"])
        row = self.state.db.execute("SELECT acquired FROM weights").fetchone()
        self.assertEqual(row[0], 1)
        self.assertEqual(self.state.run["started"], self.now)
        self.assertEqual(self.state.run["deadline"], self.now + 604800)

    def test_server_ignoring_range_restarts_same_reservation(self):
        self.partial(self.server.body[:1024])
        self.state.reserve_weight("test-weight", self.artifact["bytes"], "download", str(self.path))
        self.server.mode = "ignore_range"
        self.transfer()
        self.assertEqual(self.path.read_bytes(), self.server.body)
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], self.artifact["bytes"])

    def test_malformed_content_range_rejected_without_touching_partial(self):
        self.partial(self.server.body[:21])
        self.server.mode = "bad_range"
        with self.assertRaisesRegex(RuntimeError, "invalid_content_range"):
            self.transfer()
        self.assertEqual(self.path.with_name(self.path.name + ".part").read_bytes(), self.server.body[:21])
        self.assertFalse(self.path.exists())

    def test_wrong_content_length_rejected(self):
        self.server.mode = "bad_length"
        with self.assertRaisesRegex(RuntimeError, "content_length"):
            self.transfer()
        self.assertFalse(self.path.exists())

    def test_corrupt_complete_partial_deleted_without_budget_refund(self):
        self.partial(b"x" * self.artifact["bytes"])
        with self.assertRaisesRegex(RuntimeError, "hash_mismatch"):
            self.transfer()
        self.assertEqual(self.server.requests, [])
        self.assertIsNone(self.state.run["started"])
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], self.artifact["bytes"])
        self.assertFalse(self.path.with_name(self.path.name + ".part").exists())

    def test_complete_verified_partial_promoted_without_network(self):
        self.partial(self.server.body)
        self.transfer()
        self.assertEqual(self.path.read_bytes(), self.server.body)
        self.assertEqual(self.server.requests, [])
        self.assertIsNone(self.state.run["started"])

    def test_existing_completed_corruption_preserved(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"personal existing file")
        with self.assertRaisesRegex(RuntimeError, "completed_artifact"):
            self.transfer()
        self.assertEqual(self.path.read_bytes(), b"personal existing file")
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)

    def test_partial_with_wrong_identity_rejected(self):
        self.partial(self.server.body[:20])
        sidecar = self.path.with_name(self.path.name + ".part.json")
        atomic_json(sidecar, {"sha256": "b" * 64})
        with self.assertRaisesRegex(RuntimeError, "identity_mismatch"):
            self.transfer()
        self.assertEqual(self.server.requests, [])

    def test_verified_harness_required_before_first_network(self):
        self.state.control("harness_verified", False)
        with self.assertRaisesRegex(RuntimeError, "verified harness"):
            self.transfer()
        self.assertEqual(self.server.requests, [])
        self.assertIsNone(self.state.run["started"])

    def test_disk_limits_checked_without_starting_clock(self):
        with self.assertRaisesRegex(RuntimeError, "disk_headroom_c"):
            download(self.config, self.state, self.artifact, self.path, weight=True,
                     opener=self.opener, disk_usage=lambda root: SimpleNamespace(free=49 if root == "test-c" else 1000000000))
        self.assertEqual(self.server.requests, [])
        self.assertIsNone(self.state.run["started"])

    def test_cumulative_weight_budget_not_refunded_by_deletion(self):
        self.config["limits"]["new_model_weight_bytes"] = self.artifact["bytes"] + 1
        self.transfer()
        self.path.unlink()
        other = dict(self.artifact, artifact_id="another-weight")
        with self.assertRaisesRegex(RuntimeError, "weight_budget_exhausted"):
            download(self.config, self.state, other, self.path.with_name("another.gguf"),
                     weight=True, opener=self.opener, disk_usage=self.disk)
        self.assertEqual(len(self.server.requests), 1)

    def test_per_chunk_deadline_check_preserves_resumable_partial(self):
        self.server.body = b"z" * (3 * 1024 * 1024)
        self.artifact.update(bytes=len(self.server.body), sha256=hashlib.sha256(self.server.body).hexdigest())
        check = self.state.check_budget
        calls = 0
        def expire(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                self.now += 604800
            return check(*args, **kwargs)
        self.state.check_budget = expire
        with self.assertRaisesRegex(RuntimeError, "budget_exhausted"):
            self.transfer()
        self.assertEqual(self.path.with_name(self.path.name + ".part").stat().st_size, 1024 * 1024)
        self.assertFalse(self.path.exists())

    def test_credentials_and_mutable_urls_rejected_without_network(self):
        for url in ("http://huggingface.co/model", "https://user:secret@huggingface.co/model",
                    "https://huggingface.co/model?token=secret", "https://127.0.0.1/model",
                    "https://example.com/model"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    download(self.config, self.state, dict(self.artifact, url=url), self.path,
                             weight=True, opener=self.opener, disk_usage=self.disk)
        self.assertEqual(self.server.requests, [])

    def test_installed_gguf_and_exact_hf_snapshot_reused_in_place(self):
        candidate = {"id": "fake-model", "repo": "testing/model", "revision": "a" * 40,
                     "filename": "fake.gguf", "bytes": self.artifact["bytes"], "sha256": self.artifact["sha256"]}
        installed = self.root / "installed" / "fake.gguf"
        installed.parent.mkdir()
        installed.write_bytes(self.server.body)
        result = acquire_gguf(self.config, self.state, candidate)
        self.assertEqual(result["path"], str(installed))
        self.assertEqual(result["reuse_new_bytes"], 0)
        installed.unlink()
        cached = self.root / "cache" / ("a" * 40) / "fake.gguf"
        cached.parent.mkdir(parents=True)
        cached.write_bytes(self.server.body)
        atomic_json(self.root / "artifacts" / "inventory.json",
                    {"hf_snapshots": [{"snapshot": str(cached.parent), "files": [{"path": str(cached)}]}]})
        self.assertEqual(acquire_gguf(self.config, self.state, candidate)["path"], str(cached))
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)
        self.assertIsNone(self.state.run["started"])

    def test_gguf_new_download_pinned_and_recorded(self):
        candidate = {"id": "fake-model", "repo": "testing/model", "revision": "a" * 40,
                     "filename": "fake.gguf", "bytes": self.artifact["bytes"], "sha256": self.artifact["sha256"]}
        real_download = download
        def fake_download(config, state, artifact, path, **options):
            return real_download(config, state, artifact, path, opener=self.opener, disk_usage=self.disk, **options)
        with patch("localbench.acquisition.download", side_effect=fake_download):
            result = acquire_gguf(self.config, self.state, candidate)
        self.assertEqual(result["reuse_new_bytes"], self.artifact["bytes"])
        self.assertEqual(Path(result["path"]).read_bytes(), self.server.body)
        self.assertIn("a" * 40, result["path"])

    def make_zip(self, names):
        archive = self.root / "test.zip"
        with zipfile.ZipFile(archive, "w") as output:
            for name, value in names:
                entry = zipfile.ZipInfo()
                entry.filename = name
                output.writestr(entry, value)
        return archive

    def test_safe_zip_extract_does_not_execute_and_rejects_escapes(self):
        destination = self.root / "runtime" / "extract"
        archive = self.make_zip([("bin/llama-server.exe", b"not actually executable"), ("install.bat", b"exit 99")])
        manifest = safe_extract(self.config, self.state, archive, destination, disk_usage=self.disk)
        self.assertEqual(len(manifest), 2)
        self.assertEqual((destination / "install.bat").read_bytes(), b"exit 99")
        for name in ("../escape", "/escape", "C:/escape", "bin/../../escape", "bin\\escape", "CON", "trailing."):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    safe_extract(self.config, self.state, self.make_zip([(name, b"x")]), destination, disk_usage=self.disk)
        self.assertIsNone(self.state.run["started"])

    def test_zip_symlink_and_duplicate_casefold_destinations_rejected(self):
        archive = self.root / "links.zip"
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(info, "../escape")
        with self.assertRaisesRegex(ValueError, "links"):
            safe_extract(self.config, self.state, archive, self.root / "runtime", disk_usage=self.disk)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            safe_extract(self.config, self.state, self.make_zip([("Same.dll", b"x"), ("same.dll", b"y")]),
                         self.root / "runtime", disk_usage=self.disk)

    def test_runtime_soft_cap_separate_from_weight_ledger(self):
        self.config["limits"]["runtime_and_build_soft_cap_bytes"] = 1
        path = self.root / "runtime" / "fake.zip"
        with self.assertRaisesRegex(RuntimeError, "runtime_soft_cap"):
            download(self.config, self.state, self.artifact, path, opener=self.opener, disk_usage=self.disk)
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)
        self.assertEqual(self.server.requests, [])

    def test_upstream_rejects_unreviewed_pin_before_download(self):
        atomic_json(self.root / "artifacts" / "engine-discovery.json",
                    {"engines": [{"id": "upstream-llama-nightly", "pin": {"release_tag": "b11041", "commit": "a" * 40}}]})
        with self.assertRaisesRegex(ValueError, "pin changed"):
            acquire_upstream(self.config, self.state)
        self.assertEqual(self.server.requests, [])

    def test_hf_root_subset_excludes_original_and_remote_code(self):
        candidate = {"repo": "testing/model", "revision": "a" * 40,
                     "root_weight_files": ["model-00000.safetensors", "model-00001.safetensors"]}
        siblings = [{"rfilename": n, "size": 20, "blobId": "b" * 40}
                    for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "README.md", "LICENSE", "model.safetensors.index.json", "modeling_custom.py")]
        for n in (*candidate["root_weight_files"], "original/model.safetensors", "extra.gguf"):
            siblings.append({"rfilename": n, "size": 100, "lfs": {"sha256": "c" * 64, "size": 100}})
        metadata = {"sha": candidate["revision"], "siblings": siblings}
        index = {"weight_map": {"layer.0": "model-00000.safetensors", "layer.1": "model-00001.safetensors"}}
        result = plan_hf_subset(candidate, metadata, index)
        names = {entry["filename"] for entry in result["files"]}
        self.assertEqual(result["weight_bytes"], 200)
        self.assertNotIn("original/model.safetensors", names)
        self.assertNotIn("extra.gguf", names)
        self.assertNotIn("modeling_custom.py", names)
        self.assertIn("LICENSE", names)
        with self.assertRaisesRegex(ValueError, "immutable revision"):
            plan_hf_subset(candidate, dict(metadata, sha="d" * 40), index)
        with self.assertRaisesRegex(ValueError, "root weight list"):
            plan_hf_subset(candidate, metadata, {"weight_map": {"layer": "original/model.safetensors"}})



    def hf_fixture(self):
        candidate = {"id": "fake-hf", "repo": "testing/hf", "revision": "a" * 40,
                     "root_weight_files": ["model-00000.safetensors", "model-00001.safetensors"]}
        index = {"weight_map": {"a": "model-00000.safetensors", "b": "model-00001.safetensors"}}
        files = {"config.json": b"{}", "tokenizer.json": b"{}", "README.md": b"public model card",
                 "LICENSE": b"MIT", "model.safetensors.index.json": json.dumps(index).encode(),
                 "model-00000.safetensors": b"first fake shard" * 1000,
                 "model-00001.safetensors": b"second fake shard" * 1000}
        siblings = []
        for name, value in files.items():
            item = {"rfilename": name, "size": len(value)}
            if name.endswith(".safetensors"):
                item["lfs"] = {"sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
            else:
                item["blobId"] = hashlib.sha1(("blob " + str(len(value)) + "\0").encode() + value).hexdigest()
            siblings.append(item)
        return candidate, {"sha": candidate["revision"], "siblings": siblings}, index, files

    def subset_opener(self, metadata, files, callback=None):
        requested = []
        def open_response(request, timeout):
            filename = request.full_url.rsplit("/", 1)[-1]
            if "/api/models/" in request.full_url:
                filename = "api-metadata"
                body = json.dumps(metadata).encode()
            else:
                body = files[filename]
            requested.append(filename)
            if callback:
                callback(filename)
            stream = io.BytesIO(body)
            return SimpleNamespace(status=200, headers={"Content-Length": str(len(body))},
                                   read=stream.read, close=stream.close)
        return open_response, requested

    def cached_subset(self, candidate, files, omitted=()):
        root = self.root / "cache" / candidate["revision"]
        root.mkdir(parents=True)
        for name, value in files.items():
            if name not in omitted:
                (root / name).write_bytes(value)
        atomic_json(self.root / "artifacts" / "inventory.json",
                    {"hf_snapshots": [{"snapshot": str(root),
                                      "files": [{"path": str(root / name)} for name in files if name not in omitted]}]})
        return root

    def test_hf_complete_cached_snapshot_reused_with_all_hashes(self):
        candidate, metadata, index, files = self.hf_fixture()
        cached = self.cached_subset(candidate, files)
        opener, requested = self.subset_opener(metadata, files)
        result = acquire_hf_subset(self.config, self.state, candidate, metadata, index,
                                   opener=opener, disk_usage=self.disk)
        self.assertEqual(result["path"], str(cached))
        self.assertEqual(result["reuse_new_bytes"], 0)
        self.assertEqual(requested, [])
        self.assertIsNone(self.state.run["started"])
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)

    def test_hf_automatic_pinned_metadata_and_index_only_selected_files(self):
        candidate, metadata, index, files = self.hf_fixture()
        expected = sum(len(value) for name, value in files.items() if name.endswith(".safetensors"))
        def reserved_before_transfer(filename):
            if filename not in ("api-metadata", "model.safetensors.index.json"):
                self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], expected)
        opener, requested = self.subset_opener(metadata, files, reserved_before_transfer)
        result = acquire_hf_subset(self.config, self.state, candidate,
                                   opener=opener, disk_usage=self.disk)
        self.assertEqual(result["reuse_new_bytes"], expected)
        self.assertEqual(result["weight_bytes"], expected)
        self.assertEqual(requested[0:2], ["api-metadata", "model.safetensors.index.json"])
        self.assertEqual(requested.count("model.safetensors.index.json"), 1)
        self.assertTrue(all(Path(result["path"], name).read_bytes() == value for name, value in files.items()))
        self.assertTrue(all(entry["sha256"] == hashlib.sha256(files[entry["filename"]]).hexdigest()
                            for entry in result["files"]))

    def test_hf_partial_snapshot_hardlinks_have_zero_new_weight_bytes(self):
        candidate, metadata, index, files = self.hf_fixture()
        cached = self.cached_subset(candidate, files, omitted=("config.json",))
        opener, requested = self.subset_opener(metadata, files)
        result = acquire_hf_subset(self.config, self.state, candidate, metadata, index,
                                   opener=opener, disk_usage=self.disk)
        self.assertEqual(requested, ["config.json"])
        self.assertEqual(result["reuse_new_bytes"], 0)
        self.assertTrue(Path(result["path"], "model-00000.safetensors").samefile(cached / "model-00000.safetensors"))
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], 0)

    def test_hf_cross_volume_copy_reserved_before_any_bytes(self):
        candidate, metadata, index, files = self.hf_fixture()
        self.cached_subset(candidate, files, omitted=("README.md",))
        opener, requested = self.subset_opener(metadata, files)
        with patch("localbench.acquisition.os.link", side_effect=OSError("cross volume")):
            result = acquire_hf_subset(self.config, self.state, candidate, metadata, index,
                                       opener=opener, disk_usage=self.disk)
        expected = sum(len(value) for name, value in files.items() if name.endswith(".safetensors"))
        self.assertEqual(result["reuse_new_bytes"], expected)
        self.assertEqual(self.state.snapshot()["new_weight_bytes_reserved"], expected)
        self.assertEqual(requested, ["README.md"])
        self.assertEqual({row[0] for row in self.state.db.execute("SELECT purpose FROM weights")}, {"staging_copy"})
        self.assertEqual({row[0] for row in self.state.db.execute("SELECT acquired FROM weights")}, {1})
        self.assertTrue(all(Path(result["path"], name).read_bytes() == value for name, value in files.items()))

    def test_hf_insufficient_weight_budget_stops_before_artifact_transfers(self):
        candidate, metadata, index, files = self.hf_fixture()
        expected = sum(len(value) for name, value in files.items() if name.endswith(".safetensors"))
        self.config["limits"]["new_model_weight_bytes"] = expected - 1
        opener, requested = self.subset_opener(metadata, files)
        with self.assertRaisesRegex(RuntimeError, "weight_budget_exhausted"):
            acquire_hf_subset(self.config, self.state, candidate, metadata, index,
                              opener=opener, disk_usage=self.disk)
        self.assertEqual(requested, [])
        self.assertIsNone(self.state.run["started"])

if __name__ == "__main__":
    unittest.main()

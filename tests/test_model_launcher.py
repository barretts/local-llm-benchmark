"""CPU-only serving contracts; no weights, GPU probes, or benchmark state."""

import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from launcher import api
from localbench.state import gpu_lock


PROFILES = {
    "engine": {"path": "test-engine.exe", "commit": "test-commit"},
    "profiles": {
        "qwen": {"path": "test-qwen.gguf", "label": "Qwen", "contexts": [16384, 65536]},
        "oxcoder": {"path": "test-oxcoder.gguf", "label": "OxCoder", "contexts": [16384]},
    },
}


class ModelLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="model-launcher-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.runtime = self.root / "launcher" / "runtime"
        for name, value in (("ROOT", self.root), ("RUNTIME", self.runtime),
                            ("STATUS", self.runtime / "api-status.json")):
            patch = mock.patch.object(api, name, value)
            patch.start()
            self.addCleanup(patch.stop)
        patch = mock.patch.object(api, "settings", return_value=PROFILES)
        patch.start()
        self.addCleanup(patch.stop)

    def saved_status(self, **changes):
        saved = {
            "owner": "local-model-launcher", "session": "a" * 32,
            "controller_pid": 12345, "controller_created": 123456789,
            "profile": "qwen", "label": "Qwen", "context": 16384,
            "port": 8080, "status": "ready", "model": "local-coder",
            "api_base": "http://127.0.0.1:8080/v1",
        }
        saved.update(changes)
        api.write_json(api.STATUS, saved)
        return saved

    def process_for(self, saved):
        process = mock.Mock(spec=api.WindowsProcess)
        process.alive.return_value = True
        process.created.return_value = saved["controller_created"]
        process.argv.return_value = api.controller_argv(
            saved["profile"], saved["context"], saved["port"], saved["session"])
        return process

    def option(self, command, name):
        self.assertEqual(command.count(name), 1, name)
        index = command.index(name)
        self.assertLess(index + 1, len(command), name)
        return command[index + 1]

    def private_path(self, command, name):
        path = Path(self.option(command, name)).resolve()
        self.assertTrue(path.is_relative_to(self.runtime.resolve()), str(path))
        return path

    def make_private_aider(self):
        python = self.runtime / "aider-env" / "Scripts" / "python.exe"
        python.parent.mkdir(parents=True)
        python.touch()
        return python

    def test_foreign_loopback_listener_is_refused_and_remains_usable(self):
        with socket.socket() as listener:
            if os.name == "nt":
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(2)
            port = listener.getsockname()[1]
            with mock.patch.object(api, "stop") as stop, \
                    mock.patch.object(api.subprocess, "Popen") as spawn, \
                    mock.patch.object(api, "verify_files") as verify, \
                    mock.patch.object(api, "gpu_memory") as probe:
                with self.assertRaisesRegex(RuntimeError, "occupied"):
                    api.start("qwen", 16384, port)
                stop.assert_not_called()
                spawn.assert_not_called()
                verify.assert_not_called()
                probe.assert_not_called()
            with socket.create_connection(("127.0.0.1", port), timeout=2) as client:
                with listener.accept()[0] as accepted:
                    client.sendall(b"foreign listener survived")
                    self.assertEqual(accepted.recv(100), b"foreign listener survived")
        self.assertFalse(self.runtime.exists())
        self.assertFalse((self.root / "state").exists())

    def test_benchmark_gpu_lease_prevents_launch_without_opening_database(self):
        config = {"paths": {"state": str(self.root / "state")}}
        with gpu_lock(config):
            with mock.patch.object(api, "port_available"), \
                    mock.patch.object(api, "stop") as stop, \
                    mock.patch.object(api.subprocess, "Popen") as spawn, \
                    mock.patch.object(api, "verify_files") as verify, \
                    mock.patch.object(api, "gpu_memory") as probe:
                with self.assertRaisesRegex(RuntimeError, "GPU lock"):
                    api.start("qwen", 16384, 8080)
                stop.assert_not_called()
                spawn.assert_not_called()
                verify.assert_not_called()
                probe.assert_not_called()
        # Contention must leave the existing lease releasable and reusable.
        with gpu_lock(config):
            pass
        self.assertEqual({path.name for path in (self.root / "state").iterdir()}, {"gpu.lock"})
        self.assertFalse(self.runtime.exists())

    def test_unsupported_context_is_rejected_before_inspecting_or_stopping_model(self):
        saved = self.saved_status()
        with mock.patch.object(api, "WindowsProcess") as process, \
                mock.patch.object(api, "StopEvent") as event, \
                mock.patch.object(api, "stop") as stop, \
                mock.patch.object(api, "port_available") as port, \
                mock.patch.object(api.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(ValueError, "unsupported context"):
                api.start("oxcoder", 65536, 8080)
            for operation in (process, event, stop, port, spawn):
                operation.assert_not_called()
        self.assertEqual(api.read_json(api.STATUS), saved)

    def test_server_argv_keeps_loopback_full_precision_single_slot_and_context_boundaries(self):
        for profile, context in (("qwen", 16384), ("qwen", 65536), ("oxcoder", 16384)):
            with self.subTest(profile=profile, context=context):
                command = api.argv_for(profile, context, 8081)
                self.assertEqual(self.option(command, "--host"), "127.0.0.1")
                self.assertEqual(self.option(command, "--port"), "8081")
                self.assertEqual(self.option(command, "--alias"), "local-coder")
                self.assertEqual(self.option(command, "--ctx-size"), str(context))
                self.assertEqual(self.option(command, "--parallel"), "1")
                self.assertEqual(self.option(command, "--cache-type-k"), "f16")
                self.assertEqual(self.option(command, "--cache-type-v"), "f16")
                self.assertIn("--no-context-shift", command)

    def test_invalid_ports_cannot_reach_process_control(self):
        for port in (0, 1023, 65536):
            with self.subTest(port=port), \
                    mock.patch.object(api, "owned_status") as owned, \
                    mock.patch.object(api, "stop") as stop, \
                    mock.patch.object(api.subprocess, "Popen") as spawn:
                with self.assertRaisesRegex(ValueError, "port"):
                    api.start("qwen", 16384, port)
                owned.assert_not_called()
                stop.assert_not_called()
                spawn.assert_not_called()

    def test_owned_status_accepts_matching_birth_and_exact_controller_argv(self):
        saved = self.saved_status()
        process = self.process_for(saved)
        with mock.patch.object(api, "WindowsProcess", return_value=process) as factory:
            self.assertEqual(api.owned_status(), saved)
        factory.assert_called_once_with(saved["controller_pid"])
        process.created.assert_called_once_with()
        process.argv.assert_called_once_with()
        process.close.assert_called_once_with()
        process.terminate.assert_not_called()

    def test_reused_pid_birth_mismatch_cannot_open_or_signal_stop_event(self):
        saved = self.saved_status()
        process = self.process_for(saved)
        process.created.return_value += 1
        with mock.patch.object(api, "WindowsProcess", return_value=process), \
                mock.patch.object(api, "StopEvent") as event:
            with self.assertRaisesRegex(RuntimeError, "identity"):
                api.stop()
            event.assert_not_called()
        process.close.assert_called_once_with()
        process.terminate.assert_not_called()

    def test_argv_mismatches_cannot_open_or_signal_stop_event(self):
        saved = self.saved_status()
        expected = api.controller_argv("qwen", 16384, 8080, saved["session"])
        mutations = [expected + ["--extra"], expected[:-1] + ["b" * 32],
                     ["foreign-python.exe", *expected[1:]],
                     [*expected[:3], "oxcoder", *expected[4:]]]
        for argv in mutations:
            with self.subTest(argv=argv):
                process = self.process_for(saved)
                process.argv.return_value = argv
                with mock.patch.object(api, "WindowsProcess", return_value=process), \
                        mock.patch.object(api, "StopEvent") as event:
                    with self.assertRaisesRegex(RuntimeError, "identity"):
                        api.stop()
                    event.assert_not_called()
                process.close.assert_called_once_with()
                process.terminate.assert_not_called()

    def test_dead_saved_controller_cannot_open_stop_event(self):
        saved = self.saved_status()
        process = self.process_for(saved)
        process.alive.return_value = False
        with mock.patch.object(api, "WindowsProcess", return_value=process), \
                mock.patch.object(api, "StopEvent") as event:
            api.stop()
            event.assert_not_called()
        process.created.assert_not_called()
        process.argv.assert_not_called()
        process.close.assert_called_once_with()
        process.terminate.assert_not_called()

    def test_foreign_owner_record_cannot_inspect_or_stop_a_process(self):
        self.saved_status(owner="some-other-application")
        with mock.patch.object(api, "WindowsProcess") as process, \
                mock.patch.object(api, "StopEvent") as event:
            api.stop()
            process.assert_not_called()
            event.assert_not_called()

    def test_verified_stop_signals_event_only_after_birth_and_argv_checks(self):
        saved = self.saved_status()
        controller = self.process_for(saved)
        waited = mock.Mock(spec=api.WindowsProcess)
        waited.alive.return_value = False
        trace = []
        controller.created.side_effect = lambda: (trace.append("birth"), saved["controller_created"])[1]
        command = controller.argv.return_value
        controller.argv.side_effect = lambda: (trace.append("argv"), command)[1]
        event = mock.Mock(spec=api.StopEvent)
        event.set.side_effect = lambda: trace.append("signal")

        def open_event(session):
            trace.append("event")
            self.assertEqual(session, saved["session"])
            return event

        with mock.patch.object(api, "WindowsProcess", side_effect=[controller, waited]), \
                mock.patch.object(api, "StopEvent", side_effect=open_event) as factory:
            api.stop()
        factory.assert_called_once_with(saved["session"])
        self.assertLess(trace.index("birth"), trace.index("event"))
        self.assertLess(trace.index("argv"), trace.index("event"))
        self.assertLess(trace.index("event"), trace.index("signal"))
        event.set.assert_called_once_with()
        event.close.assert_called_once_with()
        controller.close.assert_called_once_with()
        waited.close.assert_called_once_with()
        controller.terminate.assert_not_called()
        waited.terminate.assert_not_called()

    def test_aider_requires_private_environment_without_falling_back_to_global_python(self):
        repo = self.root / "project"
        repo.mkdir()
        saved = self.saved_status()
        with mock.patch.object(api, "write_json") as write:
            with self.assertRaisesRegex(RuntimeError, "private Aider"):
                api.aider_command(repo, saved)
            write.assert_not_called()
        self.assertFalse((self.runtime / "aider-home").exists())

    def test_aider_metadata_reserves_output_inside_each_context_budget(self):
        self.make_private_aider()
        repo = self.root / "project"
        repo.mkdir()
        for context in (16384, 65536):
            with self.subTest(context=context):
                saved = {"context": context, "api_base": "http://127.0.0.1:8080/v1"}
                command, environment = api.aider_command(repo, saved)
                metadata_path = self.private_path(command, "--model-metadata-file")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                self.assertEqual(set(metadata), {"openai/local-coder"})
                limits = metadata["openai/local-coder"]
                self.assertEqual(limits["max_input_tokens"], context - 4096)
                self.assertEqual(limits["max_output_tokens"], 4096)
                self.assertEqual(limits["max_tokens"], 4096)
                self.assertEqual(limits["max_input_tokens"] + limits["max_output_tokens"], context)
                self.assertEqual(limits["input_cost_per_token"], 0)
                self.assertEqual(limits["output_cost_per_token"], 0)
                settings_path = self.private_path(command, "--model-settings-file")
                model_settings = json.loads(settings_path.read_text(encoding="utf-8"))
                self.assertEqual(len(model_settings), 1)
                model = model_settings[0]
                self.assertEqual(model["name"], "openai/local-coder")
                self.assertEqual(model["weak_model_name"], model["name"])
                self.assertEqual(model["editor_model_name"], model["name"])
                self.assertEqual(model["extra_params"]["max_tokens"], 4096)
                self.assertEqual(environment["LITELLM_LOCAL_MODEL_COST_MAP"], "True")

    def test_aider_configuration_history_and_caches_stay_private(self):
        python = self.make_private_aider()
        repo = self.root / "project"
        repo.mkdir()
        (repo / ".git").mkdir()
        outside = self.root / "original-home"
        outside.mkdir()
        (outside / ".aider.conf.yml").write_text("original home config", encoding="utf-8")
        (repo / ".env").write_text("OPENAI_API_KEY=external-key", encoding="utf-8")
        saved = {"context": 16384, "api_base": "http://127.0.0.1:8080/v1"}
        with mock.patch.dict(os.environ, {"USERPROFILE": str(outside), "HOME": str(outside),
                                         "OPENAI_API_KEY": "external-key"}):
            command, environment = api.aider_command(repo, saved, files=("example.py",),
                                                     message="edit the example", yes=True)
            second, _ = api.aider_command(repo, saved)
        self.assertEqual(command[:3], [str(python), "-m", "aider"])
        for name in ("--model", "--weak-model", "--editor-model"):
            self.assertEqual(self.option(command, name), "openai/local-coder")
        self.assertEqual(self.option(command, "--openai-api-base"), saved["api_base"])
        self.assertEqual(self.option(command, "--openai-api-key"), "local-only")
        for flag in ("--no-auto-commits", "--no-dirty-commits", "--no-gitignore",
                     "--no-add-gitignore-files", "--no-analytics", "--no-check-update",
                     "--no-auto-test", "--no-auto-lint", "--no-detect-urls"):
            self.assertIn(flag, command)
        self.assertEqual(self.option(command, "--message"), "edit the example")
        self.assertIn("--exit", command)
        self.assertIn("--yes-always", command)
        self.assertEqual(command[-1], "example.py")
        for name in ("--config", "--model-settings-file", "--model-metadata-file",
                     "--input-history-file", "--chat-history-file"):
            self.private_path(command, name)
        config = self.private_path(command, "--config")
        self.assertEqual(config.read_text(encoding="utf-8"), "{}\n")
        self.assertNotEqual(self.option(command, "--chat-history-file"),
                            self.option(second, "--chat-history-file"))
        self.assertEqual(environment["PYTHON_DOTENV_DISABLED"], "1")
        self.assertEqual(environment["LITELLM_MODE"], "PRODUCTION")
        self.assertEqual(environment["NO_PROXY"], "127.0.0.1,localhost")
        self.assertEqual(environment["no_proxy"], environment["NO_PROXY"])
        for key in ("USERPROFILE", "TIKTOKEN_CACHE_DIR", "HF_HOME"):
            self.assertTrue(Path(environment[key]).resolve().is_relative_to(self.runtime.resolve()), key)
        self.assertNotIn("OPENAI_API_KEY", environment)
        self.assertEqual((outside / ".aider.conf.yml").read_text(encoding="utf-8"), "original home config")
        self.assertEqual({path.name for path in outside.iterdir()}, {".aider.conf.yml"})
        self.assertEqual((repo / ".env").read_text(encoding="utf-8"), "OPENAI_API_KEY=external-key")
        self.assertFalse((self.root / "state").exists())

    def test_child_environment_scrubs_credentials_and_execution_overrides(self):
        safe = {"PATH": "ordinary-binary-path", "SYSTEMROOT": "C:/Windows", "TEMP": "temporary-path"}
        forbidden = {
            "GITHUB_TOKEN": "credential", "AWS_SECRET_ACCESS_KEY": "credential",
            "DB_PASSWORD": "credential", "CUSTOM_API_KEY": "credential",
            "OTHER_APIKEY": "credential", "AIDER_AUTO_COMMITS": "1",
            "LLAMA_ARG_HOST": "0.0.0.0", "LITELLM_LOG": "DEBUG",
            "OPENAI_BASE_URL": "https://external.invalid", "PYTHONPATH": "foreign-modules",
            "PYTHONHOME": "foreign-python", "CUDA_VISIBLE_DEVICES": "1",
            "GGML_CUDA_FORCE_MMQ": "1",
        }
        with mock.patch.dict(os.environ, {**safe, **forbidden}, clear=True):
            environment = api.clean_environment()
        for key in forbidden:
            self.assertNotIn(key, environment, key)
        for key, value in safe.items():
            self.assertEqual(environment[key], value)


if __name__ == "__main__":
    unittest.main()

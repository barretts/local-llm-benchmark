"""CPU-only regression checks for filtered existing Windows compiler facts."""
import os
import unittest
from unittest.mock import patch

from localbench.runtime_build import _environment


class WindowsBuildEnvironmentTests(unittest.TestCase):
    def test_preserves_existing_os_paths_and_architecture_without_secrets(self):
        existing = {"SYSTEMDRIVE": "C:", "PROGRAMDATA": r"C:\ProgramData",
                    "ALLUSERSPROFILE": r"C:\ProgramData", "PROCESSOR_ARCHITECTURE": "AMD64",
                    "PROCESSOR_ARCHITEW6432": "AMD64", "SYSTEMROOT": r"C:\Windows",
                    "HF_TOKEN": "mock-credential", "LM_BENCH_TOKEN": "mock-credential",
                    "HTTPS_PROXY": "https://mock:credential@proxy.invalid",
                    "PYTHONPATH": "untrusted startup imports", "NODE_OPTIONS": "untrusted startup imports",
                    "GIT_CONFIG_GLOBAL": "personal git config"}
        with patch.dict(os.environ, existing, clear=True):
            env = _environment()
        for key in ("SYSTEMDRIVE", "PROGRAMDATA", "ALLUSERSPROFILE", "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432", "SYSTEMROOT"):
            self.assertEqual(env[key], existing[key])
        for key in ("HF_TOKEN", "LM_BENCH_TOKEN", "HTTPS_PROXY", "PYTHONPATH", "NODE_OPTIONS"):
            self.assertNotIn(key, env)
        self.assertEqual(env["GIT_CONFIG_GLOBAL"], os.devnull)
        self.assertEqual(env["GIT_CONFIG_SYSTEM"], os.devnull)

    def test_does_not_synthesize_missing_windows_facts(self):
        with patch.dict(os.environ, {}, clear=True):
            env = _environment()
        for key in ("SYSTEMDRIVE", "PROGRAMDATA", "ALLUSERSPROFILE", "PROCESSOR_ARCHITECTURE", "PROCESSOR_ARCHITEW6432"):
            self.assertNotIn(key, env)


if __name__ == "__main__":
    unittest.main()

"""Operational API/Aider smoke tests; not benchmark qualification or timing."""
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from launcher.api import RUNTIME, start, request, aider_command, write_json, clean_environment
from launcher.windows_job import OwnedJob

DOCKER = r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"
SOURCE = "def clamp(value, lower, upper):\n    return min(max(value, upper), lower)\n"
CHECKS = '''import unittest
from calc import clamp

class ClampTests(unittest.TestCase):
    def test_cases(self):
        for value, lower, upper, expected in [(-4, 0, 10, 0), (3, 0, 10, 3),
                (14, 0, 10, 10), (0, 0, 10, 0), (10, 0, 10, 10),
                (-5, -10, -2, -5), (-15, -10, -2, -10), (1, 2, 2, 2)]:
            with self.subTest(value=value, lower=lower, upper=upper):
                self.assertEqual(clamp(value, lower, upper), expected)
    def test_reversed_bounds(self):
        with self.assertRaises(ValueError):
            clamp(5, 10, 0)

if __name__ == "__main__":
    unittest.main(verbosity=2)
'''
MESSAGE = ("Fix calc.py only. clamp(value, lower, upper) must return value when in the "
           "inclusive bounds, lower when below, upper when above, and raise ValueError "
           "when lower > upper. Keep the existing function signature. Apply your edits.")


def run_checks(repo, grader, image):
    return subprocess.run([DOCKER, "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "64",
        "--memory", "256m", "--cpus", "1", "--user", "65534:65534",
        "--mount", f"type=bind,source={repo},target=/source,readonly",
        "--mount", f"type=bind,source={grader},target=/checks,readonly",
        "--workdir", "/checks", "--env", "PYTHONPATH=/source", "--env", "PYTHONDONTWRITEBYTECODE=1",
        image, "python", "-I", "-c",
        "import sys; sys.path.insert(0, '/source'); sys.path.insert(0, '/checks'); "
        "import unittest; suite=unittest.defaultTestLoader.loadTestsFromName('test_calc'); "
        "result=unittest.TextTestRunner(verbosity=2).run(suite); "
        "sys.exit(0 if result.wasSuccessful() and result.testsRun == 2 else 1)"],
        capture_output=True, text=True, timeout=60, env=clean_environment())


def main():
    profiles = sys.argv[1:] or ["oxcoder", "qwen"]
    image = subprocess.run([DOCKER, "image", "inspect", "python:3.13.15-slim-bookworm",
                            "--format", "{{.Id}}"], capture_output=True, text=True, check=True).stdout.strip()
    receipt = {"scope": "operational smoke only; no benchmark qualification", "image": image, "profiles": {}}
    for profile in profiles:
        saved = start(profile)
        response = request(8080, "/v1/chat/completions", {"model": "local-coder",
                           "messages": [{"role": "user", "content": "Reply with exactly API_READY."}],
                           "temperature": 0, "max_tokens": 512}, timeout=90)
        if response["choices"][0]["message"]["content"].strip() != "API_READY":
            raise RuntimeError("API response smoke failed")
        try:
            request(8080, "/v1/chat/completions", {"model": "local-coder", "messages": [
                {"role": "user", "content": " z" * 18000}], "max_tokens": 16}, timeout=30)
        except urllib.error.HTTPError as error:
            overflow = {"status": error.code, "body": error.read().decode()}
            if error.code != 400 or "context" not in overflow["body"].lower():
                raise RuntimeError("overflow did not fail with an explicit context error") from error
        else:
            raise RuntimeError("oversized prompt was not rejected")
        workspace = RUNTIME / ("smoke-" + profile + "-" + uuid.uuid4().hex)
        repo, grader = workspace / "repo", workspace / "checks"
        repo.mkdir(parents=True)
        grader.mkdir()
        (repo / "calc.py").write_text(SOURCE, encoding="utf-8")
        (grader / "test_calc.py").write_text(CHECKS, encoding="utf-8")
        baseline = run_checks(repo, grader, image)
        if baseline.returncode == 0:
            raise RuntimeError("planted bug did not fail isolated checks")
        command, environment = aider_command(repo, saved, ["calc.py"], MESSAGE, yes=True, no_git=True)
        # Own the venv redirector and actual interpreter together on timeout.
        # CreateProcess inherits our cwd; restore it immediately after creation.
        import os
        original_cwd = Path.cwd()
        with OwnedJob() as job, (workspace / "aider-output.txt").open("ab", buffering=0) as log:
            try:
                os.chdir(repo)
                job.spawn(command, env=environment, stdout=log)
            finally:
                os.chdir(original_cwd)
            deadline = time.monotonic() + 300
            while job.alive():
                if time.monotonic() > deadline:
                    raise TimeoutError("owned Aider edit exceeded 300 seconds")
                time.sleep(0.2)
            aider_exit = job.exit_code()
        repaired = run_checks(repo, grader, image)
        (workspace / "isolated-checks.txt").write_text(repaired.stdout + repaired.stderr, encoding="utf-8")
        outcome = {"context": saved["context"], "gpu_loaded": saved["gpu_loaded"],
                   "api_response": response, "overflow": overflow, "aider_exit": aider_exit,
                   "baseline_failed": baseline.returncode != 0, "isolated_tests_passed": repaired.returncode == 0,
                   "workspace": str(workspace), "after_source": (repo / "calc.py").read_text(),
                   "grader_unchanged": (grader / "test_calc.py").read_text() == CHECKS}
        receipt["profiles"][profile] = outcome
        write_json(RUNTIME / "smoke-tests.json", receipt)
        if aider_exit != 0 or repaired.returncode != 0 or not outcome["grader_unchanged"]:
            raise RuntimeError(f"{profile} Aider smoke failed; see {workspace}")
        print(f"PASS: {profile} API, overflow rejection and actual Aider repair with isolated tests.")
    print("Operational tests complete. Last model remains running on 127.0.0.1:8080.")


if __name__ == "__main__":
    main()

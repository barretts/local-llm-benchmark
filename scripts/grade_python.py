"""Trusted container-only unittest driver; source executes only inside Docker."""
import json
import sys
import unittest

sys.path.insert(0, "/workspace")


class Results(unittest.TextTestResult):
    def __init__(self, *args):
        super().__init__(*args)
        self.records = []
    def startTest(self, test):
        super().startTest(test)
        self.records.append({"id":test.id(),"status":"running"})
    def _record(self, test, status):
        for record in reversed(self.records):
            if record["id"] == test.id():
                record["status"] = status
                return
        self.records.append({"id":test.id(),"status":status})
    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self._record(test, "failed" if issubclass(err[0], test.failureException) else "error")
    def addSuccess(self, test):
        super().addSuccess(test); self._record(test, "passed")
    def addFailure(self, test, err):
        super().addFailure(test,err); self._record(test, "failed")
    def addError(self, test, err):
        super().addError(test,err); self._record(test, "error")
    def addSkip(self, test, reason):
        super().addSkip(test,reason); self._record(test, "skipped")
    def addExpectedFailure(self, test, err):
        super().addExpectedFailure(test,err); self._record(test, "expected_failure")
    def addUnexpectedSuccess(self, test):
        super().addUnexpectedSuccess(test); self._record(test, "unexpected_success")


suite = unittest.defaultTestLoader.discover("/tests", pattern="test_*.py")
expected = suite.countTestCases()
result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=Results).run(suite)
print("\nLOCALBENCH_TESTS_JSON:" + json.dumps({"tests_run":result.testsRun,"expected_discovered":expected,"records":result.records,"successful":result.wasSuccessful()}), flush=True)
sys.exit(0 if result.wasSuccessful() and expected>0 and len(result.records)==expected else 1)

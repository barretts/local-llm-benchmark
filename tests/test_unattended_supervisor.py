"""Mock-only safety and checkpoint continuation tests; never launch an engine."""
import copy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'unattended-supervisor.py'
spec = importlib.util.spec_from_file_location('unattended_supervisor', SCRIPT)
supervisor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(supervisor)


def utc(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).strftime('%Y-%m-%dT%H:%M:%S') + '.0000000Z'


class Handle:
    def __init__(self, pid, created, parent, argv):
        self.pid, self.created, self.parent_pid, self.argv = pid, created, parent, argv
        self.live, self.code, self.closed = True, 0, False

    def alive(self): return self.live
    def exit_code(self): return None if self.live else self.code
    def close(self): self.closed = True


class MockBackend:
    def __init__(self):
        self.handles = {}
        self.supervisor_created = 12345
        self.preflight = Mock()
        self.launch_helper = Mock(return_value=Mock(pid=301, poll=Mock(return_value=None)))
        self.inspected = []

    def inspect(self, pid):
        self.inspected.append(pid)
        handle = self.handles[pid]
        if not handle.live: raise RuntimeError('registered_process_not_live_or_unverifiable')
        return handle


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for directory in ('scripts', 'localbench', '.logs', 'artifacts', 'state', 'weights'):
            (self.root / directory).mkdir()
        (self.root / 'scripts' / 'credential-controller.ps1').write_text('reviewed helper', encoding='utf8')
        (self.root / 'localbench' / 'quality.py').write_text('reviewed module', encoding='utf8')
        self.project_patch = patch.object(supervisor, 'PROJECT', self.root)
        self.project_patch.start()
        self.config = {'_execution_hash': 'execution', 'limits': {'new_model_weight_bytes': 300000000000,
            'per_task_timeout_seconds': 900}, 'paths': {'new_model_root': str(self.root / 'weights'),
            'logs': str(self.root / '.logs'), 'artifacts': str(self.root / 'artifacts'), 'state': str(self.root / 'state')}}
        self.now = 1001.0
        self.backend = MockBackend()
        self.records = []
        self.snapshot = {'run': {'id': 'localbench-aaaaaaaaaaaa', 'spec_hash': 'frozen', 'created': 999,
            'started': 1000, 'deadline': 1000000, 'stop_requested': 0},
            'controls': {'doctor_verified': True, 'harness_verified': True, 'fixtures_verified': True,
                'active_job': {'kind': 'quality', 'fixture': 'py01'}, 'ownership_recovery': [], 'managed_load_preflight': []},
            'controller': {'pid': 101, 'command': 'resume', 'started': 1001, 'protocol_source_hashes': {}},
            'pending': 1, 'running': 0, 'weights': [], 'failures': [], 'last_attempt': 0, 'completed_attempts': 0,
            'progress': 'initial', 'execution_hash': 'execution'}
        self.set_worker(101, 201, 1000, 'resume')
        self.subject = supervisor.Supervisor(self.config, lambda: copy.deepcopy(self.snapshot), self.backend,
            lambda row, event: self.records.append((row, event)), clock=lambda: self.now)
        self.addCleanup(self.subject.close)
        self.addCleanup(self.project_patch.stop)
        self.addCleanup(self.temp.cleanup)

    def set_worker(self, pid, launcher, created, command='sweep'):
        argv = [str(self.root / '.venv' / 'Scripts' / 'python.exe'), '-X', 'utf8', '-m', 'localbench', command]
        base = supervisor.filetime_utc(utc(created))
        self.backend.handles[pid] = Handle(pid, base + 500000, launcher, argv.copy())
        self.backend.handles[launcher] = Handle(launcher, base, 901, argv.copy())
        self.snapshot['controller'] = {'pid': pid, 'command': command, 'started': created + 1, 'protocol_source_hashes': {}}
        out, err = self.root / '.logs' / f'worker-{pid}-out.log', self.root / '.logs' / f'worker-{pid}-err.log'
        out.write_text(json.dumps({'status': 'current_measurements_pending'}), encoding='utf8')
        err.write_text('', encoding='utf8')
        self.snapshot['helper'] = {'run_id': self.snapshot['run']['id'], 'status': 'launched', 'parent_pid': 901,
            'child_pid': launcher, 'child_created_utc': utc(created), 'stdout_log': str(out), 'stderr_log': str(err)}

    def adopt(self): self.subject.adopt(copy.deepcopy(self.snapshot), 101)

    def finish(self, status='current_measurements_pending', progress=None, code=0):
        Path(self.snapshot['helper']['stdout_log']).write_text(json.dumps({'status': status}), encoding='utf8')
        for handle in self.subject.watches: handle.live = False; handle.code = code
        if progress is not None:
            self.snapshot['progress'] = progress
            self.snapshot['last_attempt'] += 1
            self.snapshot['completed_attempts'] += 1
        self.subject.step()

    def launch_and_adopt(self, pid=102, launcher=202):
        self.now += 46
        self.subject.step()
        self.assertIsNotNone(self.subject.launching)
        self.set_worker(pid, launcher, int(self.now), 'sweep')
        self.subject.step()
        self.assertEqual(self.subject.last_worker_pid, pid)
        self.assertFalse(self.subject.done)

    def test_filetime_preserves_seventh_fractional_digit(self):
        self.assertEqual(supervisor.filetime_utc('2026-09-18T23:22:36.7014347Z'), 134342473567014347)

    def test_exact_launcher_adoption_holds_both_lifetimes(self):
        self.adopt()
        self.assertEqual([h.pid for h in self.subject.watches], [101, 201])
        row = self.records[-1][0]
        self.assertEqual(row['adoption_proof'], 'exact_helper_launcher_parent')
        self.assertEqual(row['launch_receipt']['child_pid'], 201)
        self.assertEqual(row['phase'], 'quality')
        self.assertEqual(row['new_weight_bytes_reserved'], 0)
        self.assertNotIn('argv', json.dumps(row))

    def test_initial_stop_rejects_before_process_inspection(self):
        self.snapshot['run']['stop_requested'] = 1
        with self.assertRaisesRegex(RuntimeError, '^stop_requested$'):
            self.subject.adopt(self.snapshot, 101)
        self.assertEqual(self.backend.inspected, [])
        self.backend.launch_helper.assert_not_called()

    def test_wrong_command_never_exports_arbitrary_secret_arguments(self):
        self.backend.handles[101].argv[-1] = 'private-secret-value'
        with self.assertRaisesRegex(RuntimeError, '^registered_command_identity_mismatch$'): self.adopt()
        self.assertTrue(self.backend.handles[101].closed)
        self.assertNotIn('private-secret-value', json.dumps(self.records))

    def test_unmatched_parent_cannot_borrow_stale_receipt_logs(self):
        self.backend.handles[101].parent_pid = 999
        with self.assertRaisesRegex(RuntimeError, '^helper_parent_identity_mismatch$'): self.adopt()
        self.assertTrue(self.backend.handles[101].closed)

    def test_kernel_creation_mismatch_is_not_same_pid_adoption(self):
        self.backend.handles[201].created += 1
        with self.assertRaisesRegex(RuntimeError, '^helper_parent_creation_receipt_mismatch$'): self.adopt()

    def test_live_worker_quiet_polls_do_not_count_as_failed_cycles(self):
        self.adopt()
        for _ in range(20): self.now += 45; self.subject.step()
        self.assertFalse(self.subject.done)
        self.assertEqual(self.subject.no_progress, 0)
        self.backend.launch_helper.assert_not_called()

    def test_waits_for_launcher_too_and_requires_real_zero_exit(self):
        self.adopt()
        self.subject.worker.live = False
        self.subject.step()
        self.assertIsNone(self.subject.next_launch)
        self.subject.watches[1].live = False
        self.subject.watches[1].code = 2
        self.subject.step()
        self.assertTrue(self.subject.done)
        self.assertEqual(self.records[-1][0]['reason'], 'unexpected_nonzero_exit')
        self.backend.launch_helper.assert_not_called()

    def test_two_pending_cycles_launch_twice_and_update_adopted_pid(self):
        self.adopt()
        self.finish(progress='completed-one')
        self.assertEqual(self.records[-1][0]['status'], 'pending_backoff')
        self.backend.launch_helper.assert_not_called()
        self.launch_and_adopt()
        self.finish(progress='completed-two')
        self.launch_and_adopt(103, 203)
        self.assertEqual(self.backend.launch_helper.call_count, 2)
        self.assertEqual(self.subject.last_worker_pid, 103)
        self.assertEqual(self.subject.no_progress, 0)

    def test_stop_during_helper_launch_is_preserved_without_recovery(self):
        self.adopt(); self.finish(progress='completed-one')
        def launch(_):
            self.snapshot['run']['stop_requested'] = 1
            return Mock(pid=301, poll=Mock(return_value=None))
        self.backend.launch_helper.side_effect = launch
        self.now += 46; self.subject.step(); self.subject.step()
        self.assertTrue(self.subject.done)
        self.assertEqual(self.snapshot['run']['stop_requested'], 1)
        self.assertEqual(self.records[-1][0]['reason'], 'stop_requested')
        helper_source = (SCRIPT.parent / 'credential-controller.ps1').read_text(encoding='utf8')
        self.assertIn("[ValidateSet('resume', 'sweep')]", helper_source)
        self.assertIn("'localbench', $WorkerCommand", helper_source)

    def test_stop_alive_only_observes_then_blocks_future_launch(self):
        self.adopt(); self.snapshot['run']['stop_requested'] = 1; self.subject.step()
        self.assertFalse(self.subject.done)
        self.assertEqual(self.records[-1][0]['status'], 'observing_stop_requested')
        for handle in self.subject.watches: handle.live = False
        self.subject.step(); self.assertTrue(self.subject.done)
        self.backend.launch_helper.assert_not_called()

    def test_unknown_review_and_unavailable_exit_do_not_retry(self):
        for outcome in ('pending_source_review', 'pending_unknown', 'unavailable'):
            with self.subTest(outcome=outcome):
                self.subject.done = False; self.subject.watches = []; self.set_worker(101, 201, 1000, 'resume')
                self.adopt(); self.finish(outcome)
                self.assertTrue(self.subject.done)
                self.backend.launch_helper.assert_not_called()

    def test_three_completed_cycles_without_progress_require_attention(self):
        self.adopt(); self.finish(); self.launch_and_adopt()
        self.finish(); self.launch_and_adopt(103, 203); self.finish()
        self.assertTrue(self.subject.done)
        self.assertEqual(self.records[-1][0]['reason'], 'no_progress_limit')
        self.assertEqual(self.backend.launch_helper.call_count, 2)

    def test_complete_requires_durable_coverage_and_never_launches_endpoint(self):
        self.adopt()
        self.snapshot['controls'].update(search_covered=True, remaining_search=[], unfinished_current_measurement_jobs=[])
        self.finish('covered_search_complete', 'finished')
        self.assertEqual(self.records[-1][0]['status'], 'complete_final_review_required')
        self.backend.launch_helper.assert_not_called()

    def test_running_or_unresolved_owned_load_blocks_continuation(self):
        self.adopt(); self.finish(progress='finished'); self.snapshot['running'] = 1
        self.now += 46; self.subject.step()
        self.assertEqual(self.records[-1][0]['reason'], 'running_measurements_require_manual_recovery')
        self.backend.launch_helper.assert_not_called()
        self.subject.done = False; self.snapshot['running'] = 0
        self.snapshot['controls']['ownership_recovery'] = [{'status': 'pending_load'}]
        self.subject.step()
        self.assertEqual(self.records[-1][0]['reason'], 'owned_recovery_unresolved')

    def test_new_safety_failure_or_original_run_mutation_blocks(self):
        self.adopt(); self.snapshot['failures'] = [(1, 'disk_headroom_e')]; self.finish(progress='finished')
        self.assertEqual(self.records[-1][0]['reason'], 'safety_outcome_requires_attention')
        self.subject.done = False; self.snapshot['run']['deadline'] += 1; self.subject.step()
        self.assertEqual(self.records[-1][0]['reason'], 'original_run_or_execution_identity_changed')

    def test_cumulative_ledger_and_new_weight_root_are_enforced(self):
        self.snapshot['weights'] = [{'id': 'new', 'bytes': 1, 'purpose': 'model', 'path': str(self.root / 'escaping'), 'acquired': 0}]
        self.subject.step()
        self.assertEqual(self.records[-1][0]['reason'], 'new_weight_destination_changed')
        self.subject.done = False
        self.snapshot['weights'] = [{'id': 'owned', 'bytes': 10, 'purpose': 'model', 'path': str(self.root / 'weights' / 'owned'), 'acquired': 0}]
        self.subject.check(self.snapshot)
        self.snapshot['weights'] = []
        with self.assertRaisesRegex(RuntimeError, 'cumulative_weight_ledger_changed'): self.subject.check(self.snapshot)

    def test_stale_managed_ownership_refusal_blocks_before_next_launch(self):
        self.adopt()
        self.snapshot['failures'] = [(1, 'stale_managed_instance_identity:refusing_unload')]
        self.finish(progress='finished')
        self.assertEqual(self.records[-1][0]['reason'], 'safety_outcome_requires_attention')
        self.backend.launch_helper.assert_not_called()

    def test_source_change_and_disk_preflight_failure_block(self):
        self.adopt(); self.finish(progress='finished'); self.now += 46
        (self.root / 'localbench' / 'quality.py').write_text('different', encoding='utf8')
        self.subject.step()
        self.assertEqual(self.records[-1][0]['reason'], 'worker_source_changed_requires_review')
        self.backend.launch_helper.assert_not_called()
        self.subject.done = False
        self.subject.source_hashes = {str(p.relative_to(self.root)): supervisor.file_hash(p) for p in (self.root / 'localbench').rglob('*.py')}
        self.backend.preflight.side_effect = RuntimeError('disk_headroom_e')
        self.subject.step()
        self.assertEqual(self.records[-1][0]['reason'], 'disk_headroom_e')
        self.backend.launch_helper.assert_not_called()

    def test_known_deadline_blocks_before_launch(self):
        self.adopt(); self.finish(progress='finished'); self.now = self.snapshot['run']['deadline'] - 8000
        self.subject.step()
        self.assertEqual(self.records[-1][0]['reason'], 'benchmark_budget_limit')
        self.backend.launch_helper.assert_not_called()

    def test_singleton_prevents_a_second_supervisor(self):
        lock = self.root / 'artifacts' / 'singleton.lock'
        with supervisor.Singleton(lock):
            with self.assertRaisesRegex(RuntimeError, 'supervisor_already_running'):
                with supervisor.Singleton(lock): pass
        with supervisor.Singleton(lock): pass

    def test_helper_launch_is_hidden_sweep_and_only_passes_authorized_path(self):
        ready = self.root / 'artifacts' / 'credential-controller-ready.json'
        ready.write_text(json.dumps({'run_id': self.snapshot['run']['id'], 'ready': True}), encoding='utf8')
        backend = object.__new__(supervisor.WindowsBackend)
        backend.config, backend.pwsh = self.config, 'approved-pwsh.exe'
        with patch.object(supervisor.subprocess, 'Popen') as popen, patch.object(supervisor.subprocess, 'CREATE_NO_WINDOW', 0x08000000, create=True):
            backend.launch_helper(self.snapshot['run']['id'])
        args, kwargs = popen.call_args
        command = args[0]
        self.assertEqual(command[command.index('-WorkerCommand') + 1], 'sweep')
        self.assertEqual(command[command.index('-CredentialFile') + 1], supervisor.CREDENTIAL_FILE)
        self.assertEqual(kwargs['creationflags'], 0x08000000)
        self.assertNotIn('env', kwargs)
        self.assertNotIn('private-secret', str(command))

    def test_snapshot_is_sqlite_readonly_and_compact(self):
        dbpath = self.root / 'state' / 'benchmark.sqlite3'
        db = sqlite3.connect(dbpath)
        db.executescript('CREATE TABLE runs(id TEXT,spec_hash TEXT,created REAL,started REAL,deadline REAL,stop_requested INTEGER);'
            'CREATE TABLE controls(key TEXT,value TEXT); CREATE TABLE entities(kind TEXT,id TEXT,data TEXT);'
            'CREATE TABLE jobs(status TEXT,key_json TEXT); CREATE TABLE attempts(id INTEGER,finished REAL,status TEXT,reason TEXT);'
            'CREATE TABLE weights(id TEXT,bytes INTEGER,purpose TEXT,path TEXT,acquired INTEGER);')
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)', tuple(self.snapshot['run'][k] for k in ('id','spec_hash','created','started','deadline','stop_requested')))
        for k in ('doctor_verified','harness_verified','fixtures_verified'): db.execute('INSERT INTO controls VALUES(?,?)', (k, 'true'))
        db.execute('INSERT INTO entities VALUES(?,?,?)', ('controller', self.snapshot['run']['id'], json.dumps(self.snapshot['controller'])))
        db.commit(); db.close()
        before = dbpath.read_bytes()
        original_connect = sqlite3.connect
        def readonly_connect(*args, **kwargs):
            self.assertTrue(args[0].endswith('?mode=ro')); self.assertTrue(kwargs['uri'])
            connection = original_connect(*args, **kwargs)
            with self.assertRaises(sqlite3.OperationalError): connection.execute('CREATE TABLE forbidden(x)')
            return connection
        with patch.object(supervisor.sqlite3, 'connect', side_effect=readonly_connect), patch.object(supervisor, 'load', return_value=self.config):
            result = supervisor.read_snapshot(self.config)
        self.assertEqual(result['running'], 0)
        self.assertEqual(result['completed_attempts'], 0)
        self.assertEqual(result['run']['deadline'], 1000000)
        self.assertEqual(dbpath.read_bytes(), before)


if __name__ == '__main__': unittest.main()

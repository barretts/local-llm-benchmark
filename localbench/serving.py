"""Restricted ordinary serving; benchmark clocks, grades and weights stay immutable."""
from contextlib import contextmanager
import copy
import http.client
import inspect
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import threading
import time
import uuid


class ServingState:
    """State authority granted only to one registered, qualified handoff plan.

    The caller holds gpu_lock. There is deliberately no general method/DB
    forwarding: serving cannot recover or enqueue benchmark work.
    """
    def __init__(self, config, state, plan_path, monotonic=time.monotonic):
        from .handoff import load_plan, _config_identity
        from .report import Report
        self._state, self._monotonic = state, monotonic
        self.config = copy.deepcopy(config)
        self._original_limits = copy.deepcopy(config['limits'])
        self.plan_path = Path(plan_path).resolve(strict=True)
        self.envelope = load_plan(self.plan_path)
        self.plan = self.envelope['plan']
        self.identifier = self.plan['configuration_id']
        run = state.run
        self._previous_stop = bool(run['stop_requested'])
        if run['started'] is None or run['deadline'] is None:
            raise RuntimeError('serving_requires_started_qualified_benchmark')
        if self.plan['run_id'] != run['id'] or self.plan['config_identity'] != _config_identity(config,state):
            raise RuntimeError('serving_run_or_launch_config_changed')
        registry = self._entity('handoff_plan',self.identifier)
        if not registry or registry.get('plan_id') != self.envelope['plan_id'] or Path(registry.get('path','')).resolve() != self.plan_path or (registry.get('original_instance') or {}).get('stopped') is not True:
            raise RuntimeError('serving_registered_stopped_plan_required')
        configuration = self._entity('configuration',self.identifier)
        if configuration is None:
            raise RuntimeError('serving_existing_configuration_required')
        report = Report(config,state)
        rows = [row for row in report.attempts() if row['configuration_id'] == self.identifier]
        if report._configuration(self.identifier,configuration,rows)['qualified'] is not True:
            raise RuntimeError('serving_current_qualification_required')
        demo = self._entity('demo',self.identifier)
        if not demo or any(demo.get(k) is not True for k in ('passed','valid','isolated')) or demo.get('plan_id') != self.envelope['plan_id'] or (demo.get('demo_instance') or {}).get('stopped') is not True:
            raise RuntimeError('serving_verified_demo_required')
        self._demo = demo
        self._configuration_identity = configuration.get('identity')
        self._timeout = config['limits']['server_start_timeout_seconds']
        if type(self._timeout) not in (int,float) or not math.isfinite(self._timeout) or self._timeout <= 0:
            raise ValueError('serving_startup_timeout_invalid')
        self._deadline = monotonic()+self._timeout
        self._phase, self._io_active, self._activated = 'startup', False, False
        self.session_id = 'serving-'+uuid.uuid4().hex
        database = (Path(config['paths']['state'])/'benchmark.sqlite3').resolve()
        self.db = sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=30)
        self.db.row_factory = sqlite3.Row
        def read_only(action,arg1,arg2,db_name,trigger):
            allowed = {sqlite3.SQLITE_SELECT,sqlite3.SQLITE_READ,sqlite3.SQLITE_FUNCTION,sqlite3.SQLITE_RECURSIVE}
            if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() == 'load_extension':
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY
        self.db.set_authorizer(read_only)
        self._refresh_limits()

    def _entity(self,kind,identifier):
        row = self._state.db.execute('SELECT data FROM entities WHERE kind=? AND id=?',(kind,identifier)).fetchone()
        return json.loads(row['data']) if row else None

    @property
    def run(self):
        return self._state.run

    def clock(self):
        return self._state.clock()

    def get_control(self,key):
        return self._state.get_control(key)

    def _refresh_limits(self):
        remaining = max(.001,self._deadline-self._monotonic())
        for key,value in self._original_limits.items():
            if key.endswith('_timeout_seconds') and type(value) in (int,float) and value > 0:
                self.config['limits'][key] = min(value,remaining)

    def remaining_seconds(self):
        if self.run['stop_requested']:
            raise RuntimeError('stop_after_current')
        remaining = self._deadline-self._monotonic()
        if self._phase != 'startup' or remaining <= 0:
            raise RuntimeError('serving_startup_timeout')
        return remaining

    def check_budget(self,estimated_seconds=0,reserve_report=True):
        if type(estimated_seconds) not in (int,float) or not math.isfinite(estimated_seconds) or not 0 <= estimated_seconds <= self._timeout:
            raise RuntimeError('serving_startup_request_exceeds_existing_timeout')
        self.remaining_seconds()
        self._refresh_limits()

    def activate_restart(self):
        # Explicit endpoint invocation revokes only the earlier stop request;
        # future State.stop calls are still read live through self.run.
        if self._activated:raise RuntimeError('serving_restart_already_activated')
        run = self.run
        if run['stop_requested'] and not self._previous_stop:
            raise RuntimeError('stop_after_current')
        with self._state.db:
            self._state.db.execute('UPDATE runs SET stop_requested=0 WHERE id=?',(run['id'],))
        self._activated = True
        self._state.control('serving_session',{'id':self.session_id,'run_id':run['id'],
            'configuration_id':self.identifier,'plan_id':self.envelope['plan_id'],
            'status':'starting','startup_timeout_seconds':self._timeout,
            'previous_stop_requested':bool(run['stop_requested']),'started':self.clock()})
        self._state.snapshot()

    def verify_demo_pins(self):
        from .handoff import _pin
        transcript = self._demo.get('transcript') or {}
        if not transcript.get('files') or not transcript.get('manifest'):
            raise RuntimeError('serving_demo_transcript_proof_missing')
        for pin in transcript['files']+[transcript['manifest']]:
            _pin(pin['path'],pin['role'],pin['sha256'],pin['bytes'],self)

    def start_execution(self,reason):
        allowed = {'runtime_probe:'+self.plan['engine']}
        if self.plan['runtime']['kind'] in ('ollama','lm-studio'):
            allowed.add('runtime_probe:owned-cpu-tokenizer-helper')
        if reason not in allowed:
            raise RuntimeError('serving_acquisition_or_setup_forbidden')
        self.check_budget()
        # Original execution timestamps and clock_start_reason are untouched.

    def entity(self,kind,identifier,data):
        operational = {'engine','launch','tokenizer_helper','tokenizer_verification','handoff_launch','ollama_weight_import'}
        if kind == 'engine' and data.get('engine_id') != self.plan['engine']:
            raise RuntimeError('serving_unpinned_engine_forbidden')
        if kind == 'configuration':
            if identifier != self.identifier or self._entity(kind,identifier) is None or data.get('identity') != self._configuration_identity:
                raise RuntimeError('serving_new_configuration_forbidden')
        elif kind == 'model':
            if identifier != self.plan['model']['id'] or self._entity(kind,identifier) is None or data.get('sha256') != self.plan['model_identity']:
                raise RuntimeError('serving_changed_model_forbidden')
        elif kind not in operational:
            raise RuntimeError('serving_entity_authority_forbidden')
        if kind == 'ollama_weight_import' and data.get('additional_physical_weight_bytes') != 0:
            raise RuntimeError('serving_physical_weight_copy_forbidden')
        self._state.entity(kind,identifier,data)

    def control(self,key,value):
        if key == 'live_agent_endpoint' or key.startswith('serving_') or key.startswith('last_container_recovery:'):
            self._state.control(key,value)
            return
        if key.startswith('model_hash:') and key == 'model_hash:'+str(self.plan['model']['path']) and value.get('sha256') == self.plan['model_identity']:
            self._state.control(key,value)
            return
        if key.startswith('runtime_first_probe:'):
            # Serving cannot establish a new benchmark/setup first-probe clock.
            return
        raise RuntimeError('serving_control_authority_forbidden')

    def snapshot(self):
        return self._state.snapshot()

    def stop(self):
        self._state.stop()

    def _forbidden(self,*args,**kwargs):
        raise RuntimeError('serving_benchmark_or_acquisition_mutation_forbidden')
    reserve_weight = enqueue = begin = finish = recover = _forbidden

    def finish_startup(self):
        self.check_budget()
        self._phase = 'serving'
        self.config['limits'] = copy.deepcopy(self._original_limits)
        session = self._state.get_control('serving_session') or {}
        self._state.control('serving_session',{**session,'status':'serving','ready':self.clock()})

    def _cleanup_call(self):
        frame = inspect.currentframe()
        try:
            while frame is not None:
                if frame.f_code.co_name in ('unload_owned','recover_owned') and str(frame.f_globals.get('__name__','')).startswith(('localbench.adapters.','localbench.tokenizer_helper')):
                    return True
                frame = frame.f_back
            return False
        finally:
            del frame

    def _io_remaining(self):
        # Startup failure must still be able to stop its verified owned runtime.
        return self._timeout if self._cleanup_call() else self.remaining_seconds()

    @contextmanager
    def startup_io(self):
        """Cap synchronous startup commands and every HTTP read to one deadline.

        Hooks affect only this controller thread and are restored before serving.
        Other threads (including the resource collector) use original methods.
        """
        if self._io_active:
            yield
            return
        owner = threading.get_ident()
        original_run = subprocess.run
        original_request = http.client.HTTPConnection.request
        original_getresponse = http.client.HTTPConnection.getresponse
        state = self
        def bounded_run(*args,**kwargs):
            if threading.get_ident() != owner:
                return original_run(*args,**kwargs)
            remaining = state._io_remaining()
            argv = args[0] if args else kwargs.get('args')
            if not isinstance(argv,(list,tuple)) or kwargs.get('shell'):
                raise RuntimeError('serving_shell_command_forbidden')
            words = [str(word).lower() for word in argv]
            executable = Path(str(argv[0])).name.lower()
            if (executable.startswith('docker') and any(word in ('pull','build','manifest','compose') for word in words[1:2])) or (executable.startswith(('ollama','lms')) and any(word in ('pull','download','get') for word in words[1:2])) or executable.startswith(('curl','wget','pip')) or ('-m' in words and 'pip' in words):
                raise RuntimeError('serving_acquisition_or_build_command_forbidden')
            kwargs['timeout'] = min(kwargs.get('timeout') or remaining,remaining)
            result = original_run(*args,**kwargs)
            if not state._cleanup_call():state.remaining_seconds()
            return result
        def cap_socket(connection):
            remaining = state._io_remaining()
            connection.timeout = min(connection.timeout or remaining,remaining)
            if connection.sock is not None:connection.sock.settimeout(connection.timeout)
        def bounded_request(connection,*args,**kwargs):
            if threading.get_ident() == owner:cap_socket(connection)
            return original_request(connection,*args,**kwargs)
        def bounded_getresponse(connection,*args,**kwargs):
            if threading.get_ident() != owner:return original_getresponse(connection,*args,**kwargs)
            cap_socket(connection)
            return _StartupResponse(original_getresponse(connection,*args,**kwargs),state)
        self._io_active = True
        try:
            subprocess.run = bounded_run
            http.client.HTTPConnection.request = bounded_request
            http.client.HTTPConnection.getresponse = bounded_getresponse
            yield
        finally:
            subprocess.run = original_run
            http.client.HTTPConnection.request = original_request
            http.client.HTTPConnection.getresponse = original_getresponse
            self._io_active = False

    def close(self):
        self.db.close()


class _StartupResponse:
    def __init__(self,response,state):self._response,self._state = response,state
    def __getattr__(self,name):return getattr(self._response,name)
    def __enter__(self):return self
    def __exit__(self,*args):self._response.close()
    def read1(self,amount=65536):
        remaining = self._state._io_remaining()
        fp = getattr(self._response,'fp',None)
        sock = getattr(getattr(fp,'raw',None),'_sock',None)
        if sock is not None:
            existing = sock.gettimeout()
            sock.settimeout(min(existing,remaining) if type(existing) in (int,float) else remaining)
        result = self._response.read1(amount)
        if not self._state._cleanup_call():self._state.remaining_seconds()
        return result
    def read(self,amount=None):
        pieces = []
        left = None if amount is None or amount < 0 else amount
        while left is None or left > 0:
            chunk = self.read1(65536 if left is None else min(65536,left))
            if not chunk:break
            pieces.append(chunk)
            if left is not None:left -= len(chunk)
        return b''.join(pieces)


def run_endpoint(config,state,plan_path):
    """Run under runner's GPU lease; no scheduler or benchmark job mutation."""
    from .doctor import doctor
    from .endpoint import serve
    from .recovery import recover_processes
    from .resources import ResourceCollector
    serving = ServingState(config,state,plan_path)
    collector = None
    try:
        serving.activate_restart()
        with serving.startup_io():
            serving.verify_demo_pins()
            recovery = recover_processes(serving.config,state)
            serving.control('serving_ownership_recovery',recovery)
            if any(item.get('status') in ('refused','pending_descendants','pending_auth','pending_load') for item in recovery):
                raise RuntimeError('owned_recovery_unresolved')
            inspected = doctor(serving.config,hash_weights=False)
            serving.check_budget()
            if not inspected['disk_headroom_ok']:raise RuntimeError('disk_headroom_failed')
            host = json.loads((Path(config['paths']['artifacts'])/'host.json').read_text(encoding='utf-8'))
            try:loaded = json.loads(host['commands']['lms_loaded']['output'])
            except ValueError:loaded = None
            if loaded:
                serving.control('serving_waiting_for_idle',{'reason':'unrelated_loaded_models'})
                raise RuntimeError('unrelated_loaded_models')
        collector = ResourceCollector(config,state).start()
        return serve(serving.config,serving,collector,plan_path)
    finally:
        try:
            if collector:collector.stop()
            session = state.get_control('serving_session') or {}
            if session.get('id') == serving.session_id:
                endpoint = state.get_control('live_agent_endpoint') or {}
                outcome = endpoint.get('status') if endpoint.get('serving_session_id') == serving.session_id else 'startup_failed'
                state.control('serving_session',{**session,'status':outcome,'finished':state.clock()})
        finally:serving.close()

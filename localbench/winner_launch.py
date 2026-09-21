"""Export a launcher only for a currently qualified, immutable pinned winner."""
import json
from pathlib import Path
import uuid

from .config import atomic_json,canonical,digest,file_hash

HEADER = '# localbench managed winner launcher v1'


def _ps_literal(value):
    value = str(value)
    if any(char in value for char in ('\r','\n','\x00','"')):
        raise ValueError('winner_launch_literal_invalid')
    return "'"+value.replace("'","''")+"'"


def _script(config,envelope,plan_file):
    plan = envelope['plan']
    project = Path(config['paths']['project']).resolve()
    logs = Path(config['paths']['logs']).resolve()
    if not logs.is_relative_to(project):raise ValueError('winner_logs_must_be_project_owned')
    literal = _ps_literal
    lines = [HEADER,'[CmdletBinding()]','param()','$ErrorActionPreference = '+literal('Stop'),
        '$benchmarkProject = '+literal(project),
        '$benchmarkPython = '+literal(project/'.venv'/'Scripts'/'python.exe'),
        '$benchmarkLogs = '+literal(logs),
        '$benchmarkPlanFile = '+literal(plan_file),
        '$benchmarkPlanId = '+literal(envelope['plan_id']),
        '$benchmarkConfigurationId = '+literal(plan['configuration_id']),
        '# Engine identity: '+canonical(plan['engine_identity']),
        '# Model path: '+str(plan['model']['path']),
        '# Model identity: '+plan['model_identity'],
        '# Exact requested settings: '+canonical(plan['requested_settings']),
        '# Local API: '+plan['interface']['endpoint'],
        '$benchmarkRequiresLMToken = '+('$true' if plan['runtime']['kind']=='lm-studio' else '$false'),
        '$benchmarkHadToken = Test-Path -LiteralPath Env:\\LM_BENCH_TOKEN',
        '$benchmarkPreviousToken = $env:LM_BENCH_TOKEN',
        '$benchmarkSecureToken = $null',
        '$benchmarkTokenMemory = [IntPtr]::Zero',
        'try {',
        "    if (-not (Test-Path -LiteralPath $benchmarkPython -PathType Leaf) -or -not (Test-Path -LiteralPath $benchmarkPlanFile -PathType Leaf)) { throw 'Pinned local launcher files are unavailable.' }",
        '    $benchmarkEnvelope = [IO.File]::ReadAllText($benchmarkPlanFile) | ConvertFrom-Json',
        "    if ($benchmarkEnvelope.plan_id -cne $benchmarkPlanId -or $benchmarkEnvelope.plan.configuration_id -cne $benchmarkConfigurationId) { throw 'Pinned local launch plan identity changed.' }",
        '    if ($benchmarkRequiresLMToken) {',
        '        if ([string]::IsNullOrWhiteSpace($env:LM_BENCH_TOKEN)) {',
        "            $benchmarkSecureToken = Read-Host 'LM Studio local API key (process memory only)' -AsSecureString",
        '            $benchmarkTokenMemory = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($benchmarkSecureToken)',
        '            $env:LM_BENCH_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($benchmarkTokenMemory)',
        "            if ([string]::IsNullOrWhiteSpace($env:LM_BENCH_TOKEN)) { throw 'LM Studio process credential is unavailable.' }",
        '        }',
        '    } else { Remove-Item -LiteralPath Env:\\LM_BENCH_TOKEN -ErrorAction SilentlyContinue }',
        '    [void][IO.Directory]::CreateDirectory($benchmarkLogs)',
        "    $benchmarkStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')",
        "    $benchmarkStdout = Join-Path $benchmarkLogs ('winner-endpoint-' + $benchmarkStamp + '.out.log')",
        "    $benchmarkStderr = Join-Path $benchmarkLogs ('winner-endpoint-' + $benchmarkStamp + '.err.log')",
        "    $benchmarkArguments = @('-X', 'utf8', '-m', 'localbench', 'agent-endpoint', '--plan', $benchmarkPlanFile)",
        "    # All arguments are fixed tokens or a Windows .json path; quote each",
        "    # literal so spaces and apostrophes cannot split the pinned PlanFile.",
        "    if ($benchmarkArguments | Where-Object { $_.Contains([char]34) -or $_.Contains([char]13) -or $_.Contains([char]10) }) { throw 'Invalid local launcher argument.' }",
        "    $benchmarkArgumentLine = ($benchmarkArguments | ForEach-Object { [char]34 + $_ + [char]34 }) -join ' '",
        '    $benchmarkProcess = Start-Process -FilePath $benchmarkPython -ArgumentList $benchmarkArgumentLine -WorkingDirectory $benchmarkProject -WindowStyle Hidden -RedirectStandardOutput $benchmarkStdout -RedirectStandardError $benchmarkStderr -PassThru',
        '    # The child inherited the credential; restore the caller environment',
        '    # immediately. Readiness/interface are recorded by the owned controller.',
        '    if ($benchmarkHadToken) { $env:LM_BENCH_TOKEN = $benchmarkPreviousToken } else { Remove-Item -LiteralPath Env:\\LM_BENCH_TOKEN -ErrorAction SilentlyContinue }',
        "    [pscustomobject]@{ status = 'starting'; pid = $benchmarkProcess.Id; created_utc = $benchmarkProcess.StartTime.ToUniversalTime().ToString('o'); configuration_id = $benchmarkConfigurationId; plan_file = $benchmarkPlanFile; stdout_log = $benchmarkStdout; stderr_log = $benchmarkStderr } | ConvertTo-Json",
        "} catch { throw 'Local winner endpoint launch failed; no credential details are emitted.' }",
        'finally {',
        '    if ($benchmarkTokenMemory -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($benchmarkTokenMemory) }',
        '    if ($null -ne $benchmarkSecureToken) { $benchmarkSecureToken.Dispose() }',
        '    if ($benchmarkHadToken) { $env:LM_BENCH_TOKEN = $benchmarkPreviousToken } else { Remove-Item -LiteralPath Env:\\LM_BENCH_TOKEN -ErrorAction SilentlyContinue }',
        '    $benchmarkPreviousToken = $null',
        '}','']
    return '\n'.join(lines)


def generate_winner_launcher(config,state,summary):
    """Read grades/pins and write owned export files; never launch or change DB."""
    from .handoff import load_plan,_config_identity,_demo_key,_entity,_pin,_validate_files
    from .report import Report
    from .scheduler import engine_identity,tokenizer_identity
    project = Path(config['paths']['project']).resolve()
    destination = project/'scripts'/'launch-winner.ps1'
    receipt = Path(config['paths']['artifacts'])/'winner-launcher.json'
    try:previous = json.loads(receipt.read_text(encoding='utf-8'))
    except (OSError,ValueError):previous = {}
    def unchanged_owned():
        return destination.is_file() and previous.get('script_sha256') == file_hash(destination) and destination.read_text(encoding='utf-8').splitlines()[0] == HEADER
    def unavailable(reason):
        result = {'status':'unavailable','reason':reason,'script':None}
        if unchanged_owned():destination.unlink()
        elif destination.exists():result['existing_script_preserved'] = True
        atomic_json(receipt,result)
        return result
    identifier = summary.get('winner_id')
    if not identifier:return unavailable('no_unambiguous_qualified_winner')
    if state.run['started'] is None:return unavailable('no_real_execution')
    report = Report(config,state)
    configuration = _entity(state,'configuration',identifier)
    rows = [row for row in report.attempts() if row['configuration_id'] == identifier]
    if configuration is None or report._configuration(identifier,configuration,rows)['qualified'] is not True:
        return unavailable('current_winner_qualification_missing')
    registry = _entity(state,'handoff_plan',identifier)
    if not registry:return unavailable('registered_handoff_plan_missing')
    try:
        plan_file = Path(registry['path']).resolve(strict=True)
        artifacts = Path(config['paths']['artifacts']).resolve()
        if not plan_file.is_relative_to(artifacts/'launch-plans'):
            raise ValueError('winner_plan_not_project_artifact')
        envelope = load_plan(plan_file)
        plan = envelope['plan']
        if registry['plan_id'] != envelope['plan_id'] or plan['configuration_id'] != identifier or plan['run_id'] != state.run['id'] or plan['config_identity'] != _config_identity(config,state) or (registry.get('original_instance') or {}).get('stopped') is not True:
            raise ValueError('winner_plan_registry_or_run_mismatch')
        if configuration.get('requested_settings') != plan['requested_settings'] or configuration.get('model_sha256') != plan['model_identity'] or engine_identity(configuration) != plan['engine_identity'] or tokenizer_identity(configuration) != plan['tokenizer_identity']:
            raise ValueError('winner_plan_configuration_mismatch')
        demo_job = state.db.execute('SELECT status,result FROM jobs WHERE id=?',(digest(_demo_key(config,plan,envelope['plan_id'])),)).fetchone()
        demo = json.loads(demo_job['result'] or '{}') if demo_job else {}
        if not demo_job or demo_job['status'] != 'passed' or any(demo.get(key) is not True for key in ('passed','valid','isolated')) or demo.get('plan_id') != envelope['plan_id'] or (demo.get('demo_instance') or {}).get('stopped') is not True:
            raise ValueError('winner_current_same_plan_demo_missing')
        _validate_files(plan)
        transcript = demo.get('transcript') or {}
        if not transcript.get('files') or not transcript.get('manifest'):
            raise ValueError('winner_actual_transcript_missing')
        for pin in transcript['files']+[transcript['manifest']]:
            _pin(pin['path'],pin['role'],pin['sha256'],pin['bytes'])
        text = _script(config,envelope,plan_file)
    except (OSError,ValueError,RuntimeError,KeyError,TypeError):
        return unavailable('winner_plan_or_pinned_proof_invalid')
    if destination.exists() and not unchanged_owned():
        return unavailable('launcher_path_occupied_or_edited')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(text,encoding='utf-8',newline='\n')
        temporary.replace(destination)
    finally:temporary.unlink(missing_ok=True)
    result = {'status':'generated','script':str(destination),'script_sha256':file_hash(destination),
        'configuration_id':identifier,'plan_file':str(plan_file),'plan_id':envelope['plan_id'],
        'runtime_probe_executed':False,'credential':'process environment or masked prompt when invoked' if plan['runtime']['kind']=='lm-studio' else 'none'}
    atomic_json(receipt,result)
    return result

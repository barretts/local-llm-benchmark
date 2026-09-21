"""Export this pinned experiment's evidence without changing benchmark state."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'
BASELINES = {
    ('q8_0', 'q8_0'): '09b315f77ce6ffc3221823a5eb429ec399d5548d7870e1f6400dca6d4c10c03d',
    ('q4_0', 'q4_0'): '918a55d7d22dd5094226a3e7dc5b6f96058b2cf0361af196e14dd4033111e32c',
}
MODES = ('off', 'draft-mtp', 'draft-dflash')
SETTING_FIELDS = (
    'context', 'slots', 'context_shift', 'gpu_layers', 'flash_attention', 'cache_k',
    'cache_v', 'batch', 'ubatch', 'threads', 'threads_batch', 'reasoning', 'jinja',
    'speculation', 'draft_n_max', 'draft_cache_k', 'draft_cache_v', 'draft_gpu_layers',
    'speculation_baseline_configuration_id', 'speculation_contract_sha256',
)


def atomic(path, content):
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(content, encoding='utf-8')
    os.replace(temp, path)


def safe_argv(argv):
    secret_flags = {'--api-key', '--api-key-file', '--auth-token', '--hf-token'}
    answer, redact = [], False
    for arg in argv:
        if redact:
            answer.append('[REDACTED]')
            redact = False
        elif arg in secret_flags:
            answer.append(arg)
            redact = True
        elif any(arg.startswith(flag + '=') for flag in secret_flags):
            answer.append(arg.split('=', 1)[0] + '=[REDACTED]')
        else:
            answer.append(arg)
    return answer


def main():
    spec = json.loads((ROOT / 'benchmark-spec.json').read_text(encoding='utf-8-sig'))
    plan_path = ARTIFACTS / 'speculation-comparison-plan.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    phase = json.loads((ARTIFACTS / 'speculation-comparison-status.json').read_text(encoding='utf-8-sig'))
    repair = json.loads((ARTIFACTS / 'speculation-repair-status.json').read_text(encoding='utf-8-sig'))
    smoke_ids = {job['fixture'] for job in spec['grading']['smoke_jobs']}
    con = sqlite3.connect((ROOT / 'state/benchmark.sqlite3').as_uri() + '?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    con.execute('BEGIN')
    run = dict(con.execute('SELECT id,started,deadline,stop_requested FROM runs ORDER BY created DESC LIMIT 1').fetchone())
    configurations = {}
    for row in con.execute("SELECT id,data FROM entities WHERE kind='configuration'"):
        data = json.loads(row['data'])
        requested = data.get('requested_settings', {})
        cache = requested.get('cache_k'), requested.get('cache_v')
        baseline = BASELINES.get(cache)
        mode = requested.get('speculation', 'off')
        if row['id'] != baseline and not (
            baseline and mode in ('draft-mtp', 'draft-dflash')
            and requested.get('speculation_baseline_configuration_id') == baseline
            and requested.get('speculation_contract_sha256') == plan['contract_sha256']
            and requested.get('draft_n_max') == plan['draft_widths'][mode]
            and requested.get('reasoning') == 'default'
        ):
            continue
        effective = data.get('effective_settings', {})
        item = {
            'configuration_id': row['id'], 'mode': mode, 'cache': list(cache),
            'version': data.get('version_output'), 'binary': data.get('binary'),
            'binary_sha256': data.get('binary_sha256'), 'model_path': data.get('model_path'),
            'model_sha256': data.get('model_sha256'), 'identity': data.get('identity'),
            'requested_settings': {key: requested[key] for key in SETTING_FIELDS if key in requested},
            'draft_model': requested.get('draft_model'),
            'launch_argv': safe_argv(effective.get('launch_argv', [])),
            'effective_target_offload': effective.get('offload_layers'),
            'effective_draft_offload': effective.get('draft_offload_layers'),
            'capacity': [], 'coding_smokes': [], 'fresh_timing': [],
            'screen_outcome': phase.get('arm_outcomes', {}).get(mode + ':' + ','.join(cache)),
        }
        query = """
        SELECT a.id,a.status,a.reason,
          json_extract(j.key_json,'$.kind') AS kind,
          json_extract(j.key_json,'$.target_tokens') AS target_tokens,
          json_extract(j.key_json,'$.seed') AS seed,
          json_extract(j.key_json,'$.mode') AS request_mode,
          json_extract(j.key_json,'$.replicate') AS replicate,
          json_extract(a.result,'$.fixture') AS fixture,
          json_extract(a.result,'$.fixture_version') AS fixture_version,
          json_extract(a.result,'$.valid') AS valid,
          json_extract(a.result,'$.passed') AS passed,
          json_extract(a.result,'$.markers_passed') AS markers_passed,
          json_extract(a.result,'$.metrics.first_action_valid') AS first_action_valid,
          json_extract(a.result,'$.metrics.first_action_seconds') AS first_action_seconds,
          json_extract(a.result,'$.metrics.decode_tokens_per_second') AS decode_tokens_per_second,
          json_extract(a.result,'$.metrics.prompt_seconds') AS prompt_seconds,
          json_extract(a.result,'$.metrics.generated_tokens') AS generated_tokens,
          json_extract(a.result,'$.metrics.speculation_counters_verified') AS speculation_counters_verified,
          json_extract(a.result,'$.metrics.speculation_drafted_tokens') AS drafted_tokens,
          json_extract(a.result,'$.metrics.speculation_accepted_tokens') AS accepted_tokens,
          json_extract(a.result,'$.metrics.speculation_verification_steps') AS verification_steps,
          json_extract(a.result,'$.token_evidence.expected_prompt_tokens') AS expected_prompt_tokens,
          json_extract(a.result,'$.token_evidence.effective_context_tokens') AS effective_context_tokens,
          json_extract(a.result,'$.token_evidence.cache_isolation_verified') AS cache_isolation_verified,
          json_extract(a.result,'$.timing_valid') AS timing_valid,
          json_extract(a.result,'$.timing_invalid_reason') AS timing_invalid_reason,
          json_extract(a.result,'$.latency_diagnostic_only') AS latency_diagnostic_only,
          json_extract(a.result,'$.resources.foreign_workload_overlap') AS foreign_workload_overlap,
          json_extract(a.result,'$.resources.foreign_workload_evidence_available') AS foreign_workload_evidence_available,
          json_extract(a.result,'$.metrics.speculation_evidence_path') AS speculation_evidence_path,
          json_extract(a.result,'$.source_diff') AS source_diff,
          a.finished-a.started AS elapsed_seconds
        FROM attempts a JOIN jobs j ON j.id=a.job_id
        WHERE json_extract(j.key_json,'$.configuration_id')=?
          AND a.id=(SELECT MAX(b.id) FROM attempts b WHERE b.job_id=a.job_id)
        ORDER BY a.id
        """
        for row in con.execute(query, (item['configuration_id'],)):
            entry = dict(row)
            if entry['kind'] == 'capacity':
                item['capacity'].append(entry)
            elif entry['kind'] == 'quality' and entry['seed'] == 42 and entry['fixture'] in smoke_ids:
                item['coding_smokes'].append(entry)
            elif entry['kind'] == 'timing' and entry['request_mode'] == 'fresh':
                item['fresh_timing'].append(entry)
        key = mode + ':' + ','.join(cache)
        if key in configurations:
            raise RuntimeError('ambiguous_configuration_for_pinned_comparison:' + key)
        configurations[key] = item
    con.close()
    payload = {
        'schema': 'localbench-speculation-evidence-v1',
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'run': run, 'phase_status': phase['status'],
        'phase_deadline': phase['deadline'], 'active_arm': phase.get('active_arm'),
        'comparison_plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
        'harness_sources_sha256': repair['sources_sha256'],
        'harness_test_count': repair['harness_test_count'],
        'qualification_ids': phase.get('qualification_ids', []),
        'qualification_claimed_by_this_export': False,
        'definitions': {
            'capacity': 'Retrieval/tool evidence; 61440 formatted input tokens in a 65536 context, with the frozen tolerance and 4096-token reserve.',
            'coding_smokes': 'The six literal seed-42 jobs; all must pass for a speculative arm to enter expensive qualification.',
            'fresh_timing': 'Separate native useful-action timing; capacity decode samples do not determine the winner ranking.',
            'off_controls': 'Failed coding controls remain diagnostic and cannot enter the winner ranking.',
            'timing_validity': 'Capacity functional validity is separate from timing validity. Every foreign-workload-overlap timing is diagnostic; raw speed values remain preserved and must not be used for speed comparisons or ranking.',
        },
        'configurations': configurations,
    }
    lines = ['# Measured speculative comparison', '', 'Snapshot: ' + payload['generated_utc'], '',
             'Phase: ' + payload['phase_status'] + '. Full coding qualification is not claimed by this export.', '',
             '| Target cache | Mode | 64K markers passed | Clean 64K capacity timings | Coding smokes passed / finished | Valid fresh timings | Screen |',
             '| --- | --- | ---: | ---: | ---: | ---: | --- |']
    for cache in BASELINES:
        for mode in MODES:
            item = configurations.get(mode + ':' + ','.join(cache))
            if item is None:
                lines.append('| ' + '/'.join(cache) + ' | ' + mode + ' | pending | pending | pending | pending | pending |')
                continue
            caps = [row for row in item['capacity'] if row['target_tokens'] == 61440 and row['valid'] == 1]
            markers = sum(row['markers_passed'] or 0 for row in caps)
            clean_caps = sum(row['foreign_workload_overlap'] == 0 and row['foreign_workload_evidence_available'] == 1
                             and row['first_action_valid'] == 1 for row in caps)
            smokes = [row for row in item['coding_smokes'] if row['valid'] == 1]
            fresh = sum(row['valid'] == 1 and row['passed'] == 1 and row['first_action_valid'] == 1 for row in item['fresh_timing'])
            outcome = item['screen_outcome'] or {}
            screen = 'diagnostic control' if mode == 'off' else outcome.get('status', 'pending')
            lines.append(f"| {'/'.join(cache)} | {mode} | {markers}/9 | {clean_caps}/3 | {sum(row['passed']==1 for row in smokes)}/{len(smokes)} | {fresh} | {screen} |")
    lines += ['', 'Capacity throughput is auxiliary. Primary fresh-action timing, regular/long coding quality, real drafting and stability gates remain required.', '',
              'Capacity retrieval can pass while its timing is contaminated. Samples flagged foreign_workload_overlap remain diagnostic and are excluded from speed comparisons; the JSON retains their raw values and flags.', '',
              'The JSON export contains individual attempt IDs, verdicts, counters, hashes, exact argv and settings. Raw SSE, grading and resource files remain preserved in the benchmark workspace.', '']
    atomic(ARTIFACTS / 'speculation-measured-results.json', json.dumps(payload, indent=2, allow_nan=False) + '\n')
    atomic(ARTIFACTS / 'speculation-measured-results.md', '\n'.join(lines))
    print(json.dumps({'written': ['artifacts/speculation-measured-results.json', 'artifacts/speculation-measured-results.md'],
                      'configurations': len(configurations), 'phase': payload['phase_status']}))


if __name__ == '__main__':
    main()

"""Read-only intake for bounded reasoning trials after clean marker censoring."""
from pathlib import Path
import json
import re

from .report import context_reasons


def marker_reasoning_eligible(config, state, candidate, model):
    """A full input was verified; the default output alone exhausted its reserve.

    This grants consideration within the existing three-family tuning selection,
    never capacity/quality qualification or proof of effective budget enforcement.
    """
    runtime, settings = candidate.get('runtime', {}), candidate.get('settings', {})
    if runtime.get('kind') != 'native' or settings.get('reasoning') != 'default':
        return False
    from .search import pending_for_stage, stage_key
    marker = 'planned_screen:' + stage_key(runtime, model, None)
    done = state.get_control(marker)
    if candidate.get('planned_screen_key') != marker or not done \
            or done.get('configuration_id') != candidate.get('configuration_id') \
            or done.get('settings') != settings or pending_for_stage(state, candidate.get('configuration_id')):
        return False
    small = candidate.get('small_capacity', {})
    if small.get('passed') is not True or small.get('valid') is not True or small.get('target_tokens') != 16384:
        return False
    capacities = candidate.get('capacity', [])
    seeds = config['grading']['capacity_marker_seeds']
    if len(capacities) != len(seeds) or {item.get('seed') for item in capacities} != set(seeds):
        return False
    censored = False
    target = config['measurement']['full_prompt_tokens']
    tolerance = config['measurement']['prompt_target_absolute_tolerance_tokens']
    for item in capacities:
        metrics = item.get('metrics', {})
        observed = metrics.get('prompt_tokens')
        if item.get('kind') != 'capacity' or item.get('target_tokens') != target or item.get('valid') is not True \
                or metrics.get('stream_complete') is not True \
                or type(metrics.get('invalid_tool_calls')) is not int or metrics['invalid_tool_calls'] != 0 \
                or metrics.get('completion_reason') not in ('stop', 'tool_calls', 'length') \
                or type(observed) is not int or not target-tolerance <= observed <= target \
                or observed != item.get('token_evidence', {}).get('prompt_tokens') \
                or context_reasons(item, config, 'fresh'):
            return False
        if item.get('passed') is True and item.get('reason') == 'passed':
            continue
        if item.get('passed') is not False or item.get('reason') != 'capacity_marker_or_evidence_failure':
            return False
        if metrics.get('completion_reason') == 'length' and metrics.get('generated_tokens') == 4096 \
                and metrics.get('first_action_valid') is False and metrics.get('first_action_seconds') is None:
            censored = True
    if not censored:
        return False
    row = state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?",
                           (candidate.get('configuration_id'),)).fetchone()
    metadata = json.loads(row['data']) if row else {}
    help_path = Path(metadata.get('help_log', ''))
    return help_path.is_file() and bool(re.search(r'(?m)^\s*--reasoning-budget\s+N\b',
        help_path.read_text(encoding='utf-8-sig', errors='replace')))

"""Strict optional GGUF discovery queue; metadata never awards qualification."""
from pathlib import Path
import json
import re
from urllib.parse import parse_qs, unquote, urlsplit

from .config import canonical, digest
from .doctor import gguf_metadata

FIELDS = {'id', 'repo', 'revision', 'filename', 'bytes', 'sha256',
          'native_context_tokens', 'context_extension', 'format', 'sources'}
OPTIONAL = {'provenance_artifact'}
NATIVE_ENGINES = ('bundled-llama', 'upstream-llama-nightly')


def public_source(url):
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) < 33 for c in url):
        raise ValueError('invalid_public_discovery_source')
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname not in {
            'huggingface.co', 'github.com', 'api.github.com', 'raw.githubusercontent.com'} \
            or parsed.username is not None or parsed.password is not None or parsed.fragment \
            or parsed.port not in (None, 443):
        raise ValueError('discovery_source_must_be_public_primary_https')
    if parsed.query and not (parsed.hostname == 'huggingface.co' and parsed.path.startswith('/api/models/')
                            and parse_qs(parsed.query, keep_blank_values=True) == {'blobs': ['true']}):
        raise ValueError('discovery_source_query_or_credentials_forbidden')
    decoded = unquote(unquote(parsed.path))
    if '\\' in decoded or any(part in ('.', '..') for part in decoded.split('/')):
        raise ValueError('discovery_source_path_traversal')


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'candidates'} \
            or type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise ValueError('invalid_extra_discovery_manifest')
    candidates = manifest['candidates']
    if not isinstance(candidates, list) or len(candidates) > 4:
        raise ValueError('extra_discovery_queue_must_have_at_most_four_candidates')
    ids = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not FIELDS <= set(candidate) or set(candidate)-FIELDS-OPTIONAL:
            raise ValueError('unexpected_or_missing_extra_discovery_fields')
        if any(not isinstance(candidate[field], str) for field in ('id', 'repo', 'revision', 'sha256', 'filename')):
            raise ValueError('discovery_identity_fields_must_be_strings')
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,95}', candidate['id']) or candidate['id'] in ids:
            raise ValueError('invalid_or_duplicate_discovery_id')
        ids.add(candidate['id'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}', candidate['repo']):
            raise ValueError('invalid_discovery_repository')
        if not re.fullmatch(r'[a-f0-9]{40}', candidate['revision']) or not re.fullmatch(r'[a-f0-9]{64}', candidate['sha256']):
            raise ValueError('discovery_requires_immutable_commit_and_weight_sha256')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\.gguf', candidate['filename']):
            raise ValueError('discovery_requires_one_root_gguf_filename')
        if type(candidate['bytes']) is not int or candidate['bytes'] <= 0:
            raise ValueError('discovery_requires_positive_exact_weight_bytes')
        if type(candidate['native_context_tokens']) is not int or candidate['native_context_tokens'] < 65536 \
                or candidate['context_extension'] is not False or candidate['format'] != 'gguf':
            raise ValueError('discovery_requires_native_64k_gguf_without_extension')
        sources = candidate['sources']
        if not isinstance(sources, list) or not 1 <= len(sources) <= 12:
            raise ValueError('discovery_requires_public_provenance_sources')
        for source in sources:
            public_source(source)
        provenance = candidate.get('provenance_artifact')
        if provenance is not None and (not isinstance(provenance, str)
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}\.json', provenance)):
            raise ValueError('provenance_artifact_must_be_one_local_artifact_filename')
    return candidates


def load_manifest(config):
    path = Path(config['paths']['artifacts'])/'extra-discovered-gguf.json'
    if not path.exists():
        return []
    if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()) or path.stat().st_size > 512000:
        raise ValueError('extra_discovery_manifest_must_be_a_bounded_ordinary_file')
    return validate_manifest(json.loads(path.read_text(encoding='utf-8-sig')))


def pinned_variants(state, candidates):
    # Preflight every existing pin before allowing any acquisition from a queue.
    by_id = {candidate['id']: candidate for candidate in candidates}
    for row in state.db.execute("SELECT id,data FROM entities WHERE kind='extra_discovered_weight_pin'"):
        if row['id'] not in by_id or json.loads(row['data']) != by_id[row['id']]:
            raise ValueError('discovered_candidate_identity_changed:'+row['id'])


def select_variant(config, state, candidate):
    with state.db:
        selected = state.get_control('discovered_weight_variants') or []
        if not isinstance(selected, list) or any(not isinstance(item, str) for item in selected) or len(set(selected)) != len(selected):
            raise RuntimeError('invalid_durable_discovery_variant_ledger')
        row = state.db.execute("SELECT data FROM entities WHERE kind='extra_discovered_weight_pin' AND id=?",
                               (candidate['id'],)).fetchone()
        if row is not None and json.loads(row['data']) != candidate:
            raise ValueError('discovered_candidate_identity_changed:'+candidate['id'])
        if candidate['id'] not in selected:
            if len(selected) >= config['limits']['maximum_new_discovery_weight_variants']:
                return False
            selected = selected+[candidate['id']]
        elif row is None:
            raise ValueError('discovery_id_already_owned_by_another_candidate:'+candidate['id'])
        state.db.execute('INSERT OR REPLACE INTO controls VALUES(?,?)',
                         ('discovered_weight_variants', canonical(selected)))
        state.db.execute('INSERT OR REPLACE INTO entities VALUES(?,?,?)',
                         ('extra_discovered_weight_pin', candidate['id'], canonical(candidate)))
    return True


def extra_gguf_screens(config, state, collector, runtimes, baseline_screens, *,
                       acquire_model, memory_screen, unavailable):
    candidates = load_manifest(config)
    pinned_variants(state, candidates)
    if not candidates:
        return []
    from .search import pending_for_stage
    if not baseline_screens or state.get_control('installed_baselines_completed') is not True \
            or any(not screen.get('planned_screen_key')
                   or not state.get_control(screen['planned_screen_key'])
                   or pending_for_stage(state, screen.get('configuration_id')) for screen in baseline_screens):
        raise RuntimeError('current_installed_baselines_required_before_discovery_acquisition')
    native = [runtimes[engine] for engine in NATIVE_ENGINES
              if engine in runtimes and runtimes[engine].get('kind') == 'native']
    if not native:
        for candidate in candidates:
            unavailable(config, state, 'extra-gguf-discovery', candidate, 'no_existing_native_runtime_for_discovered_gguf', evidence=candidate)
        return []
    outcomes = []
    for candidate in candidates:
        state.check_budget(1800)
        if not select_variant(config, state, candidate):
            unavailable(config, state, 'extra-gguf-discovery', candidate, 'discovery_variant_budget_exhausted', evidence=candidate)
            continue
        model = acquire_model(config, state, candidate)
        if model is None:
            continue
        if any(model.get(field) != candidate[field] for field in ('id', 'revision', 'sha256', 'bytes')):
            raise RuntimeError('discovered_gguf_acquisition_identity_mismatch')
        try:
            header = gguf_metadata(model['path'])
            architecture = header.get('general.architecture')
            context = header.get(str(architecture)+'.context_length')
            template = header.get('tokenizer.chat_template')
            if not isinstance(architecture, str) or type(context) is not int or context < 65536:
                raise ValueError('discovered_gguf_native_context_header_unverified')
            if not isinstance(template, str) or not template.strip():
                raise ValueError('discovered_gguf_embedded_chat_template_unverified')
        except (OSError, ValueError) as error:
            unavailable(config, state, 'extra-gguf-discovery', candidate, str(error), evidence=candidate)
            continue
        model = {**model, 'discovered_candidate': candidate, 'actual_native_context_tokens': context,
                 'gguf_architecture': architecture, 'embedded_chat_template_sha256': digest(template)}
        state.entity('model', model['id'], model)
        for runtime in native:
            outcomes.extend(memory_screen(config, state, collector, runtime, model))
    return outcomes

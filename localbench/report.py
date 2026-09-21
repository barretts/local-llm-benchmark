"""Evidence-based qualification and exports. Every attempt remains in raw results."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
import statistics

from .config import atomic_json, canonical, digest
from .doctor import sanitize
from .metrics import decode_throughput, sample_summary


def _redact(value):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("_", "").replace("-", "")
            result[key] = "[REDACTED]" if normalized in {
                "authorization", "apikey", "password", "secret", "credential", "accesstoken",
                "lmbenchtoken"} else _redact(item)
        return result
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return sanitize(value) if isinstance(value, str) else value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _passed(row):
    result = row["result"]
    return row["status"] == "passed" and result.get("passed") is True and result.get("valid") is True \
        and (row.get("kind") in {"quality","capacity"} or result.get("resources", {}).get("foreign_workload_overlap") is not True)


def context_reasons(result, config, mode=None):
    """A declared context size alone cannot qualify a full-context measurement."""
    evidence = result.get("token_evidence", {})
    measurement = config["measurement"]
    target, tolerance = measurement["full_prompt_tokens"], measurement["prompt_target_absolute_tolerance_tokens"]
    reasons = []
    if evidence.get("effective_context_tokens") != measurement["context_window_tokens"]:
        reasons.append("effective_64k_context_unverified")
    if evidence.get("tokenization_verified") is not True:
        reasons.append("exact_tokenization_unverified")
    expected, observed = evidence.get("expected_prompt_tokens"), evidence.get("prompt_tokens")
    if type(expected) is not int or not target - tolerance <= expected <= target:
        reasons.append("full_prompt_target_unverified")
    if type(observed) is not int or not target - tolerance <= observed <= target:
        reasons.append("actual_full_prompt_count_unverified")
    if type(expected) is int and type(observed) is int and abs(expected - observed) > tolerance:
        reasons.append("actual_prompt_count_mismatch")
    if evidence.get("truncated") is not False:
        reasons.append("truncation_not_excluded")
    if evidence.get("context_shift") is not False:
        reasons.append("context_shift_not_excluded")
    if mode == "fresh":
        cached = evidence.get("cached_tokens")
        if cached is not None and (type(cached) is not int or cached < 0 or cached >
                measurement["cold_prefix_expected_cached_tokens_max"]):
            reasons.append("fresh_prefix_reuse")
        if evidence.get("cache_isolation_verified") is not True:
            reasons.append("fresh_cache_isolation_unverified")
    if mode == "cached":
        cached = evidence.get("cached_tokens")
        if type(cached) is not int or cached < 0:
            reasons.append("controlled_cached_counter_unverified")
        if evidence.get("prefix_reuse_verified") is not True and \
                evidence.get("controlled_prefix_experiment_verified") is not True:
            reasons.append("controlled_prefix_experiment_unverified")
    return reasons


class Report:
    def __init__(self, config, state):
        self.config, self.state = config, state

    def attempts(self):
        configurations = {row["id"]: json.loads(row["data"]) for row in self.state.db.execute(
            "SELECT id,data FROM entities WHERE kind='configuration'")}
        output = []
        for row in self.state.db.execute("""SELECT a.*,j.key_json,j.status AS job_status
                FROM attempts a JOIN jobs j ON j.id=a.job_id ORDER BY a.id"""):
            item = dict(row)
            item["job_key"] = json.loads(item.pop("key_json"))
            item["result"] = json.loads(item.get("result") or "{}")
            identifier = item["result"].get("configuration_id", item["job_key"].get("configuration_id"))
            item["configuration_id"] = identifier
            item["configuration"] = configurations.get(identifier, {})
            item["kind"] = item["result"].get("kind", item["job_key"].get("kind"))
            item["metrics"] = item["result"].get("metrics", {})
            output.append(_redact(item))
        return output

    def _latest_completed(self, rows):
        # Qualification uses current logical jobs, including pending replacements
        # with no attempt yet. Historical attempts remain untouched in exports.
        if not rows:
            return []
        latest, fixture_by_hash = {}, {}
        for row in rows:
            if row["job_id"] not in latest or row["id"] > latest[row["job_id"]]["id"]:
                latest[row["job_id"]] = row
            if row["kind"] == "quality":
                fixture = row["result"].get("fixture", row["result"].get("fixture_id"))
                if fixture:
                    fixture_by_hash[row["job_key"].get("fixture_prompt_hash")] = fixture
        identifiers = sorted({row["configuration_id"] for row in rows if row.get("configuration_id")})
        if not identifiers:
            return []
        placeholders = ",".join("?" for _ in identifiers)
        current = list(self.state.db.execute("SELECT jobs.rowid AS job_order,jobs.*, "
            "(SELECT COUNT(*) FROM attempts WHERE attempts.job_id=jobs.id) AS attempt_count FROM jobs WHERE "
            "json_extract(key_json,'$.configuration_id') IN (" + placeholders + ") ORDER BY jobs.rowid", identifiers))
        current_by_id = {job["id"]: job for job in current}
        logical = {}
        for job in current:
            key = json.loads(job["key_json"])
            attempt = latest.get(job["id"])
            result = json.loads(job["result"] or "{}")
            kind = key.get("kind", result.get("kind", attempt["kind"] if attempt else None))
            if self._retired_unstarted_placeholder(job, key, result, current_by_id):
                continue
            if not self._current_protocol(key, kind):
                continue
            real_protocol = "logical_plan_hash" in key
            if kind == "quality":
                # A demo's extra coding run cannot replace a required grade.
                if real_protocol and (key.get("mode") != "fresh" or key.get("replicate") != 0):
                    continue
                fixture = result.get("fixture", result.get("fixture_id")) or fixture_by_hash.get(key.get("fixture_prompt_hash"))
                bucket = (key.get("configuration_id"), kind, fixture or key.get("fixture_prompt_hash"), key.get("seed"))
            elif kind == "demo":
                bucket = (key.get("configuration_id"), kind)
            else:
                bucket = (key.get("configuration_id"), kind, key.get("seed") if kind == "capacity" else None,
                    key.get("mode"), key.get("replicate"), key.get("target_tokens"))
            logical[bucket] = (job, attempt)
        return [attempt for job, attempt in logical.values() if attempt is not None
            and job["status"] in {"passed", "failed"} and attempt["status"] == job["status"]
            and attempt["finished"] is not None]

    @staticmethod
    def _retired_unstarted_placeholder(job, key, result, current_by_id):
        reason = "stop_before_execution_unpacked_placeholder"
        if key.get("kind") != "capacity" or job["status"] != "skipped" or job["reason"] != reason \
                or result.get("reason") != reason or result.get("unstarted") is not True \
                or job["attempt_count"] != 0:
            return False
        replacement = current_by_id.get(result.get("superseded_by"))
        if replacement is None or replacement["id"] == job["id"]:
            return False
        base = {name: value for name, value in key.items() if name != "fixture_prompt_hash"}
        replacement_key = json.loads(replacement["key_json"])
        replacement_base = {name: value for name, value in replacement_key.items() if name != "fixture_prompt_hash"}
        # The reconciliation receipt identifies a generated pre-packing key,
        # rather than permitting arbitrary skipped replacements to disappear.
        return base == replacement_base and key.get("fixture_prompt_hash") == digest({"unpacked_plan": base}) \
            and replacement_key.get("fixture_prompt_hash") not in {None, key["fixture_prompt_hash"]}

    def _current_protocol(self, key, kind):
        if "logical_plan_hash" not in key or kind not in {"quality", "capacity", "timing", "throughput", "common_byte"}:
            # Legacy synthetic harness records lack scheduler protocol metadata.
            # Handoff demos have their own immutable launch-plan hash contract.
            return True
        # Scheduler imports Report before freezing these constants: import only
        # when qualification is called, never at report module import time.
        from .scheduler import PROTOCOL_HASHES
        files = ("quality.py", "tools.py", "schema.py") if kind == "quality" else \
            ("common_byte.py", "measurement.py") if kind == "common_byte" else \
            ("measurement.py", "prompts.py", "regional_prompt.py")
        expected = digest({"protocol": {name: PROTOCOL_HASHES[name] for name in files},
            "run": self.state.run["id"], "fixture_hash": key.get("fixture_prompt_hash") if kind == "quality" else None,
            "kind": kind, "seed": key.get("seed"), "mode": key.get("mode"),
            "replicate": key.get("replicate"), "target": key.get("target_tokens")})
        return key["logical_plan_hash"] == expected

    def _configuration(self, identifier, data, all_rows):
        rows = self._latest_completed(all_rows)
        grading = self.config["grading"]
        regular_required = {(fixture, seed) for fixture in grading["regular_fixture_ids"] for seed in grading["seeds"]}
        long_required = {(fixture, seed) for fixture in grading["long_fixture_ids"] for seed in grading["seeds"]}
        regular, long = {}, {}
        capacities = {}
        timing = {"fresh": {}, "cached": {}}
        throughputs = []
        throughput_samples = 0
        evidence_failures = []
        demo = False
        for row in rows:
            result, kind = row["result"], row["kind"]
            if kind == "quality":
                key = (result.get("fixture", result.get("fixture_id")), result.get("seed", row["job_key"].get("seed")))
                group = result.get("group")
                if group == "regular" and key in regular_required:
                    regular[key] = _passed(row)
                if group == "long" and key in long_required:
                    reasons = context_reasons(result, self.config)
                    long[key] = _passed(row) and not reasons
                    evidence_failures.extend({"attempt_id": row["id"], "reason": reason} for reason in reasons)
            if kind == "capacity":
                seed = result.get("seed", row["job_key"].get("seed"))
                if seed in grading["capacity_marker_seeds"]:
                    reasons = context_reasons(result, self.config)
                    values = result.get("retrieved_values")
                    capacities[seed] = _passed(row) and not reasons and result.get("markers_passed") == 3 \
                        and isinstance(values, list) and len(values) == 3
                    evidence_failures.extend({"attempt_id": row["id"], "reason": reason} for reason in reasons)
            if kind == "timing" and result.get("final_validation") is True:
                mode = result.get("mode", row["metrics"].get("mode"))
                if mode in timing:
                    reasons = context_reasons(result, self.config, mode)
                    metrics = row["metrics"]
                    first_action = metrics.get("first_action_seconds")
                    valid = _passed(row) and not reasons and result.get("stability_passed") is True \
                        and metrics.get("first_action_valid") is True and _finite(first_action) \
                        and first_action >= 0 and metrics.get("invalid_tool_calls", 0) == 0
                    replicate = result.get("replicate", row["job_key"].get("replicate"))
                    if valid and type(replicate) is int:
                        timing[mode][replicate] = row
                    evidence_failures.extend({"attempt_id": row["id"], "reason": reason} for reason in reasons)
            if kind == "throughput" and _passed(row) and not context_reasons(result, self.config):
                metrics = row["metrics"]
                generated = metrics.get("generated_tokens")
                if type(generated) is int and generated >= 64:
                    throughput_samples += 1
                rate = decode_throughput(generated, metrics.get("decode_seconds"))
                if type(generated) is int and generated >= 64 and rate is not None \
                        and metrics.get("sustained_throughput_qualified") is True:
                    throughputs.append(rate)
            if kind == "demo" and _passed(row) and result.get("isolated") is True:
                demo = True
        stability_failures = []
        runtime_reasons = ("crash", "oom", "numeric", "loop", "capacity", "timeout", "truncat", "context_shift")
        for row in all_rows:
            if not self._current_protocol(row["job_key"], row["kind"]):
                continue
            result = row["result"]
            if result.get("final_validation") is not True:
                continue
            reason = str(row.get("reason") or "")
            if result.get("stability_passed") is False or row["status"] == "failed" or \
                    (row["status"] == "invalid" and any(part in reason.lower() for part in runtime_reasons)):
                stability_failures.append({"attempt_id": row["id"], "status": row["status"], "reason": reason})
        regular_passes, long_passes = sum(regular.values()), sum(long.values())
        required_fresh = self.config["measurement"]["final_fresh_repetitions"]
        required_cached = self.config["measurement"]["final_cached_repetitions"]
        gates = {
            "regular_coverage": set(regular) == regular_required,
            "regular_quality": regular_passes >= grading["regular_minimum_passes"],
            "long_coverage": set(long) == long_required,
            "long_quality": long_passes >= grading["long_minimum_passes"],
            "capacity_all_nine_values": all(capacities.get(seed) for seed in grading["capacity_marker_seeds"]),
            "twenty_valid_fresh": len(timing["fresh"]) >= required_fresh,
            "twenty_valid_cached": len(timing["cached"]) >= required_cached,
            "full_context_throughput_sample": throughput_samples > 0,
            "runtime_stability": not stability_failures,
            "isolated_agent_demo": demo,
        }
        if data.get('requested_settings',{}).get('speculation','off') in ('draft-mtp','draft-dflash'):
            gates['speculation_real_drafting']=any(row.get('metrics',{}).get('speculation_counters_verified') is True
                and row['metrics'].get('speculation_drafted_tokens',0)>0 and _passed(row) for row in rows)
            gates['speculation_cold_latency_regression']=False
        latency = {}
        for mode, replicates in timing.items():
            mode_rows = list(replicates.values())
            latency[mode] = {"first_action": sample_summary(row["metrics"]["first_action_seconds"] for row in mode_rows),
                "first_stream_token": sample_summary(row["metrics"]["first_stream_token_seconds"] for row in mode_rows
                    if _finite(row["metrics"].get("first_stream_token_seconds"))),
                "samples": [{"attempt_id": row["id"], "replicate": replicate, "metrics": row["metrics"]}
                    for replicate, row in sorted(replicates.items())]}
        peak = max((row["result"].get("resources", {}).get("peak_dedicated_gpu_bytes") for row in rows
                    if _finite(row["result"].get("resources", {}).get("peak_dedicated_gpu_bytes"))), default=None)
        resource_uncertainty = [{"attempt_id": row["id"], "reason":
            "resource_interval_incomplete" if row["result"].get("resources", {}).get("resource_window_complete") is False else
            "gpu_workload_attribution_uncertain_or_unavailable"} for row in rows
            if row["kind"] in {"capacity", "timing", "throughput"} and (
                row["result"].get("resources", {}).get("foreign_workload_attribution_uncertain") is True or
                row["result"].get("resources", {}).get("foreign_workload_evidence_available") is False or
                row["result"].get("resources", {}).get("resource_window_complete") is False)]
        return {"configuration_id": identifier, "configuration": data, "qualified": all(gates.values()),
            "gates": gates, "failed_gates": [gate for gate, passed in gates.items() if not passed],
            "regular": {"passed": regular_passes, "attempted": len(regular), "total": len(regular_required),
                "missing": [{"fixture": fixture, "seed": seed} for fixture, seed in sorted(regular_required - set(regular))]},
            "long": {"passed": long_passes, "attempted": len(long), "total": len(long_required),
                "missing": [{"fixture": fixture, "seed": seed} for fixture, seed in sorted(long_required - set(long))]},
            "capacity": {str(seed): capacities.get(seed, False) for seed in grading["capacity_marker_seeds"]},
            "latency": latency, "throughput": {**sample_summary(throughputs),
                "actual_count_samples": throughput_samples,
                "median": statistics.median(throughputs) if throughputs else None,
                "exact_sustained_64k": bool(throughputs)},
            "peak_dedicated_gpu_bytes": peak, "stability_failures": stability_failures,
            "evidence_failures": evidence_failures, "resource_attribution_uncertainty": resource_uncertainty,
            "attempt_count": len(all_rows)}

    def summarize(self, rows=None):
        rows = self.attempts() if rows is None else rows
        configurations = {row["id"]: _redact(json.loads(row["data"])) for row in self.state.db.execute(
            "SELECT id,data FROM entities WHERE kind='configuration'")}
        for row in rows:
            if row["configuration_id"]:
                configurations.setdefault(row["configuration_id"], row["configuration"])
        scores = [self._configuration(identifier, data, [row for row in rows if row["configuration_id"] == identifier])
                  for identifier, data in sorted(configurations.items())]
        by_id={score['configuration_id']:score for score in scores}
        for score in scores:
            settings=score['configuration'].get('requested_settings',{})
            if settings.get('speculation','off') not in ('draft-mtp','draft-dflash'):continue
            baseline=by_id.get(settings.get('speculation_baseline_configuration_id'))
            actual=score['latency']['fresh']['first_action']['p95']
            reference=baseline['latency']['fresh']['first_action']['p95'] if baseline else None
            valid=baseline is not None and baseline['gates']['twenty_valid_fresh'] and score['gates']['twenty_valid_fresh'] \
                and _finite(reference) and reference>0 and _finite(actual)
            limit=self.config['tuning']['speculation_cold_p95_regression_limit_fraction']
            score['gates']['speculation_cold_latency_regression']=bool(valid and actual<=reference*(1+limit))
            score['speculation_comparison']={'baseline_configuration_id':settings.get('speculation_baseline_configuration_id'),
                'fresh_p95_ratio':actual/reference if valid else None,'regression_limit_fraction':limit,
                'draft_weight_bytes':settings.get('draft_model',{}).get('bytes')}
            score['qualified']=all(score['gates'].values())
            score['failed_gates']=[gate for gate,passed in score['gates'].items() if not passed]
        qualified = sorted((score for score in scores if score["qualified"]), key=lambda score:
            score["latency"]["fresh"]["first_action"]["p95"])
        ranking, uncertainty = [], []
        top_tie_unresolved = False
        pending = list(qualified)
        while pending:
            best = pending[0]["latency"]["fresh"]["first_action"]["p95"]
            tied = [score for score in pending if
                score["latency"]["fresh"]["first_action"]["p95"] - best <= 1. and
                score["latency"]["fresh"]["first_action"]["p95"] <= best * 1.1]
            exact = all(score["throughput"]["exact_sustained_64k"] for score in tied)
            if len(tied) > 1 and not exact:
                if not ranking:
                    top_tie_unresolved = True
                uncertainty.append("Latency tie has incomplete comparable exact sustained 64K throughput: " +
                                   ", ".join(score["configuration_id"] for score in tied))
            elif len(tied) > 1:
                tied.sort(key=lambda score: (-score["throughput"]["median"],
                    score["latency"]["cached"]["first_action"]["p95"],
                    -score["regular"]["passed"] - score["long"]["passed"],
                    score["peak_dedicated_gpu_bytes"] if score["peak_dedicated_gpu_bytes"] is not None else float("inf")))
            ranking.extend(tied)
            pending = [score for score in pending if score not in tied]
        uncertain = [score["configuration_id"] for score in scores if score["resource_attribution_uncertainty"]]
        if uncertain:
            uncertainty.append("GPU workload attribution is uncertain or unavailable, or a resource interval is incomplete, for: " +
                ", ".join(uncertain) + ". Timing comparisons carry this measurement limit; aggregate WSL/Docker activity does not identify individual process ownership.")
        failures = {}
        for row in rows:
            if row["status"] in {"failed", "invalid", "skipped"}:
                reason = row.get("reason") or row["status"]
                failures[reason] = failures.get(reason, 0) + 1
        pareto = []
        for score in qualified:
            quality = score["regular"]["passed"] / score["regular"]["total"] + score["long"]["passed"] / score["long"]["total"]
            latency = score["latency"]["fresh"]["first_action"]["p95"]
            speed = score["throughput"]["median"]
            dominated = False
            for other in qualified:
                if other is score or speed is None or other["throughput"]["median"] is None:
                    continue
                other_quality = other["regular"]["passed"] / other["regular"]["total"] + other["long"]["passed"] / other["long"]["total"]
                other_latency = other["latency"]["fresh"]["first_action"]["p95"]
                other_speed = other["throughput"]["median"]
                if other_quality >= quality and other_latency <= latency and other_speed >= speed and \
                        (other_quality > quality or other_latency < latency or other_speed > speed):
                    dominated = True
            if not dominated:
                pareto.append(score["configuration_id"])
        return {"run": self.state.run, "attempt_count": len(rows), "configurations": scores,
            "ranking": [score["configuration_id"] for score in ranking],
            "winner_id": ranking[0]["configuration_id"] if ranking and not top_tie_unresolved else None,
            "qualification_status": "qualified_tie_unresolved" if top_tie_unresolved else
                "qualified" if ranking else "nothing_qualifies", "ranking_uncertainty": uncertainty,
            "pareto_alternatives": pareto, "failure_counts": failures,
            "coverage_note": "Finite recorded model/engine/settings search. Skipped or unavailable setups are not measured losers. "
                "Fresh and cached timing remain separate. Twenty samples support nearest-rank p50/p95, not universal superiority."}

    def write(self):
        artifacts = Path(self.config["paths"]["artifacts"])
        artifacts.mkdir(parents=True, exist_ok=True)
        rows = self.attempts()
        summary = self.summarize(rows)
        atomic_json(artifacts / "results.json", rows)
        csv_path = artifacts / "results.csv"
        fields = ("id", "job_id", "number", "status", "started", "finished", "reason", "job_status",
                  "configuration_id", "kind", "job_key", "configuration", "metrics", "result")
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: canonical(row[field]) if isinstance(row.get(field), (dict, list)) else row.get(field)
                                 for field in fields})
        manifest = {"engines": [], "configurations": []}
        for row in self.state.db.execute("SELECT kind,id,data FROM entities WHERE kind IN ('engine','configuration') ORDER BY kind,id"):
            manifest["engines" if row["kind"] == "engine" else "configurations"].append(
                {"id": row["id"], **_redact(json.loads(row["data"]))})
        atomic_json(artifacts / "engine-manifest.json", manifest)
        from .winner_launch import generate_winner_launcher
        summary["winner_launcher"] = generate_winner_launcher(self.config,self.state,summary)
        atomic_json(artifacts / "ranking.json", summary)
        markdown = self._recommendation(summary)
        (artifacts / "recommendation.md").write_text(markdown, encoding="utf-8")
        (artifacts / "report.html").write_text(self._html(summary, markdown), encoding="utf-8")
        return summary

    generate = write

    def _recommendation(self, summary):
        winner = summary["winner_id"]
        if winner:
            opening = "Qualified 64K recommendation: **" + winner + "**."
        elif summary["ranking"]:
            opening = "64K candidates qualify, but a latency tie cannot be resolved from comparable exact sustained throughput."
        else:
            opening = "**Nothing qualifies as a verified 64K coding-agent winner.**"
        if summary["run"]["started"] is None:
            opening += " No real runtime probe or new-weight acquisition has started; these are harness records, not a measured ranking."
        text = [opening, summary["coverage_note"],
            "Ranking uses fresh 64K p95 time to the first completely parsed schema-valid tool action. "
            "Latency ties require both 10% and 1 second; exact sustained decode throughput then breaks ties. "
            "Cached latency is a separate secondary metric. Shared GPU allocation is reported as a delta from idle baseline and does not prove spill."]
        table = ["| Configuration | Qualifies | Regular | Long | Fresh action p95 (s) | Cached action p95 (s) | Sustained 64K tok/s median | Failed gates |",
                 "|---|---|---|---|---:|---:|---:|---|"]
        order = {identifier: index for index, identifier in enumerate(summary["ranking"])}
        for score in sorted(summary["configurations"], key=lambda score: (order.get(score["configuration_id"], 100000), score["configuration_id"])):
            number = lambda value: "unknown" if value is None else format(value, ".3f")
            table.append("| " + " | ".join((score["configuration_id"], str(score["qualified"]),
                str(score["regular"]["passed"]) + "/" + str(score["regular"]["total"]),
                str(score["long"]["passed"]) + "/" + str(score["long"]["total"]),
                number(score["latency"]["fresh"]["first_action"]["p95"]),
                number(score["latency"]["cached"]["first_action"]["p95"]),
                number(score["throughput"]["median"]), ", ".join(score["failed_gates"]))) + " |")
        text.append("\n".join(table))
        text.extend(summary["ranking_uncertainty"])
        text.append("All " + str(summary["attempt_count"]) + " attempts, including retries, invalid samples and skips, are retained in results.json/results.csv. "
                    "Exact configuration/version/launch metadata is in engine-manifest.json; raw logs remain in .logs.")
        if summary["failure_counts"]:
            text.append("Recorded failures and unavailable reasons: " + canonical(summary["failure_counts"]))
        text.append("The aspirations (fresh stream <10 s, fresh action p95 <15 s, cached action p95 <2 s, decode ≥40 tok/s) are targets, not claims. "
                    "A smaller-context endpoint, stable cached prefix or repository retrieval can be useful separately; none establishes fresh 64K qualification.")
        return "\n\n".join(text) + "\n"

    def _html(self, summary, markdown):
        escape = html.escape
        body = "<h1>Local coding-agent benchmark</h1><pre>" + escape(markdown) + "</pre>"
        for score in summary["configurations"]:
            body += "<details><summary>" + escape(score["configuration_id"]) + " — evidence and exact configuration</summary><pre>" \
                + escape(json.dumps(score, ensure_ascii=False, indent=2)) + "</pre></details>"
        body += "<details><summary>All attempt evidence</summary><p><a href='results.json'>JSON results</a> · " \
                "<a href='results.csv'>CSV results</a> · <a href='engine-manifest.json'>Engine manifest</a></p></details>"
        return "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>" \
            "<title>Local coding-agent benchmark</title><style>body{font:16px system-ui;max-width:1200px;margin:2rem auto;padding:0 1rem;" \
            "color:#17202b;background:#f8fafc}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;background:white;padding:1rem;" \
            "border:1px solid #dae0e6;border-radius:8px}details{margin:1rem 0}summary{cursor:pointer;font-weight:600}</style><body>" + body + "</body></html>\n"


def report(config, state):
    """CLI-compatible entry point; report creation never starts the run clock."""
    return Report(config, state).write()

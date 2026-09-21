"""Resumable coordinate exploration. Only measured, gated screens can advance."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
import statistics

from .config import atomic_json, digest
from .doctor import sanitize
from .report import Report
from .search_core import checkpoint_model, quality_qualified, regular_and_long, screen_latency
from .native_marker_reasoning import marker_reasoning_eligible


def _smoke_passed(config, candidate):
    jobs = config["grading"]["smoke_jobs"]
    rows = candidate.get("smoke", [])
    expected = {(job["fixture"], job["seed"]) for job in jobs}
    found = {(row.get("fixture", row.get("fixture_id")), row.get("seed")) for row in rows
        if row.get("passed") is True and row.get("valid") is True}
    return len(rows) == len(jobs) and found == expected


def _promising(config, candidate):
    latency = screen_latency(candidate)
    return candidate.get("eligible") is True and _smoke_passed(config, candidate) and math.isfinite(latency)


def _section(help_text, flag):
    lines = help_text.splitlines()
    pattern = re.compile(re.escape(flag) + r"(?![a-z0-9-])")
    for index, line in enumerate(lines):
        if not pattern.search(line):
            continue
        section = [line]
        for following in lines[index + 1:]:
            if following.lstrip().startswith("-") and re.search(r"--[a-z0-9-]+", following):
                break
            section.append(following)
        return "\n".join(section)
    return ""


def _classify(reason):
    value = str(reason).lower()
    for category, markers in (("memory", ("oom", "memory", "wsl_cap")),
            ("unsupported_control", ("unsupported", "unknown argument", "unrecognized")),
            ("quality_or_tools", ("smoke", "quality", "tool", "action")),
            ("context", ("context", "capacity", "truncat")),
            ("runtime", ("crash", "timeout", "numeric", "loop")),
            ("budget", ("budget", "stop_after_current"))):
        if any(marker in value for marker in markers):
            return category
    return "infrastructure_or_measurement"


class _Tuner:
    def __init__(self, config, state, collector, screens):
        self.config, self.state, self.collector = config, state, collector
        self.screens = list(screens)
        self.input_screens = list(screens)
        self.models = {}
        self.phase = state.get_control("tuning_phase")
        if self.phase is None:
            now = state.clock()
            deadline = now + config["phase_budget_hours"]["tuning_engine_comparisons"] * 3600
            if state.run["deadline"] is not None:
                deadline = min(deadline, state.run["deadline"] - 7200)
            self.phase = {"started": now, "deadline": deadline, "selected_families": None, "status": "active"}
            state.control("tuning_phase", self.phase)
        self.stopped = False
        self.new_configurations_exhausted = False
        self.qualified_ids = {item["configuration_id"] for item in quality_qualified(config, state, self.screens)}
        self.scores = {item["configuration_id"]: item for item in Report(config, state).summarize()["configurations"]}
        for row in state.db.execute("SELECT data FROM entities WHERE kind='tuning_proposal'"):
            proposal = json.loads(row[0])
            if self.proposal_current(proposal):
                self._append(proposal["result"])
        self.qualified_ids = {item["configuration_id"] for item in quality_qualified(config, state, self.screens)}

    def model(self, candidate):
        key = candidate["model"]
        if key not in self.models:
            self.models[key] = checkpoint_model(self.state, key, self.config["installed_model_candidates"])
        return self.models[key]

    def screen_key(self, runtime, model, settings):
        # Import lazily: search loads this module after its baseline stages.
        from .search import stage_key
        return 'planned_screen:' + stage_key(runtime, model, settings)

    def proposal_current(self, proposal):
        result = proposal.get("result")
        if proposal.get("status") != "completed" or not result or not proposal.get("runtime"):
            return False
        try:
            model = checkpoint_model(self.state, proposal["model"], self.config["installed_model_candidates"])
            key = self.screen_key(proposal["runtime"], model, proposal["settings"])
        except (KeyError, OSError, RuntimeError, ValueError):
            return False
        from .search import pending_for_stage
        return result.get("planned_screen_key") == key and bool(self.state.get_control(key)) \
            and not pending_for_stage(self.state, result.get("configuration_id"))

    def family(self, candidate):
        model = self.model(candidate)
        return str(candidate.get("weight_family") or model.get("weight_family") or model.get("family") or model.get("base_model") or model["id"])

    def group(self, candidate):
        return digest([candidate["engine"], candidate["model"], candidate.get("runtime", {})])

    def rank(self, candidate):
        score = self.scores.get(candidate.get("configuration_id"), {})
        quality = sum(score.get(group, {}).get("passed", 0) / max(1, score.get(group, {}).get("total", 1))
            for group in ("regular", "long"))
        return (candidate.get("configuration_id") not in self.qualified_ids, -quality,
            not _smoke_passed(self.config, candidate), screen_latency(candidate), candidate["engine"], candidate["model"])

    def _append(self, candidate):
        identifier = candidate.get("configuration_id")
        if identifier:
            for index, item in enumerate(self.screens):
                if item.get("configuration_id") == identifier:
                    self.screens[index] = candidate
                    return
            self.screens.append(candidate)
        elif candidate not in self.screens:
            self.screens.append(candidate)

    def decision(self, candidate, stage, reason, settings=None, status="considered_skip"):
        data = {"stage": stage, "engine": candidate.get("engine"), "model": candidate.get("model"),
            "settings": settings, "status": status, "reason": sanitize(reason), "reason_class": _classify(reason)}
        self.state.entity("tuning_decision", digest(data), data)
        self.export()

    def export(self):
        rows = [{"id": row["id"], **json.loads(row["data"])} for row in self.state.db.execute(
            "SELECT id,data FROM entities WHERE kind IN ('tuning_proposal','tuning_decision') ORDER BY kind,id")]
        atomic_json(Path(self.config["paths"]["artifacts"]) / "tuning-exploration.json",
            {"phase": self.phase, "proposals_and_decisions": rows,
             "note": "Screen timing is exploratory. Each prospective final settings/profile needs full quality qualification; synthetic speed never qualifies it."})

    def available(self, candidate, stage, estimated_seconds=0):
        if self.stopped:
            return False
        try:
            self.state.check_budget(estimated_seconds=estimated_seconds)
        except RuntimeError as error:
            self.decision(candidate, stage, str(error))
            self.stopped = True
            self.phase["status"] = "stopped_budget_or_request"
            self.state.control("tuning_phase", self.phase)
            return False
        if self.state.clock() + estimated_seconds > self.phase["deadline"]:
            self.decision(candidate, stage, "tuning_phase_budget_exhausted")
            self.stopped = True
            self.phase["status"] = "stopped_phase_budget"
            self.state.control("tuning_phase", self.phase)
            return False
        return True

    def trial(self, anchor, changes, stage):
        settings = {**anchor["settings"], **changes}
        if anchor["runtime"]["kind"] == "tabby" and "cache_mode" in changes:
            settings.pop("cache_k", None)
            settings.pop("cache_v", None)
        runtime, model = anchor["runtime"], self.model(anchor)
        screen_key = self.screen_key(runtime, model, settings)
        identifier = digest([self.state.run["id"], anchor["engine"], runtime,
            model.get("sha256") or model.get("revision") or model["id"], settings, screen_key])
        row = self.state.db.execute("SELECT data FROM entities WHERE kind='tuning_proposal' AND id=?", (identifier,)).fetchone()
        if row:
            existing = json.loads(row[0])
            if self.proposal_current(existing):
                result = existing.get("result")
                if result:
                    self._append(result)
                return result
        # Reuse exact completed baseline settings, including their actual smoke
        # outcomes. A failed smoke cannot acquire exploratory timing eligibility.
        from .search import pending_for_stage
        same = next((item for item in self.screens if self.group(item) == self.group(anchor) and item.get("settings") == settings
            and not pending_for_stage(self.state, item.get("configuration_id"))), None)
        if same:
            # Baseline stages use settings=None; tuning stages use explicit
            # settings. Either current marker must prove these actual settings.
            for key in (screen_key, self.screen_key(runtime, model, None)):
                done = self.state.get_control(key)
                if done and done.get("configuration_id") == same.get("configuration_id") and done.get("settings") == settings:
                    current = {**done, "planned_screen_key": key}
                    self._append(current)
                    return current
        estimate = self.config["limits"]["server_start_timeout_seconds"] + self.config["limits"]["per_64k_request_timeout_seconds"]
        if not self.available(anchor, stage, estimate):
            return None
        count = self.state.db.execute("SELECT COUNT(*) FROM entities WHERE kind='configuration'").fetchone()[0]
        if self.new_configurations_exhausted or count >= self.config["limits"]["maximum_unique_runtime_configurations"]:
            self.decision(anchor, stage, "configuration_budget_exhausted", settings)
            self.new_configurations_exhausted = True
            self.phase["configuration_limit_reached"] = True
            return None
        proposal = {"engine": anchor["engine"], "model": anchor["model"], "family": self.family(anchor),
            "runtime": runtime, "settings": settings, "stage": stage, "status": "running"}
        self.state.entity("tuning_proposal", identifier, proposal)
        self.export()
        try:
            from .search import run_screen
            result = run_screen(self.config, self.state, self.collector, runtime, model, settings=settings, tuning=True)
        except (OSError, RuntimeError, ValueError) as error:
            if str(error) in {"budget_exhausted", "stop_after_current"}:
                self.decision(anchor, stage, str(error), settings)
                self.stopped = True
                # Keep the running proposal to resume its scheduler checkpoints.
                return None
            result = {"engine": anchor["engine"], "model": anchor["model"], "runtime": runtime, "settings": settings,
                "eligible": False, "valid": False, "reason": sanitize(str(error))}
        status = "pending_measurements" if pending_for_stage(self.state, result.get("configuration_id")) else "completed"
        proposal.update(status=status, result=result, reason=result.get("reason"), reason_class=_classify(result.get("reason")))
        self.state.entity("tuning_proposal", identifier, proposal)
        self._append(result)
        self.export()
        if "configuration_budget_exhausted" in str(result.get("reason")):
            self.new_configurations_exhausted = True
            self.phase["configuration_limit_reached"] = True
        return result

    def metadata(self, candidate):
        row = self.state.db.execute("SELECT data FROM entities WHERE kind='configuration' AND id=?", (candidate.get("configuration_id"),)).fetchone()
        data = json.loads(row[0]) if row else {}
        text = ""
        path = data.get("help_log")
        if path and Path(path).is_file():
            text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
        return data, text, set(re.findall(r"--[a-z0-9-]+", text))

    def best(self, candidates, default=None):
        eligible = [item for item in candidates if item and _promising(self.config, item)]
        return min(eligible, key=screen_latency) if eligible else default

    def cache_and_prefill(self, anchor):
        engine, kind = anchor["engine"], anchor["runtime"]["kind"]
        metadata, help_text, flags = self.metadata(anchor)
        pool = [anchor]
        if kind == "tabby":
            for mode in ("FP16", "8,8", "8,4", "4,4"):
                pool.append(self.trial(anchor, {"cache_mode": mode, "chunk_size": 2048}, "tabby_cache"))
            return self.best(pool)
        if engine in {"vllm", "sglang"}:
            fraction_flag = "--gpu-memory-utilization" if engine == "vllm" else "--mem-fraction-static"
            prefill_flag = "--max-num-batched-tokens" if engine == "vllm" else "--chunked-prefill-size"
            if fraction_flag in flags:
                for fraction in self.config["tuning"]["linux_gpu_memory_utilization_candidates"]:
                    pool.append(self.trial(anchor, {"gpu_memory_utilization": fraction}, "linux_memory"))
            else:
                self.decision(anchor, "linux_memory", "unsupported_memory_fraction_in_pinned_help")
            center = self.best(pool, anchor)
            if prefill_flag in flags:
                for tokens in self.config["tuning"]["vllm_max_num_batched_tokens"]:
                    pool.append(self.trial(center, {"prefill_tokens": tokens}, "linux_prefill"))
            else:
                self.decision(anchor, "linux_prefill", "unsupported_prefill_budget_in_pinned_help")
            if "--kv-cache-dtype" in flags:
                choices = set(re.findall(r"\b(?:auto|fp8|fp8_e4m3|fp8_e5m2)\b", _section(help_text, "--kv-cache-dtype")))
                center = self.best(pool, anchor)
                for dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
                    if dtype in choices:
                        pool.append(self.trial(center, {"kv_cache_dtype": dtype}, "linux_cache"))
            else:
                self.decision(anchor, "linux_cache", "unsupported_cache_dtype_in_pinned_help")
            return self.best(pool)
        native = kind == "native" or engine == "ik-llama"
        if native:
            choices_k = set(re.findall(r"\b(?:f16|q8_0|q4_0)\b", _section(help_text, "--cache-type-k")))
            choices_v = set(re.findall(r"\b(?:f16|q8_0|q4_0)\b", _section(help_text, "--cache-type-v")))
        else:
            choices_k = choices_v = set(metadata.get("supported_cache_types", []))
        for key, value in self.config["tuning"]["kv_pairs_primary"]:
            if key in choices_k and value in choices_v:
                pool.append(self.trial(anchor, {"cache_k": key, "cache_v": value}, "cache"))
            elif "cache_k" in anchor["settings"]:
                self.decision(anchor, "cache", "unsupported_cache_pair_in_observed_controls", {"cache_k": key, "cache_v": value})
        center = self.best(pool, anchor)
        oom = any(item and _classify(item.get("reason")) == "memory" for item in pool)
        if native and (oom or center["settings"].get("cache_k") != "f16"):
            for key, value in self.config["tuning"]["kv_pairs_followup"]:
                if key in choices_k and value in choices_v:
                    pool.append(self.trial(anchor, {"cache_k": key, "cache_v": value}, "asymmetric_cache"))
        else:
            self.decision(anchor, "asymmetric_cache", "no_useful_supported_asymmetric_cache_lead")
        center = self.best(pool, anchor)
        if native and {"--batch-size", "--ubatch-size"} <= flags:
            for batch, ubatch in self.config["tuning"]["batch_ubatch_initial"]:
                if ubatch <= batch:
                    pool.append(self.trial(center, {"batch": batch, "ubatch": ubatch}, "batch_pair"))
        elif native:
            self.decision(anchor, "batch_pair", "unsupported_batch_controls_in_pinned_help")
        return self.best(pool)

    def deepen(self, anchor):
        metadata, _, flags = self.metadata(anchor)
        kind, engine = anchor["runtime"]["kind"], anchor["engine"]
        pool = [anchor]
        native = kind == "native" or engine == "ik-llama"
        if native and {"--batch-size", "--ubatch-size"} <= flags:
            s = anchor["settings"]
            logical = sorted({pair[0] for pair in self.config["tuning"]["batch_ubatch_initial"]})
            physical = sorted({pair[1] for pair in self.config["tuning"]["batch_ubatch_initial"]})
            for field, values, fixed in (("batch", logical, s["ubatch"]), ("ubatch", physical, s["batch"])):
                lower = [value for value in values if value < s[field]]
                upper = [value for value in values if value > s[field]]
                for value in ([max(lower)] if lower else []) + ([min(upper)] if upper else []):
                    if (field == "batch" and value >= fixed) or (field == "ubatch" and value <= fixed):
                        pool.append(self.trial(anchor, {field: value}, "batch_coordinate"))
        center = self.best(pool, anchor)
        effective = metadata.get("effective_settings", {})
        layers = effective.get("offload_layers")
        residency_known = isinstance(layers, list) and len(layers) == 2 and all(type(value) is int for value in layers) and layers[1] > 0
        host_work = effective.get("host_work_verified") is True or effective.get("cpu_prefill_observed") is True or \
            (residency_known and layers[0] < layers[1])
        if native and "--threads" in flags and host_work:
            for threads in self.config["tuning"]["threads_initial"]:
                pool.append(self.trial(center, {"threads": threads}, "generation_threads"))
        else:
            self.decision(anchor, "generation_threads", "no_supported_host_or_partial_offload_evidence")
        center = self.best(pool, center)
        if native and "--threads-batch" in flags and (residency_known or host_work):
            for threads in self.config["tuning"]["threads_batch_initial"]:
                pool.append(self.trial(center, {"threads_batch": threads}, "batch_threads"))
        else:
            self.decision(anchor, "batch_threads", "no_supported_prefill_or_residency_evidence")
        center = self.best(pool, center)
        pool.extend(self.reasoning_profiles(center))
        return [item for item in pool if item and _promising(self.config, item)]

    def reasoning_profiles(self, anchor):
        metadata, _, flags = self.metadata(anchor)
        kind, engine = anchor["runtime"]["kind"], anchor["engine"]
        native = kind == "native" or engine == "ik-llama"
        center = anchor
        pool = [anchor]
        profiles = {"default": "checkpoint default"}
        if kind == "tabby":
            profiles.update(reduced="reasoning_budget_tokens=256", disabled="enable_thinking=false;reasoning_budget_tokens=0")
        elif native:
            if "--reasoning-budget" in flags:
                profiles.update(reduced="native numeric reasoning budget=256")
                if kind == "native":
                    profiles.update(disabled="native numeric reasoning budget=0")
            if engine == "ik-llama" and "--reasoning" in flags:
                profiles.update(disabled="native --reasoning off")
        elif kind == "ollama" and "thinking" in metadata.get("model_api_details", {}).get("capabilities", []):
            architecture = self.model(anchor).get("metadata", {}).get("general.architecture", "").replace("-", "")
            if "gptoss" in architecture:
                profiles.update(reduced="documented think=low;not an exact token budget")
            else:
                profiles.update(disabled="documented think=false")
        for profile in ("default", "reduced", "disabled"):
            if profile in profiles:
                self.decision(anchor, "reasoning_method", profiles[profile], {"reasoning": profile}, status="considered_supported")
                pool.append(self.trial(center, {"reasoning": profile}, "reasoning_profile"))
            else:
                self.decision(anchor, "reasoning_profile", "unsupported_actual_reasoning_control", {"reasoning": profile})
        model = self.model(anchor)
        architecture = model.get("metadata", {}).get("general.architecture", "").replace("-", "").lower()
        gptoss = model["id"].lower().startswith("gptoss") or architecture == "gptoss"
        if kind == "container" and engine in {"vllm", "sglang"} and gptoss and center["settings"].get("reasoning_effort") == "medium":
            # Harmony effort is a documented request control, not a numeric
            # reasoning-token cap or a claim that thinking can be disabled.
            changes = {"reasoning": "default", "reasoning_effort": "low"}
            self.decision(anchor, "reasoning_method", "documented Harmony reasoning_effort=low;numeric256_budget_unsupported",
                changes, status="considered_supported")
            pool.append(self.trial(center, changes, "harmony_effort"))
        from .native_speculation import TARGET
        reason = 'handled_by_pinned_mtp_dflash_comparison' if anchor.get('model')==TARGET and self.state.get_control('speculation_comparison_enabled') else 'no_verified_adapter_launch_contract_and_eligible_concrete_draft_lead;speculation_remains_off'
        self.decision(anchor, "speculation", reason)
        return [item for item in pool if item and _promising(self.config, item)]

    def qualify(self, candidates):
        unique = {item["configuration_id"]: item for item in candidates if _promising(self.config, item)}
        prospective = sorted(unique.values(), key=screen_latency)[:self.config["limits"]["maximum_final_configurations"]]
        changed = [item for item in prospective if item["configuration_id"] not in self.qualified_ids]
        durations = [row[0] for row in self.state.db.execute(
            "SELECT finished-started FROM attempts WHERE finished IS NOT NULL AND status IN ('passed','failed') AND json_extract(result,'$.kind')='quality'") if row[0] > 0]
        task_estimate = statistics.median(durations) * 1.5 if durations else self.config["limits"]["per_task_timeout_seconds"]
        startups = [item.get("configuration", {}).get("effective_settings", {}).get("startup_seconds") for item in self.scores.values()]
        startups = [value for value in startups if type(value) in (int, float) and math.isfinite(value) and value >= 0]
        startup = statistics.median(startups) * 1.5 if startups else self.config["limits"]["server_start_timeout_seconds"]
        total_tasks = self.config["grading"]["regular_total"] + self.config["grading"]["long_total"]
        per_candidate = total_tasks * (task_estimate + startup)
        batch = []
        if not self.available(prospective[0] if prospective else {"engine": "all", "model": "all"}, "full_quality"):
            for candidate in changed:
                self.decision(candidate, "full_quality", "qualification_budget_unavailable;exploratory_only")
            return
        remaining = self.phase["deadline"] - self.state.clock()
        if self.state.run["deadline"] is not None:
            remaining = min(remaining, self.state.run["deadline"] - self.state.clock() - 7200)
        for candidate in changed:
            if per_candidate * (len(batch) + 1) <= remaining:
                batch.append(candidate)
            else:
                break
        if batch:
            regular_and_long(self.config, self.state, self.collector, batch)
            qualified = {item["configuration_id"] for item in quality_qualified(self.config, self.state, batch)}
            for candidate in batch:
                passed = candidate["configuration_id"] in qualified
                self.decision(candidate, "full_quality", "full_quality_qualified" if passed else "full_quality_not_qualified",
                    status="qualified" if passed else "not_qualified")
        for candidate in changed[len(batch):]:
            self.decision(candidate, "full_quality", "qualification_budget_unavailable;exploratory_only")
        self.state.control("tuning_prospective_finals", [item["configuration_id"] for item in prospective])

    def run(self):
        if self.phase.get("anchors") is None:
            anchors = {}
            for candidate in sorted(self.input_screens, key=self.rank):
                if not candidate.get("runtime") or not candidate.get("settings"):
                    continue
                if not (candidate.get("capacity_qualified") is True or candidate.get("eligible") is True
                        or marker_reasoning_eligible(self.config, self.state, candidate, self.model(candidate))):
                    continue
                anchors.setdefault(self.group(candidate), candidate)
            self.phase["anchors"] = list(anchors.values())
            self.state.control("tuning_phase", self.phase)
        anchors = self.phase["anchors"]
        if self.phase["selected_families"] is None:
            families = []
            for candidate in sorted(anchors, key=self.rank):
                family = self.family(candidate)
                if family not in families:
                    families.append(family)
                if len(families) == 3:
                    break
            self.phase["selected_families"] = families
            self.state.control("tuning_phase", self.phase)
        selected = set(self.phase["selected_families"])
        centers = []
        rescue = []
        promising = []
        marker_anchors = [anchor for anchor in sorted(anchors, key=self.rank)
            if self.family(anchor) in selected and
            marker_reasoning_eligible(self.config, self.state, anchor, self.model(anchor))]
        # Probe supported output profiles before tuning a censored default's
        # cache/batch settings. The same three-family and phase budgets apply.
        for anchor in marker_anchors[:3]:
            if self.available(anchor, "reasoning_rescue"):
                promising.extend(self.reasoning_profiles(anchor))
        marker_groups = {self.group(anchor) for anchor in marker_anchors}
        for anchor in sorted(anchors, key=self.rank):
            if self.family(anchor) not in selected:
                self.decision(anchor, "family_selection", "bounded_three_family_limit")
                continue
            if self.group(anchor) in marker_groups:
                if anchor not in marker_anchors[:3]:
                    self.decision(anchor, "reasoning_rescue", "bounded_three_anchor_limit")
                continue
            if not self.available(anchor, "coordinate_search"):
                break
            center = self.cache_and_prefill(anchor)
            if center:
                centers.append(center)
            elif anchor.get("capacity_qualified") is True:
                rescue.append(anchor)
        # A failed default-thinking smoke is a reason to probe supported
        # reasoning controls, while every timing still requires all six smokes.
        for anchor in sorted(rescue, key=self.rank)[:max(0, 3-len(marker_anchors[:3]))]:
            if self.available(anchor, "reasoning_rescue"):
                promising.extend(self.reasoning_profiles(anchor))
        for center in sorted(centers, key=screen_latency)[:3]:
            if self.available(center, "deepening"):
                promising.extend(self.deepen(center))
        self.qualify(promising or centers)
        if not self.stopped:
            self.phase["status"] = "exploration_complete"
            self.state.control("tuning_phase", self.phase)
        self.export()
        return self.screens


def tune(config, state, collector, screens):
    """Sequential runtime calls occur only through screen and full qualification."""
    return _Tuner(config, state, collector, screens).run()

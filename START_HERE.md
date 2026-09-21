# Local coding-agent benchmark: start here

Saved 2026-09-18. Status: reviewed execution plan only. No new harness, downloads, engine installation, or benchmark sweep has been started.

## Goal and locked decisions

Find the fastest reliable local coding-agent setup on this computer, with a genuinely usable 65,536-token context window. Correct code and working tools are hard requirements. Rank qualifying setups primarily by fresh 61,440-token time to first useful action, then generation speed; also measure cached follow-up turns.

Stable, nightly, alpha, beta, community forks, community quantizations, and coding finetunes are all eligible. "Official" is not a ranking advantage. Claims from other GPUs are leads, not results.

The user accepts up to 168 hours (seven days) of unattended benchmark execution and up to 300 GB (300,000,000,000 bytes) of additional model weights. These are ceilings, not requirements to spend them. Reuse installed models first. Stop early when the covered search is exhausted and final verification is complete.

The user reported painful approximately 35-second Gemma TTFT and is willing to wait days for a better result. Do not silently substitute a shorter-context winner.

## Read order

1. Read [EXECUTION_PLAN.md](EXECUTION_PLAN.md) completely.
2. Read [FIXTURES.md](FIXTURES.md) completely before implementing grading.
3. Load [benchmark-spec.json](benchmark-spec.json) as configuration; do not improvise replacement defaults.
4. Read [RESEARCH.md](RESEARCH.md) when probing the corresponding engine or deciding TTFT experiments.

None of these documents depend on the removed Expert / 00-init / 01-router skills. Do not reinstall or invoke them. They were uninstalled at the user's request; recovery copies are outside this project.

## New-session prompt: copy everything in this block

> Implement and execute the local coding-agent benchmark saved at C:\Users\barrett\local-llm-benchmark.
>
> Read START_HERE.md, EXECUTION_PLAN.md, FIXTURES.md, benchmark-spec.json, and RESEARCH.md first. Follow their phases, grading contracts, measurement definitions, budgets, resumability, and safety rules. I want a real 64K-capable coding agent, quality first, then the lowest fresh-context time to first useful code/tool action and high tok/s. Cached TTFT is a separate secondary metric. I do not care whether the best engine/model is official, stable, beta, alpha, nightly, or a community fork.
>
> Reinspect hardware and running workloads before measuring. Reuse installed models before downloading more. Additional model weights are capped at 300 GB decimal; benchmark execution is capped at 168 elapsed hours starting with the first real runtime probe or model download after harness verification, whichever occurs first, not with plan reading or harness implementation. Work sequentially on the one GPU. Resume checkpoints instead of restarting the whole search.
>
> First build and test the resumable harness and deterministic Python/TypeScript fixtures. Then establish installed-model baselines and probe upstream llama.cpp, ik_llama.cpp, an eligible CUDA TurboQuant fork, ExLlamaV3/TabbyAPI, vLLM, SGLang, LM Studio, and Ollama as specified. Include bounded discovery of newer models/engines. Experimental candidates must pass the same coding, tool, long-context, and stability gates. Do not pick a winner from synthetic throughput alone.
>
> Do not modify my driver, global Python/Node installs, global WSL configuration, power/clock settings, existing app configuration, or personal repositories. Do not kill unrelated processes. Keep APIs local and credentials out of files/logs. If a significant system change or new authority is necessary, finish unaffected work and ask me.
>
> Deliver the measured ranking, raw results, exact versions/settings/launch commands, a working local agent endpoint, and an honest explanation of remaining limits. If nothing qualifies, say so; do not invent a successful 64K result. Keep progress updates concise.

## What should happen first

- Confirm the host facts in the JSON. Inventory available model files, engine versions, disk space, WSL/Docker GPU access, and unrelated GPU workloads.
- Create the isolated harness, tests, logs, and state under this directory; model downloads and large engine environments belong on F:, not C:.
- Run harness unit tests and fixture gold/bug verification before any costly sweep.
- A suggested future CLI is defined in EXECUTION_PLAN.md. It does not exist yet: do not claim the listed commands have already run.

## Authentication and Windows notes

The previously pasted LM Studio API key must not be copied into this project or reused by default. If authentication is needed, ask for a new key through an unlogged prompt and keep it only in the current process environment.

The current session's normal Windows command sandbox failed with SetTokenInformation(TokenDefaultDacl), error 1344. Read-only commands worked through the host's approved elevated execution path. This is an execution-environment problem, not an LLM-engine result. If it recurs, request the normal scoped escalation; do not disable security or alter user ACLs. File changes still use apply_patch.

If apply_patch can add a file but cannot read it for an update, the tested fallback is the installed executable C:\Users\barrett\AppData\Local\nvm\v24.13.1\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe with --codex-run-as-apply-patch. Pass ONE normal multiline patch argument through an approved PowerShell execution, in the project parent working directory. Build it from an array of quoted lines joined with [char]10; escape literal apostrophes by doubling them. Keep each shell command below Windows command-line length limits (small update patches, not whole-document replacements). This remains apply_patch, not a shell content-writing workaround. Verify changed files afterwards. Paths/packages may change next session: confirm the executable still exists before use.

## Completion checklist

- Harness tests pass; each intentional fixture bug fails and each golden solution passes.
- Exact fresh/cached token counts and effective per-slot context are verified.
- Every advertised winner passes coding and long-context quality gates and the final stability run.
- Experimental and unsupported candidates have explicit results or logged skip reasons.
- State, raw streams, resource logs, scoring details, and human-readable recommendations are saved.
- Winner can be reproduced from a pinned model file/revision and engine commit/image digest.

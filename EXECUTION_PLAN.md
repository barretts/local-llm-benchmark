# Execution plan: fastest reliable 64K local coding agent

This is an implementation specification, not a report of completed benchmarks. Read START_HERE.md for the new-session prompt and benchmark-spec.json for literal defaults, paths, candidate metadata and limits. Read FIXTURES.md before implementing the grader. Removed Expert skills and router rules are not dependencies.

## 1. Outcome and constraints

The deliverable is a reproducible, locally hosted coding-agent endpoint and a measured ranking of model + weight quantization + engine version + effective settings. A small-context speed record, a model-card score, or a stream of early reasoning tokens is not the requested result.

A qualifying configuration must:
- Provide 65,536 effective tokens to ONE request slot, without automatic truncation, context shift, or unvalidated RoPE/context-extension tricks.
- Process a fresh 61,440-token fully formatted prompt and leave 4,096 tokens for output.
- Pass at least 27/36 regular coding tasks and 9/12 long-context tasks, plus every capacity-marker check.
- Use working tools through the interface that will actually be handed off.
- Finish the final fresh/cached stability sequence with no crashes, OOMs, invalid numeric output, uncontrolled loops, or failed capacity checks.

These gates do not promise error-free coding on arbitrary repositories. They supply repeatable evidence and expose failures. Tests remain necessary during real use.

Rank qualified configurations by lowest cold 64K p95 time to the first completely parsed, permitted tool call. Then compare output tok/s, cached follow-up latency, quality score and memory. Measure first streamed token separately; never substitute it for first action.

All release channels compete equally: stable, nightly, alpha, beta, community forks, coding finetunes and community weight/cache quantizations. Freeze versions before comparisons. Optimizations earn their place through local measurements and full requalification.

Budget: 168 elapsed hours of benchmark execution; 300,000,000,000 bytes of additional model weights. Start the durable execution clock at the first real runtime probe/model download after harness verification, whichever occurs first. Plan reading and harness implementation are outside that clock. Time continues through pauses and session changes; do not reset it on resume. Check caps before starting each job. Stop launching expensive work when the remaining budget cannot cover it. Leave time for reporting.

The weight ledger is cumulative acquisition/conversion/staging reservations, not a quota reset by deleting old downloads. Resuming one partial artifact consumes its existing reservation, not another full copy. Reusing an installed file in place costs zero new-weight bytes; a newly created physical weight copy consumes budget. Enforce disk headroom separately.

## 2. System baseline and safety

Observed on 2026-09-18: Windows; RTX 4060 Ti 16 GB (16,380 MiB reported); driver 591.86; Threadripper 1950X, 16 cores/32 threads; approximately 128 GiB RAM. F: held the models and had approximately 569 GiB free. WSL2 Ubuntu was stopped; Docker Desktop's WSL distribution was running. Global WSL limits were 32 GiB RAM, 16 processors, 8 GiB swap. Reinspect rather than trusting this snapshot.

Keep the small controller, logs and reports under C:\Users\barrett\local-llm-benchmark. Put new model weights under F:\models\localbench and large runtime environments/builds under F:\local-llm-benchmark-runtime. Preserve at least 64 GiB free on F: and 50 GiB on C:. The runtime/build soft cap is 100 GiB in addition to the weight cap. Count conversion inputs, converted weights and temporary weight copies; do not hide them as "runtime."

Before changes, record existing loaded models, server ports, app/backend versions, container names, WSL state, and running GPU workloads. Do not unload or terminate an unrelated active job silently. Ask a nonblocking question or wait for an agreed idle window while implementing safe parts. Use only owned instances and explicitly identified benchmark containers/PIDs thereafter.

Do not modify drivers, global Python/Node packages, global WSL configuration, BIOS, clocks, power limits, permanent app preferences, firewall rules or personal repositories. A desired optimization requiring one of these is a reported optional follow-up, not an unattended step. Existing skill uninstallation is already complete; do not revisit it.

Bind host APIs to 127.0.0.1. Use private ports from the JSON; do not steal the user's port 1234. Docker may listen on all interfaces inside a container, but publish ONLY 127.0.0.1:hostport:containerport. Do not expose APIs on the LAN.

New credentials come from an unlogged interactive prompt or a process-local environment variable. Mask Authorization headers, request URLs containing secrets and config secrets before logging. Never save the previous LM Studio key, credentials in command lines, or raw environment dumps. Do not pass inference credentials into test containers.

Downloaded engines are untrusted software: use pinned public source/package/image metadata, isolated directories, no administrator runtime, no privileged Docker, no Docker socket mounts. Read custom installer/build scripts before execution. Do not run a curl-pipe-shell installer. Default to trust_remote_code=false; if a model needs arbitrary remote Python, quarantine the probe or choose another candidate.

### Generated code must be isolated too

A fixed "run tests" tool is NOT a sandbox: Python/TypeScript code can execute arbitrary actions during import/tests. Never execute agent-generated code on the host.

Run graders in owned, CPU-only Docker containers:
- Pinned Python 3.13 and Node 24 Linux images (digests saved), with TypeScript compiler preinstalled in a separately built pinned test image.
- No network, no GPU, no credentials, no personal directory mounts.
- Unprivileged user, cap-drop ALL, no-new-privileges, read-only container root, bounded writable /tmp.
- Mount only that fixture workspace read/write and controller-owned tests/dependencies read-only.
- Limit each test container to 2 CPU cores, 2 GiB RAM, 128 processes and a 60-second test timeout.
- Disable Python bytecode writes and Node telemetry; build TypeScript into the fixture workspace or /tmp.
- Remove only containers bearing this benchmark's run/job ownership labels.

The agent can edit source paths only; public/held-out tests and grading contracts are not editable. Held-out tests are not reachable through agent tools. During a task, give only public-test feedback; run held-out tests after the final response. Use structured assertions, not natural-language self-reported success. If CPU Docker isolation is unavailable, continue non-executing inference measurements, but do not run generated code directly or declare a coding-agent winner. Report the missing isolation requirement.

## 3. Implementation order and gates

The following phases are sequential gates; implement a working narrow vertical slice before adding every engine.

| Phase | Work | Gate to continue |
| --- | --- | --- |
| A | Doctor, configuration, paths, ownership, durable state | Host report and caps verified |
| B | Mock engine, streaming parser, timers, tools, scheduler | Harness unit tests all pass |
| C | Python/TS fixtures, public/held-out graders, gold/bug verification | Every gold passes; every planted bug fails meaningful tests |
| D | One installed GGUF on native bundled/upstream server | Real streamed tools, exact token counts, 64K capacity checks |
| E | Installed models, selected downloads, alternative/experimental engines | Probe results or specific logged skips |
| F | Quality qualification and bounded setting search | Candidate-specific scores and valid fresh/cached measurements |
| G | Requalify finalists, 20 fresh + 20 cached runs | Stable fully specified winner(s) or explicit no-winner result |
| H | Report, exact launch scripts and endpoint demo | Reproduction and isolated coding-agent task pass |

Suggested execution allocation: discovery/probes/downloads 24h; baseline quality 72h; tuning/comparisons 48h; final validation/reporting 24h. Unused time rolls forward; never exceed the global 168h clock. Reserve the final 2h for report generation even if the search is incomplete. If all planned feasible work completes sooner, finish sooner.

### A. Doctor and inventory

Implement doctor first. Save complete command output to .logs/doctor-*.log BEFORE parsing it; never pipe command output directly into a filter.

Read-only checks:
- nvidia-smi: GPU identity, total/free VRAM, driver, compute capability if queryable, temperature, power, utilization and processes. Verify CUDA capability through an isolated probe if a field is unsupported.
- Windows CPU/RAM/free disk; WSL distro versions/state and .wslconfig limits; docker info and a bounded GPU-access probe using a compatible pinned CUDA image.
- Absolute paths and versions for Python, Node/npm.cmd, git, lms, Ollama, Docker, WSL, CMake, compiler, CUDA toolkit and existing llama launchers.
- Installed GGUF paths/sizes/hashes; readable GGUF architecture/template metadata; LM Studio model keys; HF cache snapshots, completeness and hashes.
- LM Studio readiness via real API health/model listing, not CLI exit status alone. Earlier lms ps --json returned [] with exit 1. The bundled llama-server.exe is a small launcher requiring adjacent DLLs; a llama-bench.exe was not confirmed.

Do not start a user's stopped distribution merely for an inventory command unless a Linux probe is now in scope; record that transition. GPU Docker/WSL compatibility is a probe, not proof that any particular engine supports this model.

Save host.json, inventory.json and installed-weight-ledger.json. No model loading is required just to inspect hardware. Do not treat a busy/encoding GPU as a valid performance environment.

### B. Harness scaffold and interfaces

Use the existing Python 3.13 interpreter to create a project-local .venv. Keep the controller mostly standard-library: argparse, dataclasses, sqlite3, json, hashlib, pathlib, subprocess, urllib/http.client, threading/queue, time and unittest. Pin any added tokenizer/HTTP packages in a lock file after checking actual available versions. Do not guess an obsolete dependency version.

Suggested source layout:
- localbench/__main__.py: CLI and defaults loaded from benchmark-spec.json.
- config.py / schema.py: strict input/output validation.
- ownership.py / paths.py: scoped processes, containers and safe paths.
- state.py: SQLite jobs/attempts/weight reservations plus atomic summary snapshots.
- streams.py / metrics.py: SSE and Ollama NDJSON parsing, timers and counters.
- resources.py: one persistent lightweight GPU/CPU/RAM/shared-memory collector.
- prompts.py: deterministic corpus generation, token packing, cache experiments.
- tools.py: controller-owned tool executor and allowed-path/test-command rules.
- fixtures.py / grading.py: generation, hashes, public feedback, held-out scoring.
- adapters/base.py, native.py, lms.py, ollama.py, openai_linux.py, tabby.py.
- runner.py / scheduler.py: jobs, global GPU lock, deadlines, retries, checkpoints.
- report.py: JSON/CSV/HTML/Markdown reports without a frontier-model judge.
- tests/: harness unit tests and fake streaming server.
- scripts/: logged PowerShell launch/test helpers; Docker grading definitions.

EngineAdapter contract:
1. discover_capabilities() -> supported load, template, sampling, tool, cache and token-count controls.
2. launch(config) -> owned handle, endpoint, version/digest, actual effective configuration.
3. health() -> real readiness or structured failure.
4. tokenize(messages, tools) -> fully templated count and provenance.
5. stream_chat(request) -> normalized reasoning/content/tool/usage events plus raw stream.
6. reset_cache() -> verified success, unsupported, or owned-server restart requirement.
7. unload_owned() -> stop only the handle launched/loaded by this adapter.
8. metadata() -> exact model/engine/config hashes and defaults affecting output.

Do not silently ignore unsupported options. Record requested and effective configurations separately. A cache or reasoning toggle accepted by HTTP is not proof it actually took effect.

Minimum tests before real inference:
- SSE fragmented UTF-8, multiple data lines, split events, blank/role-only events, DONE, usage-only tail, reasoning fields and text <think> regions.
- Split/parallel tool arguments, malformed JSON, missing required fields, unexpected tool names and premature stream closure.
- Ollama NDJSON chunks, final counters, nanosecond duration conversion and absent optional fields.
- Fake clock: TTFT starts before send and stops at the correct event; first valid tool action occurs only after complete schema-valid arguments.
- No SSE-chunk-to-token conversion; no cached/fresh metric conflation.
- Token packing includes system/tools/template and rejects over-window or truncated results.
- Safe path handling rejects absolute/UNC/drive escapes, ../, junctions/symlinks to outside directories and test/grader writes.
- Tool execution rejects arbitrary shell, unsupported executables and commands not on the fixed allowlist.
- Inject crash midway through a job; resume preserves finished results, marks stale running work retryable and respects original deadline/disk reservations.
- Stale PID reuse, unrelated containers and occupied ports cannot be killed/claimed.
- Deterministic prompt/job hashes and report sorting/percentile math.
- 10-minute fake sweep exercises stop/resume, retries, reports and resource logs.

### State and CLI contract

SQLite records run, engine, model, configuration, fixture version, job and attempt rows. Use transactions for status changes; append raw event JSONL and flush it at completion/checkpoints. Export state/summary.json atomically using write-temp + replace. A separate run lock prevents two controllers using the GPU.

job_id = SHA256(canonical JSON of engine binary/digest, model SHA/revision, effective settings, sampler/reasoning profile, fixture/prompt hash, tokenizer version, seed, fresh/cached mode and replicate). Reuse only a completed job with an identical key. Changing engine commit or any meaningful setting makes a new job.

Keep attempts with status pending/running/passed/failed/invalid/skipped and a reason code. Infrastructure errors are not coding failures; retry up to two transient attempts, then record an explicit unavailable result. A repeatable model OOM or unsupported kernel is not transient.

CLI to implement:
- python -m localbench doctor
- python -m localbench init-fixtures
- python -m localbench verify-fixtures
- python -m localbench probe --engine ENGINE --model MODEL
- python -m localbench sweep
- python -m localbench status
- python -m localbench resume
- python -m localbench stop --after-current
- python -m localbench report
- python -m localbench agent-demo --configuration CONFIG_ID

These are future commands, not already installed functionality. Default configuration is benchmark-spec.json in the project. status reads durable state only; it must not load a model or contact a stopped engine. stop is a durable control request, not an unrelated-process killer. resume continues the original run deadline and preserves valid results.

Launch long operations through the host's nonblocking path. On Windows use Start-Process with -WindowStyle Hidden, an explicit working directory and separate stdout/stderr log paths; save the PID and creation time. Poll status and inspect full saved logs. Do not use a foreground multi-hour command or long blocking sleep. The user should receive a concise progress update at least once per minute during an active agent turn; overnight controller work is resumable.

### C. Fixtures before measuring models

Implement the exact contracts in FIXTURES.md. Each task has buggy initial source, a controller-only golden implementation, public tests and held-out deterministic tests. Use Python unittest and TypeScript tsc + Node's built-in test runner; pin the TypeScript package and grading images. No production services or paid external API calls.

Each planted bug must fail at least two meaningful assertions overall, including a public assertion and a held-out assertion. The gold must pass every test for all three seeds. Archive hashes and test output before benchmarking; do not modify contracts/grading after seeing model results without making a new fixture version and invalidating comparisons.

Tools: list_files, read_file, write_file, apply_patch and run_tests. For long answer tasks add submit_answer. Native OpenAI-style tool calls are primary. Tool names/JSON schemas and agent system instructions are identical across engines unless an adapter must serialize them into a required template; record that transformation.

Regular agent loop: fresh fixture workspace, initial contract + public failure description, max 16 assistant turns, max 32,768 total generated tokens, 900s task timeout, source-only edits. Tool output is capped at 2,048 tokens with an explicit truncation notice, never an undisclosed cut. Stop if another turn would overflow 65,536 tokens; no hidden rolling context, compaction or summarization in this benchmark. Score held-out tests once the agent finishes or exhausts limits.

Do not award a pass for saying "tests pass." Public and held-out tests must all pass and no tool/path/budget rule may be violated. Save original source, resulting diff, public feedback, hidden aggregate result and timestamps for first action, first edit, first public pass and final completion.

## 4. Measurement protocol

### Exact context

Generate a fixed, deterministic fake repository/code corpus, with no personal data. A fully formatted prompt includes system text, tool definitions, all chat messages, model template and assistant-generation prefix.

Targets: 4,096, 16,384 and 61,440 input tokens. Aim exactly at target; accept at most 64 fewer tokens, NEVER more than target for the full-context run. Retokenize after each padding/trim operation. Do not estimate tokens from characters or words.

Use an engine's tokenize/apply-template endpoints where verified, or the exact checkpoint tokenizer/template pinned to that model. A native GGUF tokenization helper can serve the same GGUF tested through another engine, but confirm final actual usage. Engine outputs may distinguish total, cached and evaluated tokens; map these explicitly rather than assuming one counter means all three.

Cross-check server-reported input counts against expected counts. A difference over tolerance, unexplained context clipping, auto-fit reducing context, or missing sufficient evidence makes the run invalid. If tokenization cannot be established, record only exploratory timings, not a qualifying 64K result.

Use a common deterministic corpus family and equal token budgets across tokenizers; report a common-byte workload separately to reveal tokenizer effects. Never compare a 48K baseline to 64K as though their TTFTs were directly interchangeable.

At each candidate's 64K context, run three capacity probes (seeds 17, 101, 1009): unique random markers near 5%, 50% and 95% of the tokenized corpus, with a final tool request to return all three exact values in order. Verify each and inspect actual prompt counts. All nine retrieved values must be correct. This checks the configured window, not just its memory allocation.

### Cold loaded-model latency

Model loading/startup are separate metrics. "Cold" means a brand-new long prompt with no useful prefix-cache reuse, on a loaded model with warmed kernels.

For each replicate:
1. Launch/health-check owned engine and confirm effective settings.
2. Warm with unrelated 512-token input and up to 32 output tokens. Warmup must not share the test's system-prefix nonce.
3. Disable/reset verified prefix cache; otherwise restart the owned engine, repeat short disjoint warmup, then measure.
4. Put a unique nonce at the beginning of system CONTENT (after any fixed template tokens), before tools/repository text. Use new repository data/markers.
5. Send the exact-length streamed latency probe asking for a specific valid read_file call against the fixture. Continue through end/usage; do not stop logging at first token.
6. Record evaluated/cached input tokens when observable. At most 64 fixed-prefix tokens may be cached. If counters are unavailable, isolation must be proven by fresh owned process + disjoint warmup. Otherwise label cache state unverified.
7. Time first streamed token and first fully parsed allowed tool action with time.perf_counter_ns, beginning immediately before request send.

Empty role deltas, heartbeats, whitespace-only intent and partial JSON do not count as first action. If reasoning arrives first, report its latency and how much later the tool action arrives. Missing action is failure/censored, not zero seconds.

### Cached reuse

Separate controlled-prefix experiment: stable fully templated prefix near 59,392 tokens + changed suffix near 2,048 = total 61,440. Prime once, then change only the suffix for measured requests. Pack the actual template to total target; arithmetic alone is not tokenization.

Enable actual prefix caching, preserve the same owned engine/slot, and log observable cached/evaluated counts. Hybrid/recurrent architectures may need checkpoint support and may reuse only part of the prefix. Never pretend the configured prefix length is the measured hit count. Also record cached timing during real multi-turn fixture runs; label it separately from the controlled-prefix benchmark.

### Decode throughput

Use a separate reproducible code-output request, fully formatted at the same input targets, with up to 256 generated tokens. Track total/reasoning/visible counts where available, final wall time, server evaluation duration, inter-event latency and completion reason.

Preferred decode tok/s uses server's actual generated token count and decode-only duration. Ollama's durations are nanoseconds. Record client total-output-tokens / (last token arrival - first token arrival) as an auxiliary arrival-rate estimate, not exact decode speed. Never use number of SSE messages as tokens. Retokenized text is an explicitly approximate fallback, not a silent replacement for unavailable usage.

Early EOS is permitted but a sample under 64 generated tokens is not comparable for sustained-throughput ranking. Retry with a longer requested code artifact; if it remains short, leave sustained tok/s unqualified. Do not force ignoring EOS for coding-quality tests.

### Resources and sample count

One persistent collector samples nvidia-smi about every second, Windows dedicated/shared GPU memory about every five seconds, process/host RAM, CPU load and GPU temperature/power/utilization. Avoid launching a heavyweight PowerShell counter subprocess every second. Collect idle baseline and mark foreign workload overlap.

Report dedicated memory and DELTA in shared GPU memory. A preallocated shared-memory baseline is not evidence of active spill. A growing shared delta, flat/full dedicated memory and slower timings is a diagnostic lead; correlate before blaming it. CPU/offload configurations are eligible only if they actually win and pass quality; label their memory/residency honestly.

Screen with three fresh samples per important 64K configuration. Finalists require 20 fresh + 20 cached samples, plus quality/capacity checks. Compute p50/p95 using nearest-rank; publish all samples, min/max and N. With only 20 samples, avoid exaggerated precision or claims of universal superiority. Repeat invalid foreign-workload samples without hiding their count.

## 5. Engine and model search

### Candidate selection

Start with installed priority-1 GGUFs, then priority-2, then speed controls. Parse real file metadata; LM Studio package sizes can include multimodal projectors and are not the text-weight size. Prefer text-only loading; exclude mmproj unless an engine needs it, then log its memory cost.

Discover and reuse existing HF snapshots before downloads. Pins and hashes in JSON are starting candidates, not a claim they have been run. For a new selection, save HF API metadata, repo commit, exact filenames/sizes/LFS hashes, tokenizer/template/config and license. Never download an entire mixed-format repo blindly.

Download priority-1 small plain Qwen/Ornith first. Higher-bit upgrades and 35B/dense comparisons follow observed quality/performance needs. A 35B MoE with 3B active parameters still needs resident/offloaded weights for all experts. Ornith 1.5 35B Q4 alone is approximately 21.7 GB; it is not a full-GPU 16GB candidate. Its 1.0 low-bit comparison is a different model version.

Do not reject a smaller model from size alone; it can win only after the same full quality gates. Do not assume "uncensored/heretic" improves coding. Do not promote an advertised 64K window without actual checks. Exclude sub-64K defaults such as an unextended Qwen2.5-Coder-14B from the primary ranking.

### Bounded discovery of newer options

Spend at most 4h total researching additional current candidates during execution. Search primary maintainer docs/source, HF model/quant cards and reproduced CUDA results. Search seeds are in JSON; extend them if a newer architecture/finetune genuinely fits the machine.

Add at most four newly discovered weight variants beyond the pinned list; prioritize small dense or low-active-parameter MoE models with real tools/coding training, verified >=64K context and workable Ada kernels. Discovery may also select a newer engine implementation, but still uses the configuration/time/disk caps. Record considered/rejected leads and concrete reasons. Do not use popularity or a claimed benchmark as the winner criterion.

Pin each engine commit/binary/image digest before recording comparable results. One follow-up revision is allowed for a diagnosed compatibility bug or specific performance lead, with old results retained separately. Do not endlessly chase today's master and destroy reproducibility.

### Probe ladder for every engine

Within the engine's probe cap:
1. Save --help/version or current documented config, exact dependencies/image digest and model format requirements.
2. Check GPU access/architecture support and launch a small compatible model with one slot.
3. Confirm stream, real tool-call round trip, actual sampler/reasoning controls, input counters and cache behavior.
4. Run exact 16K and 64K memory/capacity probes.
5. Run the six literal smoke_jobs in the JSON before expensive tuning.
6. Complete full qualification for promising runtime/settings; no final pass from smoke tests alone.

Failures have reason codes: installation, missing compiler/toolkit, unsupported architecture/format/template/cache type, insufficient RAM/VRAM, WSL cap, invalid/truncated context, malformed tools, runtime crash, timeout, quality failure, or budget exhausted. Save the full underlying logs.

A compatibility cap is not a performance conclusion. If an optional build/conversion has a concrete promising path, spend its separate bounded budget or remaining phase time rather than declaring it slower merely because setup took effort.

### Native upstream / bundled llama.cpp

Use bundled native server for an initial slice; compare a pinned current upstream CUDA nightly or source build. A stable build may be a useful control, not a preferred winner.

Read actual --help before emitting flags. Initial INTENT: model GGUF, context 65536, parallel=1, full GPU offload, GPU KV, Flash Attention on, f16 K/V, logical batch 2048, physical/ubatch 512, CPU threads=16, batch threads=16, Jinja tools, context shift disabled, host 127.0.0.1 and private port. Confirm loaded per-slot context, layer residency, cache buffers and effective settings in logs.

If building, use isolated source/build directories; detect compute capability rather than blindly compiling somebody else's sm_86/sm_120 preset. Enable all-quant Flash Attention only when the pinned source supports the build option. Missing compiler/toolkit favors an isolated compatible container/prebuilt, not a global toolchain install. If llama-bench is unavailable, real server measurements remain authoritative; do not invent a benchmark executable.

### ik_llama.cpp

Probe early: it specifically targets CPU/CUDA performance and current hybrid architectures. Use a plain non-XL Q4_K_M or Q6_K GGUF also tested by upstream, to isolate engine differences. The maintainer warns some Unsloth _XL files contain incompatible f16 tensors; do not spend hours forcing the installed _XL file to load.

Read the fork's parameters rather than copying upstream controls. Native Windows or an isolated cu12-server container is acceptable. Check AVX2 baseline on the 1950X and real CUDA kernels on this GPU. If partial MoE offload yields incoherent text with graph splitting, try the documented graphs=0 diagnostic once and requalify; do not accept gibberish because tok/s rose. Avoid the documented CPU repacking pitfall for partially offloaded K-quants. Test advanced KV/speculation only after a sane baseline.

### Experimental CUDA TurboQuant fork

Initial lead: jarkevithwlad/llama.cpp-turboquant-cuda; x6nux/llama-cpp and Madreag/turbo3-cuda are alternatives. Choose one with an actual supported NVIDIA implementation and model head dimension. Do not copy Radeon/ROCm speed claims onto the 4060 Ti.

For the initial lead, documented K=tbqp3 and V=tbq3 are a required pair with CUDA KV offload and Flash Attention. Verify this at the pinned commit. First test f16 or q8 controls in THAT fork, then TurboQuant, with MTP/speculation off; compare the same GGUF/template/sampler. Build for the detected GPU and use real supported head dimensions.

Quality evaluation must include deep 64K retrieval/coding, not just 2K perplexity. Reject NaNs, symbol loops, bad tools, and context-dependent output degradation. Other forks' incompatible cache labels are not interchangeable. Never enable synthetic speculative acceptance flags or test-only shortcuts in scored runs.

### ExLlamaV3 + TabbyAPI

This is a first-class candidate, not dismissed because it is unofficial. It uses EXL3 weights, not GGUF or EXL2. Pin the runtime and API layer. Read actual model support and current example config. Gemma E2B/E4B support differs from main Gemma; do not assume every model in a family works.

Prefer ready-made community EXL3 for the chosen 9B model at approximately 3.5/4/5 bpw, validating exact base revision and format. If no suitable public quant exists, allow ONE isolated conversion of a promising 9B checkpoint within a 6h cap and the weight/disk budget; use maintained conversion scripts and their calibration defaults, record them, and preserve resumable work. A conversion unable to finish is a logged incomplete path, not a fabricated throughput result.

Native Windows requires compatible triton-windows/Python/Torch/CUDA packages. WSL/container is an equal option under the existing 32GiB WSL cap. Do not replace global Python 3.13 to satisfy an engine; use a self-contained engine environment/image. Verify real SSE tools through TabbyAPI and KV bit-width/context settings from the loaded model.

### vLLM / SGLang

Linux via WSL2/Docker, not an assumed native-Windows pip install. Start with prequantized Qwen3.5/Ornith 9B checkpoints in JSON. Read config and use the actual compressed-tensors/AWQ representation. BF16 9B weights generally exceed available VRAM plus 64K state; do not start with an impossible BF16 baseline.

Test supported current release, then an eligible nightly when model support/performance warrants. Ornith 1.5's card gives starting minimum versions, but newer packages can change requirements. Save current support evidence. BitsAndBytes may require an out-of-tree plugin; prefer a known prequantized format instead of turning installation into an open-ended plugin project.

Initial intent: max model length 65536, max simultaneous requests 1, GPU memory utilization around 0.85, appropriate tool/reasoning parser, compatible quant kernels. Probe memory fractions 0.80/0.85/0.90; tune prefill token/chunk budget 4096/8192/16384 where supported. vLLM defaults that optimize concurrent serving may not optimize one user's cold TTFT. If disabling chunked prefill requires a token budget >= max model length, account for that memory demand; do not force a tiny budget and claim an engine failure.

For Qwen-derived Ornith, current cards point to vLLM qwen3_xml and SGLang qwen3_coder tool parsers plus qwen3 reasoning parser. Validate a real round trip; parser names are version-dependent. Avoid attention kernels requiring Hopper/Blackwell when this Ada GPU cannot run them.

For gpt-oss, download only index-referenced ROOT safetensors and supporting config/tokenizer/template. The repository contains duplicate original/ weights; whole-repo snapshots waste the budget. Validate actual MXFP4/kernel support and 64K KV fit before making it a Linux finalist.

### LM Studio / Ollama

These are practical controls and may win if measurements support them. LM Studio adapter loads/unloads an owned instance and records backend build and effective load config. Use only actual exposed API/SDK controls; do not edit private app files to pretend physical batch/cache toggles are supported. Current/old response identifiers differ; normalize only fields actually observed.

Ollama adapter uses a private owned daemon/model name, an existing GGUF via a local Modelfile, a process-scoped models directory if needed, verified num_ctx=65536, and native streamed /api/chat NDJSON stats. Preserve the user's daemon/models/settings. Record actual think profile and cached/input counters rather than assuming all OpenAI-compatible wrappers expose the same fields.

## 6. Tuning and quality scheduling

Never run two GPU jobs together. Warmups, resource collectors and CPU grading must not secretly add GPU work.

Baseline installed models first; screen both 16K and full 64K. Run regular quality tasks in round-robin order across candidates/seeds to avoid temperature/time drift. Complete 36 regular + 12 long tasks for finalists. Early elimination is valid only after 10 regular failures or 4 long failures, because the candidate then mathematically cannot reach its gate. Infrastructure failures do not enter that count.

Select up to three promising model/weight families for deep setting exploration, while keeping newly supported alternative engines eligible. Avoid a full Cartesian product. Maximum 120 distinct effective runtime configurations across the run; probes still consume time.

Coordinate search for native GGUF:
1. Start full GPU residency, one slot, Flash Attention on, speculation off.
2. Compare K/V f16/f16, q8/q8 and q4/q4 only where supported. Try asymmetric pairs from JSON if useful. Cache quantization is a memory/speed/accuracy tradeoff, not guaranteed TTFT improvement.
3. Compare logical/physical batch pairs: (512,128), (1024,256), (2048,512), (2048,1024), (4096,2048). Physical cannot exceed logical. Hold slots/template/cache constant.
4. Around the best pair, vary one dimension once; test generation threads 8/16/32 separately when host work/offload matters, and batch threads 2/8/16/32 separately for full-GPU or CPU-prefill behavior.
5. Only if full residency cannot fit, lower memory costs or test bounded partial GPU/CPU-MoE offload; record which operations moved to CPU and resulting long-prompt cost.
6. Reduced/default/disabled thinking are separate output profiles; probe an exact 256-token reasoning budget where supported, otherwise documented low effort with the actual method recorded. Every prospective winner must independently pass quality. Start with common sampling from JSON; allow at most one producer-recommended alternative per model if quality/support warrants. Avoid unsupported or producer-discouraged greedy settings.
7. Tune cache/recurrent checkpoints for FOLLOW-UP reuse separately from fresh prompt latency.
8. Try ngram/draft/MTP/DFlash/DSpark only when real support and compatible drafts exist. Recount memory. They primarily optimize decoding, not the cost of reading a brand-new 64K prompt.
9. Keep speculative settings only if they preserve quality and do not worsen cold p95 action latency >10%; report acceptance and draft overhead, never synthetic acceptances.

Use smaller 16K jobs to find crashes/OOMs, but reject settings only on real full-context evidence. A large physical batch can increase scratch memory and hurt performance; the prior Gemma experiment did not demonstrate a TTFT gain.

For each changed cache quant, model quant, reasoning profile or experimental kernel, run smoke quality before expensive timing. A smoke pass permits exploration, not final recommendation. Fully requalify each final effective configuration. Reuse completed jobs only if their exact key is unchanged.

## 7. Final validation, ranking and handoff

Freeze up to six final configurations; prioritize finishing the best three completely over partially testing six. Each advertised finalist needs:
- Complete 36/12 quality scores at its exact effective settings.
- All capacity-marker probes; no truncation/context shift.
- 20 valid fresh + 20 valid controlled-cached samples at full context.
- Sustained-throughput samples with actual token counts.
- No stability failures in the final sequence; any transient infrastructure retry is visible.
- A real isolated agent-demo task through the endpoint/format handed to the user.

Primary winner = smallest qualifying full-context cold p95 FIRST ACTION. Treat candidates within BOTH 10% and 1 second of the minimum as a latency tie, then prefer higher median sustained 64K decode tok/s. If throughput lacks exact comparable counters, report the uncertainty instead of making a false precision-based tie-break. Show quality/latency/throughput Pareto alternatives, including any materially higher-quality runner-up.

Aspirations, not fabricated promises: fresh first streamed token <10s; fresh p95 first action <15s; cached first action <2s; decode >=40 tok/s. If measured results remain around/above the painful 35s, say those targets were not achieved. The historical 35s was at 48,046 tokens and included reasoning: use a newly measured 64K Gemma control for direct comparisons.

If nothing qualifies, deliver the best measured alternatives and exact failed gates; do not silently call a 16K/32K setup a 64K winner. Explain practical secondary workflows (stable prompt prefixes, warm engine, smaller active context, selective repo retrieval) without counting them as a cold 64K success.

Save:
- artifacts/host.json, inventory.json, engine-manifest.json, weight-ledger.json and fixture-manifest.json.
- artifacts/results.json and results.csv: one row per attempt, model SHA/revision, engine commit/digest, requested/effective settings, prompt/usage/cache counts, seed, scores, metrics, resource peaks and validity reasons.
- .logs/: complete process/test logs, sanitized requests, raw streams/events and timestamped resource samples.
- state/: durable database, run deadline, pending/completed jobs and resumable conversions.
- artifacts/report.html and recommendation.md: ranking, failures, coverage limits, latency breakdown and model/context tradeoffs.
- scripts/launch-winner.ps1 (only if a winner qualifies): literal verified model paths/engine version/settings and local bind; no unresolved placeholders.
- A sample local agent-demo command and successful isolated diff/test transcript.

Do not claim an absolute "best in the world": state the finite candidates/settings covered. A skipped setup is not a measured loser. Keep raw contrary results.

At execution handoff, clearly state whether any OWNED winner server remains running, its PID/container and stop command; restore other owned temporary/app state. Plan-saving alone starts no server. Test the winner launch script from a stopped owned instance before calling it reproducible.

## 8. Common pitfalls to actively prevent

- Allocated context != useful effective context, especially with multiple slots.
- A model file fitting in 16GB != model + KV/recurrent state + activations + scratch fitting.
- Shared-memory allocation != demonstrated spill; measured deltas and timing matter.
- Prefix caching helps unchanged prefixes, not arbitrary new long prompts.
- Speculative decoding or reduced thinking != inherently faster prefill.
- First reasoning token != first useful tool action.
- SSE messages != tokens; decode duration != whole-request wall time.
- GGUF, EXL3, EXL2, AWQ/compressed-tensors and MXFP4 are not interchangeable file formats.
- Alpha software may win, but speed without correct tools/64K quality is not a qualifying result.
- Generated-code tests need OS/container isolation, not only a restricted tool name.
- A restart must not reset the run budget or erase completed evidence.

These checks are core implementation requirements, not optional TODOs.

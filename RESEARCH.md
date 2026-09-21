# Research and TTFT experiment notes

Researched 2026-09-18. These are compatibility leads and experiment rationale, not locally measured winners. Recheck changed software documentation during execution, then pin the actual version used. Experimental/community sources are eligible; the relevant distinction is reproducible working code versus an untested claim.

## 1. Historical Gemma baseline: what was actually measured

The existing [Gemma log](../.logs/gemma-prefill-f16-repeat-20260918.log) identifies google/gemma-4-12b-qat, GGUF Q4_0, configured context 49,152 and actual prompt 48,046 tokens.

| Physical batch | Parallel slots | Server-reported TTFT | Output tok/s | Request wall time |
| --- | --- | --- | --- | --- |
| 512 | 4 | 34.787s | 25.78 | 40.681s |
| 1024 | 4 | 35.871s | 23.05 | 41.834s |
| 2048 | 1 | 35.423s | 24.52 | 40.758s |

Other raw files are [batch 1024](../.logs/gemma-prefill-batch1024-20260918.log) and [batch 2048](../.logs/gemma-prefill-batch2048-20260918.log). Three runs with changed slots are not a controlled batch-size study.

The first log records 128 generated tokens, of which 125 were reasoning. The test used a non-streaming client, so these are server-reported first-token times, NOT client-stream arrival or first useful tool-action measurements. It also did not establish a 64K context. The config excerpt does not explicitly record K/V quantization formats; do not infer them solely from the log filename.

Peak dedicated GPU memory was approximately 10.0–10.3 GiB across these runs, and shared allocation was flat within each run. That is not evidence that active WDDM spill caused the latency.

Dividing 48,046 by 34.787 gives approximately 1,381 input tokens per second as a ROUGH aggregate diagnostic, not pure kernel prefill throughput. This suggests investigating model/engine prompt-processing paths, rather than expecting a physical-batch toggle alone to remove tens of seconds. Additional reasoning can delay first useful output even after first-token time improves. Both require fresh streamed measurements.

## 2. TTFT: separate the costs before optimizing

For a loaded engine, latency includes request/queue overhead, template/tokenization, data movement, prompt processing, first decode and stream delivery. First action additionally includes any reasoning and generation needed to finish valid tool JSON. Startup/model loading is a different measurement.

Experiment priority:
1. GPU-resident small/hybrid models with sufficient coding quality, compared across GGUF CUDA, optimized forks, EXL3 and quantized Linux kernels. This is the main cold-prefill hypothesis.
2. Actual CUDA Flash Attention and supported quant/GEMM paths; check logs for CPU fallback and real architecture support.
3. Batch/physical batch and scratch-memory balance with one request slot held constant.
4. Lower-bit KV/weights only where they make residency/kernels better and retain 64K coding quality.
5. Reduced/disabled reasoning for time-to-action, with complete quality requalification.
6. Prefix reuse and recurrent-state checkpoints for follow-up turns, never counted as a fresh-prompt win.
7. Speculative decoding for output speed, guarded against fresh TTFT regression and added VRAM.

Keeping a server loaded and warmed improves startup experience, but does not erase reading a brand-new long prompt. Cache hits can skip repeated prompt computation, not arbitrary new code. vLLM documents this distinction in [automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/).

vLLM's prefill scheduling is worth testing rather than accepting serving defaults: chunk/token budget trades prefill latency against decode/concurrency behavior. Read the exact release's controls and constraints in [optimization guidance](https://docs.vllm.ai/en/stable/configuration/optimization/). The test is one interactive coding agent, not maximum multi-user throughput.

Do not reduce reported input length, silently compress the prompt, or enable context shift to make the "64K" number look faster. Retrieval/smaller active context is a separately labelled practical alternative if fresh full-context targets prove unattainable.

## 3. Memory/context planning

For an ordinary all-full-attention architecture, a rough unquantized KV estimate is:

KV bytes ~= 2 * full_attention_layers * kv_heads * head_dimension * bytes_per_value * cached_tokens * concurrent_slots

This is not a universal model-memory formula. Hybrid/linear-attention, sliding windows, cache scales/padding, recurrent state, logits, activations, graph buffers and scratch memory change it. The fork research explicitly describes Qwen3.5's hybrid cache behavior in [TurboQuant model-specific notes](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant/blob/master/docs/turboquant.md); do not transfer that fork's Radeon throughput to NVIDIA.

Use model/config metadata and actual loaded-engine buffer/peak logs as authority. Confirm PER-SLOT context when parallel slots are enabled. Treat f16, q8, q4 and experimental caches as separate quality profiles. Turning quantization off often increases memory use without guaranteeing a speed improvement.

Keep headroom for desktop VRAM and peak prefill buffers. A weight file below 16GB is necessary for straightforward residency but not sufficient. A MoE's active parameter count describes compute, not storage of all experts.

## 4. Upstream llama.cpp and speculation

Native server/template/cache control reference: [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). Build options/toolchains: [build guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md). Speculation types/drafts: [speculative decoding guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md).

Read actual --help and effective startup logs; these links track moving branches. Native generation is authoritative even if llama-bench is missing. Prefix-cache reset and context-shift controls must be capability-tested in the selected build.

A reported Qwen3.5/MTP prompt-processing slowdown compares compiler/CUDA setups on other hardware, not this machine. It is a reason to compare MTP OFF/ON and inspect compiler effects after a reproducible local regression, not a promised fix: [maintainer issue 28790](https://github.com/ggml-org/llama.cpp/issues/28790).

A speculative-prefill feature request was closed rather than establishing an implemented universal cold-prefill accelerator: [issue 19082](https://github.com/ggml-org/llama.cpp/issues/19082). Do not advertise ordinary draft decoding as though it skips evaluating a new 64K input.

## 5. ik_llama.cpp: concrete probe instructions

Compatibility warnings and model support must come from the [maintainer README](https://github.com/ikawrakow/ik_llama.cpp). Engine-specific controls are documented in [parameters.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/parameters.md); do not mix flag names from other forks.

Useful test leads from the parameter reference:
- --reasoning on/off/auto and --reasoning-budget provide explicit thinking profiles.
- --peg is a Qwen3.5 tool-parser option worth checking against real malformed/nonexistent-tool handling.
- --ctx-checkpoints, interval and tolerance influence reuse for recurrent architectures.
- --cache-ram stores reusable prompt state in host RAM; bound it under WSL's 32GiB limit.
- Batch threads can be lower for full-GPU inference; include 2 alongside the regular thread sweep.
- --spec-type uses canonical types/payloads; legacy speculative aliases may be rejected.

These are version-specific leads, not commands to run blindly. Cache/checkpoint improvements are follow-up metrics. Source/build/image hashes go into the engine manifest.

## 6. ExLlamaV3 / TabbyAPI

Use current [ExLlamaV3 compatibility/install/conversion documentation](https://github.com/turboderp-org/exllamav3). It is a different format/kernel candidate, not a GGUF backend; preserve exact base-model provenance when comparing quantizations.

The maintained API layer is [TabbyAPI](https://github.com/theroyallab/tabbyAPI), which supports OpenAI-style tools and is a rolling release. Its Docker guidance distinguishes CUDA tags and recommends adequate shared memory. For a container probe, set the documented shared-memory allowance, use a local-only host port, mount only the selected model/config directories, and pin the image digest. Avoid third-party impersonation/download sites.

The runtime's maintained converter has create/resume forms:
- convert.py -i INPUT_DIR -o OUTPUT_DIR -w WORK_DIR -b BITRATE
- convert.py -w WORK_DIR -r

Execute with the isolated compatible engine interpreter. These are argument contracts, not actual installed paths. Read convert.py -h and the linked conversion/self-calibration guide at the pinned commit. Account for a full converted copy in WORK_DIR while building OUTPUT_DIR. Start at approximately 4 bpw for one selected 9B model; upgrade/downgrade only if remaining budget and observed quality/memory justify it. Prefer existing compatible community EXL3 to unnecessary duplicate conversion.

## 7. Experimental CUDA TurboQuant

Initial lead is the [CUDA-focused fork](https://github.com/jarkevithwlad/llama.cpp-turboquant-cuda). Its documented cache pair is tbqp3 K / tbq3 V with GPU KV and Flash Attention. The README's example hardware/preset is not this GPU; detect/build for the local architecture and test the selected model's head dimension.

Alternative leads: [x6nux CUDA fork](https://github.com/x6nux/llama-cpp) and [Madreag turbo3-cuda](https://github.com/Madreag/turbo3-cuda). They have distinct implementations/cache labels; select one initially and inspect actual kernels/tests/help before investing the larger sweep budget.

ROCm results are not CUDA validation. A different fork explicitly labels CUDA hardware untested and notes a deep-context decode tradeoff: [Pascal-SAPUI5 limitations](https://github.com/Pascal-SAPUI5/llama.cpp-turboquant). This is why the plan requires full-depth retrieval, code tests and actual local decode measurements, rather than assuming compressed KV must improve every metric.

Run experimental code without system-wide changes. Symbol loops, incoherent long-context answers or broken tool parsing disqualify a setting even if memory/throughput look excellent.

## 8. Linux quantized engines and checkpoints

vLLM's supported installation/environment baseline is [GPU installation documentation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/). For SGLang use [installation](https://docs.sglang.io/docs/get-started/install) and [quantization support](https://docs.sglang.io/docs/advanced_features/quantization). Inspect current Ada/model/format support and package/image requirements; do not assume all Linux wheels or kernels work on Windows/WSL.

The pinned [Qwen3.5 9B INT4 checkpoint](https://huggingface.co/cyankiwi/Qwen3.5-9B-AWQ-4bit) is advertised as AWQ but config representation must decide loading. The pinned [Ornith 1.5 9B INT4 checkpoint](https://huggingface.co/cyankiwi/Ornith-1.5-9B-AWQ-INT4) is a separate format probe. Save exact config/index/tokenizer/shards at the JSON revisions; let a supported engine detect the actual quantization scheme.

Do not casually fall back to BitsAndBytes: current vLLM guidance describes a plugin integration rather than a guaranteed built-in path. Check [BitsAndBytes integration](https://docs.vllm.ai/en/stable/features/quantization/bnb/) if this bounded fallback is genuinely needed.

For [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), index-driven root-shard selection avoids duplicate original/ weights. The three root files/13.76GB total in JSON are metadata, not proof its 64K cache fits. Confirm exact MXFP4 kernel path and coding-tool quality locally.

## 9. Ornith and Hugging Face selection/pinning

[Ornith 1.5 9B's producer card](https://huggingface.co/ornith-ai/Ornith-1.5-9B) gives Qwen-derived architecture/context and initial package/parser/sampling guidance. The reviewed card listed minimum Transformers 5.8.1, vLLM 0.19.1 and SGLang 0.5.9; recheck newer compatibility instead of treating those numbers as timeless installation locks.

The [9B GGUF repository](https://huggingface.co/ornith-ai/Ornith-1.5-9B-GGUF) and [35B GGUF repository](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF) provided pinned file metadata. Smaller community 35B quantizations or new finetunes are eligible if found, with independent provenance and quality checks. Do not confuse 1.0 and 1.5 models.

Selection algorithm for any extra HF candidate:
1. Query public model metadata including blobs; record returned commit SHA, file sizes and LFS SHA256 values.
2. Read config, tokenizer_config, chat template, model card and license without executing remote code.
3. Verify architecture/native context/tools and a runnable compatible local format.
4. Check installed/cache roots for complete matching files; reuse them.
5. Reserve required new weight bytes and disk headroom before download, including conversion staging.
6. Select exact filenames or index-referenced shards plus support files; exclude duplicate formats, projectors for text-only runs, training checkpoints and original/ copies.
7. Download revision-pinned files with resumable partials; validate size/SHA before use.
8. Save acquisition metadata and actual format/base-model mapping in the ledger.

For newly discovered EXL3, verify the artifact really is EXL3 and corresponds to the desired base revision; a search match or "4bit" filename alone is inadequate. A newer experimental quant may beat the pinned list, but it still goes through the same gates.

## 10. Managed engine controls

Use the [LM Studio load API](https://lmstudio.ai/docs/developer/rest/load) to learn which controls are actually exposed in the installed version. Treat effective config and owned instance identity as authoritative; do not use private-file edits for a hidden knob.

Ollama's [native chat API](https://docs.ollama.com/api/chat) provides streaming and final runtime stats. Context behavior is documented in [context-length guidance](https://docs.ollama.com/context-length). Observe the installed daemon's actual counters and num_ctx; missing cached counts are unknown, not zero.

No source above establishes a local winner. The future reports must replace compatibility hypotheses with this machine's logged timings, functional scores, exact configurations and honest skipped/incomplete coverage.

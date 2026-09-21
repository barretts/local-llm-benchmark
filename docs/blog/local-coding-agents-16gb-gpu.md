# My Search for a Reliable Local Coding Agent on a 16 GB GPU

I wanted a local coding assistant that could read a useful amount of repository context, use its tools, and finish a change correctly. Then I wanted it to be fast. That order turned out to matter quite a bit.

In [my earlier LM Studio experiment](https://sosuke.com/running-codex-with-lm-studio-taking-it-offline-by-stopping-the-callouts/), I got a local model to create a file, read it back, and verify it. That was a useful connection test. This time I wanted to find out how far a local coding agent could get on harder, repeatable work.

The machine was a Windows desktop with an RTX 4060 Ti 16 GB, a Threadripper 1950X, and 128 GiB of RAM. I allowed a seven-day execution ceiling and 300 GB of additional model weights, reused installed files where possible, and ran one model workload at a time.

The recorded work spanned about 56 hours of elapsed time, including downloads and pauses. It produced 63 registered configurations and 1,093 attempt records. Those records include coding tasks, capacity checks, timing, retries, and interrupted attempts. They are not 1,093 different programming problems.

**Nothing passed the complete qualification contract.** I did find two configurations I would keep for supervised coding, a genuinely interesting compression result, and several failures that needed a more careful explanation than "the model is bad."

## What counted as a passing coding agent

The original target was a 65,536-token context window, with 61,440 tokens of fully templated input and 4,096 tokens reserved for output. I wanted to measure a nearly full, fresh prompt, rather than load a large context setting and send it a tiny question.

A qualifying configuration needed:

- At least 27 passes across 36 regular Python and TypeScript coding runs, with complete coverage.
- At least nine passes across 12 long-context coding runs.
- All nine capacity markers retrieved across three seeded requests.
- Twenty valid fresh measurements and twenty separately measured cached follow-ups.
- A working isolated agent demonstration.

The harness had public tests, hidden checks, and known-good implementations. I used Codex to build and operate it, while the local models being benchmarked handled the fixture tasks. Candidate code ran in isolated CPU Docker graders.

The coding runs already allowed iteration: up to 16 tool turns, a generated-token budget, and a 900-second task deadline. An agent could inspect source, patch it, run public tests, and revise. These scores describe the final outcome after that opportunity.

The gates are demanding. They are also the gates I chose before seeing which model looked good.

## The three speed numbers I kept separate

Time to first token was not enough.

The first streamed event could be reasoning. The agent might start talking almost immediately and still leave me waiting for a usable tool call. I therefore measured first output and first useful action separately. A useful action had to be completely parsed and valid for the advertised tool schema.

Decode speed was a third number. It described verified output tokens per decode second, not the time needed to ingest a fresh repository.

For three original 64K configurations, the screen looked like this:

| Configuration | Fresh first output | Fresh first useful action | Sustained decode |
| --- | --- | --- | --- |
| Qwen3.6 27B IQ2_M, upstream llama.cpp | 119.54-119.68 s | 123.33-124.55 s | 17.30 tok/s |
| Qwen3.6 27B IQ2_M, bundled llama.cpp | 107.05-107.19 s | 110.75-112.83 s | 17.13 tok/s |
| OxCoder 9B Q4_K_M, upstream llama.cpp | 35.23-35.40 s | 37.62-37.81 s | 31.79 tok/s |

These are screening results: three fresh samples per configuration and one sustained full-context throughput sample. They are not the required final twenty-sample p95. Request timing also excludes model launch startup.

Cached follow-ups remained a separate metric. No candidate reached and completed the final fresh/cached qualification sequence, so I do not have a qualified cached-latency winner to announce.

A screenshot showing a high output rate can be perfectly real and still answer a different question from the one I was asking.

## Qwen3.6 was the closest to what I wanted

The strongest regular coding result came from [Qwen3.6 27B in the i1-IQ2_M representation](https://huggingface.co/mradermacher/Qwen3.6-27B-i1-GGUF). Both measured llama.cpp builds passed **28 of 36 regular coding runs**.

The upstream build had one malformed or unpermitted tool failure in those 36 runs. The bundled build had two. Those are small counts, so I would not claim that they prove one engine is generally more reliable.

Both configurations also retrieved all nine full-context capacity markers.

Then the long-context contract stopped them. They passed four of eight completed long runs, and four failures made the nine-of-twelve target unreachable. The final four runs were skipped.

For upstream, all four failed long runs violated the requirement for exactly one final action. The bundled build had three such failures and one timeout. That is important: I cannot turn those records into a claim that the model forgot everything in the repository. It failed the required agent behavior.

Qwen3.6 IQ2_M remains my quality-first starting point for supervised work. It passed the regular gate and had relatively few regular tool failures. It did not earn the full 64K agent label.

The faster option I would keep is [OxCoder 9B Q4_K_M](https://huggingface.co/prithivMLmods/OxCoder-9B-GGUF) on upstream llama.cpp. It passed 11 of 21 completed coding runs, with **zero tool-error runs**. Its ten failures were deterministic coding-test failures.

That makes its limitations easier to work with: the tool loop functioned, but the final code often needed better decisions. The benchmark had already allowed revisions, so another prompt is not a guaranteed cure. Still, this is the faster configuration I would choose for small edits with tests and review.

## Dropping to 16K found fast models, not a qualified agent

I eventually authorized a separate 16K pass. It used a 16,384-token window, 12,288 tokens of input, and the same 4,096-token output reserve. It also required full GPU layer offload, f16 KV cache, and at least 512 MiB of measured memory headroom.

For the installed family models, I added a quick tool-use screen: stop after four coding runs if two or more ended in tool-call errors.

Here is where that queue finished:

| Installed model | Coding passes / completed runs | Tool-error runs |
| --- | --- | --- |
| Devstral Small 2 24B Q3_K_L | 2/12 | 3 |
| Ministral 3 14B Reasoning Q4_K_M | 4/14 | 3 |
| GPT-OSS 20B Q4_K_S | 0/4 | 4 |
| Gemma 4 12B QAT Q4_0 | 2/12 | 6 |
| Gemma 4 E4B Q6_K | 5/15 | 5 |
| Nemotron 3 Nano 4B Q6_K | 2/12 | 3 |

These are early-stopped subsets, not a complete-suite leaderboard. Tool failures overlap with coding failures. Only GPT-OSS triggered the first-four cutoff; the other models later reached ten regular task failures.

Gemma E4B and Nemotron were responsive in the diagnostic workload. Their median fresh useful actions were about 5.11 and 3.91 seconds respectively, with native 256-output-token screens around 50.4 and 53.5 tok/s. Those measurements used the smaller 12,288-token input, so they should not be compared directly with the original 61,440-token results.

They were fast. Their coding and tool reliability were still insufficient.

The Nemotron result applies to the installed Nano 4B file. It is not a verdict on every Nemotron model.

## GPT-OSS did not simply return null

I consider broken tool use a more serious obstacle than incorrect code. A functioning tool loop can run a test and show me the failure. A rejected action can prevent the agent from making the next useful change.

But the GPT-OSS records deserve precision.

All four failed coding runs successfully used native file-listing and reading tools. The next patch call supplied JSON-valid arguments containing the Codex-style `*** Begin Patch` and `*** Update File` format.

The frozen patch tool required standard numbered unified diffs. It rejected that different patch dialect.

There was also a separate native Harmony parser error in the third capacity request. The first two seeded requests succeeded, retrieving six of the nine required markers.

The tested integration failed. That is a narrower and more useful finding than saying GPT-OSS cannot call tools. A bounded retest with a clearly followed patch contract and compatible parsing is worth considering. Silently accepting the wrong format and keeping the same grade would make the benchmark less honest.

## The model file is only part of GPU memory

I initially thought the main benefit of shorter context would be room for higher quantization. That benefit exists, but shorter actual input also means less fresh prefill work.

The memory boundary was less intuitive than a file listing made it look:

| 16K trial | Weight file size, decimal GB | Measured free headroom |
| --- | --- | --- |
| Qwen3.8 27B UD-Q4_K_M | 16.46 | 290 MiB |
| Qwen3.8 27B UD-Q4_K_S | 15.36 | 364 MiB |
| Qwen3.8 27B UD-IQ4_XS | 14.25 | 1,354 MiB |
| Qwen3.6 27B IQ4_XS | 15.44 | 312 MiB |
| Qwen3.6 27B UD-Q3_K_XL | 14.47 | 1,164 MiB |

The two Qwen3.8 Q4 files and Qwen3.6 IQ4_XS missed the headroom rule and their full-prompt probes timed out. Qwen3.8 IQ4_XS and Qwen3.6 Q3_K_XL fit and passed all nine capacity markers.

A GGUF file is a disk representation. GPU weight buffers, KV cache, compute buffers, runtime allocations, and Windows activity have to fit too. I also corrected a headroom calculation after finding about 270 MiB of reserved memory that "total minus used" had incorrectly treated as available.

The fitting Qwen3.6 Q3_K_XL stack finished at eight passes in eighteen coding runs, including nine tool failures. Qwen3.8 IQ4_XS had six passes in eleven completed runs, including four tool failures, before I stopped it.

Higher precision did not automatically rescue these configurations. Context and weight representation changed together, so this is not a controlled claim that one quantization caused better or worse reasoning.

## Bonsai made the memory claim interesting

[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) was the compression experiment I most wanted to check. Its PTQ1_0 weight file was about **5.95 GB decimal** and needed the Prism ternary fork.

It fully offloaded to the GPU, passed all nine 64K capacity markers, and recorded roughly 10,426-10,554 MiB of peak GPU use in those full-context capacity requests.

The response time was less exciting. With 61,440 fresh input tokens, the three clean capacity samples took **167.5-169.2 seconds to first output** and **187.5-192.1 seconds to the useful action**.

A smaller probe took 40.7 seconds to first output and 54.1 seconds to useful action. That probe contained 16,384 input tokens while the engine was still configured for 65,536. It was not the separate 16K-window benchmark.

Only five coding smokes finished before I stopped Bonsai: three passed and two failed on tool streams. An interrupted sixth run is not a completed failure.

I would keep it for memory-efficiency research. I would not make it my everyday coding pick from these results.

## MTP and DFlash actually drafted tokens

The speculative tests used Qwen3.6 IQ2_M with q8 and q4 target-cache profiles. Native counters confirmed real drafting and accepted draft tokens.

MTP acceptance was approximately 90-92%. DFlash acceptance was approximately 58-59%.

There were promising auxiliary decode samples: roughly 24.2-25.5 tok/s for MTP, and 36.7 tok/s for one clean DFlash q4-cache capacity sample. The clean sample counts were tiny, and several other intervals overlapped with foreign GPU activity.

Every speculative arm failed screening. None produced a qualifying set of fresh useful-action measurements.

I still think these are useful leads. They add draft weights and cache pressure, though, and a long fresh prompt still has to be processed. Draft acceptance and decode speed do not replace a functioning coding agent.

## Some engine comparisons remained unavailable

The search covered upstream and bundled llama.cpp paths, an eligible CUDA TurboQuant fork, and attempts to compare ik_llama.cpp, ExLlamaV3/TabbyAPI, vLLM, SGLang, LM Studio, and Ollama.

Several comparisons stopped at integration or setup barriers: a missing CUDA library in the ik container, missing snapshot metadata for Tabby, Docker storage headroom for vLLM and SGLang, and unresolved managed loading or request controls for LM Studio and Ollama. Bounded setup budgets also expired.

LM Studio authentication was not its final blocker.

Those engines are unavailable comparisons in this experiment. I do not have measurements establishing that they are slower or worse. A fair engine ranking still needs those barriers resolved and the same tasks run successfully.

## What I would use, and what I would test next

For supervised coding quality, I would start with **Qwen3.6 27B IQ2_M**. For a faster assistant whose tools behaved consistently across its completed regular tasks, I would keep **OxCoder 9B Q4_K_M on upstream llama.cpp**.

GPT-OSS is an interface-repair lead. Bonsai is a memory research lead. The faster MoE and small-model output rates were interesting, but their coding or tool failures kept them out of my practical shortlist.

The next experiment I would run is a small, clearly versioned GPT-OSS interface retest. After that, a matched 16K comparison of Qwen3.6 IQ2_M and OxCoder would make the latency tradeoff easier to judge. I would give speculative decoding another pass once the target agent earns confidence in its tools.

The useful result from this search is a set of measured limits and reasons to prefer particular configurations. I still do not have the fully qualified local coding agent I set out to find.


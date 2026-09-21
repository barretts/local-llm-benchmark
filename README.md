# Local coding-agent benchmark

Resumable Windows benchmark for local coding agents. Coding quality, valid tool use, context capacity, and stability are qualification gates. Fresh-context time to the first useful action is the primary speed metric; cached latency is measured separately.

Read [START_HERE.md](START_HERE.md), [EXECUTION_PLAN.md](EXECUTION_PLAN.md), [FIXTURES.md](FIXTURES.md), [benchmark-spec.json](benchmark-spec.json), and [RESEARCH.md](RESEARCH.md) for the original frozen 64K contract. The planning documents retain their original pre-implementation wording. The harness is now implemented.

## Source layout

For everyday local serving and Aider, double-click `Local-Models.cmd`. See [the small launcher guide](launcher/README.md) for the local API, model switching and coding client. These practical profiles are separate from benchmark qualification.

- `localbench/`: controller, adapters, measurement, grading, resource collection, reports, acquisition, recovery, and serving.
- `scripts/`: setup, verification, supervision, and focused experiment controllers.
- `tests/`: controller and adapter tests, with synthetic measurement data.
- `fixtures/`: deterministic Python and TypeScript agent tasks.
- `grader/`: public and hidden checks, gold implementations, and fixture metadata. Agents receive only the permitted fixture files; grader material stays outside their workspaces.

## Local execution

The configuration contains absolute paths for the original Windows host, its installed runtimes, and model files. This repository is a source snapshot, not a portable runtime bundle. Model weights, binaries, environments, credentials, logs, raw measurements, and durable checkpoints remain local and are ignored by Git. Review the documented safety and budget rules before executing on another host.

With the project's private Python environment available, inspect CLI commands using:

```powershell
.\.venv\Scripts\python.exe -m localbench --help
```

The original verified harness had 597 tests with one skipped test. Focused controllers verify that the parent source hashes remain unchanged and perform additional scope checks before starting runtime probes.

## Separate 16K family experiment

`scripts/run-families-16k.py` sequentially tests the installed Devstral, Ministral, GPT-OSS 20B, Gemma 12B QAT, Gemma E4B, and Nemotron files. It requires full GPU offload, f16 KV cache, and at least 512 MiB of measured GPU headroom, using a 12,288-token prompt and a 4,096-token output reserve.

The first pass stops a model after four coding runs if at least two have tool-call errors. Ordinary coding failures are recorded separately. Surviving models still need to pass the full coding, long-context, capacity, stability, and demo gates. A 16K result never qualifies as a 64K result. All experiments share the original cumulative weight ledger and elapsed-time deadline.

```powershell
.\.venv\Scripts\python.exe scripts/run-families-16k.py --self-test
.\.venv\Scripts\python.exe scripts/run-families-16k.py
```

Queue status and per-model results are written to `artifacts/families-16k/` on the local host. Resume from the existing checkpoints; do not restart the entire search.

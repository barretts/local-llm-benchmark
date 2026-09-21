# Local API + Aider

For this 16 GB machine, start with **Qwen3.6 + Aider at 16K**. OxCoder is the faster fallback. The launcher reuses the installed, pinned llama.cpp engine and weights. It adds no API proxy or web framework.

Double-click **Local-Models.cmd** in the project folder. Choose Qwen, OxCoder, or Qwen at 64K, then choose **Open Aider** and enter your coding folder. Switching models stops the current launcher-owned model first. Closing the menu leaves the API running; **Stop-Local-Model.cmd** releases it.

You can also drag a coding folder onto **Aider.cmd**; that starts Qwen at 16K and opens Aider there. Aider can edit files in the folder you choose. Automatic Git commits, automatic lint/test execution, analytics and update checks are disabled. Add files with `/add`, request a change, review it with `/diff`, and run your checks with `/test YOUR_TEST_COMMAND`. Use `/clear` when starting a separate task and `/drop` for files no longer needed.

## API

| Setting | Value |
| --- | --- |
| OpenAI-compatible base URL | `http://127.0.0.1:8080/v1` |
| Model name | `local-coder` |
| API key if your client requires one | `local-only` (a dummy value) |
| Default context | 16,384 tokens, one slot |
| Aider output limit | 4,096 tokens |

The API has no authentication and binds to loopback only. It works with local OpenAI-compatible clients. No LM Studio credential is needed. Native tool use is available through the model's template; Aider uses textual SEARCH/REPLACE edits instead.

From the project folder:

```powershell
.\.venv\Scripts\python.exe launcher\api.py start qwen
.\.venv\Scripts\python.exe launcher\api.py start oxcoder
.\.venv\Scripts\python.exe launcher\api.py start qwen --context 65536
.\.venv\Scripts\python.exe launcher\api.py status
.\.venv\Scripts\python.exe launcher\api.py stop
.\.venv\Scripts\python.exe launcher\api.py aider --repo C:\path\to\your\project
```

Add specific files with repeated `--file filename.py`. Choose a different model with `aider oxcoder --repo ...`. Use `--no-git` to work on selected files without discovering an enclosing Git repository. Override the API port with `--port 8081` on the same command. No global environment variables or app configuration need changing.

## Safety and runtime

The launcher holds the benchmark's existing GPU file lock for the server's lifetime. It refuses an occupied port and a GPU already using more than 2 GiB; it requires at least 512 MiB of free GPU memory after loading. It does not open or change benchmark checkpoints, stop flags, grades or budgets.

Servers start suspended and enter a Windows kill-on-close Job Object before executing. Model descendants are owned together, including the llama.cpp loader's children. Stop verifies the controller's original creation time and exact command before signaling its session event. Closing or crashing the controller closes its job and terminates its model processes. Unrelated applications are not stopped.

`profiles.json` records model SHA-256, size, engine SHA-256 and commit. The binary is checked each launch. Models are fully hashed on first launch, then the successful check is reused while the path, size and modification time remain unchanged. Delete `runtime/verified-models.json` if you want to force a new full model hash check.

State, server logs, Aider histories, tokenizer caches, Python and Aider packages stay in ignored `launcher/runtime/`. Aider uses a private home directory and disables dotenv loading. Explicit local model metadata prevents remote pricing/model metadata lookups. It can still honor settings in the coding repository you select, with launcher flags taking precedence. Review your existing repository configuration before use.

## Versions and installation

The installed version is Aider 0.86.2 with private Python 3.12.13. The API launcher uses the benchmark's existing private Python 3.13 environment. llama.cpp is build 11040, commit `5b335f413e4f73b0809c4fe39af894efbcc6a0d2`; exact settings are in `api.py` and the current `runtime/api-status.json`.

All dependencies are already installed on this host. If rebuilding the Aider environment, run `launcher/setup-aider.ps1`. It uses the existing uv executable with private Python and cache directories; it never changes global Python, PATH or persistent environment variables. The repository does not bundle weights, llama.cpp binaries or environments. Paths in `profiles.json` are specific to this machine.

## Limits and verification

Neither practical pick qualified under the full coding-agent benchmark. A simple Aider repair passing checks demonstrates this integration; it does not establish general coding reliability. Qwen's optional 64K window also does not make it a qualified 64K agent. The default 16K profile is intended for selected files and small changes.

The local integration checks passed for both models on September 21, 2026: actual API inference, explicit context overflow rejection, and an Aider repair checked in an isolated container. OxCoder needed no format correction; Qwen needed one automatic retry for a missing edit filename. Qwen's 64K option also loaded and answered a short request. The 21 CPU safety tests passed. See [verification.json](verification.json); full operational logs remain in ignored `runtime/`.

Aider's token estimates use a different tokenizer from the model. Metadata describes the input/output allowance; Aider does not strictly enforce it. The server disables context shifting and rejects oversized prompts. Keep file selection small, inspect the token estimate with `/tokens`, and clear old task history when needed. Long responses can still end at the configured output limit.

Run the operational smoke test with:

```powershell
.\.venv\Scripts\python.exe scripts\verify-local-launcher.py
```

It switches models sequentially, checks API inference and explicit overflow rejection, then requests an actual Aider repair in disposable folders. Model-written Python executes only in a CPU-only, network-disabled, read-only container. This requires the already installed Python Docker image and leaves the final model running. CPU ownership/contract tests are `tests/test_launcher_job.py` and `tests/test_model_launcher.py`.

References: [Aider local APIs](https://aider.chat/docs/llms/openai-compat.html), [edit formats](https://aider.chat/docs/more/edit-formats.html), [options](https://aider.chat/docs/config/options.html), [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

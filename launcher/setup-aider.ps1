$ErrorActionPreference = 'Stop'
$taskProject = Split-Path $PSScriptRoot -Parent
$taskRuntime = Join-Path $PSScriptRoot 'runtime'
$taskUv = 'C:\Users\barrett\.local\bin\uv.exe'
if (-not (Test-Path -LiteralPath $taskUv)) { throw 'The existing private uv executable is missing.' }
New-Item -ItemType Directory -Force -Path $taskRuntime | Out-Null
$taskOldPythonDir = $env:UV_PYTHON_INSTALL_DIR
$taskOldCacheDir = $env:UV_CACHE_DIR
try {
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $taskRuntime 'python'
    $env:UV_CACHE_DIR = Join-Path $taskRuntime 'cache\uv'
    & $taskUv --no-config python install --no-bin 3.12.13
    if ($LASTEXITCODE -ne 0) { throw 'Private Python installation failed.' }
    $taskEnvironment = Join-Path $taskRuntime 'aider-env'
    if (-not (Test-Path -LiteralPath (Join-Path $taskEnvironment 'Scripts\python.exe'))) {
        & $taskUv --no-config venv --python 3.12.13 --managed-python $taskEnvironment
        if ($LASTEXITCODE -ne 0) { throw 'Private Aider environment creation failed.' }
    }
    $taskPython = Join-Path $taskEnvironment 'Scripts\python.exe'
    & $taskUv --no-config pip install --python $taskPython -r (Join-Path $PSScriptRoot 'aider-lock.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Aider dependency installation failed.' }
    & $taskUv --no-config pip check --python $taskPython
    if ($LASTEXITCODE -ne 0) { throw 'Aider dependency verification failed.' }
    & $taskPython -c "import os,sys; os.environ['TIKTOKEN_CACHE_DIR']=sys.argv[1]; import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base'); from importlib.metadata import version; print('Aider '+version('aider-chat'))" (Join-Path $taskRuntime 'cache\tiktoken')
    if ($LASTEXITCODE -ne 0) { throw 'Aider tokenizer verification failed.' }
} finally {
    $env:UV_PYTHON_INSTALL_DIR = $taskOldPythonDir
    $env:UV_CACHE_DIR = $taskOldCacheDir
}

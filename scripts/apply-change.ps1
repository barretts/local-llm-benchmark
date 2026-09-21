param([Parameter(Mandatory=$true)][string]$PatchFile)
$ErrorActionPreference = 'Stop'
$project = 'C:\Users\barrett\local-llm-benchmark'
$resolved = [IO.Path]::GetFullPath($PatchFile)
if (-not $resolved.StartsWith($project + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Patch file must be inside benchmark workspace' }
$patch = Get-Content -LiteralPath $resolved -Raw
if ($patch.Length -gt 24000) { throw 'Keep patch below Windows command-line limit' }
foreach ($line in ($patch -split "`n")) {
    if ($line -match '^\*\*\* (?:Add|Update|Delete) File: (.+)$' -or $line -match '^\*\*\* Move to: (.+)$') {
        $target = $Matches[1].Trim().Replace('/', '\')
        if (-not [IO.Path]::IsPathRooted($target)) { $target = Join-Path $project $target }
        $target = [IO.Path]::GetFullPath($target)
        if (-not $target.StartsWith($project + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Patch target outside benchmark workspace' }
    }
}
$codexPatch = 'C:\Users\barrett\AppData\Local\nvm\v24.13.1\node_modules\@openai\codex\node_modules\@openai\codex-win32-x64\vendor\x86_64-pc-windows-msvc\bin\codex.exe'
if (-not (Test-Path -LiteralPath $codexPatch -PathType Leaf)) { throw 'Documented apply-patch executable missing' }
Push-Location -LiteralPath $project
try { & $codexPatch --codex-run-as-apply-patch $patch; if ($LASTEXITCODE -ne 0) { throw 'apply-patch failed' } }
finally { Pop-Location }

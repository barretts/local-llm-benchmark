param(
    [Parameter(Mandatory=$true)][ValidateSet('doctor','self-test','prepare-graders','init-fixtures','verify-fixtures','sweep','resume','probe','report','agent-endpoint')][string]$Operation,
    [string]$Engine,
    [string]$Model,
    [string]$Plan
)
$ErrorActionPreference = 'Stop'
$project = 'C:\Users\barrett\local-llm-benchmark'
$python = Join-Path $project '.venv\Scripts\python.exe'
$logs = Join-Path $project '.logs'
$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ')
$arguments = @('-X', 'utf8', '-m', 'localbench', $Operation)
if ($Operation -eq 'probe') {
    if ($Engine -notmatch '^[a-z0-9-]+$' -or $Model -notmatch '^[a-z0-9-]+$') { throw 'Invalid engine or model id' }
    $arguments += @('--engine', $Engine, '--model', $Model)
}
$stdout = Join-Path $logs ($Operation + '-' + $stamp + '.out.log')
if ($Operation -eq 'agent-endpoint') {
    $resolvedPlan = (Resolve-Path -LiteralPath $Plan -ErrorAction Stop).Path
    if (-not $resolvedPlan.StartsWith((Join-Path $project 'artifacts\'), [StringComparison]::OrdinalIgnoreCase)) { throw 'Launch plan must be inside benchmark artifacts' }
    $arguments += @('--plan', $resolvedPlan)
}
$stderr = Join-Path $logs ($Operation + '-' + $stamp + '.err.log')
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
[pscustomobject]@{pid=$process.Id; created=$process.StartTime.ToUniversalTime().ToString('o'); operation=$Operation; stdout=$stdout; stderr=$stderr} | ConvertTo-Json

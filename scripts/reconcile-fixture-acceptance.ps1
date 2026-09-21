param([Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$ExpectedVersion)
$ErrorActionPreference = 'Stop'
$project = [IO.Path]::GetFullPath('C:\Users\barrett\local-llm-benchmark')
function Assert-OwnedPath([string]$Path) {
    $resolved = [IO.Path]::GetFullPath($Path)
    if (-not $resolved.StartsWith($project + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Archive operation outside benchmark workspace' }
    return $resolved
}
$manifestPath = Assert-OwnedPath (Join-Path $project 'artifacts\fixture-manifest.json')
$verificationPath = Assert-OwnedPath (Join-Path $project 'artifacts\fixture-verification.json')
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$verification = Get-Content -LiteralPath $verificationPath -Raw | ConvertFrom-Json
$summary = Get-Content -LiteralPath (Join-Path $project 'state\summary.json') -Raw | ConvertFrom-Json
if ($manifest.version -ne $ExpectedVersion -or $verification.fixture_version -ne $ExpectedVersion) { throw 'Unexpected fixture version' }
if ($verification.expected_combinations -ne 174 -or @($verification.records.psobject.Properties).Count -ne 174) { throw 'Complete the current acceptance check before reconciliation' }
if ($null -ne $summary.run.started -or $null -ne $summary.run.deadline -or @($summary.jobs.psobject.Properties).Count -ne 0) { throw 'Reconciliation only authorized before runtime measurement' }
$archive = Assert-OwnedPath (Join-Path $project ('artifacts\fixture-versions\' + $ExpectedVersion))
if (Test-Path -LiteralPath $archive) { throw 'Archive already exists; do not overwrite historical evidence' }
New-Item -ItemType Directory -Path $archive | Out-Null
foreach ($relative in @('START_HERE.md','EXECUTION_PLAN.md','FIXTURES.md','benchmark-spec.json','localbench\fixtures.py','localbench\fixture_python.py','localbench\fixture_typescript.py','localbench\grading.py','scripts\grade_python.py','scripts\grade_typescript.sh','scripts\node_reporter.cjs','fixtures','grader')) {
    $source = Assert-OwnedPath (Join-Path $project $relative)
    $destination = Assert-OwnedPath (Join-Path $archive $relative)
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $source -Destination $destination -Recurse
}
# These are two explicitly checked files, not recursive directory moves.
Move-Item -LiteralPath $manifestPath -Destination (Assert-OwnedPath (Join-Path $archive 'fixture-manifest.json'))
Move-Item -LiteralPath $verificationPath -Destination (Assert-OwnedPath (Join-Path $archive 'fixture-verification.json'))
[pscustomobject]@{archivedVersion=$ExpectedVersion; archive=$archive; priorAccepted=$verification.accepted_combinations; priorExpected=$verification.expected_combinations; executionClock=$summary.run.started; retainedRawLogs=(Join-Path $project '.logs')} | ConvertTo-Json

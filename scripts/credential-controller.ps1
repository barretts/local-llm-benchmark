param(
    [Parameter(Mandatory = $true)][ValidatePattern('^localbench-[a-f0-9]{12}$')][string]$RunId,
    [string]$CredentialFile,
    [ValidateSet('resume', 'sweep')][string]$WorkerCommand = 'resume',
    [ValidateRange(1, 604800)][int]$MaximumWaitSeconds = 604800
)

# Launch this script with PowerShell 7 -STA. The user explicitly authorized
# the visible password dialog. No credential is read from or written to disk.
$ErrorActionPreference = 'Stop'
$benchmarkProject = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$benchmarkSpec = Get-Content -LiteralPath (Join-Path $benchmarkProject 'benchmark-spec.json') -Raw | ConvertFrom-Json
$benchmarkArtifacts = [IO.Path]::GetFullPath($benchmarkSpec.paths.artifacts)
$benchmarkLogs = [IO.Path]::GetFullPath($benchmarkSpec.paths.logs)
foreach ($benchmarkDirectory in @($benchmarkArtifacts, $benchmarkLogs)) {
    if (-not $benchmarkDirectory.StartsWith($benchmarkProject + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Credential controller artifacts must remain within the benchmark workspace.'
    }
    [void][IO.Directory]::CreateDirectory($benchmarkDirectory)
}
$benchmarkSummaryPath = Join-Path $benchmarkSpec.paths.state 'summary.json'
$benchmarkReadyPath = Join-Path $benchmarkArtifacts 'credential-controller-ready.json'
$benchmarkStatusPath = Join-Path $benchmarkArtifacts 'credential-controller-status.json'
$benchmarkPython = Join-Path $benchmarkProject '.venv\Scripts\python.exe'
$benchmarkParentCreated = (Get-Process -Id $PID).StartTime.ToUniversalTime().ToString('o')
$benchmarkStdout = $null
$benchmarkStderr = $null
$benchmarkAuthReadiness = 'unknown'
$benchmarkAuthStatusCode = $null

function Write-BenchmarkCredentialStatus {
    param([string]$Status, [Nullable[int]]$ChildPid = $null, [string]$ChildCreated = $null, [string]$Reason = $null)
    # Keep this literal allowlist: do not serialize forms, exceptions or env.
    $benchmarkRecord = [ordered]@{
        run_id = $RunId
        status = $Status
        timestamp_utc = [DateTime]::UtcNow.ToString('o')
        parent_pid = $PID
        parent_created_utc = $benchmarkParentCreated
        child_pid = $ChildPid
        child_created_utc = $ChildCreated
        stdout_log = $benchmarkStdout
        stderr_log = $benchmarkStderr
        auth_readiness = $benchmarkAuthReadiness
        auth_http_status = $benchmarkAuthStatusCode
        auth_unavailable = ($Status -eq 'cancelled' -or $Status -eq 'failed')
        reason = $Reason
        worker_command = $WorkerCommand
    }
    $benchmarkTemporaryStatus = $benchmarkStatusPath + '.' + $PID + '.tmp'
    [IO.File]::WriteAllText($benchmarkTemporaryStatus, ($benchmarkRecord | ConvertTo-Json -Depth 3), [Text.UTF8Encoding]::new($false))
    [IO.File]::Move($benchmarkTemporaryStatus, $benchmarkStatusPath, $true)
}

function Test-BenchmarkRunIdentity {
    if (-not (Test-Path -LiteralPath $benchmarkSummaryPath -PathType Leaf)) { return $false }
    try {
        $benchmarkSummary = Get-Content -LiteralPath $benchmarkSummaryPath -Raw | ConvertFrom-Json
        return ($benchmarkSummary.run.id -ceq $RunId)
    }
    catch { return $false }
}

function Test-BenchmarkLMAuth {
    $benchmarkAuthHandler = $null
    $benchmarkAuthClient = $null
    $benchmarkAuthRequest = $null
    $benchmarkAuthResponse = $null
    try {
        $benchmarkAuthHandler = [Net.Http.HttpClientHandler]::new()
        $benchmarkAuthHandler.UseProxy = $false
        $benchmarkAuthClient = [Net.Http.HttpClient]::new($benchmarkAuthHandler)
        $benchmarkAuthClient.Timeout = [TimeSpan]::FromSeconds(10)
        $benchmarkAuthRequest = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Get, 'http://127.0.0.1:1234/api/v1/models')
        $benchmarkAuthRequest.Headers.Authorization = [Net.Http.Headers.AuthenticationHeaderValue]::new('Bearer', $env:LM_BENCH_TOKEN)
        $benchmarkAuthResponse = $benchmarkAuthClient.Send($benchmarkAuthRequest, [Net.Http.HttpCompletionOption]::ResponseHeadersRead)
        $benchmarkAuthCode = [int]$benchmarkAuthResponse.StatusCode
        $benchmarkAuthState = if ($benchmarkAuthCode -eq 200) { 'accepted' } elseif ($benchmarkAuthCode -in @(401, 403)) { 'rejected' } else { 'unavailable' }
        return @{ readiness = $benchmarkAuthState; http_status = $benchmarkAuthCode }
    }
    catch { return @{ readiness = 'unavailable'; http_status = $null } }
    finally {
        # Never read, serialize or log the response body or request headers.
        if ($null -ne $benchmarkAuthResponse) { $benchmarkAuthResponse.Dispose() }
        if ($null -ne $benchmarkAuthRequest) { $benchmarkAuthRequest.Dispose() }
        if ($null -ne $benchmarkAuthClient) { $benchmarkAuthClient.Dispose() }
        elseif ($null -ne $benchmarkAuthHandler) { $benchmarkAuthHandler.Dispose() }
    }
}

$benchmarkDialog = $null
$benchmarkPassword = $null
try {
    if ($PSVersionTable.PSVersion.Major -lt 7 -or ([string]::IsNullOrEmpty($CredentialFile) -and [Threading.Thread]::CurrentThread.GetApartmentState() -ne 'STA')) {
        Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'requires_powershell7_sta'
        exit 2
    }
    if (-not (Test-BenchmarkRunIdentity) -or -not (Test-Path -LiteralPath $benchmarkPython -PathType Leaf)) {
        Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'original_run_or_isolated_python_unavailable'
        exit 2
    }
    # An inherited credential is never used as a default or shown in the UI.
    Remove-Item -LiteralPath Env:\LM_BENCH_TOKEN -ErrorAction SilentlyContinue
    if (-not [string]::IsNullOrEmpty($CredentialFile)) {
        # The user explicitly authorized this one existing file. Read it only
        # here; never echo the contents, modify the file or include key in argv.
        if (-not [IO.Path]::GetFullPath($CredentialFile).Equals('C:\Users\barrett\lmstudio-local.txt', [StringComparison]::OrdinalIgnoreCase)) {
            Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'credential_file_path_not_authorized'
            exit 2
        }
        $env:LM_BENCH_TOKEN = [IO.File]::ReadAllText($CredentialFile).Trim()
        if ([string]::IsNullOrWhiteSpace($env:LM_BENCH_TOKEN)) {
            Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'authorized_credential_file_empty'
            exit 2
        }
    }
    else {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing
    [Windows.Forms.Application]::EnableVisualStyles()
    $benchmarkDialog = [Windows.Forms.Form]::new()
    $benchmarkDialog.Text = 'Local benchmark: LM Studio API key'
    $benchmarkDialog.ClientSize = [Drawing.Size]::new(530, 186)
    $benchmarkDialog.StartPosition = 'CenterScreen'
    $benchmarkDialog.FormBorderStyle = 'FixedDialog'
    $benchmarkDialog.MaximizeBox = $false
    $benchmarkDialog.MinimizeBox = $false
    $benchmarkDialog.TopMost = $true

    $benchmarkLabel = [Windows.Forms.Label]::new()
    $benchmarkLabel.Text = 'Enter the new LM Studio API key. It stays in process memory.'
    $benchmarkLabel.Location = [Drawing.Point]::new(18, 18)
    $benchmarkLabel.Size = [Drawing.Size]::new(493, 38)
    $benchmarkDialog.Controls.Add($benchmarkLabel)

    $benchmarkPassword = [Windows.Forms.TextBox]::new()
    $benchmarkPassword.Location = [Drawing.Point]::new(18, 64)
    $benchmarkPassword.Size = [Drawing.Size]::new(493, 28)
    $benchmarkPassword.UseSystemPasswordChar = $true
    $benchmarkPassword.MaxLength = 8192
    $benchmarkDialog.Controls.Add($benchmarkPassword)

    $benchmarkContinue = [Windows.Forms.Button]::new()
    $benchmarkContinue.Text = 'Continue'
    $benchmarkContinue.Location = [Drawing.Point]::new(311, 122)
    $benchmarkContinue.Size = [Drawing.Size]::new(96, 32)
    $benchmarkContinue.Add_Click({
        if ([string]::IsNullOrWhiteSpace($benchmarkPassword.Text)) {
            [void][Windows.Forms.MessageBox]::Show($benchmarkDialog, 'Enter a key or choose Cancel.', 'LM Studio API key')
            return
        }
        $benchmarkDialog.DialogResult = [Windows.Forms.DialogResult]::OK
        $benchmarkDialog.Close()
    })
    $benchmarkDialog.Controls.Add($benchmarkContinue)
    $benchmarkDialog.AcceptButton = $benchmarkContinue

    $benchmarkCancel = [Windows.Forms.Button]::new()
    $benchmarkCancel.Text = 'Cancel'
    $benchmarkCancel.Location = [Drawing.Point]::new(415, 122)
    $benchmarkCancel.Size = [Drawing.Size]::new(96, 32)
    $benchmarkCancel.DialogResult = [Windows.Forms.DialogResult]::Cancel
    $benchmarkDialog.Controls.Add($benchmarkCancel)
    $benchmarkDialog.CancelButton = $benchmarkCancel
    $benchmarkDialog.Add_Shown({ $benchmarkPassword.Focus() })
    Write-BenchmarkCredentialStatus -Status 'prompt_open'
    if ($benchmarkDialog.ShowDialog() -ne [Windows.Forms.DialogResult]::OK) {
        Write-BenchmarkCredentialStatus -Status 'cancelled' -Reason 'user_cancelled_password_prompt'
        exit 0
    }
    $env:LM_BENCH_TOKEN = $benchmarkPassword.Text
    $benchmarkPassword.Clear()
    $benchmarkDialog.Dispose()
    $benchmarkDialog = $null
    $benchmarkPassword = $null
    }
    if ($env:LM_BENCH_TOKEN -match '[^\x21-\x7e]') {
        Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'credential_header_format_invalid'
        exit 2
    }
    $benchmarkAuthResult = Test-BenchmarkLMAuth
    $benchmarkAuthReadiness = $benchmarkAuthResult.readiness
    $benchmarkAuthStatusCode = $benchmarkAuthResult.http_status
    Write-BenchmarkCredentialStatus -Status 'waiting_for_verified_controller'

    $benchmarkWaitStarted = [DateTime]::UtcNow
    while ($true) {
        if (([DateTime]::UtcNow - $benchmarkWaitStarted).TotalSeconds -gt $MaximumWaitSeconds) {
            Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'controller_readiness_timeout'
            exit 2
        }
        if (-not (Test-BenchmarkRunIdentity)) {
            Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'original_run_identity_changed_or_unavailable'
            exit 2
        }
        $benchmarkReady = $null
        if (Test-Path -LiteralPath $benchmarkReadyPath -PathType Leaf) {
            try { $benchmarkReady = Get-Content -LiteralPath $benchmarkReadyPath -Raw | ConvertFrom-Json }
            catch { $benchmarkReady = $null }
        }
        if ($null -ne $benchmarkReady -and $benchmarkReady.run_id -ceq $RunId -and
            $benchmarkReady.ready -is [bool] -and $benchmarkReady.ready -eq $true) { break }
        Start-Sleep -Milliseconds 500
    }
    $benchmarkStamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ')
    $benchmarkStdout = Join-Path $benchmarkLogs ('credential-controller-' + $benchmarkStamp + '-stdout.log')
    $benchmarkStderr = Join-Path $benchmarkLogs ('credential-controller-' + $benchmarkStamp + '-stderr.log')
    # Start-Process inherits the current process environment; the key never
    # appears in arguments, status JSON, console output or temporary files.
    $benchmarkChild = Start-Process -FilePath $benchmarkPython -ArgumentList @('-X', 'utf8', '-m', 'localbench', $WorkerCommand) -WorkingDirectory $benchmarkProject -WindowStyle Hidden -RedirectStandardOutput $benchmarkStdout -RedirectStandardError $benchmarkStderr -PassThru -ErrorAction Stop
    Remove-Item -LiteralPath Env:\LM_BENCH_TOKEN -ErrorAction SilentlyContinue
    $benchmarkChildCreated = $null
    try { $benchmarkChildCreated = $benchmarkChild.StartTime.ToUniversalTime().ToString('o') } catch {}
    Write-BenchmarkCredentialStatus -Status 'launched' -ChildPid $benchmarkChild.Id -ChildCreated $benchmarkChildCreated
}
catch {
    # Exception strings can contain unexpected content. Emit a fixed code only.
    Write-BenchmarkCredentialStatus -Status 'failed' -Reason 'credential_controller_failed'
    exit 2
}
finally {
    Remove-Item -LiteralPath Env:\LM_BENCH_TOKEN -ErrorAction SilentlyContinue
    if ($null -ne $benchmarkPassword) { $benchmarkPassword.Clear() }
    if ($null -ne $benchmarkDialog) { $benchmarkDialog.Dispose() }
}

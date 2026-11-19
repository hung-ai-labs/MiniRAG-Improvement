# Watchdog for the Phi C0 run. EDIT THE PARAMS BLOCK for your machine / Modal workspace.
param(
    [string]$Root = (Resolve-Path "$PSScriptRoot\..\..\..").Path,          # MiniRAG-Improvement checkout
    [string]$Py = 'python',                                              # Python 3.13 env (see pip_freeze_minirag_venv.txt)
    [string]$Profile = '<your-modal-profile>',                           # modal profile of YOUR workspace
    [string]$PhiUrl = 'https://<your-workspace>--minirag-slm-phi-ropefix-serve.us-east.modal.run',
    [string]$DotEnv = '<path to your .env with MINIRAG_SLM_KEY and GEMINI_API_KEY_*>',
    [double]$CostStop = 29.0                                             # stop at ~1 USD below a 30 USD cap
)
$Out = Join-Path $Root 'review\mhr_phi_20261118'
$env:MINIRAG_DOTENV = $DotEnv
$env:PHI_SLM_URL = $PhiUrl
$Log = Join-Path $Out 'watchdog_c0.log'
$lastCost = 0
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$restarts = 0; $tick = 0

function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $Log }
function Procs { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'reproduce\.mhr_phi\.c0' }) }
function State { try { (Get-Content (Join-Path $Out 'c0\status.json') -Raw | ConvertFrom-Json).state } catch { '' } }
function Cost {
    $env:MODAL_PROFILE = $Profile
    $t = & $Py -m modal billing summary 2>$null | Out-String
    Remove-Item Env:MODAL_PROFILE
    if ($t -match 'Metered Cost:\D*([0-9]+\.[0-9]+)') { [double]$Matches[1] } else { -1 }
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Win32.Power]::SetThreadExecutionState([uint32]'0x80000003')
Say 'watchdog started (keep-awake held)'
while ($true) {
    if ((State) -eq 'completed' -and (Procs).Count -eq 0) { Say 'C0 completed, exiting'; break }
    if ($tick % 3 -eq 0 -or $lastCost -ge 27) {
        $c = Cost; $lastCost = $c
        Say "modal metered cost this month: $c"
        if ($c -ge $CostStop) {
            Procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
            "cost $c >= $CostStop at $(Get-Date -Format s) during C0" | Out-File -Encoding utf8 (Join-Path $Out 'PAUSED_LOW_CREDIT')
            Say "PAUSED: cost $c >= $CostStop; C0 stopped (resumable)"; break
        }
    }
    if ((Procs).Count -eq 0 -and (State) -ne 'completed') {
        if ($restarts -ge 8) { Say 'restart budget used up' }
        else {
            Say "C0 not running (state=$(State)), starting execute"
            Start-Process -FilePath $Py -ArgumentList '-B', '-u', '-m', 'reproduce.mhr_phi.launch', 'reproduce.mhr_phi.c0', 'execute' -WorkingDirectory $Root `
                -RedirectStandardOutput (Join-Path $Out "c0_execute_$restarts.out.log") -RedirectStandardError (Join-Path $Out "c0_execute_$restarts.err.log") -WindowStyle Hidden
            $restarts++
        }
    }
    $tick++
    Start-Sleep 300
}

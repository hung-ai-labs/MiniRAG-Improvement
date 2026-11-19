# Watchdog for the MHR +set test run (no LLM). Every 5 min:
#  - finish dead, or judging with every judge journal silent > 20 min -> restart finish
#  - execute dead before status.json says completed -> restart execute
# Both resumable (fresh attempt dirs). Max 6 restarts each. Log: review/mhr_set_20261117/watchdog.log
$Root = 'E:\Project\MiniRAG research\MiniRAG-Improvement'
$Out = Join-Path $Root 'review\mhr_set_20261117'
$Py = 'E:\Project\MiniRAG research\MiniRAG\.venv\Scripts\python.exe'
$Log = Join-Path $Out 'watchdog.log'
$env:MINIRAG_DOTENV = 'E:/Project/MiniRAG research/MiniRAG/.env'
$env:PYTHONUTF8 = '1'
# 2026-11-17 21:xx: workspace nguyenhuuanhtai36 disabled by Modal; same app (reproduce/modal_slm.py, unchanged)
# redeployed on workspace nguyenhuuanhtai426. Env wins over .env (load_dotenv override=False).
$env:MINIRAG_SLM_URL = 'https://nguyenhuuanhtai426--minirag-slm-serve.us-east.modal.run/v1'
$restarts = @{ finish = 0; execute = 0 }

function Say($m) { "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m" | Out-File -Append -Encoding utf8 $Log }
function Procs($pattern) { @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match $pattern }) }
function Done($name) { try { (Get-Content (Join-Path $Out $name) -Raw | ConvertFrom-Json).state -eq 'completed' } catch { $false } }
function LastEvent($stem) {
    $f = Get-ChildItem (Join-Path $Out "${stem}_events") -Recurse -Filter event.json -ErrorAction SilentlyContinue | Sort-Object LastWriteTime | Select-Object -Last 1
    if ($f) { try { [pscustomobject]@{ v = (Get-Content $f.FullName -Raw | ConvertFrom-Json); t = $f.LastWriteTime } } catch { $null } }
}
function NewestJournal {
    Get-ChildItem (Join-Path $Out 'test'), (Join-Path $Out 'secondary_checks') -Recurse -Filter judge.jsonl -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
[void][Win32.Power]::SetThreadExecutionState([uint32]'0x80000003')
Say 'watchdog started (keep-awake request held)'
while ($true) {
    $jd = Done 'status_judge.json'; $sd = Done 'status.json'
    if ($jd -and $sd) { Say 'all done, exiting'; break }

    if (-not $jd) {
        $fin = Procs 'mhr_set\.online finish'
        $stale = $false
        $e = LastEvent 'status_judge'
        if ($e -and $e.v.step -in @('judge_SET_delta', 'judge_CE_drift') -and ((Get-Date) - $e.t).TotalMinutes -gt 20) {
            $j = NewestJournal
            $stale = (-not $j) -or (((Get-Date) - $j.LastWriteTime).TotalMinutes -gt 20)
        }
        if ($fin.Count -eq 0 -or $stale) {
            if ($restarts.finish -ge 6) { Say 'finish: restart budget used up, leaving it' }
            else {
                Say "finish: restarting (running=$($fin.Count) stale=$stale step=$($e.v.step) seed=$($e.v.seed))"
                Procs 'mhr_set\.online finish|mhr_set\.judge_entry|mhr_set\.run finish' | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                Start-Sleep 3
                Start-Process -FilePath $Py -ArgumentList '-u', '-m', 'reproduce.mhr_set.online', 'finish', '--rounds', '20' -WorkingDirectory $Root `
                    -RedirectStandardOutput (Join-Path $Out "finish_wd$($restarts.finish).out.log") -RedirectStandardError (Join-Path $Out "finish_wd$($restarts.finish).err.log") -WindowStyle Hidden
                $restarts.finish++
            }
        }
    }

    if (-not $sd) {
        $ex = Procs 'mhr_set\.(online execute|run answers)'
        if ($ex.Count -eq 0) {
            if ($restarts.execute -ge 6) { Say 'execute: restart budget used up, leaving it' }
            else {
                $e = LastEvent 'status'
                Say "execute: not running (last event=$($e.v.state)/$($e.v.step)/$($e.v.seed)), restarting"
                Start-Process -FilePath $Py -ArgumentList '-u', '-m', 'reproduce.mhr_set.online', 'execute' -WorkingDirectory $Root `
                    -RedirectStandardOutput (Join-Path $Out "execute_wd$($restarts.execute).out.log") -RedirectStandardError (Join-Path $Out "execute_wd$($restarts.execute).err.log") -WindowStyle Hidden
                $restarts.execute++
            }
        }
    }
    Start-Sleep 300
}

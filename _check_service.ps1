$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'main\.py' }
foreach ($p in $procs) {
    Write-Output ("running pid={0} cmd={1}" -f $p.ProcessId, $p.CommandLine)
}
if (-not $procs) { Write-Output "no main.py process running" }

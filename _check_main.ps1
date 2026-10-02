Start-Sleep -Seconds 5
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'main\.py' }
if (-not $procs) {
    Write-Output "no main.py process found"
} else {
    foreach ($p in $procs) {
        Write-Output ("pid={0} ppid={1} cmd={2}" -f $p.ProcessId, $p.ParentProcessId, $p.CommandLine)
    }
}

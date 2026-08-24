# 查找并终止运行 main.py 的 python 进程（用于重启服务）
$procs = Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'main\.py' }
foreach ($p in $procs) {
    Write-Output ("killing pid={0} cmd={1}" -f $p.ProcessId, $p.CommandLine)
    Stop-Process -Id $p.ProcessId -Force
}
if (-not $procs) { Write-Output "no main.py process found" }

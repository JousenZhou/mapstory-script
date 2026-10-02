# 列出所有运行 main / main_debug / web_main 的 python 进程（含父进程，用于区分派生子进程与残留进程）
$query = 'SELECT ProcessId,ParentProcessId,Name,CommandLine FROM Win32_Process WHERE Name=''python.exe'' OR Name=''pythonw.exe'''
$procs = Get-CimInstance -Query $query | Where-Object { $_.CommandLine -match 'main' }
if (-not $procs) {
    Write-Output "no main process found"
} else {
    foreach ($p in $procs) {
        Write-Output ("pid={0} ppid={1} name={2} cmd={3}" -f $p.ProcessId, $p.ParentProcessId, $p.Name, $p.CommandLine)
    }
}

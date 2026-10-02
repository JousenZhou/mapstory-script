$procs = Get-Process | Where-Object { $_.MainWindowTitle -ne '' -and ($_.ProcessName -eq 'python' -or $_.ProcessName -like '*main*') }
if (-not $procs) {
    Write-Output "no GUI window found for python processes yet"
} else {
    foreach ($p in $procs) {
        Write-Output ("pid={0} name={1} title={2}" -f $p.Id, $p.ProcessName, $p.MainWindowTitle)
    }
}

# 以当前（已提权）会话启动生产模式 main.py，使用仓库本地虚拟环境解释器
$venvPy = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$workdir = $PSScriptRoot
$p = Start-Process -FilePath $venvPy -ArgumentList 'main.py' -WorkingDirectory $workdir -PassThru
Write-Output ("started pid={0} workdir={1}" -f $p.ProcessId, $workdir)

# 以管理员身份启动 main.py（生产模式，虚拟环境解释器）
$venvPy = 'd:\workspace\mapstory-script\.venv\Scripts\python.exe'
Start-Process -FilePath $venvPy `
    -ArgumentList 'main.py' `
    -WorkingDirectory 'd:\workspace\mapstory-script' `
    -Verb RunAs
Write-Output "elevation request sent; please confirm UAC prompt"

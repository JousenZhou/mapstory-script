# Build Windows portable edition (PyInstaller onedir) using the local .venv.
# Usage:
#   .\build_exe.ps1                   # build into dist\goodgoodstudydaydayup
#   .\build_exe.ps1 -Version v1.2.3   # inject version string
#   .\build_exe.ps1 -Zip              # also produce goodgoodstudydaydayup-win-x64.zip
param(
    [string]$Version = "",
    [switch]$Zip
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# Use the repo-local venv Python (use-local-venv convention).
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "venv not found: .venv\Scripts\python.exe. Create the venv and install requirements first."
}

# Auto-install PyInstaller when missing.
& $python -c "import PyInstaller"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing PyInstaller ..."
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $python -m pip install --index-url https://pypi.org/simple/ pyinstaller 2>&1 | ForEach-Object { Write-Host $_ }
    $ErrorActionPreference = $prevEAP
    if ($LASTEXITCODE -ne 0) { throw "Failed to install PyInstaller." }
}

# Version: argument first, otherwise the latest git tag.
if (-not $Version) {
    $Version = (git describe --tags --abbrev=0 2>$null)
    if (-not $Version) { $Version = "" }
}
if ($Version) {
    Write-Host "Building version: $Version"
}

# Clean previous output; stop a running instance first (may require elevation,
# e.g. when the packaged app or its adb server runs as administrator).
$oldDist = Join-Path $root "dist\goodgoodstudydaydayup"
if (Test-Path $oldDist) {
    Stop-Process -Name "goodgoodstudydaydayup" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
    Remove-Item $oldDist -Recurse -Force -ErrorAction SilentlyContinue
    if (Test-Path $oldDist) {
        throw "Cannot clean $oldDist. Close the running goodgoodstudydaydayup.exe (and adb.exe) first, then retry."
    }
}

# Run PyInstaller. Its INFO logs go to stderr, so relax error handling and
# rely on the exit code instead of $ErrorActionPreference.
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $python -m PyInstaller build_exe.spec --clean --noconfirm 2>&1 | ForEach-Object { Write-Host $_ }
$ErrorActionPreference = $prevEAP
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$distDir = Join-Path $root "dist\goodgoodstudydaydayup"

# Copy project resource dirs next to the exe; the app reads them by paths
# relative to the exe working directory.
foreach ($res in @("i18n", "ok_templates", "icons", "assets")) {
    $src = Join-Path $root $res
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $distDir -Recurse -Force
    }
}

# Pre-create writable runtime dir; user configs are generated on first launch.
New-Item -ItemType Directory -Force -Path (Join-Path $distDir "configs") | Out-Null

# Write version file read by main.py at startup.
if ($Version) {
    Set-Content -Path (Join-Path $distDir "VERSION.txt") -Value $Version -Encoding ascii -NoNewline
}

if ($Zip) {
    $zipPath = Join-Path $root "dist\goodgoodstudydaydayup-win-x64.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Write-Host "Compressing to $zipPath ..."
    Compress-Archive -Path (Join-Path $distDir "*") -DestinationPath $zipPath -CompressionLevel Optimal
}

Write-Host ""
Write-Host "Build finished: $distDir"
Write-Host "Run as administrator: $distDir\goodgoodstudydaydayup.exe"

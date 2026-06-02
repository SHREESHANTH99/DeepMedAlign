param(
    [switch]$SkipInstall,
    [switch]$SkipTests,
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot '.venv'
$pythonExe = Join-Path $venvPath 'Scripts\python.exe'

Write-Host '==> Ensuring virtual environment'
if (-not (Test-Path $pythonExe)) {
    py -3 -m venv $venvPath
}

Write-Host '==> Upgrading packaging tools'
& $pythonExe -m pip install --upgrade pip setuptools wheel

if (-not $SkipInstall) {
    Write-Host '==> Installing Windows requirements'
    & $pythonExe -m pip install -r (Join-Path $repoRoot 'requirements-windows.txt')
}

if (-not $SkipVerify) {
    Write-Host '==> Verifying imports'
    & $pythonExe -c "import torch, SimpleITK, nibabel, voxelmorph, antspyx, wandb, hd_bet; print('  torch:', torch.__version__); print('  SimpleITK:', SimpleITK.Version()); print('  nibabel:', nibabel.__version__); print('  voxelmorph: ok'); print('  antspyx: ok'); print('  wandb:', wandb.__version__); print('  hd_bet: ok')"
}

if (-not $SkipTests) {
    Write-Host '==> Running tests'
    & $pythonExe -m pytest (Join-Path $repoRoot 'tests') -v --tb=short
}

Write-Host ''
Write-Host '✓ Environment ready. Activate with:'
Write-Host "  & $venvPath\Scripts\Activate.ps1"

param(
    [int]$Port = 8767,
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop with Linux containers first.' }
docker image inspect fashion-atlas:latest *> $null
if ($LASTEXITCODE -ne 0) {
    docker build -t fashion-atlas:latest .
    if ($LASTEXITCODE -ne 0) { throw 'Docker build failed.' }
}

$gpuOptions = @()
if (-not $Cpu) {
    docker run --rm --gpus all --entrypoint python fashion-atlas:latest -c 'import torch; assert torch.cuda.is_available(); torch.ones(1).cuda().add_(1).cpu()' *> $null
    if ($LASTEXITCODE -eq 0) { $gpuOptions = @('--gpus', 'all') }
}
if ($gpuOptions.Count) { Write-Host 'Using GPU.' } else { Write-Host 'Using CPU.' }
Write-Host "Open http://127.0.0.1:$Port - Ctrl+C stops the container."
docker run --rm --init @gpuOptions -p "127.0.0.1:${Port}:8767" fashion-atlas:latest
if ($LASTEXITCODE -notin @(0, 130, 143)) { throw 'Container failed. Check the output above and whether the port is occupied.' }

# Builds the task's own image and runs verify_first.py inside it. That is the
# only place Triton exists on a Windows host, and it also proves the shipped
# Dockerfile is genuinely able to launch a kernel. Run from the repo root:
#
#     .\tools\verify_in_docker.ps1
#
# Requires Docker Desktop with the WSL2 backend and NVIDIA GPU support.

$ErrorActionPreference = "Stop"

$tag  = "gatherlib-task5-verify:latest"
$repo = (Get-Location).Path

if (-not (Test-Path (Join-Path $repo "tools\verify_first.py"))) {
    throw "Run this from the repository root (tools\verify_first.py not found here)."
}

Write-Host "[1/3] building the task image (this also runs its build-time assertions)..." -ForegroundColor Cyan
docker build -t $tag "$repo\task5\environment"
if ($LASTEXITCODE -ne 0) {
    throw "The task image failed to build. Fix that before anything else -- Harbor builds the same Dockerfile."
}

Write-Host "[2/3] checking the container can see the GPU and reach Triton's driver shim..." -ForegroundColor Cyan
docker run --rm --gpus all $tag python -c @"
import torch, triton
assert torch.cuda.is_available(), 'no CUDA inside the container'
free, total = torch.cuda.mem_get_info()
print('device :', torch.cuda.get_device_name(0))
print('memory : %.2f GiB free of %.2f GiB' % (free / 2**30, total / 2**30))
print('triton :', triton.__version__)
from triton.runtime import driver
driver.active.get_current_device()
print('driver : shim built ok')
if free / 2**30 < 2.6:
    print('WARNING: under 2.6 GiB free; the full size fixture needs ~2.5 GiB')
"@
if ($LASTEXITCODE -ne 0) {
    throw "The container cannot initialise Triton's CUDA driver. Check Docker Desktop's WSL2 backend and NVIDIA support."
}

Write-Host "[3/3] running verify_first.py inside the image..." -ForegroundColor Cyan
docker run --rm --gpus all --shm-size=1g `
    -v "${repo}:/repo" -w /repo `
    -e TRITON_DEBUG=0 `
    $tag python tools/verify_first.py

exit $LASTEXITCODE

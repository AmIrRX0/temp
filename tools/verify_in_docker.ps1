# Runs verify_first.py inside the task's own base image, which is the only
# place Triton actually exists on a Windows host. Run from the repo root:
#
#     .\tools\verify_in_docker.ps1
#
# Requires Docker Desktop with the WSL2 backend and NVIDIA GPU support.

$ErrorActionPreference = "Stop"

$image = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
$repo  = (Get-Location).Path

if (-not (Test-Path (Join-Path $repo "tools\verify_first.py"))) {
    throw "Run this from the repository root (tools\verify_first.py not found here)."
}

Write-Host "checking the GPU is visible to Docker..." -ForegroundColor Cyan
docker run --rm --gpus all $image python -c "import torch; assert torch.cuda.is_available(), 'no CUDA inside the container'; print('cuda ok:', torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    throw "The container cannot see a GPU. Check Docker Desktop's WSL2 backend and NVIDIA support."
}

Write-Host "checking the image ships Triton..." -ForegroundColor Cyan
docker run --rm $image python -c "import triton; print('triton', triton.__version__)"
if ($LASTEXITCODE -ne 0) {
    throw "This image has no Triton. The task's Dockerfile asserts on it at build time too."
}

Write-Host "running verify_first.py inside the image..." -ForegroundColor Cyan
docker run --rm --gpus all --shm-size=1g `
    -v "${repo}:/repo" -w /repo `
    -e TRITON_CACHE_DIR=/tmp/triton-cache `
    -e TRITON_DEBUG=0 `
    $image python tools/verify_first.py

exit $LASTEXITCODE

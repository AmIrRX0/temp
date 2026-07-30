# A neutral GPU Python environment for running a solver experiment.
#
# The task image is unsuitable for this: it bakes gatherlib into
# /workspace/gatherlib and its tag names the task, both of which are hints. This
# carries only torch, Triton and the C compiler Triton needs at kernel launch.
#
#   docker build -t gpu-py:latest -f tools/solver_env.Dockerfile tools
#
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

ENV TRITON_CACHE_DIR=/tmp/triton-cache

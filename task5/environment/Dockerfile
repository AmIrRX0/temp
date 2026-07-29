FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Triton builds a small CPython extension for its CUDA driver shim the first
# time a kernel is launched, so it needs a C compiler and the Python headers at
# run time. The -runtime base image ships neither.
RUN apt-get update \
 && apt-get install -y --no-install-recommends gcc \
 && rm -rf /var/lib/apt/lists/*

# Only the library under test enters the image.
COPY gatherlib/ /workspace/gatherlib/

RUN pip install --no-cache-dir pytest==8.3.4

# Fail the build loudly rather than at verification time: assert the base image
# still ships a matching Triton, and that the compiler and headers Triton needs
# are genuinely usable together.
RUN python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)" \
 && printf '#include <Python.h>\nint main(void){return 0;}\n' > /tmp/cc_probe.c \
 && gcc /tmp/cc_probe.c -I"$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"include\"])')" -o /tmp/cc_probe \
 && rm -f /tmp/cc_probe.c /tmp/cc_probe

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    TRITON_CACHE_DIR=/tmp/triton-cache

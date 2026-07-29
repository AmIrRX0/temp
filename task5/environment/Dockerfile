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

# Fail the build loudly rather than at verification time. This resolves the
# include directory the same way Triton's own builder does, then actually
# compiles a translation unit against it, so a missing compiler or missing
# headers cannot reach the verifier.
RUN gcc --version | head -1 \
 && python -c "import os, sys, sysconfig, torch, triton; print('torch', torch.__version__, 'triton', triton.__version__); scheme = sysconfig.get_default_scheme(); scheme = 'posix_prefix' if scheme == 'posix_local' else scheme; inc = sysconfig.get_paths(scheme=scheme)['include']; open('/tmp/pyinc', 'w').write(inc); print('scheme', scheme, 'include', inc); sys.exit('MISSING ' + os.path.join(inc, 'Python.h')) if not os.path.exists(os.path.join(inc, 'Python.h')) else print('Python.h found')" \
 && printf '#include <Python.h>\nint main(void){return 0;}\n' > /tmp/cc_probe.c \
 && gcc /tmp/cc_probe.c -I"$(cat /tmp/pyinc)" -o /tmp/cc_probe \
 && rm -f /tmp/cc_probe.c /tmp/cc_probe /tmp/pyinc \
 && echo 'toolchain check passed'

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    TRITON_CACHE_DIR=/tmp/triton-cache

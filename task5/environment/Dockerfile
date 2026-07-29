FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

# Only the library under test enters the image.
COPY gatherlib/ /workspace/gatherlib/

# Fail the build loudly rather than at verification time if the base image ever
# stops shipping a Triton that matches this torch.
RUN pip install --no-cache-dir pytest==8.3.4 \
 && python -c "import torch, triton; print('torch', torch.__version__, 'triton', triton.__version__)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    TRITON_CACHE_DIR=/tmp/triton-cache

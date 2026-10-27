# syntax=docker/dockerfile:1

# Download the small Apache-2.0 DPDFNet model in a builder stage so the runtime
# image does not need curl or network access at first use.
ARG DPDFNET_MODEL_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet8.onnx
ARG DPDFNET_MODEL_SHA256=2751c1f5a4e849d23a07c675b4c838158b249b42152f10cc318522dd339134f0
FROM python:3.10-slim-bookworm AS dpdfnet-model

ARG DPDFNET_MODEL_URL
ARG DPDFNET_MODEL_SHA256

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /models \
    && python -c "import urllib.request; urllib.request.urlretrieve('${DPDFNET_MODEL_URL}', '/models/dpdfnet8.onnx')" \
    && echo "${DPDFNET_MODEL_SHA256}  /models/dpdfnet8.onnx" | sha256sum -c - \
    && python -c "import urllib.request; urllib.request.urlretrieve('https://raw.githubusercontent.com/ceva-ip/DPDFNet/main/LICENSE', '/models/DPDFNet-LICENSE')"

FROM python:3.10-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/beecave-homelab/insanely-fast-whisper-rocm"

ARG HSA_OVERRIDE_GFX_VERSION=

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV TZ=Europe/Amsterdam
ENV ROCM_PATH=/opt/rocm
ENV HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}
# The ROCm PyTorch wheels bundle the required userspace libraries. Mode 2 is
# the tested reliable MIOpen route for diarization on the target hardware.
ENV MIOPEN_FIND_MODE=2

# GPU inference and diarization defaults
ENV WHISPER_DEVICE=0
ENV WHISPER_DTYPE=float16
ENV DIARIZATION_DEVICE=cuda
ENV DIARIZATION_PRELOAD_AUDIO=true
ENV DIARIZATION_ALLOW_CPU_FALLBACK=true
ENV WHISPER_DIARIZATION_MODEL=pyannote/speaker-diarization-community-1

# Noise reduction defaults
ENV DEMUCS_DEFAULT=false
ENV DENOISE_DEFAULT=false
ENV DENOISE_BACKEND=dpdfnet
ENV DPDFNET_MODEL_PATH=/app/models/dpdfnet8.onnx

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV TORCHAUDIO_USE_SOUNDFILE=1

# Install OS-level deps needed for audio backends and ROCm PyTorch wheels.
RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the working directory in the container
WORKDIR /app

# Copy the fully resolved GPU runtime export first for Docker layer caching.
COPY requirements-gpu.txt /app/

# Install the resolved package set without dependency resolution. This preserves
# the full ROCm PyTorch wheel and excludes Demucs and unused ROCm ONNX Runtime.
RUN pip install --no-cache-dir --no-deps -r /app/requirements-gpu.txt

# Bundle the small DPDFNet model and its Apache-2.0 license.
RUN mkdir -p /app/models
COPY --from=dpdfnet-model /models/dpdfnet8.onnx /app/models/dpdfnet8.onnx
COPY --from=dpdfnet-model /models/DPDFNet-LICENSE /app/models/DPDFNet-LICENSE

# Copy project metadata, licenses, and application source.
COPY openapi.yaml /app/
COPY pyproject.toml /app/
COPY LICENSE.txt /app/
COPY THIRD_PARTY_NOTICES.md /app/
COPY ./insanely_fast_whisper_rocm /app/insanely_fast_whisper_rocm/

# Install the local package itself without re-resolving dependencies.
RUN pip install --no-cache-dir --no-deps .

# Generate the runtime package-license inventory for redistribution notices.
COPY scripts/audit_licenses.py /app/scripts/audit_licenses.py
RUN python /app/scripts/audit_licenses.py --write /app/THIRD_PARTY_LICENSES.md

# Expose default internal ports (API/WebUI). Actual bindings are controlled by Compose.
EXPOSE 8888
EXPOSE 7860

# Use the package entrypoint so host/port are controlled by env vars.
CMD ["insanely-fast-whisper-rocm"]

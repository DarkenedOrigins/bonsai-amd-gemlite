# Bonsai 1-bit + GemLite on AMD RDNA4 (RX 9070 XT / gfx1201)
#
# Slim approach: the pip PyTorch-ROCm wheels are SELF-CONTAINED -- they bundle the
# ROCm runtime libs they need (rocBLAS / hipBLASLt / MIOpen) and pull in
# pytorch-triton-rocm (which GemLite's Triton kernels require). The host only needs
# the amdgpu/KFD kernel driver, which Bazzite already provides (/dev/kfd).
# => No 40GB rocm/pytorch SDK image needed; this lands around ~10-12GB.
#
# Build:   podman build -t localhost/bonsai-rocm:latest -f Containerfile .
# Use via: distrobox create --name bonsai --image localhost/bonsai-rocm:latest
FROM docker.io/library/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# python3 on Ubuntu 24.04 is 3.12 (matches the cp312 torch wheel).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev \
        git curl wget ca-certificates \
        sudo less vim nano bash-completion \
        libgomp1 libatomic1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*
# build-essential + python3-dev: Triton JIT-compiles host-side helpers (hip_utils) and
# kernels at runtime and needs a C/C++ compiler + Python.h, or `import gemlite` fails with
# "Failed to find C compiler".

# Dedicated venv (Ubuntu's python is externally-managed; venv avoids that friction).
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip wheel

# All Python package specs live in the repo's requirements files (not hardcoded here):
#  - requirements.txt        : torch (stable ROCm 7.2 channel; the nightly deadlocks import
#                              -- see BUG_REPORT.md) + PyPI deps + typer
#  - requirements-nodeps.txt : gemlite + hqq, installed --no-deps so pip doesn't pull a
#                              generic `triton` that collides with torch's bundled triton-rocm
COPY requirements.txt requirements-nodeps.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-deps -r requirements-nodeps.txt

# NOTE: no build-time `import torch` check -- the build sandbox has no /dev/kfd, so torch's
# HIP init can block. ALL torch/GPU validation (import, get_arch_list, gfx1201, generation)
# is done at RUNTIME inside the distrobox, where /dev/kfd is present. See run_minimal.py.

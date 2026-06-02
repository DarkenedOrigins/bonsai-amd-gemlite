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

# torch (+ pytorch-triton-rocm) from the STABLE ROCm 7.2 release channel.
# NOTE: do NOT use the nightly/rocm7.2 wheel -- 2.13.0.dev hard-links libtorch_rocshmem,
# whose global constructor calls exit() on a single consumer GPU and deadlocks `import torch`
# (see BUG_REPORT.md). Stable 2.12.0 doesn't hard-link it and imports fine; arch list
# includes gfx1201 and it detects the RX 9070 XT.
RUN pip install --no-cache-dir torch==2.12.0 \
        --index-url https://download.pytorch.org/whl/rocm7.2

# Demo deps from PyPI. gemlite + hqq from the Dropbox org (project moved there).
# CRITICAL: gemlite's setup declares `triton>=3.6.0`, which makes pip pull the generic
# PyPI `triton` on top of torch's `triton-rocm` (both ship a top-level `triton/` module,
# so they collide and the generic one shadows the ROCm build -> gemlite breaks at runtime).
# So we install gemlite/hqq with --no-deps and provide their non-triton deps ourselves.
RUN pip install --no-cache-dir \
        "transformers>=4.46" accelerate huggingface_hub \
        numpy tqdm einops termcolor
RUN pip install --no-cache-dir --no-deps \
        "git+https://github.com/dropbox/hqq" \
        "git+https://github.com/dropbox/gemlite"

# NOTE: no build-time `import torch` check. This ROCm nightly's libtorch_hip touches
# /dev/kfd on import, which doesn't exist in the build sandbox -> the build hangs.
# ALL torch/GPU validation (import, torch.cuda.get_arch_list(), gfx1201, generation)
# is done at RUNTIME inside the distrobox, where /dev/kfd is present. See run_minimal.py.

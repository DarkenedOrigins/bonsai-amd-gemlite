# bonsai-amd-gemlite

Running the [GemLite 1-bit Bonsai inference gist](https://gist.github.com/mobicham/ef350332cf0669a3b140dc81882fdeec)
on an AMD **Radeon RX 9070 XT** (RDNA4 / `gfx1201`) under Bazzite.

The gist is written for NVIDIA (RTX 5090, ~660 tok/s). The motivation is the PyTorch
blog [*Accelerating LLM Inference with GemLite, TorchAO and SGLang*](https://pytorch.org/blog/accelerating-llm-inference/):
GemLite's low-bit GEMV kernel makes batch-1 token generation much faster. The question
this repo answers — **can that win be reached on consumer RDNA4?** — is **yes** (see Results).

## Results

1-bit Bonsai-1.7B, single RX 9070 XT, ~1024 new tokens, warm (post-compile):

| Config | tok/s | vs baseline |
|---|---:|---:|
| bf16 / plain `generate` | 22.7 | 1.0× |
| fp16 / plain `generate` | 22.1 | 1.0× |
| fp16 / `torch.compile` (max-autotune) | 33.7 | 1.5× |
| **fp16 / `torch.compile` + HIP graphs** (full gist path) | **166.2** | **7.3×** |

fp16 accumulation barely helps the plain path — batch-1 decode is overhead-bound, not
compute-bound. The big wins are `torch.compile` (fuses ops, cuts Python/launch overhead)
and especially HIP/CUDA graphs (eliminate launch overhead).

> **Note on the number:** 166.2 was a single early sample. A 6-run A/B later showed the
> true figure is **~180 tok/s ± 10%** (run-to-run spread ~20 tok/s). That's roughly 27% of
> the 5090's ~660 — reasonable for consumer RDNA4 vs flagship Blackwell. The NVIDIA 660
> used `max` autotune + fp16 acc; ours uses gemlite's default (`fast`) autotune (see below).

## Can we beat it? (RDNA4 tuning investigation)

Short answer: **no meaningful headroom from autotuning** — gemlite's default is already optimal here.

- **Where the GPU time goes** (CUDA-event breakdown, `amd_optimized.py --breakdown`): 1-bit
  GEMV **58.7%**, attention (sdpa) **1.9%**, everything else (norms/RoPE/dequant/elementwise)
  **39.4%**. So the GEMV kernel is the bottleneck — autotuning it is the right lever to test.
- **gemlite `GEMV_REVSPLITK` autotune config counts (AMD):** `default`=1, `fast`=16, `max`=540
  (`3 warps × 2 stages × 3 waves_per_eu × 6 N × 5 K`). The stock path already uses **`fast`**;
  NVIDIA's 660 used **`max`**. A full `max` run is ~9–10 h on this stack (AMD Triton compiles
  each new config slowly) and never completed — exhaustive tuning is effectively intractable here.
- **Targeted probe:** Bonsai has only **4 unique decode shapes**; `fast` picks
  `N=64,K=64,warps=1,stages=1,waves=2` for all. A principled ~36-config "small" probe
  (`--autotune small`, centered on that winner, extending into `max`-only territory incl. `K=128`)
  found per-shape-different winners — but **`K=128` never won**, and a 6-run A/B vs `fast`
  came out **statistically identical** (medians 184.3 vs 181.8 tok/s; ±10% noise ≫ the 2.5 tok/s gap).
- **Conclusion:** config tuning is a **dead lever** on RDNA4 for this workload. The gap to the
  ~2000 tok/s bandwidth roofline is **kernel codegen efficiency + the 39% non-GEMV overhead**,
  neither of which autotuning can fix. `HipKittens` doesn't help (CDNA-only, GEMM/attention-oriented).

**Profiling note:** `torch.profiler` doesn't populate GPU kernel times on this ROCm stack
(kineto/roctracer gap), and `rocprofv3` can't coexist with pip-torch's bundled ROCm
(ABI clash → `SIGABRT`). The working approach was **`torch.cuda.Event` module timing**
(`--breakdown`) — zero-install, runs inside torch's process.

## Stack

- **Model:** `prism-ml/Bonsai-1.7B-unpacked` (full bf16 ~3.3 GB; 1-bit HQQ quant happens on the fly)
- **Kernels:** [`gemlite`](https://github.com/dropbox/gemlite) (Triton low-bit GEMM/GEMV)
- **Quant + generation:** [`hqq`](https://github.com/dropbox/hqq) + transformers `HFGenerator`
- **Runtime:** **stable** `torch==2.12.0+rocm7.2` (cp312) + `pytorch-triton-rocm`

> Note: `gemlite` and `hqq` now live under the **dropbox** GitHub org (not `mobiusml`).
> On ROCm the AMD GPU is exposed as `cuda:0` (HIP masquerades as CUDA), so the gist's
> `device='cuda:0'` works unchanged.

## Layout

| File | Purpose |
|------|---------|
| `Containerfile` | Slim Ubuntu 24.04 + stable pip torch (rocm7.2) + gemlite/hqq + C toolchain |
| `run_minimal.py` | Stage-1 run: 1-bit GemLite generation, plain path (no compile / no graphs) |
| `bench.py` | Parameterized benchmark: `--acc bf16\|fp16 --mode plain\|compile\|cudagraph --new N` |
| `amd_optimized.py` | RDNA4 tuning toolkit: `--breakdown` (GPU-time split), `--profile`, `--tune-only`, `--autotune fast\|small\|gemv\|max`, resumable logging to `tmp/` |
| `BUG_REPORT.md` | Write-up of the nightly-wheel import deadlock (for filing upstream) |

Bazzite is immutable, so ROCm/PyTorch live in a container, not on the base OS. The pip
PyTorch-ROCm wheel is self-contained (it bundles the rocBLAS/hipBLASLt/MIOpen runtime and
pulls `pytorch-triton-rocm`), so the base stays a plain Ubuntu (~16 GB) instead of the
~40 GB `rocm/pytorch` SDK image. The host only needs the amdgpu/KFD kernel driver
(`/dev/kfd`), which Bazzite already provides.

## Build & run

```bash
# 1. Build the image
podman build -t localhost/bonsai-rocm:latest -f Containerfile .

# 2. Create a distrobox from it (shares /dev, so the GPU passes through automatically)
distrobox create --name bonsai-rocm --image localhost/bonsai-rocm:latest --yes
distrobox enter bonsai-rocm

# 3. Plain path — confirm the 1-bit kernel runs (first run JIT-compiles, ~5 min)
python run_minimal.py

# 4. Full fast path benchmark
python bench.py --acc fp16 --mode cudagraph --new 1024
```

## Gotchas discovered (the hard-won bits)

1. **Do NOT use the `nightly/rocm7.2` torch wheel.** `2.13.0.dev` hard-links
   `libtorch_rocshmem.so`, whose global constructor calls `exit()` on a single consumer
   GPU and **deadlocks `import torch`** (loader-lock vs rocprofiler atexit). Stable
   `2.12.0+rocm7.2` doesn't hard-link it and works. See `BUG_REPORT.md`.
2. **Triton needs a C compiler at runtime.** It JIT-builds host helpers + kernels, so the
   image must include `build-essential` + `python3-dev` or `import gemlite` fails with
   *"Failed to find C compiler"*.
3. **Don't reinstall `triton`.** gemlite declares `triton>=3.6.0`; let pip pull only the
   torch-bundled `triton-rocm` (install gemlite/hqq with `--no-deps`) — a generic PyPI
   `triton` collides with it.
4. **Never run a GPU probe at image-build time.** The build sandbox has no `/dev/kfd`;
   `torch.cuda.*` (and even `import torch` on the nightly) blocks on it. Do GPU checks at
   runtime in the distrobox.
5. The venv is root-owned (built as root); to pip-install inside the box use `sudo`.

# `import torch` deadlocks at startup on ROCm wheels: `libtorch_rocshmem.so` global constructor calls `exit()` during `dlopen`, deadlocking against the dynamic-loader lock via rocprofiler's atexit handler

## Summary

On a single consumer AMD GPU (Radeon RX 9070 XT, gfx1201/RDNA4), `import torch` hangs **forever** at process startup with the ROCm 7.2 nightly wheel. It never returns and consumes 0% CPU (deadlocked, not busy).

Root cause (confirmed via gdb): a **global/static constructor** in the bundled `libtorch_rocshmem.so` runs `rocshmem::NUMAWrapper::NUMAWrapper()` during `dlopen()` of `_C.cpython-*.so`. That constructor hits a failure path and calls **`exit()` while the dynamic-loader lock is held**. `exit()` then runs atexit handlers, one of which is `rocprofiler::registration::finalize()`, which tries to `join()` a rocprofiler worker thread. That worker thread is blocked in `__cxa_thread_atexit_impl` waiting for the **same loader lock** the main thread still holds → classic deadlock.

`libtorch_rocshmem.so` is a hard `NEEDED` dependency of `libtorch_hip.so`, `libtorch.so`, and `_C.*.so`, so it is loaded eagerly on every `import torch`; there is no way to opt out. rocSHMEM is a multi-node RDMA/symmetric-memory library and is meaningless on a single consumer GPU with no RDMA NIC, yet its constructor runs unconditionally.

## Environment

- GPU: AMD Radeon RX 9070 XT (Navi 48, RDNA4, **gfx1201**)
- OS: Bazzite (Fedora Kinoite based), kernel 6.17
- Container: rootless podman 5.8 + distrobox 1.8 (Ubuntu 24.04 base); `/dev/kfd` and `/dev/dri/renderD128` passed through and world-rw
- torch: `2.13.0.dev20260601+rocm7.2` (cp312) from `https://download.pytorch.org/whl/nightly/rocm7.2`
- Python 3.12, venv

## Reproduction

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm7.2
python -c "import torch"   # hangs forever, 0% CPU
```

## Evidence (gdb `thread apply all bt` on the hung process)

```
Thread 1 (main) — holds loader lock, called exit() from a constructor:
  #3  rocshmem::NUMAWrapper::NUMAWrapper()                       (libtorch_rocshmem.so)
  #11 __GI_exit
  #10 rocprofiler::registration::finalize()                     (librocprofiler-sdk.so)
  #5  PTL::ThreadPool::destroy_threadpool()
  #4  std::thread::join()                                        ← waiting on Thread 2
  ...
  #13 rocshmem::NUMAWrapper::NUMAWrapper()                       (libtorch_rocshmem.so)
  #14 _GLOBAL__sub_I_numa_wrapper.cpp                            ← static initializer
  #17 _dl_init / call_init                                       ← running ELF ctors
  #22 _dl_open ".../torch/_C.cpython-312-x86_64-linux-gnu.so"
  #28 dlopen
  (import torch)

Thread 2 (rocprofiler PTL worker) — needs the loader lock to finish:
  #4  __cxa_thread_atexit_impl  (locking _rtld_global+2568, the loader lock)
  #6  PTL::ThreadPool::execute_thread(...)                       (librocprofiler-sdk.so)
  ...
```

So: main thread holds the loader lock (mid-`dlopen`) → `NUMAWrapper` ctor calls `exit()` → atexit → rocprofiler `finalize()` → `thread::join()` on the worker → worker needs the loader lock → **deadlock**.

Additional facts ruling out the obvious:
- `libnuma.so.1` is **bundled** in `torch/lib/` and resolves fine.
- NUMA topology is visible in the container (`/sys/devices/system/node/node0`).
- No diagnostic message is printed before the hang.
- `/dev/kfd` is **not even opened** at the time of the deadlock — it dies before touching the GPU.

## Why this is a bug

1. **Calling `exit()` from a library global constructor is unsafe** — it runs while the dynamic-loader lock is held, so any atexit handler that touches the loader (here, rocprofiler joining a thread that registers a thread-local dtor) deadlocks.
2. **rocSHMEM is initialized eagerly and unconditionally** via a hard `NEEDED` link, even on single-GPU/consumer systems where it cannot work and is never used.

## Suggested fixes (any one)

- Do not eagerly hard-link/initialize `libtorch_rocshmem.so`; load/init it lazily only when symmetric-memory/rocSHMEM is actually requested.
- Never call `exit()`/`abort()` from `libtorch_rocshmem`'s constructors — fail gracefully (return/log) so `import torch` can proceed without rocSHMEM.
- Gate rocSHMEM init behind an environment variable (opt-in), defaulting off.

## Workaround (confirmed)

Use the **stable** release wheel instead of nightly:

```bash
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

`2.12.0+rocm7.2` does not hard-link rocSHMEM, imports cleanly, has `gfx1201` in
`torch.cuda.get_arch_list()`, and runs end-to-end on the RX 9070 XT (verified: full
torch.compile + HIP-graph generation at ~166 tok/s on a 1-bit model).

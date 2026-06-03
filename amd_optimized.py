#!/usr/bin/env python3
"""
RDNA4 (gfx1201) tuning toolkit for the 1-bit GemLite Bonsai decode.

Three commands (distinct jobs, shared plumbing):

    tune       autotune the GEMV decode kernel on a plain EAGER decode (no torch.compile),
               then cache the config to tmp/.  --scope {fast,small,gemv,max}
    bench      benchmark the full fast path (torch.compile + HIP graphs); optional cached config
    breakdown  CUDA-event GPU-time split (gemlite GEMV vs attention vs other) -- the
               profiler that actually works on this ROCm stack

All output (incl. gemlite/Triton autotune lines) is tee'd to tmp/amd_optimized_<ts>.log,
with a stable tmp/amd_optimized_latest.log symlink: `tail -F tmp/amd_optimized_latest.log`.

    python amd_optimized.py tune --scope small
    python amd_optimized.py bench --load-config tmp/gemlite_gfx1201_small.json
    python amd_optimized.py breakdown --tokens 32
"""
import os
import sys
import time
import signal
import atexit
from pathlib import Path
from typing import Optional

import typer

# --------------------------------------------------------------------------------------
# logging: tee stdout+stderr into a tail-able logfile in tmp/ (captures Triton autotune
# output too). Set up at import so every command gets it.
# --------------------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent
TMP = REPO / "tmp"
TMP.mkdir(exist_ok=True)
LOGPATH = TMP / f"amd_optimized_{time.strftime('%Y%m%d_%H%M%S')}.log"
_LOGF = open(LOGPATH, "a", buffering=1)  # line-buffered -> live tail-able


class _Tee:
    def __init__(self, real, f):
        self._real, self._f = real, f

    def write(self, data):
        self._real.write(data)
        self._f.write(data)
        return len(data)

    def flush(self):
        self._real.flush()
        self._f.flush()

    def isatty(self):
        return False  # tqdm/Triton then emit clean line-based output

    def fileno(self):
        return self._real.fileno()


sys.stdout = _Tee(sys.__stdout__, _LOGF)
sys.stderr = _Tee(sys.__stderr__, _LOGF)

_LATEST = TMP / "amd_optimized_latest.log"
try:
    if _LATEST.is_symlink() or _LATEST.exists():
        _LATEST.unlink()
    _LATEST.symlink_to(LOGPATH.name)
except OSError:
    pass


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
DEVICE = "cuda:0"            # ROCm exposes the AMD GPU as cuda:0 (HIP masquerades as CUDA)
MODEL_ID = "prism-ml/Bonsai-1.7B-unpacked"
PROMPT = "Write an essay about large language models."
GROUP_SIZE = 128
BASELINE_TOKS = 180.0       # fp16/cudagraph on the RX 9070 XT (config-independent; ~+-10% noise)
DEFAULT_CONFIG = TMP / "gemlite_gfx1201.json"

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="RDNA4 (gfx1201) tuning toolkit for the 1-bit GemLite Bonsai decode.")


# --------------------------------------------------------------------------------------
# shared plumbing
# --------------------------------------------------------------------------------------
def _small_grid():
    """~36-config probe centered on the fast winner (N=64,K=64,w=1,s=1,v=2), extending into
    max-only territory on the impactful levers (BLOCK_SIZE_N/K, num_warps, waves_per_eu)."""
    import triton
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
             "A_load_order": 0, "dot_prod_mode": 0, "waves_per_eu": v},
            num_warps=w, num_stages=1)
        for N in (32, 64, 128) for K in (64, 128) for w in (1, 2) for v in (0, 2, 4)
    ]


def _prepare(acc: str = "fp16", scope: str = "off",
             load_config: Optional[Path] = None, save_config: Optional[Path] = None):
    """Load model+tokenizer, set up gemlite (+ optional autotune), 1-bit patch the model.
    Returns (torch, model, tok, gemlite). Shared by every command."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import gemlite
    from gemlite.core import DType
    from gemlite.helper import patch_model, A16W1_HQQ_INT

    assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- GPU not visible"
    log(f"GPU={torch.cuda.get_device_name(0)} | torch={torch.__version__}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cpu")

    if load_config and Path(load_config).exists():
        gemlite.load_config(str(load_config))
        log(f"loaded gemlite config <- {load_config}")

    if scope in ("fast", "small", "gemv", "max"):
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # print each kernel's chosen config
        if not (load_config and Path(load_config).exists()):
            gemlite.reset_config()  # tune from scratch (discard the NVIDIA presets)
        spec = {"fast":  {"GEMV_REVSPLITK": "fast"},
                "small": {"GEMV_REVSPLITK": "fast"},   # reload machinery; .configs overridden below
                "gemv":  {"GEMV_REVSPLITK": "max"},
                "max":   "max"}[scope]
        try:
            gemlite.set_autotune(spec, use_cuda_graph=True)
        except TypeError:
            gemlite.set_autotune(spec)

        if scope == "small":
            from gemlite.triton_kernels import gemv_revsplitK_kernels as rk
            cfgs = _small_grid()
            kobj = rk.gemv_INT_revsplitK_kernel
            n_old = len(getattr(kobj, "configs", []))
            kobj.configs = cfgs
            log(f"SMALL grid override on gemv_INT_revsplitK_kernel: {n_old} -> {len(cfgs)} configs")
        log(f"autotune scope = {scope} (use_cuda_graph=True)")

        # resumable: cache whatever's tuned so far on normal exit OR kill (SIGTERM/SIGINT)
        if save_config:
            def _save(*_):
                try:
                    gemlite.cache_config(str(save_config))
                    log(f"[save] cached gemlite config -> {save_config}")
                except Exception as e:  # noqa: BLE001
                    log(f"[save] cache failed: {e}")
            atexit.register(_save)
            for _sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(_sig, lambda *_a: sys.exit(0))

    if acc == "fp16":
        gemlite.set_acc_dtype(DType.FP16)
        log("gemlite acc dtype = FP16")

    log("patch_model (1-bit)...")
    t0 = time.time()
    patch_model(model, device=DEVICE, processor=A16W1_HQQ_INT(), group_size=GROUP_SIZE)
    log(f"patched in {time.time()-t0:.1f}s")
    return torch, model, tok, gemlite


def _eager_decode(torch, model, tok, n: int):
    """Plain eager greedy decode of n tokens + sync. Used for autotune-driving and warmup."""
    inputs = tok(PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=n, do_sample=False)
    torch.cuda.synchronize()
    return out


def _install_timers(torch, model):
    """Bracket each gemlite linear + the whole forward + the sdpa call with CUDA events.
    Returns (gemlite_ev, attn_ev, total_ev, restore_fn)."""
    import torch.nn.functional as F
    gem, attn, total = [], [], []

    def pre(store):
        def h(mod, inp):
            ev = torch.cuda.Event(enable_timing=True); ev.record(); mod._bd_s = ev
        return h

    def post(store):
        def h(mod, inp, out):
            ev = torch.cuda.Event(enable_timing=True); ev.record(); store.append((mod._bd_s, ev))
        return h

    handles, n_g = [], 0
    for m in model.modules():
        if type(m).__module__.split(".")[0] == "gemlite":
            handles += [m.register_forward_pre_hook(pre(gem)), m.register_forward_hook(post(gem))]
            n_g += 1
    handles += [model.register_forward_pre_hook(pre(total)), model.register_forward_hook(post(total))]

    orig = F.scaled_dot_product_attention

    def timed(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = orig(*a, **k); e.record(); attn.append((s, e)); return out

    F.scaled_dot_product_attention = timed

    def restore():
        F.scaled_dot_product_attention = orig
        for h in handles:
            h.remove()

    log(f"hooked {n_g} gemlite modules + attention")
    return gem, attn, total, restore


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------
@app.command()
def tune(
    scope: str = typer.Option("fast", help="fast (16 cfg = baseline) | small (~36 probe) | "
                                           "gemv (540 exhaustive) | max (every kernel)"),
    new: int = typer.Option(4, help="decode tokens to drive autotune (1 is enough)"),
    save_config: Path = typer.Option(DEFAULT_CONFIG, help="cache the tuned config here"),
    load_config: Optional[Path] = typer.Option(None, help="resume from a cached config"),
):
    """Autotune the GEMV decode kernel on a plain EAGER decode (no torch.compile/cudagraph),
    then cache the config. Eager = each kernel tunes once per shape, no inductor explosion."""
    torch, model, tok, gemlite = _prepare(scope=scope, load_config=load_config, save_config=save_config)
    log(f"tune[{scope}]: eager decode of {new} tokens "
        f"(watch tmp/amd_optimized_latest.log for per-kernel configs)...")
    t0 = time.time()
    _eager_decode(torch, model, tok, new)
    log(f"autotune eager decode done in {time.time()-t0:.1f}s")
    gemlite.cache_config(str(save_config))
    log(f"cached gemlite config -> {save_config}")


@app.command()
def bench(
    new: int = typer.Option(1024, help="tokens to generate for the timed run"),
    acc: str = typer.Option("fp16", help="gemlite accumulation dtype: bf16 | fp16"),
    load_config: Optional[Path] = typer.Option(None, help="load a cached gemlite config"),
):
    """Benchmark the full fast path: fp16 acc + torch.compile (max-autotune) + HIP graphs."""
    from hqq.utils.generation_hf import HFGenerator

    torch, model, tok, _ = _prepare(acc=acc, scope="off", load_config=load_config)
    gen = HFGenerator(
        model, tok, max_new_tokens=new, do_sample=False,
        compile="partial",
        compile_options={"mode": "max-autotune-no-cudagraphs", "fullgraph": True},
    ).enable_cuda_graph()

    log("warmup (torch.compile + HIP graph capture; slow first time)...")
    t0 = time.time()
    gen.warmup()
    log(f"warmup done in {time.time()-t0:.1f}s")

    t0 = time.time()
    out = gen.generate(PROMPT, print_tokens=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    text = out["output_text"] if isinstance(out, dict) else str(out)
    if isinstance(out, dict) and "output_tokens" in out:
        ot = out["output_tokens"]
        n = len(ot) if hasattr(ot, "__len__") else int(ot)
    else:
        n = len(tok(text, add_special_tokens=False).input_ids)
    n = max(n, 1)
    tps = n / dt
    log(f"RESULT [{acc}/cudagraph]: {n} tok in {dt:.2f}s -> {tps:.1f} tok/s "
        f"({tps/BASELINE_TOKS:.2f}x of ~{BASELINE_TOKS:.0f} baseline)")
    print("\n----- OUTPUT (first 400 chars) -----\n" + text[:400])


@app.command()
def breakdown(tokens: int = typer.Option(32, help="decode tokens to profile")):
    """CUDA-event GPU-time split: gemlite 1-bit GEMV vs attention (sdpa) vs other.

    Zero-install (runs inside torch's process). torch.profiler doesn't populate GPU kernel
    times on this ROCm stack, and rocprofv3 can't coexist with pip-torch -- this is the
    method that works. Per-module GPU time is a valid proxy for the cudagraph regime."""
    torch, model, tok, _ = _prepare(scope="off")
    log("warmup (compiles gemlite kernels w/ stock configs)...")
    _eager_decode(torch, model, tok, 8)

    gem, attn, total, restore = _install_timers(torch, model)
    log(f"breakdown: decoding {tokens} tokens (instrumented; wall tok/s will look slow)...")
    t0 = time.time()
    _eager_decode(torch, model, tok, tokens)
    wall = time.time() - t0
    restore()

    def ms(evs):
        return sum(s.elapsed_time(e) for s, e in evs)

    g, a, T = ms(gem), ms(attn), ms(total)
    other = max(T - g - a, 0.0)
    log("\n===== GPU-TIME BREAKDOWN (CUDA events) =====")
    log(f"  total model.forward : {T:9.1f} ms   100.0%   [{len(total)} fwd calls]")
    log(f"  gemlite 1-bit GEMV  : {g:9.1f} ms   {100*g/T:5.1f}%   [{len(gem)} calls]")
    log(f"  attention (sdpa)    : {a:9.1f} ms   {100*a/T:5.1f}%   [{len(attn)} calls]")
    log(f"  other (norm/rope/..): {other:9.1f} ms   {100*other/T:5.1f}%")
    log(f"  (eager wall {wall:.2f}s = {tokens/wall:.1f} tok/s, inflated by instrumentation)")


if __name__ == "__main__":
    log(f"logging to {LOGPATH}")
    app()

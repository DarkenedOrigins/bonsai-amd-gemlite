#!/usr/bin/env python3
"""
RDNA4 (gfx1201) re-autotune experiment for the 1-bit GemLite Bonsai decode.

gemlite ships configs autotuned on NVIDIA. This script throws those away and
re-autotunes the Triton kernels EXHAUSTIVELY on *this* GPU (set_autotune "max"),
caches the result, and benchmarks the warm fast path (fp16 acc + torch.compile +
HIP graphs) against the stock baseline of 166.2 tok/s.

  # tune from scratch for this GPU + save config + benchmark (SLOW warmup):
  python amd_optimized.py --autotune max --save-config gemlite_gfx1201.json

  # reuse a cached config (fast warmup):
  python amd_optimized.py --load-config gemlite_gfx1201.json
"""
import os, sys, time, argparse, traceback, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- logging: tee BOTH stdout and stderr into a tail-able logfile in repo tmp/, so we capture
#     our milestones AND gemlite/Triton autotune output (TRITON_PRINT_AUTOTUNING) + progress bars. ---
_REPO = os.path.dirname(os.path.abspath(__file__))
_LOGDIR = os.path.join(_REPO, "tmp")
os.makedirs(_LOGDIR, exist_ok=True)
LOGPATH = os.path.join(_LOGDIR, f"amd_optimized_{time.strftime('%Y%m%d_%H%M%S')}.log")
_logf = open(LOGPATH, "a", buffering=1)  # line-buffered -> live tail-able

class _Tee:
    def __init__(self, real, f): self._real, self._f = real, f
    def write(self, data):
        self._real.write(data); self._f.write(data); return len(data)
    def flush(self):
        self._real.flush(); self._f.flush()
    def isatty(self): return False            # tqdm/Triton then emit clean line-based output
    def fileno(self): return self._real.fileno()
sys.stdout = _Tee(sys.__stdout__, _logf)
sys.stderr = _Tee(sys.__stderr__, _logf)

# stable 'latest' symlink so you can always `tail -f tmp/amd_optimized_latest.log`
_LATEST = os.path.join(_LOGDIR, "amd_optimized_latest.log")
try:
    if os.path.islink(_LATEST) or os.path.exists(_LATEST):
        os.remove(_LATEST)
    os.symlink(os.path.basename(LOGPATH), _LATEST)
except OSError:
    pass

def _log_excepthook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)   # -> stderr -> teed into the logfile
sys.excepthook = _log_excepthook

def log(m): print(f"{time.strftime('%H:%M:%S')} {m}", flush=True)

log(f"logging to {LOGPATH} (captures gemlite/Triton autotune output too)")

BASELINE_TOKS = 166.2  # stock NVIDIA-tuned configs, fp16/cudagraph, on this RX 9070 XT

ap = argparse.ArgumentParser()
ap.add_argument("--new", type=int, default=1024)
ap.add_argument("--autotune", choices=["off", "fast", "small", "gemv", "max"], default="gemv",
                help="'fast'=16-config baseline; 'small'=~36-config probe around the fast winner "
                     "(N/K/warps/waves, incl. max-only K=128); 'gemv'=540-config; 'max'=all kernels; 'off'=stock")
ap.add_argument("--save-config", default=None, help="cache autotuned configs to JSON")
ap.add_argument("--load-config", default=None, help="load cached configs (skips autotune)")
ap.add_argument("--profile", action="store_true",
                help="profile a decode with stock configs and dump a per-kernel GPU-time breakdown (no autotune)")
ap.add_argument("--profile-tokens", type=int, default=64, help="tokens to generate while profiling")
ap.add_argument("--decode-only", action="store_true",
                help="bare eager decode w/ stock configs (no torch.profiler) -- trace it with rocprofv3")
ap.add_argument("--breakdown", action="store_true",
                help="CUDA-event GPU-time split: gemlite GEMV vs attention vs other (zero-install profiler)")
ap.add_argument("--tune-only", action="store_true",
                help="autotune gemlite on a plain EAGER decode (no torch.compile/cudagraph), cache, exit. "
                     "Avoids the inductor x gemlite autotune explosion. Then benchmark with --load-config.")
args = ap.parse_args()
if args.profile or args.decode_only or args.breakdown:
    args.autotune = "off"  # use stock configs (the 166 tok/s kernels); fast warmup

device = "cuda:0"
compute_dtype = torch.bfloat16
model_id = "prism-ml/Bonsai-1.7B-unpacked"
prompt = "Write an essay about large language models."

assert torch.cuda.is_available(), "GPU not visible"
log(f"GPU={torch.cuda.get_device_name(0)} | autotune={args.autotune} new={args.new}")

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=compute_dtype, attn_implementation="sdpa", device_map="cpu")

import gemlite
from gemlite.core import DType
from gemlite.helper import patch_model, A16W1_HQQ_INT

# --- config selection: load cached configs, then (optionally) keep autotuning uncached ones ---
if args.load_config and os.path.exists(args.load_config):
    gemlite.load_config(args.load_config)
    log(f"loaded gemlite config <- {args.load_config}")

if args.autotune in ("fast", "small", "gemv", "max"):
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"  # print each kernel's best config as it tunes
    if not (args.load_config and os.path.exists(args.load_config)):
        gemlite.reset_config()  # tune from scratch (discard NVIDIA presets)
    # 'small' reuses the fast machinery (reload + cudagraph), then OVERRIDES the revsplitk
    # autotuner's config list with our ~36-config probe centered on the fast winner.
    spec = {"fast":  {"GEMV_REVSPLITK": "fast"},
            "small": {"GEMV_REVSPLITK": "fast"},
            "gemv":  {"GEMV_REVSPLITK": "max"},
            "max":   "max"}[args.autotune]
    try:
        gemlite.set_autotune(spec, use_cuda_graph=True)
    except TypeError:
        gemlite.set_autotune(spec)

    if args.autotune == "small":
        import triton
        from gemlite.triton_kernels import gemv_revsplitK_kernels as _rk
        small_cfgs = [
            triton.Config(
                {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": N, "BLOCK_SIZE_K": K,
                 "A_load_order": 0, "dot_prod_mode": 0, "waves_per_eu": v},
                num_warps=w, num_stages=1)
            for N in (32, 64, 128) for K in (64, 128) for w in (1, 2) for v in (0, 2, 4)
        ]
        kobj = _rk.gemv_INT_revsplitK_kernel
        n_old = len(getattr(kobj, "configs", []))
        kobj.configs = small_cfgs
        log(f"SMALL grid override on gemv_INT_revsplitK_kernel: {n_old} -> {len(small_cfgs)} configs "
            f"(N{{32,64,128}} x K{{64,128}} x warps{{1,2}} x waves{{0,2,4}}, stages=1)")
    log(f"gemlite re-autotune = {args.autotune} (use_cuda_graph=True)")

    # RESUMABLE: cache whatever's been tuned so far on normal exit OR on kill (SIGTERM/SIGINT),
    # so a long autotune is never lost -- rerun with --load-config to continue uncached shapes.
    import signal, atexit
    def _save_cfg(*_):
        if args.save_config:
            try:
                gemlite.cache_config(args.save_config)
                log(f"[save] cached gemlite config -> {args.save_config}")
            except Exception as e:
                log(f"[save] cache failed: {e}")
    atexit.register(_save_cfg)
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, lambda *_a: sys.exit(0))  # -> triggers atexit -> _save_cfg

gemlite.set_acc_dtype(DType.FP16)

log("patch_model (1-bit)...")
t0 = time.time()
patch_model(model, device=device, processor=A16W1_HQQ_INT(), group_size=128)
log(f"patched in {time.time()-t0:.1f}s")

# --- TUNE-ONLY: drive gemlite autotune via a plain EAGER decode, then cache & exit. --------
# Eager (no torch.compile/cudagraph) means each gemlite kernel autotunes ONCE per shape on
# first call -- no inductor graph replays re-triggering it. With TRITON_PRINT_AUTOTUNING=1
# you see each kernel's chosen config land in the log as it finishes.
if args.tune_only:
    inputs = tok(prompt, return_tensors="pt").to(device)
    log("tune-only: eager decode driving gemlite GEMV autotune (watch tmp/ log for per-kernel configs)...")
    t0 = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()
    log(f"autotune eager decode done in {time.time()-t0:.1f}s")
    if args.save_config:
        gemlite.cache_config(args.save_config)
        log(f"cached gemlite config -> {args.save_config}")
    sys.exit(0)

# --- PROFILE MODE: find where GPU time actually goes, then exit ------------------
# Profiles an eager decode (stock gemlite configs). Per-kernel GPU (self-device) time
# is a valid proxy for the cudagraph regime: same kernels, same work -- graphs only
# remove the gaps/launch overhead between them. Tells us if the 1-bit GEMV is the
# bottleneck (=> autotune worth it) or if it's attention/dequant/norms/overhead.
if args.profile or args.decode_only or args.breakdown:
    inputs = tok(prompt, return_tensors="pt").to(device)
    log("warmup (compiles gemlite kernels w/ stock configs)...")
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
    torch.cuda.synchronize()

if args.breakdown:
    # Zero-install GPU-time attribution via CUDA events (no external profiler, no ROCm
    # userspace conflict). Brackets each gemlite linear + the sdpa call; "other" = the
    # rest of model.forward (norms, rope, lm_head, elementwise). Per-module GPU elapsed
    # time is a valid proxy for the cudagraph regime (same kernels, same work).
    import torch.nn.functional as F
    gemlite_ev, attn_ev, total_ev = [], [], []

    def _pre(store):
        def h(mod, inp):
            s = torch.cuda.Event(enable_timing=True); s.record(); mod._bd_s = s
        return h
    def _post(store):
        def h(mod, inp, out):
            e = torch.cuda.Event(enable_timing=True); e.record(); store.append((mod._bd_s, e))
        return h

    n_g = 0
    for m in model.modules():
        if type(m).__module__.split(".")[0] == "gemlite":
            m.register_forward_pre_hook(_pre(gemlite_ev)); m.register_forward_hook(_post(gemlite_ev)); n_g += 1
    model.register_forward_pre_hook(_pre(total_ev)); model.register_forward_hook(_post(total_ev))
    log(f"hooked {n_g} gemlite linear modules + attention")

    _orig_sdpa = F.scaled_dot_product_attention
    def _timed_sdpa(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); out = _orig_sdpa(*a, **k); e.record(); attn_ev.append((s, e)); return out
    F.scaled_dot_product_attention = _timed_sdpa

    log(f"breakdown: decoding {args.profile_tokens} tokens (instrumented; wall tok/s will look slow)...")
    t0 = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=args.profile_tokens, do_sample=False)
    torch.cuda.synchronize()
    wall = time.time() - t0
    F.scaled_dot_product_attention = _orig_sdpa

    def msum(evs): return sum(s.elapsed_time(e) for s, e in evs)
    g, a, T = msum(gemlite_ev), msum(attn_ev), msum(total_ev)
    other = max(T - g - a, 0.0)
    log("\n===== GPU-TIME BREAKDOWN (CUDA events, stock configs) =====")
    log(f"  total model.forward : {T:9.1f} ms   100.0%   [{len(total_ev)} fwd calls]")
    log(f"  gemlite 1-bit GEMV  : {g:9.1f} ms   {100*g/T:5.1f}%   [{len(gemlite_ev)} calls]")
    log(f"  attention (sdpa)    : {a:9.1f} ms   {100*a/T:5.1f}%   [{len(attn_ev)} calls]")
    log(f"  other (norm/rope/..): {other:9.1f} ms   {100*other/T:5.1f}%")
    log(f"  (eager wall: {wall:.2f}s = {args.profile_tokens/wall:.1f} tok/s, inflated by instrumentation)")
    log("VERDICT: gemlite GEMV is " + ("the dominant GPU cost -> targeted autotune is worth it"
        if g/T > 0.5 else "NOT dominant -> autotuning it has limited ceiling; bottleneck is elsewhere"))
    sys.exit(0)

if args.decode_only:
    # bare eager decode, no torch.profiler -- meant to be traced by `rocprofv3 --kernel-trace`.
    # nvtx range maps to roctx on ROCm so the decode region is markable in the trace.
    log(f"decode-only: generating {args.profile_tokens} tokens (trace me with rocprofv3)...")
    torch.cuda.nvtx.range_push("decode")
    t0 = time.time()
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=args.profile_tokens, do_sample=False)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    log(f"decode-only done in {time.time()-t0:.2f}s")
    sys.exit(0)

if args.profile:
    import torch.profiler as tp
    log(f"profiling eager decode of {args.profile_tokens} tokens...")
    with tp.profile(activities=[tp.ProfilerActivity.CPU, tp.ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=args.profile_tokens, do_sample=False)
        torch.cuda.synchronize()

    table = None
    for key in ("self_device_time_total", "self_cuda_time_total", "cuda_time_total"):
        try:
            table = prof.key_averages().table(sort_by=key, row_limit=40)
            log(f"(sorted by {key})")
            break
        except Exception:
            continue
    if table is None:
        table = prof.key_averages().table(row_limit=40)
    log("\n===== PER-KERNEL GPU TIME (top 40) =====\n" + table)

    tracepath = os.path.join(_LOGDIR, f"profile_trace_{time.strftime('%Y%m%d_%H%M%S')}.json")
    try:
        prof.export_chrome_trace(tracepath)
        log(f"chrome trace (view in perfetto.dev or chrome://tracing) -> {tracepath}")
    except Exception as e:
        log(f"(chrome trace export failed: {e})")
    log("profile done; exiting before autotune/cudagraph path")
    sys.exit(0)

# --- full fast path: torch.compile + HIP graphs (the gist's config) ---
from hqq.utils.generation_hf import HFGenerator
gen = HFGenerator(
    model, tok, max_new_tokens=args.new, do_sample=False,
    compile="partial",
    compile_options={"mode": "max-autotune-no-cudagraphs", "fullgraph": True},
).enable_cuda_graph()

log("warmup (compiles + EXHAUSTIVE gemlite autotune; this is the slow part)...")
t0 = time.time()
gen.warmup()
log(f"warmup done in {time.time()-t0:.1f}s")

# cache the freshly-autotuned configs so future runs skip the slow warmup
if args.save_config:
    gemlite.cache_config(args.save_config)
    log(f"cached gemlite config -> {args.save_config}")

# --- timed warm run ---
t0 = time.time()
o = gen.generate(prompt, print_tokens=False)
torch.cuda.synchronize()
dt = time.time() - t0
text = o["output_text"] if isinstance(o, dict) else str(o)
if isinstance(o, dict) and "output_tokens" in o:
    ot = o["output_tokens"]; n = len(ot) if hasattr(ot, "__len__") else int(ot)
else:
    n = len(tok(text, add_special_tokens=False).input_ids) - len(tok(prompt, add_special_tokens=False).input_ids)
n = max(n, 1)

new_toks = n / dt
log(f"RESULT [AMD-retuned, fp16/cudagraph]: {n} tok in {dt:.2f}s -> {new_toks:.1f} tok/s")
log(f"vs stock baseline {BASELINE_TOKS} tok/s  =>  {new_toks/BASELINE_TOKS:.2f}x")
print("\n----- OUTPUT (first 400 chars) -----")
print(text[:400])

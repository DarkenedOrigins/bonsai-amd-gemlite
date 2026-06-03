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
import os, sys, time, logging, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- proper logging: milestones go to a durable, tail-able logfile in repo tmp/ + console ---
_REPO = os.path.dirname(os.path.abspath(__file__))
_LOGDIR = os.path.join(_REPO, "tmp")
os.makedirs(_LOGDIR, exist_ok=True)
LOGPATH = os.path.join(_LOGDIR, f"amd_optimized_{time.strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOGPATH), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("amd_optimized")

# stable 'latest' symlink so you can always `tail -f tmp/amd_optimized_latest.log`
_LATEST = os.path.join(_LOGDIR, "amd_optimized_latest.log")
try:
    if os.path.islink(_LATEST) or os.path.exists(_LATEST):
        os.remove(_LATEST)
    os.symlink(os.path.basename(LOGPATH), _LATEST)
except OSError:
    pass

# capture uncaught exceptions (autotune/compile failures) in the logfile, not just console
def _log_excepthook(exc_type, exc, tb):
    logger.error("UNCAUGHT EXCEPTION", exc_info=(exc_type, exc, tb))
    sys.__excepthook__(exc_type, exc, tb)
sys.excepthook = _log_excepthook

def log(m): logger.info(m)

log(f"logging to {LOGPATH}")

BASELINE_TOKS = 166.2  # stock NVIDIA-tuned configs, fp16/cudagraph, on this RX 9070 XT

ap = argparse.ArgumentParser()
ap.add_argument("--new", type=int, default=1024)
ap.add_argument("--autotune", choices=["off", "max"], default="max",
                help="'max' = exhaustive re-autotune from scratch for this GPU (slow warmup)")
ap.add_argument("--save-config", default=None, help="cache autotuned configs to JSON")
ap.add_argument("--load-config", default=None, help="load cached configs (skips autotune)")
args = ap.parse_args()

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

# --- config selection: load cached, or set up exhaustive re-autotune ---
if args.load_config:
    gemlite.load_config(args.load_config)
    log(f"loaded gemlite config <- {args.load_config} (autotune skipped)")
elif args.autotune == "max":
    gemlite.reset_config()  # discard the NVIDIA-tuned presets; tune from scratch here
    try:
        gemlite.set_autotune("max", use_cuda_graph=True)
    except TypeError:
        gemlite.set_autotune("max")
    log("gemlite re-autotune = MAX (exhaustive, from scratch) -- warmup will be slow")

gemlite.set_acc_dtype(DType.FP16)

log("patch_model (1-bit)...")
t0 = time.time()
patch_model(model, device=device, processor=A16W1_HQQ_INT(), group_size=128)
log(f"patched in {time.time()-t0:.1f}s")

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

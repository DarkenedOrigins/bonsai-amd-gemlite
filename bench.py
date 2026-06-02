#!/usr/bin/env python3
"""
Incremental speed benchmark for 1-bit GemLite Bonsai on AMD RDNA4 (gfx1201).

Stages (add one feature at a time, compare tok/s):
  --acc bf16|fp16        GemLite accumulation dtype (fp16 is faster on consumer GPUs)
  --mode plain           plain HF model.generate (baseline path)
  --mode compile         hqq HFGenerator with torch.compile (max-autotune, no cudagraphs)
  --mode cudagraph       HFGenerator with torch.compile + CUDA/HIP graphs (full gist path)

Usage:  python bench.py --acc fp16 --mode plain
"""
import time, argparse, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--acc", choices=["bf16", "fp16"], default="bf16")
ap.add_argument("--mode", choices=["plain", "compile", "cudagraph"], default="plain")
ap.add_argument("--new", type=int, default=128, help="tokens to generate for timed run")
args = ap.parse_args()

device = "cuda:0"
compute_dtype = torch.bfloat16
model_id = "prism-ml/Bonsai-1.7B-unpacked"
prompt = "Write an essay about large language models."

assert torch.cuda.is_available(), "GPU not visible"
log(f"GPU={torch.cuda.get_device_name(0)} | acc={args.acc} mode={args.mode} new={args.new}")

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=compute_dtype, attn_implementation="sdpa", device_map="cpu")

import gemlite
from gemlite.core import DType
from gemlite.helper import patch_model, A16W1_HQQ_INT

if args.acc == "fp16":
    gemlite.set_acc_dtype(DType.FP16)
    log("GemLite acc dtype = FP16")

log("patch_model (1-bit)...")
t0 = time.time()
patch_model(model, device=device, processor=A16W1_HQQ_INT(), group_size=128)
log(f"patched in {time.time()-t0:.1f}s")

def tps(n, dt): return f"{n} tok in {dt:.2f}s -> {n/dt:.1f} tok/s"

if args.mode == "plain":
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)  # warmup
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.new, do_sample=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n = out.shape[1] - inputs["input_ids"].shape[1]
    text = tok.decode(out[0], skip_special_tokens=True)
else:
    from hqq.utils.generation_hf import HFGenerator
    gen = HFGenerator(
        model, tok, max_new_tokens=args.new, do_sample=False,
        compile="partial",
        compile_options={"mode": "max-autotune-no-cudagraphs", "fullgraph": True},
    )
    if args.mode == "cudagraph":
        gen = gen.enable_cuda_graph()
    log("warmup (compiles/autotunes; slow first time)...")
    t0 = time.time()
    gen.warmup()
    log(f"warmup done in {time.time()-t0:.1f}s")
    t0 = time.time()
    o = gen.generate(prompt, print_tokens=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    text = o["output_text"] if isinstance(o, dict) else str(o)
    # exact generated-token count from HFGenerator (it returns 'output_tokens')
    if isinstance(o, dict) and "output_tokens" in o:
        ot = o["output_tokens"]
        n = len(ot) if hasattr(ot, "__len__") else int(ot)
    else:
        n = len(tok(text, add_special_tokens=False).input_ids) - len(tok(prompt, add_special_tokens=False).input_ids)

log(f"RESULT [{args.acc}/{args.mode}]: {tps(max(n,1), dt)}")
print("\n----- OUTPUT (first 400 chars) -----")
print(text[:400])

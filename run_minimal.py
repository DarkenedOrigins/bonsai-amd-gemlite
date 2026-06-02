#!/usr/bin/env python3
"""
Stage 1: get Bonsai-1.7B (1-bit) + GemLite generating text on the AMD GPU.

This is the gist adapted for ROCm/RDNA4, with ALL speed features stripped out so
we isolate the "does the GemLite 1-bit Triton kernel even run on gfx1201?" question.
No torch.compile, no CUDA graphs. Once this works we add speed back incrementally.

On ROCm, PyTorch exposes the AMD GPU as 'cuda:0' (HIP masquerades as CUDA), so
device='cuda:0' is correct and unchanged from the NVIDIA gist.
"""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

device = "cuda:0"
compute_dtype = torch.bfloat16
cache_dir = None
model_id = "prism-ml/Bonsai-1.7B-unpacked"

# --- sanity: confirm we actually see the GPU before doing anything expensive ---
assert torch.cuda.is_available(), "torch.cuda.is_available() is False -- GPU not visible"
log(f"Device: {torch.cuda.get_device_name(0)}")
log(f"HIP: {torch.version.hip} | arch list: {torch.cuda.get_arch_list()}")

log("Loading tokenizer + model (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=compute_dtype,
    attn_implementation="sdpa",
    cache_dir=cache_dir,
    device_map="cpu",  # load on CPU first; GemLite moves layers to GPU during patch
)
log("Model loaded.")

import gemlite
from gemlite.helper import patch_model, A16W1_HQQ_INT

# FP16 accumulation is faster on consumer GPUs (quality slightly worse). Try without
# first; uncomment if you want to A/B it later.
# gemlite.set_acc_dtype(gemlite.core.DType.FP16)

# This is THE core test: 1-bit HQQ quantize + patch linears with GemLite's 1-bit kernel,
# moving them to the AMD GPU. First call also JIT-compiles the Triton kernels for gfx1201.
log("Patching model with 1-bit GemLite kernel (A16W1_HQQ_INT)...")
t0 = time.time()
patch_model(model, device=device, processor=A16W1_HQQ_INT(), group_size=128)
log(f"patch_model done in {time.time()-t0:.1f}s")

# --- plain HF generate, no compile / no cuda graph ---
prompt = "Write an essay about large language models."
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Warmup: first generate JIT-compiles + autotunes the 1-bit Triton kernels for gfx1201.
# This is slow (minutes) and NOT representative of decode speed.
log("Warmup generate (compiles/autotunes Triton kernels; slow first time)...")
t0 = time.time()
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
torch.cuda.synchronize()
log(f"warmup done in {time.time()-t0:.1f}s")

# Timed WARM run: this is the real steady-state decode speed (no compile).
log("Timed warm generate...")
t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=128, do_sample=False)
torch.cuda.synchronize()
n_new = out.shape[1] - inputs["input_ids"].shape[1]
dt = time.time() - t0
log(f"WARM: {n_new} tokens in {dt:.2f}s  ->  {n_new/dt:.1f} tok/s")
print("\n----- OUTPUT -----")
print(tokenizer.decode(out[0], skip_special_tokens=True))

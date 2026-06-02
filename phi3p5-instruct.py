"""
phi3p5-instruct.py — Multimodal LLM Loader
------------------------------------------------
Loads a vision-language model Phi-3.5-vision onto 
Intel XPU using float16 precision.

NOTE: We do NOT use BitsAndBytes (4-bit) here because that library
is NVIDIA-only. On Intel XPU we use float16 instead, which gives
a ~50% memory saving over the default float32.

Recommended models for PC with 16GB XPU (pick one):
  - microsoft/Phi-3.5-vision-instruct   (~7 GB)  ← safest, start here
  - Qwen/Qwen2-VL-2B-Instruct           (~4 GB)  ← very fast
  - llava-hf/llava-v1.6-mistral-7b-hf   (~14 GB) ← powerful but tight
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

from utils import get_device

from huggingface_hub import login
login(token="YOUR_HF_TOKEN_HERE")

# ────────────────────────────── Configuration — change the model here ─────────────────────────────────────
# Why Phi-3.5-vision? : it's small, fast, and great on Intel hardware.
MODEL_ID = "microsoft/Phi-3.5-vision-instruct"

print("Loading Phi-3.5 vision encoder...")

# ──────────────────────────────── Load processor ────────────────────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    num_crops=4,
)
# ────────────────────────────── Load model — CPU first, then move to XPU ─────────────────────────────────
# Why CPU first? device_map with XPU can silently route ops to CPU.
# Explicit .to(device) after loading guarantees everything is on XPU.
print("Loading weights to CPU first...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,     # bfloat16 — better suited to Intel iGPU
    trust_remote_code=True,
    _attn_implementation="eager",   # disable Flash Attention
    low_cpu_mem_usage=True,         # stream weights in, saves peak RAM
)
 
print(f"Moving model to {device}...")
model = model.to(device)           # explicit move — no ambiguity
model.eval()
 
# Confirm every parameter is on the right device
param_device = next(model.parameters()).device
print(f" Let's Go! Phi-3.5 loaded — all parameters on: {param_device}")
 
 
# ────────────────────────────── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n=== Model Self-Test ===")
    print(f" Device confirmed  : {next(model.parameters()).device}")
    print(f" Processor type   : {type(processor).__name__}")
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f" Parameters       : ~{total_params:.1f}B")
 
    # Quick dummy generation test to confirm XPU does the compute
    print("\n Running dummy generation to test XPU compute...")
    dummy_input = processor.tokenizer(
                                        "Hello", return_tensors="pt"
                                    ).to(device)
    with torch.no_grad():
        out = model.generate(**dummy_input, max_new_tokens=5)
    print(f" Generation works! Output shape: {out.shape}")
 

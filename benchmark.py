"""
benchmark.py — measure real tokens/sec on your XPU
Run with: python benchmark.py
"""

import torch
import time
from PIL import Image
from llava import model, processor, device

print(f"\n=== XPU Benchmark ===")
print(f"Device: {device}")

# ── Warmup — first run is always slower due to JIT compilation ────────────────
print("\n[Warmup] Running 5 token warmup...")
dummy_image = Image.new("RGB", (224, 224), color=(120, 80, 60))
prompt = "<|user|>\n<|image_1|>\nWhat is this?<|end|>\n<|assistant|>\n"
inputs = processor(text=prompt, images=dummy_image, return_tensors="pt").to(device)

with torch.no_grad():
    model.generate(**inputs, max_new_tokens=5, do_sample=False)
print("    Warmup done.")

# ── Real benchmark — 20 tokens ────────────────────────────────────────────────
print("\n[Benchmark] Generating 20 tokens...")
TOKEN_COUNT = 20

torch.xpu.synchronize()   # wait for any pending XPU work to finish
t_start = time.perf_counter()

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=TOKEN_COUNT, do_sample=False)

torch.xpu.synchronize()   # wait for XPU to finish before stopping timer
t_end = time.perf_counter()

elapsed = t_end - t_start
actual_tokens = out.shape[1] - inputs["input_ids"].shape[1]
tokens_per_sec = actual_tokens / elapsed

print(f"\n=== Results ===")
print(f"  Tokens generated : {actual_tokens}")
print(f"  Time taken       : {elapsed:.2f}s")
print(f"  Speed            : {tokens_per_sec:.2f} tokens/sec")
print(f"  Est. 200 tokens  : ~{200/tokens_per_sec:.0f}s ({200/tokens_per_sec/60:.1f} min)")

# ── dtype check ───────────────────────────────────────────────────────────────
sample_param = next(model.parameters())
print(f"\n  Model dtype      : {sample_param.dtype}")
print(f"  Model device     : {sample_param.device}")

# ── Memory usage ──────────────────────────────────────────────────────────────
if hasattr(torch.xpu, "memory_allocated"):
    mem_used = torch.xpu.memory_allocated() / 1e9
    mem_reserved = torch.xpu.memory_reserved() / 1e9
    print(f"  XPU mem used     : {mem_used:.2f} GB")
    print(f"  XPU mem reserved : {mem_reserved:.2f} GB")

# ── Suggestion ────────────────────────────────────────────────────────────────
print(f"\n=== Diagnosis ===")
if tokens_per_sec > 10:
    print("  ✅ Good speed — XPU is working well")
elif tokens_per_sec > 3:
    print("  ⚠️  Moderate speed — typical for iGPU with float16")
    print("     Try: switch to bfloat16 in llava.py")
else:
    print("  ❌ Slow — iGPU struggling with float16")
    print("     Try: switch to bfloat16 or use a smaller model (Qwen2-VL-2B)")
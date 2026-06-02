"""
debug_xpu.py — diagnose why generation falls back to CPU
Run with: python debug_xpu.py
"""

import torch
from PIL import Image
from llava import model, processor, device

print(f"\n=== XPU Fallback Debugger ===")
print(f"Device: {device}")

# ──────────────────── Step 1: confirm model parameters are on XPU ──────────────────────────────
devices_found = set(str(p.device) for p in model.parameters())
print(f"\n[1] Model parameter devices: {devices_found}")
if all("xpu" in d for d in devices_found):
    print("     All parameters on XPU")
else:
    print("    xxxxx Some parameters NOT on XPU xxxxxxxxxxx   ")

# ──────────────────── Step 2: check input tensors ───────────────────────────────────────────────
print("\n[2] Checking input tensor device...")
dummy_image = Image.new("RGB", (224, 224), color=(120, 80, 60))
prompt = "<|user|>\n<|image_1|>\nWhat is this?<|end|>\n<|assistant|>\n"

inputs = processor(text=prompt, images=dummy_image, return_tensors="pt")
print(f"    Before .to(device):")
for k, v in inputs.items():
    print(f"      {k}: {v.device}")

inputs = inputs.to(device)
print(f"    After .to(device):")
for k, v in inputs.items():
    print(f"      {k}: {v.device}")

# ──────────────────── Step 3: hook to catch any CPU tensor during forward pass ──────────────────
print("\n[3] Running forward pass with device hook...")

cpu_ops_caught = []

def make_hook(name):
    def hook(module, input, output):
        for i, t in enumerate(input):
            if isinstance(t, torch.Tensor) and t.device.type == "cpu":
                cpu_ops_caught.append(f"{name} — input[{i}] is on CPU")
        if isinstance(output, torch.Tensor) and output.device.type == "cpu":
            cpu_ops_caught.append(f"{name} — output is on CPU")
    return hook

# Register hooks on all submodules
hooks = []
for name, module in model.named_modules():
    h = module.register_forward_hook(make_hook(name))
    hooks.append(h)

# Run a tiny generation (just 3 tokens to keep it fast)
print("    Generating 3 tokens — watching for CPU tensors...")
try:
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
    print(f"    Output device: {out.device}")
except Exception as e:
    print(f"    xxxx Error during generate: {e} xxxx ")

# Remove hooks
for h in hooks:
    h.remove()

# ────────────────────── Step 4: report findings ───────────────────────────────────────────────────
print(f"\n[4] CPU fallback ops found: {len(cpu_ops_caught)}")
if cpu_ops_caught:
    print("    First 10 offending ops:")
    for op in cpu_ops_caught[:10]:
        print(f"   xxxxxx  {op}")
else:
    print("     No CPU fallback detected — generation should be on XPU")

print("\nDone.")
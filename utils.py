import torch


def get_device():
    """
    Returns the best available device string.
    Priority: XPU (Intel) > CUDA (NVIDIA) > CPU
    """
    if torch.xpu.is_available():
        print(" Intel XPU detected")
        return "xpu"
    elif torch.cuda.is_available():
        print(" NVIDIA CUDA detected")
        return "cuda"
    else:
        print("  No GPU found, falling back to CPU (will be slow)")
        return "cpu"


def print_device_info():
    """Prints a summary of your hardware."""
    device = get_device()
    if device == "xpu":
        print(f"   Device name : {torch.xpu.get_device_name(0)}")
        print(f"   Device count: {torch.xpu.device_count()}")
    elif device == "cuda":
        print(f"   Device name : {torch.cuda.get_device_name(0)}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   VRAM        : {total_mem:.1f} GB")
    return device


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Device Check ===")
    print_device_info()

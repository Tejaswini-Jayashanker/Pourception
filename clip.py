"""
clip.py — Vision Encoder 
---------------------------------------
Uses OpenAI's CLIP model (free, on HuggingFace) to convert an alcohol
image into a 512-dimensional feature vector (embedding).

"""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from utils import get_device

from huggingface_hub import login
login(token="YOUR_HF_TOKEN_HERE")

# ────────────────────────── Load model once at import time ────────────────────────────
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

print("Loading CLIP vision encoder...")

clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID)
clip_model = clip_model.to(device)
clip_model.eval()

print(f" CLIP loaded on {device}")

# ────────────────────────────── Main function ─────────────────────────────────────────────────────────────
def get_image_features(image: Image.Image) -> torch.Tensor:
    """
    Takes a PIL Image, returns a 512-dim feature tensor.
 
    Args:
        image: PIL.Image.Image — your alcohol photo
 
    Returns:
        torch.Tensor of shape [1, 512]
    """
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
 
    with torch.no_grad():
        output = clip_model.get_image_features(**inputs)
 
    # get_image_features() can return either a plain tensor or a
    # ModelOutput object depending on the transformers version.
    # We always pull out the raw tensor before doing math on it.
    if hasattr(output, "image_embeds"):
        features = output.image_embeds        # ModelOutput path
    elif hasattr(output, "pooler_output"):
        features = output.pooler_output       # BaseModelOutputWithPooling path
    else:
        features = output                     # already a plain tensor
 
    # Normalise so values sit on a unit sphere (-1 to 1 range)
    features = features / features.norm(dim=-1, keepdim=True)
    return features
 
 
def load_image(image_path: str) -> Image.Image:
    """Helper: load an image from a file path and convert to RGB."""
    return Image.open(image_path).convert("RGB")
 
 
# ────────────────────────────── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
 
    # Use a provided path or create a tiny test image
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        image = load_image(img_path)
        print(f"Testing with image: {img_path}")
    else:
        # Create a blank test image (100x100 white square)
        print("No image path given — using a blank test image")
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))
 
    features = get_image_features(image)
    print(f" CLIP works! Feature tensor shape: {features.shape}")
    print(f"   First 5 values: {features[0, :5].tolist()}")

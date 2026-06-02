# 🍾 Pourception -  Alcohol Vision + LLM Assistant

A multimodal AI assistant that identifies alcohol from images and answers questions about origin, production, benefits, and health risks.

Built with HuggingFace open-source models, PyTorch or PyTorch with XPU.
---

## How it works

```
📷 Image  ──►  CLIP vision encoder  ──►  image embeddings
                                                │
💬 Question ─────────────────────────►  Prompt builder
                                                │
                                         LLaVA / Phi-3.5
                                                │
                                    ◄──  Answer to user
```


## Project structure

```
alcohol-vision-llm/
├── utils.py                    ← device detection (XPU / CUDA / CPU)
├── clip.py                     ← CLIP vision encoder
├── phi3p5-instruct.py          ← multimodal LLM loader
├── prompt.py                   ← full pipeline + Gradio UI  ← run this
├── requirements.txt
└── example_output.png
```

---

## Setup

### 1. Install PyTorch with XPU support

Visit https://pytorch-extension.intel.com/installation and follow the instructions for your OS. It will give you a command like:

### 2. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify your device

```bash
python utils.py
```

---

## Running the app

```bash
python prompt.py
```

Opens a Gradio UI at http://localhost:7860

---

## Testing individual modules

```bash
python utils.py               # check XPU is detected
python clip.py                # test vision encoder
python clip.py my_image.jpg   # test with a real image
python phi3p5-instruct.py     # test model loads onto XPU
```

---

## Model options

Edit `MODEL_ID` in `phi3p5-instruct.py` to switch models:

| Model | VRAM (float16) | Notes |
|---|---|---|
| `microsoft/Phi-3.5-vision-instruct` | ~7 GB | ✅ Default — best for XPU |
| `Qwen/Qwen2-VL-2B-Instruct` | ~4 GB | Fastest |
| `llava-hf/llava-v1.6-mistral-7b-hf` | ~14 GB | Most capable |

---

## Example questions

- "What alcohol is this and where is it from?"
- "How is this drink made?"
- "What are the health risks of drinking this?"
- "What food does this pair well with?"
- "What is the history of this spirit?"

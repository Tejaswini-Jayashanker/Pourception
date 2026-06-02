"""
prompt.py — Full Pipeline + Gradio UI
----------------------------------------------------------
Run with:
    python prompt.py

Opens Gradio UI at http://localhost:7860
"""

import torch
from PIL import Image
import gradio as gr

from phi3p5-instruct import model, processor, device

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert sommelier, spirits historian, and health educator.
When shown an image of an alcohol bottle, label, or drink, you always:
1. Identify the type and brand of alcohol
2. Describe its country or region of origin
3. Briefly explain how it is made
4. List 2-3 moderate cultural or culinary benefits
5. List 2-3 honest health risks and harms
Be factual, educational, balanced, and friendly."""


# ── Core function ─────────────────────────────────────────────────────────────
def ask_about_alcohol(image: Image.Image, question: str) -> str:
    """
    Takes a PIL image and a text question, returns the model's answer.
    Uses Phi-3.5-vision's native prompt format with <|image_1|> token.
    """
    if image is None:
        return "!!! Please upload an image first !!! "

    if not question.strip():
        question = "What alcohol is this? Tell me its origin, how it's made, any benefits, and health risks."

    # Phi-3.5 uses <|image_1|> as the image placeholder in the prompt.
    # The format is: <|user|>\n<|image_1|>\nQuestion<|end|>\n<|assistant|>
    prompt_text = (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n<|image_1|>\n{question}<|end|>\n"
        f"<|assistant|>\n"
    )

    # Process image + text into tensors together
    inputs = processor(
        text=prompt_text,
        images=image,
        return_tensors="pt",
    ).to(device)

    # Generate answer
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    # Slice off the input prompt — only decode the new tokens (the answer)
    answer_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    return answer.strip()


# ── Gradio UI ─────────────────────────────────────────────────────────────────
def build_ui():
    with gr.Blocks(title="🍾 Alcohol Vision Assistant") as demo:
        gr.Markdown("# 🍾 Alcohol Vision Assistant")
        gr.Markdown(
            "Upload a photo of any alcohol — bottle, label, or glass — "
            "and ask anything about it."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="Upload alcohol image",
                )
                question_input = gr.Textbox(
                    label="Your question",
                    placeholder="What country is this from? What are the health risks?",
                    lines=2,
                )
                submit_btn = gr.Button("Analyse 🔍", variant="primary")

            with gr.Column():
                answer_output = gr.Textbox(
                    label="AI Answer",
                    lines=15,
                )

        gr.Examples(
            examples=[
                [None, "What type of alcohol is this and where is it from?"],
                [None, "How is this alcohol made?"],
                [None, "What are the health risks of drinking this?"],
                [None, "What food does this pair well with?"],
            ],
            inputs=[image_input, question_input],
        )

        submit_btn.click(
            fn=ask_about_alcohol,
            inputs=[image_input, question_input],
            outputs=answer_output,
        )

    return demo


# ────────────────────────────────────────── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n !=!=!=! Pourception - Your Alcohol Vision Assistant !=!=!=! ")
    print(f"Device: {device}")
    print("Starting Gradio UI...")

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )
"""Gradio side-by-side demo: GPT-2 (LoRA) vs TinyLlama (QLoRA).

    python app/demo.py --share

The `DomainChatModel` class is imported, not redefined. The previous version of
this file carried its own copy that had drifted from the training notebook's, so
the demo silently generated with different settings than the evaluation measured.

Adapters resolve from `$HF_USER` (see `medqa.config`), so a clone of this repo
loads *your* adapters, not someone else's.
"""

from __future__ import annotations

import argparse
import json

import gradio as gr

from medqa import config, models

DISCLAIMER = (
    "⚠️ **Not medical advice.** This is an educational demonstration of LLM "
    "fine-tuning. Both models are small, were trained for one epoch, and will state "
    "false things fluently. Do not use their output for any health decision."
)


def measured_results() -> str:
    """Report what was actually measured, or say plainly that nothing was."""
    if not config.METRICS_PATH.exists():
        return "_No evaluation on record — run `python -m medqa.evaluate` to populate this._"
    runs = json.loads(config.METRICS_PATH.read_text())
    rows = " · ".join(f"**{name}** {m['bits_per_byte']:.3f}" for name, m in sorted(runs.items()))
    return (
        f"Held-out bits per byte (lower is better): {rows}\n\n"
        "_Bits per byte, not perplexity — the two models use different tokenizers, "
        "so per-token perplexity is not comparable between them._"
    )


def build_demo(preset: models.GenerationPreset = models.DEMO) -> gr.Blocks:
    print("loading models…")
    gpt2 = models.DomainChatModel(config.GPT2, preset=preset)
    tinyllama = models.DomainChatModel(config.TINYLLAMA, preset=preset)
    print("ready ✅")

    def compare(question: str) -> tuple[str, str]:
        if not question.strip():
            return "Please enter a question.", "Please enter a question."
        return gpt2.generate(question), tinyllama.generate(question)

    with gr.Blocks(title="Medical Q&A: GPT-2 vs TinyLlama") as demo:
        gr.Markdown("# 🩺 Medical Q&A — GPT-2 vs TinyLlama (LoRA / QLoRA)")
        gr.Markdown("Same medical dataset, two architectures, two adaptation methods.")
        gr.Markdown(measured_results())
        question = gr.Textbox(
            label="Ask a medical question",
            value="What are the symptoms of Tourette syndrome?",
        )
        button = gr.Button("Generate", variant="primary")
        with gr.Row():
            left = gr.Textbox(label="GPT-2 (124M · LoRA baseline)", lines=8)
            right = gr.Textbox(label="TinyLlama (1.1B · QLoRA)", lines=8)
        button.click(compare, inputs=question, outputs=[left, right])
        gr.Markdown(DISCLAIMER)

    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share", action="store_true", help="public gradio.live URL (~72h)")
    parser.add_argument("--preset", default="demo", choices=sorted(models.PRESETS))
    args = parser.parse_args(argv)
    build_demo(models.PRESETS[args.preset]).launch(share=args.share)


if __name__ == "__main__":
    main()

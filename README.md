# LoRA vs QLoRA on medical Q&A

Fine-tuning two small language models on the same medical Q&A dataset, and measuring
honestly whether the adaptation actually did anything.

- **GPT-2** (124M) with **LoRA** — the baseline arm
- **TinyLlama-1.1B-Chat** with **QLoRA** (4-bit NF4, double-quantised) — the treatment arm
- Dataset: [`keivalya/MedQuad-MedicalQnADataset`](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) — 16,407 Q&A pairs, split 90/10 with a fixed seed
- Adapters: [`gpt2-medqa-lora`](https://huggingface.co/Babblu2821/gpt2-medqa-lora) · [`tinyllama-medqa-qlora`](https://huggingface.co/Babblu2821/tinyllama-medqa-qlora)

> ⚠️ **Not medical advice.** This is an educational demonstration of fine-tuning
> technique. Both models are small, trained for a single epoch, and will state false
> things fluently and confidently. Nothing here should inform any health decision.

---

## The measurement problem this project is really about

The first version of this work reported a headline result:

> GPT-2 perplexity 5.99 → TinyLlama 2.80, a 53% reduction.

**That comparison does not hold up, for two independent reasons.**

**1. Perplexity is not comparable across different tokenizers.** Perplexity is a
per-*token* quantity. GPT-2 uses byte-level BPE with a 50,257-piece vocabulary;
TinyLlama uses SentencePiece with 32,000. The same sentence becomes a different number
of tokens in each, so the two perplexities have different denominators. They were never
on one scale, and the ratio between them means very little.

The fix is **bits per byte** — total negative log-likelihood divided by the UTF-8 byte
count of the *same* reference answers. Bytes are identical regardless of tokenizer, so
the number is genuinely comparable across models.

**2. There was no control condition.** The original comparison was fine-tuned GPT-2
against fine-tuned TinyLlama, with the difference credited to QLoRA. But TinyLlama is
roughly 9× larger and was already instruction-tuned before this project touched it.
Nothing in that design separates *"QLoRA worked"* from *"the bigger model was already
better."*

The fix is to also score both **base models, untouched**. That turns one number into a
2×2, and the only interesting quantity — how much fine-tuning moved each model — becomes
measurable. Both controls are forward passes over the eval set: no training, minutes not hours.

<!-- RESULTS:BEGIN -->
## Results

Held-out set: 1641 examples, identical rows for every arm. Lower bits/byte is better.
Perplexity is shown only for continuity with the original report — compare it *within*
a model, never across the two.

| run | fine-tuned | bits/byte ↓ | perplexity | eval rows |
|---|---|---|---|---|
| `gpt2-base` | — (control) | 0.8049 | 11.51 | 1641 |
| `gpt2-lora` | yes | 0.5970 | 6.12 | 1641 |

- **gpt2**: fine-tuning cut bits/byte by **25.8%** vs its own base model.

Measured 2026-08-07 by `python -m medqa.evaluate`, read from `outputs/metrics.json`.
Regenerate this block with `python scripts/update_readme_results.py`.

<!-- RESULTS:END -->

---

## Reproducing this

```bash
git clone https://github.com/fayazhussain2821/llm-finetuning-medqa
cd llm-finetuning-medqa
uv venv && uv pip install -r requirements.txt && uv pip install -e .
```

Training needs a CUDA GPU (QLoRA's 4-bit kernels are CUDA-only). Open
`notebooks/colab_train.ipynb` in Colab on a T4 — it clones this repo and calls into the
package, so the notebook holds no logic of its own.

```bash
python -m medqa.train --model gpt2          # LoRA
python -m medqa.train --model tinyllama     # QLoRA, 4-bit
```

Evaluation runs anywhere, including Apple Silicon (MPS) and CPU:

```bash
python -m medqa.evaluate --model gpt2                 # fine-tuned
python -m medqa.evaluate --model gpt2      --base     # control
python -m medqa.evaluate --model tinyllama
python -m medqa.evaluate --model tinyllama --base
python -m medqa.evaluate --table                      # print what was measured
```

Every result is computed into `outputs/metrics.json`; the tables above and below are
rendered from that file by `python -m medqa.evaluate --markdown`. No figure in this
README was typed by hand.

The demo:

```bash
python app/demo.py --share
```

## How it is put together

```
src/medqa/
  config.py     every constant, plus the ModelSpec for each arm
  data.py       load → format → seeded split → tokenize
  models.py     DomainChatModel and the named generation presets
  train.py      one CLI, both arms
  evaluate.py   scoring → metrics.json
app/demo.py     Gradio side-by-side
notebooks/
  colab_train.ipynb   thin driver — orchestration only
  original/           the three original Colab notebooks, kept verbatim
tests/
```

`notebooks/original/` is deliberately frozen, outputs and all. Those outputs are the
record of what the first version actually produced, including the numbers this project
went on to correct. They are exempted from `nbstripout` in `.gitattributes`.

Set `HF_USER` to point the demo and the push targets at your own Hub account:

```bash
export HF_USER=your-username
```

## Method notes

**Both arms train under the same `Trainer`.** The original ran GPT-2 through
`transformers.Trainer` and TinyLlama through TRL's `SFTTrainer`. Since the data is
pre-tokenized either way, that gave the two arms of a comparison different training
machinery for no benefit. Everything that genuinely differs between them — 4-bit loading,
optimiser, batch shape — lives in a `ModelSpec` in `config.py`.

**Effective batch size is pinned equal across arms** at 16 (GPT-2: 8×2, TinyLlama: 4×4).
The physical shapes differ to fit VRAM; the number of optimisation steps does not.
LoRA rank and alpha are shared too, so rank is not confounded with method. A test fails
if anyone tunes one side without the other.

**Scoring is restricted to the answer span.** GPT-2 sees an `### Instruction:` template
and TinyLlama sees its native chat template. Scoring the full sequence would let the
template — boilerplate neither model had to work for — move the metric.

**Evaluation runs during training**, not only at the end, so overfitting is visible
while there is still time to stop.

## Limitations

- **One epoch, one seed, one run per arm.** No variance estimate, so small differences
  between arms should not be read as real.
- **Perplexity and bits-per-byte measure likelihood, not correctness.** A model can
  score well by matching the dataset's register and phrasing while still being wrong.
  Nothing here evaluates factual accuracy, and for medical content that is the gap
  that matters most.
- **MedQuAD is authoritative but narrow** — NIH-sourced, US-centric, and frozen at
  collection time. Neither model knows anything more recent.
- **~5% of examples exceed GPT-2's 1024-token context** and are truncated. That ceiling
  is architectural for the baseline arm.
- **The two arms differ in more than the adaptation method.** TinyLlama is ~9× larger
  *and* was already instruction-tuned. The base-model controls are what make the
  fine-tuning effect within each model attributable; the cross-model gap is not.
- Evaluated on Apple Silicon in fp16 rather than 4-bit, since bitsandbytes requires
  CUDA. Both TinyLlama rows are measured identically, so the within-model comparison
  holds, but the figures are not bit-identical to a 4-bit Colab run.

## License

Code: MIT. The dataset and base models carry their own licenses.

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
| `tinyllama-base` | — (control) | 0.6120 | 5.39 | 1641 |
| `tinyllama-qlora` | yes | 0.3954 | 2.97 | 1641 |

- **gpt2**: fine-tuning cut bits/byte by **25.8%** vs its own base model.
- **tinyllama**: fine-tuning cut bits/byte by **35.4%** vs its own base model.

Measured 2026-08-07 by `python -m medqa.evaluate`, read from `outputs/metrics.json`.
Regenerate this block with `python scripts/update_readme_results.py`.

<!-- RESULTS:END -->

### What the controls change

**The original conclusion survives, but not the number, and not the reasoning.**

TinyLlama + QLoRA really is the better model here, and fine-tuning really did help
it — those parts hold. Three things look different once the controls exist:

**The headline was inflated.** The reported 53% reduction was a perplexity ratio
across two tokenizers. On bits per byte, fine-tuned GPT-2 to fine-tuned TinyLlama is
a **33.8%** reduction. Real, but two-thirds the advertised size; the rest was an
artifact of counting tokens in different units.

**Most of the cross-model gap was never about QLoRA.** Untouched TinyLlama scores
0.6120 — within a few percent of *fully fine-tuned* GPT-2 at 0.5970. A model that
never saw one row of MedQuAD essentially matches the baseline arm's finished result,
because it is ~9× larger and was instruction-tuned before this project began.
Attributing that gap to the adaptation method was the error the missing control hid.

**But QLoRA did do more work, and now that is separable.** Fine-tuning moved TinyLlama
35.4% and GPT-2 25.8%, each against its own starting point. That comparison is
attributable in a way the original 5.99-vs-2.80 never was — it holds the model fixed
and varies only the training.

So: the right claim is not *"QLoRA cut perplexity 53%"* but *"QLoRA adapted the larger
model further than LoRA adapted the smaller one, on a base that was already ahead."*
That is a weaker sentence and a true one.

As a sanity check on the rewrite, the fine-tuned perplexities here (6.12 and 2.97)
land close to the originally reported 5.99 and 2.80, despite scoring only the answer
span. The pipeline reproduces the original run; it just refuses to compare the two
numbers the way the original did.

### But does it answer the question?

Everything above is likelihood: how much probability an arm put on the reference
answer. It never asks the model to write one. That hides two failures — a model that
loops, and a model that writes fluent prose about the wrong condition — because
neither costs anything when you are only scoring the reference tokens.

So each arm was made to answer, greedily, and the text was compared with the reference.

<!-- QUALITY:BEGIN -->
Greedy decoding, ≤200 new tokens, the first 200 held-out rows —
the same questions for every arm. Likelihood says how plausible the reference was;
these say what the model actually wrote when asked.

| run | ROUGE-L F1 ↑ | token F1 ↑ | repeated 4-grams ↓ | length ratio | empty | n |
|---|---|---|---|---|---|---|
| `gpt2-base` | 0.0797 | 0.1666 | 0.0000 | 0.71 | 0.0% | 200 |
| `gpt2-lora` | 0.0971 | 0.2060 | 0.0005 | 0.80 | 0.0% | 200 |
| `tinyllama-base` | 0.1548 | 0.2718 | 0.0101 | 0.55 | 0.0% | 200 |
| `tinyllama-qlora` | 0.2337 | 0.3435 | 0.0146 | 0.52 | 0.0% | 200 |

Measured 2026-08-14 by `python -m medqa.quality`. Every generation is kept in
`outputs/generations/<run>.jsonl` — the summary is a number, that file is the evidence.

<!-- QUALITY:END -->

**The two metrics rank the arms differently, and that is the point.**

On likelihood, fine-tuned GPT-2 (0.5970 bits/byte) edges out *untouched* TinyLlama
(0.6120). Ask them both to actually write an answer and the order reverses, hard: the
untouched model beats the fully fine-tuned one by **59% on ROUGE-L** and **32% on token
F1**. A whole epoch of LoRA on MedQuAD does not buy GPT-2 what TinyLlama had before the
project started. Likelihood made that gap look like a tie; generation shows it is not.

Everything else the controls established still holds, and holds more strongly.
Fine-tuning lifts ROUGE-L **+21.8%** for GPT-2 and **+51.0%** for TinyLlama, each
against its own base — the same ordering as bits per byte, measured a completely
different way. Two independent metrics agreeing on the within-model effect is worth
more than either one alone.

**What the degeneration numbers say: nothing is looping.** Repeated 4-grams sit at
0.0000–0.0146 everywhere, so the "model stuck in a cycle" failure this metric exists to
catch did not occur in any arm. That is a real negative result, and it narrows things
usefully — the models' problem is not incoherence, it is being wrong. Reading
`outputs/generations/gpt2-lora.jsonl` shows what that looks like: fluent, well-formed,
and claiming Tourette syndrome causes "severe muscle weakness, paralysis". No automatic
metric here penalises that. A human reading the file is still the only thing that does.

**Two caveats on these numbers.** Length ratio is confounded with the 200-token
generation cap. 163 of GPT-2's 200 answers run to within a few words of the ceiling
because it rarely emits a stop token, against 12 of TinyLlama's — so 0.80-vs-0.52
measures *stopping behaviour* far more than verbosity, and should not be read as
GPT-2 matching the reference length more closely. And ROUGE-L rewards phrasing that
matches MedQuAD's house style, which fine-tuning teaches directly; part of the
within-model gain is register, not knowledge.

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

Answer quality generates rather than scores, so it is slower and runs on a subset:

```bash
python -m medqa.quality --model gpt2                  # fine-tuned
python -m medqa.quality --model gpt2       --base     # control
python -m medqa.quality --model tinyllama
python -m medqa.quality --model tinyllama  --base
python -m medqa.quality --table                       # print what was measured
```

Every result is computed into `outputs/metrics.json`, and every generation into
`outputs/generations/`. The tables above and below are rendered from that file by
`python scripts/update_readme_results.py`. No figure in this README was typed by hand.

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
  evaluate.py   likelihood scoring → metrics.json
  quality.py    generate, then ROUGE-L / token-F1 / degeneration → metrics.json
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
- **Nothing here evaluates factual accuracy.** Bits per byte measures likelihood;
  ROUGE-L and token F1 measure word overlap with one reference answer. All three
  reward matching the dataset's register and phrasing, and none of them can tell a
  correct medical claim from a fluent false one. For medical content that is the gap
  that matters most, and it is still open. `outputs/generations/` exists so the
  answers can at least be read.
- **One reference answer per question.** A correct answer phrased differently, or a
  correct answer MedQuAD happens not to give, scores as a miss.
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

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

Measured 2026-08-15 by `python -m medqa.evaluate`, read from `outputs/metrics.json`.
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
0.6120, against 0.5970 for *fully fine-tuned* GPT-2 — a 2.5% gap (interval below). A
model that never saw one row of MedQuAD comes within a few percent of the baseline
arm's finished result, because it is ~9× larger and was instruction-tuned before this
project began. Attributing that gap to the adaptation method was the error the missing
control hid.

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

### Reading the answers: the only measurement that sees factuality

ROUGE-L cannot tell a true medical claim from a fluent false one. So 20 held-out
questions were sampled, all four answers to each were **shuffled and stripped of
their labels**, and every answer was rated 1–5 for factual soundness against the
reference. `python -m medqa.review --sample 20` builds the sheet; the unblinding
key is written to a separate file that `--report` reads afterwards.

> **These ratings are a human pass.** All 80 answers were rated by the author against
> the reference, blind to which arm produced them. An earlier LLM-judge pass over the
> same sheet is kept at `outputs/review/ratings.claude.json` for comparison; the two
> are contrasted below. Neither rater is a clinician.

| run | mean 1–5 ↑ | sd | contradicts reference (1–2) ↓ | n |
|---|---|---|---|---|
| `gpt2-base` | 1.60 | 0.92 | 70% | 20 |
| `gpt2-lora` | 1.80 | 0.98 | 60% | 20 |
| `tinyllama-base` | 2.90 | 1.37 | 50% | 20 |
| `tinyllama-qlora` | 3.25 | 1.18 | 35% | 20 |

Every arm answered the same 20 questions, so the comparisons are paired. The
interval matters more than the mean at this sample size:

| comparison | mean Δ | 95% CI | W/L/T | detected? |
|---|---|---|---|---|
| `gpt2-lora` − `gpt2-base` | +0.20 | [−0.40, +0.80] | 6/4/10 | **no** |
| `tinyllama-qlora` − `tinyllama-base` | +0.35 | [−0.20, +0.90] | 7/3/10 | **no** |
| `tinyllama-base` − `gpt2-lora` | +1.10 | [+0.35, +1.85] | 13/4/3 | yes |
| `tinyllama-qlora` − `gpt2-lora` | +1.45 | [+0.80, +2.10] | 17/2/1 | yes |

**Neither fine-tuning run produced a detectable improvement in factual soundness.**
Both within-model intervals span zero. That is the sharpest disagreement in this
project: bits per byte says fine-tuning helped by 25.8% and 35.4%, ROUGE-L says
+21.8% and +51.0%, and on whether the answers are *true*, twenty questions cannot
detect any effect at all. Both point estimates are positive but small — LoRA taught
these models MedQuAD's register, and the register is what the automatic metrics score.

**What is unambiguous is the thing the project was controlling for, not testing.**
Untouched TinyLlama beats fully fine-tuned GPT-2 by +1.10 (13 wins, 4 losses), and
fine-tuned TinyLlama beats it by +1.45, winning 17 of 20 and losing 2. Model
choice dominates adaptation method so completely that the adaptation method is not
measurable underneath it.

**The absolute numbers are the real story.** `gpt2-base` contradicts the reference
or invents an entity in **70%** of its answers, `gpt2-lora` in 60%. The best arm
still does so in 35%.
Typical failures: Marfan syndrome attributed to "an infection", congenital stromal
corneal dystrophy attributed to `COL4A1` (it is `DCN`), Chagas disease transmitted
by "ticks or fleas", a fabricated `FHNV` gene, and a citation to a Johns Hopkins
doctor who does not appear to exist. All fluent. All confidently phrased. None of
it is visible in any other number on this page.

"Detected no difference" is not "there is no difference" — 20 questions is a small
instrument, and the +0.35 for TinyLlama may well be real and simply unresolved here.

**What the LLM judge got right and wrong.** The same 80 answers were rated first by
an LLM and then by a human, blind, on the same sheet. The two raters disagreed on
**36 of 80 individual scores**, and the LLM was consistently harsher — it put
`gpt2-lora` at 1.20 mean / 95% contradiction where the human pass says 1.80 / 60%,
and it alone ranked `gpt2-lora` *below* its own base model. Yet all four verdicts in
the table above are identical under both raters: neither fine-tuning effect is
detectable, and both cross-model gaps are. The one ranking they disagree on is the
one neither can resolve — `gpt2-lora` against `gpt2-base`, where the interval spans
zero either way. The judge was a reliable guide to *which comparisons resolve* and an
unreliable guide to *how bad the answers are* — worth knowing before quoting an
LLM-judged absolute rate as a finding.

### How much of this is noise?

Every number above is a point estimate. A gap is not a result until you know whether a
rerun would reproduce it, and there are two independent reasons it might not:

1. **Which questions were held out.** The 1,641 eval rows are a sample; a different
   draw would move the numbers. Measured below by resampling.
2. **Which training run happened.** Data order, LoRA initialisation, dropout. Only
   retraining measures this — see the caveat after the table.

<!-- VARIANCE:BEGIN -->
Bootstrap over the held-out rows, 10,000 resamples. Comparisons are
**paired** — one resample of row indices applied to both arms, since every arm was
scored on the same questions.

| run | bits/byte | 95% CI (eval sample) | n |
|---|---|---|---|
| `gpt2-base` | 0.8049 | [0.7977, 0.8121] | 1641 |
| `gpt2-lora` | 0.5970 | [0.5850, 0.6088] | 1641 |
| `tinyllama-base` | 0.6120 | [0.6059, 0.6185] | 1641 |
| `tinyllama-qlora` | 0.3954 | [0.3860, 0.4046] | 1641 |

| comparison | reduction | 95% CI | survives resampling? |
|---|---|---|---|
| `gpt2-base` → `gpt2-lora` | 25.8% | [24.3%, 27.5%] | yes |
| `tinyllama-base` → `tinyllama-qlora` | 35.4% | [33.8%, 37.1%] | yes |
| `gpt2-lora` → `tinyllama-qlora` | 33.8% | [33.1%, 34.5%] | yes |
| `gpt2-lora` → `tinyllama-base` | -2.5% | [-5.0%, -0.2%] | yes |

Produced by `python -m medqa.variance`. These intervals cover the eval sample only;
run-to-run spread across training seeds is a separate experiment.

<!-- VARIANCE:END -->

**Every headline gap survives resampling the questions.** 25.8% and 35.4% within each
model, and 33.8% between the fine-tuned arms, all with intervals comfortably clear of
zero. On this axis the results are solid.

**One case shows why the pairing matters.** The marginal intervals for `gpt2-lora`
([0.5850, 0.6088]) and `tinyllama-base` ([0.6059, 0.6185]) *overlap*, which by the
usual eyeball test would mean "no detectable difference". Paired, the difference is
detectable: fine-tuned GPT-2 really is ahead of untouched TinyLlama on likelihood, by
2.5% [0.2%, 5.0%]. Overlapping error bars are not a test, and comparing arms
independently when they answered identical questions throws away the pairing that
makes the comparison sharp.

**And that sharpens the contradiction.** On likelihood, `gpt2-lora` beats
`tinyllama-base` by a small but *statistically detectable* margin. On generated
answers, `tinyllama-base` beats `gpt2-lora` by 59% on ROUGE-L. On factual soundness it
wins by +1.10 [+0.55, +1.65]. The metrics do not merely disagree about size — they
reverse the ranking, and the reversal is detectable in both directions.

> **Seed variance is still unmeasured.** These intervals cover the eval sample only. A
> gap can clear them comfortably and still vanish under a different training seed.
> `python -m medqa.train --model gpt2 --seed 43` measures that, needs a CUDA GPU, and
> has not been run — so "one epoch, one seed, one run per arm" remains a live
> limitation, now narrowed to exactly one source of uncertainty instead of two.

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

How much of any gap is noise, in two parts:

```bash
python -m medqa.variance                     # intervals over the held-out sample
python -m medqa.train --model gpt2 --seed 43 # a variance run — needs a CUDA GPU
python -m medqa.evaluate --model gpt2 --seed 43
```

`--seed` moves the training run and nothing else; the split is pinned to
`config.SPLIT_SEED` so every run is scored on identical rows.

Factual soundness needs a person, so it is two commands with reading in between:

```bash
python -m medqa.review --sample 20    # blinded sheet → outputs/review/sheet.md
# ... read it, fill in outputs/review/ratings.json ...
python -m medqa.review --report       # unblind, aggregate, paired intervals
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
  review.py     blinded human rating → paired intervals → metrics.json
  variance.py   bootstrap intervals on the eval sample; spread across train seeds
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
- **No automatic metric here evaluates factual accuracy.** Bits per byte measures
  likelihood; ROUGE-L and token F1 measure word overlap with one reference answer.
  All three reward matching the dataset's register and phrasing, and none can tell a
  correct medical claim from a fluent false one. Only the blinded review above looks
  at truth, and it is 20 questions rated by the author, not a clinician.
- **The factual ratings are one non-expert rater.** Blind to arm, but a single pass
  by the repo's author against MedQuAD's reference text — not adjudicated by a second
  rater, and not a clinical judgement. Inter-rater spread is unmeasured.
- **Every arm is factually unreliable.** The best of them contradicts the reference
  or invents an entity in 35% of answers; the worst, in 70%. Nothing in this repo
  produces a model that should be read for medical content.
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

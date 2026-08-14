# ROADMAP — `llm-finetuning-medqa`

A step-by-step plan to take this project from three Colab notebooks to a
version-controlled, tested, reproducible repo that still trains on Colab's GPU.

**Working principle throughout:** `main` is never pushed to directly. Every change
goes through a branch → PR → verification → merge. Each phase below ends with a
**Verify** gate you must pass before opening the PR.

---

## Table of contents

| Phase | What it covers | Time |
|---|---|---|
| [0](#phase-0--what-is-already-done) | Machine setup already completed | done |
| [1](#phase-1--local-environment) | Python env, dependencies, pinning | 30 min |
| [2](#phase-2--repo--pr-workflow) | Git, GitHub, branch protection, PR flow | 45 min |
| [3](#phase-3--restructure-notebooks--package) | Notebooks → importable package | 3–4 h |
| [4](#phase-4--verification-tests--ci) | Tests, linting, GitHub Actions | 2 h |
| [5](#phase-5--colab-integration) | Colab pulls from GitHub | 1 h |
| [6](#phase-6--fix-the-science) | The methodological problems | 4–6 h |
| [7](#phase-7--ship-it) | HF Spaces, README, model cards | 2 h |

---

## Phase 0 — What is already done

These were completed on 2026-08-06. Recorded here so the setup is reproducible on
another machine.

### 0.1 Tooling installed

```bash
brew install rclone gh
```

- **`gh`** — GitHub's CLI. Chosen over a GitHub MCP server because `gh auth setup-git`
  wires git's credential helper, so `git push` moves a whole repo with history.
  An MCP server only reaches the API, which means file-by-file uploads and no history.
- **`rclone`** — pulls files from Google Drive selectively. Preferred over Google Drive
  Desktop, which insists on syncing whole folders.

### 0.2 GitHub authenticated

```bash
gh auth login          # GitHub.com → HTTPS → authenticate git → browser/token
gh auth setup-git      # wires git credential helper
git config --global user.email "230100932+fayazhussain2821@users.noreply.github.com"
```

The noreply address keeps your Gmail out of public commit history. GitHub still
attributes the commits to you.

> ⚠️ **Your token lacks the `workflow` scope.** Current scopes: `gist`, `read:org`, `repo`.
> Phase 4 pushes a GitHub Actions file, which will be **rejected** without it. Fix before Phase 4:
> ```bash
> gh auth refresh -s workflow
> ```

### 0.3 Google Drive connected

```bash
rclone config create gdrive drive scope=drive >/dev/null 2>&1
```

The `>/dev/null` matters — without it rclone prints your OAuth **refresh token** to
the terminal, and that token grants full Drive access until revoked.

> ⚠️ **Action item:** the token was printed during the original setup. Revoke rclone at
> [myaccount.google.com/permissions](https://myaccount.google.com/permissions), then
> re-run the command above with the redirect.

> ⚠️ rclone's shared OAuth client ID **stops working during 2026**. Register your own
> before then: [rclone.org/drive/#making-your-own-client-id](https://rclone.org/drive/#making-your-own-client-id)

Useful rclone commands:

```bash
rclone lsd gdrive:                        # list top-level folders
rclone ls "gdrive:Colab Notebooks"        # list files
rclone copy "gdrive:Colab Notebooks/LLM Fine-Tuning" ./notebooks/
rclone copy ./out "gdrive:backup" -P      # push back, with progress
```

📚 [rclone Drive docs](https://rclone.org/drive/) · [rclone filtering](https://rclone.org/filtering/)

### 0.4 Notebooks pulled

```
~/Fayaz_Repo/llm-finetuning-medqa/notebooks/
├── Training.ipynb        # 286 KB — the 5-phase pipeline
├── Loading Models.ipynb  #  10 KB — HF Hub upload utility
└── Live Demo.ipynb       #  89 KB — Gradio comparison app
```

---

## Phase 1 — Local environment

**Goal:** a reproducible Python environment where you can run everything except
GPU training.

### Step 1.1 — Install `uv`

```bash
brew install uv
```

`uv` replaces pip/venv/pip-tools. It resolves and installs 10–100× faster, and its
lockfile gives you genuinely reproducible installs — which matters here, because
`transformers` + `peft` + `trl` + `bitsandbytes` break against each other regularly.

📚 [uv docs](https://docs.astral.sh/uv/)

### Step 1.2 — Create the environment

```bash
cd ~/Fayaz_Repo/llm-finetuning-medqa
uv venv --python 3.11
source .venv/bin/activate
```

> **Why 3.11, not your system 3.13?** As of early 2026 `bitsandbytes` wheels lag
> behind the newest CPython. 3.11 is the safe floor for this stack, and it matches
> Colab, which is what you want — the whole point is that local and Colab agree.

### Step 1.3 — Declare dependencies

Create `pyproject.toml`:

```toml
[project]
name = "medqa-finetune"
version = "0.1.0"
requires-python = ">=3.11,<3.12"
dependencies = [
    "torch>=2.4",
    "transformers>=4.44",
    "datasets>=2.21",
    "accelerate>=0.34",
    "peft>=0.12",
    "trl>=0.10",
    "huggingface_hub>=0.24",
]

[project.optional-dependencies]
# bitsandbytes has no macOS build — GPU-only extras live here
gpu = ["bitsandbytes>=0.43"]
app = ["gradio>=4.44"]
dev = ["pytest>=8", "ruff>=0.6", "pre-commit>=3.8", "nbstripout>=0.7"]

[tool.ruff]
line-length = 100
target-version = "py311"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

```bash
uv pip install -e ".[app,dev]"
```

> **`bitsandbytes` will not install on your Mac** — there is no macOS wheel; it is
> CUDA-only. That is why it sits in the `gpu` extra. Locally you can run data prep,
> tests, GPT-2 inference, and the Gradio app. QLoRA training is Colab-only, by design.

### Step 1.4 — Lock it

```bash
uv pip compile pyproject.toml -o requirements.txt
uv pip compile pyproject.toml --extra gpu --extra app -o requirements-colab.txt
```

Two lockfiles: one for your Mac, one for Colab. Colab installs the second.

The exact commands used (note `--python-platform linux` — without it the Colab
lockfile is resolved for macOS and silently omits `bitsandbytes`):

```bash
uv pip compile pyproject.toml --extra app --extra dev -o requirements.txt
uv pip compile pyproject.toml --extra gpu --extra app \
    --python-platform linux --python-version 3.11 -o requirements-colab.txt
```

#### Verified resolution (2026-08-06)

| Package | Version |
|---|---|
| torch | 2.13.0 |
| transformers | **5.14.1** |
| trl | **1.9.2** |
| peft | 0.20.0 |
| datasets | 5.0.1 |
| bitsandbytes | 0.50.0 *(Colab lockfile only)* |

> Your notebooks were written against **transformers 4.x / trl 0.x**. Those are major
> version bumps — so each API the notebook calls was tested directly against the
> resolved stack rather than assumed. Results in §3.0 below.

**✅ Verify Phase 1**

```bash
python -c "import torch, transformers, peft, trl, datasets; print('ok')"
python -c "import torch; print('device:', 'mps' if torch.backends.mps.is_available() else 'cpu')"
```

Expect `ok` and `device: mps`. Apple Silicon gives you Metal acceleration for
inference — not enough for QLoRA training, but fine for testing generation.

---

## Phase 2 — Repo & PR workflow

**Goal:** `main` is protected. Nothing lands without a reviewed, verified PR.

### Step 2.1 — Ignore files before you ever add them

Create `.gitignore`:

```gitignore
.venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.DS_Store

# never commit weights or data
*.safetensors
*.bin
*.pt
*.pth
checkpoints/
outputs/
data/
gpt2-medqa-lora/
tinyllama-medqa-qlora/

# secrets
.env
*.token
```

> **The single most important line here is the weights block.** Git stores every
> version of every binary forever. Commit a 9 MB adapter ten times and the repo is
> 90 MB with no way to shrink it short of rewriting history. Weights belong on the
> HF Hub — where yours already are.

### Step 2.2 — Strip notebook outputs automatically

```bash
uv pip install nbstripout
nbstripout --install --attributes .gitattributes
```

`Training.ipynb` is 286 KB, and roughly 95% of that is embedded output — base64
images, logs, tensors. Without stripping, every re-run produces a huge meaningless
diff. With it, you diff *code*.

📚 [nbstripout](https://github.com/kynan/nbstripout)

### Step 2.3 — Initialise and make the first commit

```bash
cd ~/Fayaz_Repo/llm-finetuning-medqa
git init -b main
mkdir -p notebooks/original
git mv notebooks/*.ipynb notebooks/original/ 2>/dev/null || mv notebooks/*.ipynb notebooks/original/
git add .
git commit -m "chore: initial import of Colab notebooks

Three notebooks moved verbatim from Google Drive:
- Training.ipynb: 5-phase LoRA/QLoRA pipeline
- Loading Models.ipynb: HF Hub upload utility
- Live Demo.ipynb: Gradio comparison app

Preserved unmodified in notebooks/original/ as the reference implementation."
```

📚 [Conventional Commits](https://www.conventionalcommits.org/) — the `chore:` / `feat:` /
`fix:` prefix convention used throughout this roadmap.

### Step 2.4 — Create the GitHub repo

```bash
gh repo create llm-finetuning-medqa \
  --public \
  --source=. \
  --description "LoRA vs QLoRA fine-tuning comparison on medical Q&A (GPT-2 / TinyLlama)" \
  --push
```

> **Public or private?** Public gets you free branch protection and free Actions
> minutes, and this is portfolio-grade work worth showing. Private is fine too, but
> rulesets need GitHub Pro. Swap `--public` for `--private` if you prefer.

### Step 2.5 — Protect `main`

```bash
gh api -X POST repos/fayazhussain2821/llm-finetuning-medqa/rulesets \
  --input - <<'EOF'
{
  "name": "protect-main",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "pull_request",
     "parameters": {
       "required_approving_review_count": 0,
       "dismiss_stale_reviews_on_push": true,
       "require_code_owner_review": false,
       "require_last_push_approval": false,
       "required_review_thread_resolution": false
     }}
  ]
}
EOF
```

This forces every change through a PR while still letting you self-merge
(`required_approving_review_count: 0` — you have no second reviewer). After Phase 4
you will add the CI check as a required status.

📚 [GitHub rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets)

### Step 2.6 — The loop you will repeat for every phase below

```bash
git switch -c phase-3/extract-package     # 1. branch
# ... make changes ...
ruff check . && ruff format --check . && pytest -q   # 2. verify LOCALLY first
git add -p                                # 3. stage deliberately, hunk by hunk
git commit -m "refactor: extract data pipeline into src/medqa/data.py"
git push -u origin phase-3/extract-package
gh pr create --fill                       # 4. open PR
gh pr checks --watch                      # 5. wait for CI to go green
gh pr merge --squash --delete-branch       # 6. merge only when green
git switch main && git pull                # 7. resync
```

> `git add -p` walks you through each hunk and asks. It is the cheapest habit for
> catching a stray `print()` or a pasted token before it becomes permanent.

📚 [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow)

**✅ Verify Phase 2**

```bash
git push origin main    # MUST be rejected by the ruleset
gh repo view --web
```

If that push succeeds, protection is not active — recheck Step 2.5.

---

## Phase 3 — Restructure: notebooks → package

**Goal:** logic lives in `.py` files that git can diff and pytest can import.
Notebooks become thin drivers.

**Branch:** `phase-3/extract-package`

> **Status: done (2026-08-07).** Three deviations from the plan below, each
> deliberate:
>
> 1. **Both arms train under `Trainer`, not `Trainer` + TRL `SFTTrainer`.** Since
>    `medqa.data` pre-tokenizes both arms, `SFTTrainer` was only wrapping the same
>    loop — and a comparison whose two arms run different training machinery has a
>    confound it does not need. Everything that genuinely differs (4-bit, optimiser,
>    batch shape) now lives in `ModelSpec`. `trl` stays in the dependencies but is
>    no longer on the training path.
> 2. **`evaluate.py` computes bits-per-byte and supports `--base` from the start.**
>    Step 3.5 only asked for computed-not-hardcoded metrics, but writing a metric we
>    already know is invalid (Phase 6.1) and then rewriting it was the worse order.
>    Phase 6.1 and 6.2 are now a matter of *running* the CLI, not editing it.
> 3. **The Phase 3 verify step no longer nbconvert-executes the driver notebook.**
>    `colab_train.ipynb` asserts a CUDA GPU and shells out to `git clone` — it cannot
>    run on a Mac by design. It is verified structurally instead: valid JSON, every
>    code cell parses, all outputs stripped. Real execution belongs in Phase 5, on Colab.
>
> `pytest` is 38 passing / 1 network-marked, `ruff check .` is clean, and the
> scoring path was smoke-tested end-to-end against the real dataset.

### 3.0 — API compatibility, verified not assumed

Every API the notebooks call, tested against transformers 5.14.1 / trl 1.9.2 on
2026-08-06:

| Notebook usage | Status | Action |
|---|---|---|
| `DataCollatorForLanguageModeling(tokenizer=…, mlm=False)` | ✅ works | none |
| `SFTConfig(dataset_text_field="text", max_length=1024)` | ✅ works | none — `max_seq_length` was removed, and you already used the surviving name |
| `SFTTrainer(processing_class=tokenizer)` | ✅ works | none — `tokenizer=` was removed; you are already on the new API |
| `from_pretrained(torch_dtype=torch.float16)` | ⚠️ **deprecated** | rename to `dtype=` — 3 sites: `Training.ipynb` cells 20 & 27, `Live Demo.ipynb` cell 1 |
| `TrainingArguments(evaluation_strategy=…)` | n/a | removed in favour of `eval_strategy`; you never used it, but Step 3.4 adds it — use the new name |

The notebook is in far better shape than a two-major-version jump would suggest:
**one rename, not a rewrite.** The `torch_dtype` call still functions but emits
`` `torch_dtype` is deprecated! Use `dtype` instead! `` on every load.

> **Do not skip the lockfiles because of this.** The notebooks call
> `!pip install -q transformers datasets accelerate peft trl bitsandbytes` with no
> pins at all, so a Colab run six months from now resolves to whatever is current
> that day. The compatibility above is a snapshot, not a guarantee.

### Target layout

```
llm-finetuning-medqa/
├── pyproject.toml
├── requirements.txt / requirements-colab.txt
├── ROADMAP.md
├── README.md
├── src/medqa/
│   ├── __init__.py
│   ├── config.py        # every constant, one place
│   ├── data.py          # load → format → tokenize
│   ├── models.py        # DomainChatModel  ← single copy
│   ├── train.py         # both phases behind one CLI
│   └── evaluate.py      # perplexity → metrics.json
├── app/demo.py          # Gradio
├── notebooks/
│   ├── original/        # untouched reference
│   └── colab_train.ipynb  # thin driver
└── tests/
```

### Step 3.1 — `config.py` first

Everything currently hardcoded across cells goes here: `MODEL_NAME`, `MAX_LEN`,
`SEED = 42`, LoRA `r`/`alpha`/`dropout`, target modules per architecture, batch
sizes, learning rate, `HF_USER`.

Add the path shim that makes one codebase work in both places:

```python
import os
from pathlib import Path

IN_COLAB = "COLAB_GPU" in os.environ or Path("/content").exists()

if IN_COLAB:
    ROOT = Path("/content/llm-finetuning-medqa")
    DRIVE = Path("/content/drive/MyDrive")
else:
    ROOT = Path(__file__).resolve().parents[2]
    DRIVE = None

OUTPUT_DIR = ROOT / "outputs"
HF_USER = os.environ.get("HF_USER", "Babblu2821")
```

> **`HF_USER` via env var, not hardcoded.** Right now `Live Demo.ipynb` hardcodes
> `"Babblu2821"` while `Training.ipynb` derives it from `whoami()`. Anyone who clones
> this repo gets a demo that silently loads *your* adapters.

### Step 3.2 — `data.py`

Port cells 3–6: `load_medquad()`, `format_examples()`, `tokenize()`, `get_collator()`.
Keep the seeded split (`seed=42`) — that reproducibility is already correct.

Keep the token-length percentile probe as a function. Measuring rather than guessing
`MAX_LEN` is one of the better decisions in the notebook; it deserves to survive as
code, not as a cell someone forgets to run.

### Step 3.3 — `models.py` — **fixes a real bug**

`DomainChatModel` currently exists in **two** notebooks and they have **already
drifted apart**:

| | `Training.ipynb` cell 27 | `Live Demo.ipynb` cell 1 |
|---|---|---|
| Decoding | `do_sample=True, temp=0.7, top_p=0.9` | `do_sample=False` (greedy) |
| `max_new_tokens` | 256 | 80 |
| CPU fallback | none | `HAS_GPU` branch |

Same class name, different behaviour. **Your demo does not show what your evaluation
measured** — one samples, the other is greedy. Extract one copy; make decoding an
explicit argument with a named preset (`GenerationConfig`) so eval and demo can
share settings or differ *on purpose*.

📚 [HF generation strategies](https://huggingface.co/docs/transformers/generation_strategies)

### Step 3.4 — `train.py`

One CLI, both phases:

```bash
python -m medqa.train --model gpt2       --epochs 1
python -m medqa.train --model tinyllama  --epochs 1 --4bit
```

Add what the notebook lacks:

- `set_seed(42)` covering torch, numpy, and random
- `eval_strategy="steps"`, `eval_steps=100` — currently you only evaluate *after*
  training, so an overfit is invisible until it is too late to stop
- `load_best_model_at_end=True`
- Save the resolved config next to the adapter, so a checkpoint records how it was made

📚 [TRL `SFTTrainer`](https://huggingface.co/docs/trl/sft_trainer) · [PEFT](https://huggingface.co/docs/peft)

### Step 3.5 — `evaluate.py` — **fixes a second real bug**

Cell 30 hardcodes results:

```python
gpt2_ppl, tiny_ppl = 5.99, 2.80    # measured in Cells 9 and 16
```

Change a hyperparameter, re-run, and this cell reports the old numbers with total
confidence. Metrics must be *computed* and written to `outputs/metrics.json`; the
comparison table reads that file. Never retype a number a program produced.

**✅ Verify Phase 3**

```bash
pytest                          # 38 passed, 1 deselected (network)
ruff check .
python -c "from medqa.data import load_medquad; d=load_medquad(); print(len(d))"   # 16407
python -c "from medqa.models import DomainChatModel; print('import ok')"
python -m medqa.evaluate --table                                  # reads metrics.json
```

The driver notebook is checked structurally rather than executed — see the status
note at the top of this phase. Open the PR only once all of the above pass.

---

## Phase 4 — Verification: tests & CI

**Goal:** the PR gate becomes automatic.

**Branch:** `phase-4/ci`

> ⚠️ **Run `gh auth refresh -s workflow` first** (see Phase 0.2) or the push is rejected.

> **Status: 4.1–4.3 done (2026-08-07).** 45 tests, `.pre-commit-config.yaml`, and
> `.github/workflows/ci.yml`. Step 4.4 (required status checks) is still open — it
> needs one green CI run to exist before the check can be marked required.
>
> Three things this phase turned up that the plan did not anticipate:
>
> 1. **CI needed its own lockfile.** `requirements.txt` was resolved on macOS and
>    carries no platform markers, so installing it on ubuntu resolves `torch==2.13.0`
>    to the CUDA build and drags the whole `nvidia-*` stack into a CPU-only job.
>    `requirements-ci.txt` is the linux/cpu resolution (`torch==2.13.0+cpu`).
>    That makes three lockfiles: mac (dev), colab (GPU), ci (linux CPU).
> 2. **`ruff format` rewrites Python inside markdown fences.** It reformatted the
>    `gpt2_ppl, tiny_ppl = 5.99, 2.80` line quoted in Step 3.5 — but that line is
>    *evidence*, quoted verbatim from the notebook. `*.md` is now excluded from ruff.
> 3. **The ruff version must be pinned in three places at once** — `requirements.txt`,
>    `requirements-ci.txt`, and `.pre-commit-config.yaml`. If the hook and CI disagree
>    on a ruff version they disagree on formatting, and commits ping-pong between them.
>    All three are on 0.16.1.

### Step 4.1 — Tests worth writing

Do not chase coverage. Test the things that break silently:

| Test | Catches |
|---|---|
| `test_prompt_template_roundtrip` | `to_chat_text()` splits on `"### Response:\n"` — a stray occurrence inside an *answer* corrupts the example |
| `test_split_is_deterministic` | seed drift making eval sets incomparable across runs |
| `test_collator_masks_padding` | `labels == -100` count must equal pad count (cell 8 checks this by eye — make it a test) |
| `test_no_train_eval_overlap` | leakage, which would invalidate every perplexity number |
| `test_config_paths_resolve` | the Colab/local shim breaking on one side |

That fourth one matters most. Perplexity on a leaked eval set looks *great*.

📚 [pytest](https://docs.pytest.org/)

### Step 4.2 — Pre-commit

`.pre-commit-config.yaml` with `ruff`, `ruff-format`, `nbstripout`, and
`detect-private-key`. Then `pre-commit install`.

Catching things locally is strictly cheaper than catching them in CI.

📚 [pre-commit](https://pre-commit.com/) · [Ruff](https://docs.astral.sh/ruff/)

### Step 4.3 — GitHub Actions

`.github/workflows/ci.yml` — on PRs to `main`, run ruff + pytest on CPU only.
Never train in CI; there is no GPU and it would burn your minutes. Cache the `uv`
environment and the HF dataset to keep runs near a minute.

### Step 4.4 — Require the check

```bash
gh api -X PUT repos/fayazhussain2821/llm-finetuning-medqa/rulesets/RULESET_ID ...
```

Add `required_status_checks` for the `ci` job. Now a red build genuinely blocks merge.

**✅ Verify Phase 4** — open a throwaway PR that breaks a test on purpose; confirm
merge is blocked. Then close it.

---

## Phase 5 — Colab integration

**Goal:** Colab runs *this repo*, not a copy that has drifted from it.

**Branch:** `phase-5/colab-driver`

### Step 5.1 — The bootstrap cell

Cell 1 of `notebooks/colab_train.ipynb`:

```python
# ── Bootstrap: clone the repo and install ──
import os
REPO = "https://github.com/fayazhussain2821/llm-finetuning-medqa.git"
if not os.path.exists("/content/llm-finetuning-medqa"):
    !git clone -q {REPO} /content/llm-finetuning-medqa
%cd /content/llm-finetuning-medqa
!git pull -q                                   # always latest
!pip install -q -r requirements-colab.txt
!pip install -qe .
print("✅ ready — Runtime > Restart session, then continue")
```

For a private repo, mint a fine-grained PAT, store it via Colab's **Secrets** panel
(🔑 in the sidebar), and read it with `userdata.get()`. Never paste a token into a cell —
notebooks save their outputs to Drive.

📚 [Colab secrets](https://colab.research.google.com/notebooks/snippets/secrets.ipynb)

### Step 5.2 — Driver cells

The remaining cells only orchestrate:

```python
from medqa import config, data, train, evaluate
train.run(model="gpt2")
train.run(model="tinyllama", four_bit=True)
evaluate.compare()          # reads metrics.json — no retyped numbers
```

### Step 5.3 — Retire the Drive round-trip

Cell 25 currently backs adapters up to Drive, and `Loading Models.ipynb` restores
them. Both are obsolete now that the adapters live on the HF Hub — push at the end
of training, pull by repo ID anywhere. That deletes an entire notebook and the
Drive mount along with it.

Keep Drive only for genuinely large artifacts. For this project there are none.

### Step 5.4 — Editing from your Mac

```bash
git switch -c experiment/higher-rank
# edit src/medqa/config.py: LORA_R = 32
git commit -am "experiment: raise LoRA rank to 32"
git push -u origin experiment/higher-rank
```

Then in Colab: `!git fetch && git checkout experiment/higher-rank` and re-run.
Edit on the Mac with a real editor; execute on the GPU. That is the whole point of
the split.

**✅ Verify Phase 5** — fresh Colab runtime, run only the bootstrap cell, confirm
`import medqa` works and `nvidia-smi` shows the T4.

---

## Phase 6 — Fix the science

This phase matters more than everything above it. The engineering is sound; the
**experimental design has two flaws that undercut the headline claim**.

**Branch:** `phase-6/valid-comparison`

### 6.1 — ⚠️ Perplexity is not comparable across different tokenizers

> **GPT-2 5.99 → TinyLlama 2.80 (−53%)** is not a valid comparison.

Perplexity is `exp(mean negative log-likelihood **per token**). GPT-2 uses byte-level
BPE with a 50,257-token vocabulary; TinyLlama uses a Llama SentencePiece tokenizer
with 32,000. **They segment the same sentence into different numbers of tokens.**
Per-token loss therefore has a different denominator for each model, and the ratio
between them partly measures tokenizer granularity rather than model quality.

Coarser tokenization (fewer, bigger tokens) generally means each token carries more
information and is harder to predict; finer tokenization inflates the token count
with easy, highly-predictable pieces. Neither direction is "better" — they are just
not the same measurement.

**The fix — normalise by something tokenizer-independent:**

```python
# bits per byte: comparable across ANY tokenizer
bpb = (total_nll / math.log(2)) / total_utf8_bytes
```

Report **bits-per-byte** (or per-character perplexity) as the headline. Keep
per-token perplexity per model as a secondary, clearly labelled as
within-model-only. Then the improvement claim survives scrutiny.

📚 [HF perplexity guide](https://huggingface.co/docs/transformers/perplexity) ·
[The Pile §5 on BPB](https://arxiv.org/abs/2101.00027)

### 6.2 — ⚠️ There is no control condition

You compare *fine-tuned GPT-2* against *fine-tuned TinyLlama*, then attribute the
gap to QLoRA. But TinyLlama is **9× larger** and far better pretrained. The
experiment cannot separate "QLoRA worked" from "the bigger model was already better."

Add the missing cells of the 2×2:

| | Base (no fine-tune) | + LoRA/QLoRA |
|---|---|---|
| **GPT-2 124M** | ❌ missing | ✅ 5.99 |
| **TinyLlama 1.1B** | ❌ missing | ✅ 2.80 |

Both missing cells are cheap — a forward pass over `eval_ds`, no training. With
them you can finally state *how much fine-tuning contributed, per model*, which is
the actual research question.

> This is the single highest-value change in the roadmap. It converts a demo into
> an experiment.

### 6.3 — Perplexity alone is a weak proxy for answer quality

A model can score well and still produce fluent medical nonsense. Add at least one
task metric:

- **ROUGE-L / BERTScore** against reference answers — cheap, standard
- **LLM-as-judge** on ~100 held-out questions, scoring factuality and relevance
- A small **hand-scored** set — 20 questions rated 1–5. Tedious, and more
  informative than either automated metric.

📚 [HF Evaluate](https://huggingface.co/docs/evaluate) · [BERTScore](https://arxiv.org/abs/1904.09675)

> **Status: done (2026-08-14)** — `src/medqa/quality.py`, all four arms, 200 held-out
> rows each, greedy decoding. ROUGE-L F1 and token F1 against the reference, plus a
> repeated-4-gram rate to catch degeneration and a length/empty check. Every
> generation is kept in `outputs/generations/<run>.jsonl`, which is what makes the
> hand-scored option below possible later.
>
> **The finding this phase was worth doing for: the two metrics disagree on rank
> order.** On likelihood, fine-tuned GPT-2 (0.5970 bits/byte) beat untouched TinyLlama
> (0.6120). On generated answers, untouched TinyLlama beats fine-tuned GPT-2 by 59% on
> ROUGE-L. Phase 6.2's control looked like a near-tie and is not one.
>
> Four things worth recording:
>
> 1. **ROUGE-L was implemented here rather than pulled in.** `rouge-score` drags in
>    nltk and absl-py for ~40 lines of LCS, and both lockfiles would need recompiling.
>    The expected values in `test_quality.py` were produced by running the real
>    `rouge_score` 0.1.2 in a throwaway venv, so the check costs no dependency but is
>    still anchored to the reference implementation. One value hand-derived first was
>    wrong (0.3636, not 0.3333) — worth remembering before trusting arithmetic done in
>    a docstring.
> 2. **No degeneration anywhere** (repeated 4-grams 0.0000–0.0146). The failure this
>    metric was added to catch did not occur. The models are not incoherent, they are
>    wrong — which only reading the jsonl reveals.
> 3. **Length ratio is confounded with the token cap.** 163/200 GPT-2 answers run to
>    the 200-token ceiling versus 12/200 for TinyLlama, so that column measures
>    stopping behaviour, not verbosity.
> 4. **`write_metrics` had to stop clobbering.** It replaced an arm's whole entry, so
>    re-running the likelihood pass would have silently deleted an hour of
>    generations. It merges now, and two tests hold that.
>
> Still open from the list above: LLM-as-judge and the hand-scored set. Neither
> automatic metric here can tell a correct medical claim from a fluent false one.

### 6.4 — Report variance

Single-seed results are anecdotes. Run three seeds, report mean ± std. If the gap
between two configurations is smaller than the seed-to-seed spread, you have not
measured a difference at all.

### 6.5 — Safety disclaimer (non-negotiable)

This model answers **medical** questions, and your Gradio demo is publicly
shareable via `share=True`. It must carry a visible disclaimer stating it is a
research artifact, not medical advice, and must not be used for diagnosis or
treatment decisions. Put it in the UI, the README, and both HF model cards.

Two small fine-tuned models trained for one epoch will confidently generate
plausible-sounding false medical claims. That is a property of the setup, not a bug
you can train away at this scale.

---

## Phase 7 — Ship it

**Branch:** `phase-7/publish`

### 7.1 — README

Problem → approach → results table (read from `metrics.json`) → repro instructions →
limitations → disclaimer. Include the bits-per-byte numbers and the 2×2 from Phase 6.
State plainly what the models cannot do.

### 7.2 — Model cards

Both HF repos currently have autogenerated `README.md` stubs. Fill in: base model,
dataset, hyperparameters, eval protocol, intended use, limitations, disclaimer.

📚 [Model cards](https://huggingface.co/docs/hub/model-cards)

### 7.3 — Deploy the demo to HF Spaces

`app/demo.py` is already Hub-native — it loads adapters by repo ID. A free CPU Space
will run GPT-2 comfortably; TinyLlama in 4-bit needs a GPU Space, or swap to fp32 CPU
and accept slower generation.

This gives you a permanent public URL instead of a Colab `share=True` link that dies
in 72 hours.

📚 [HF Spaces](https://huggingface.co/docs/hub/spaces)

### 7.4 — Tag a release

```bash
git tag -a v1.0.0 -m "Reproducible LoRA/QLoRA comparison with corrected metrics"
git push origin v1.0.0
```

---

## Learning references

**The methods**
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al. 2021. §4.1 explains the rank choice.
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al. 2023. NF4, double quantization, paged optimizers — all three appear in your Cell 13.
- [PEFT docs](https://huggingface.co/docs/peft) · [TRL docs](https://huggingface.co/docs/trl)
- [HF LLM Course](https://huggingface.co/learn/llm-course) — free, chapters 3 & 11 are directly relevant.

**Evaluation**
- [HF perplexity guide](https://huggingface.co/docs/transformers/perplexity) — read before Phase 6.1.
- [HF Evaluate](https://huggingface.co/docs/evaluate)

**Engineering**
- [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow) · [rulesets](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets)
- [Conventional Commits](https://www.conventionalcommits.org/) · [uv](https://docs.astral.sh/uv/) · [Ruff](https://docs.astral.sh/ruff/) · [pre-commit](https://pre-commit.com/)
- [rclone Drive](https://rclone.org/drive/)

---

## Order of work

If time is short, this is the priority order:

1. **Phase 6.2** — the missing control. Highest value, cheapest to run.
2. **Phase 6.1** — the tokenizer problem. Your headline number depends on it.
3. **Phase 2** — version control. Everything else is safer once this exists.
4. **Phase 3.3 / 3.5** — the duplicated class and the hardcoded metrics. Both are live bugs.
5. Everything else.

Phases 1–5 make the work *maintainable*. Phase 6 makes it *correct*. If you only
do one, do Phase 6.

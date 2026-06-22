# INSTALL_MEMO — Gemma 4 model fails to load (mlx-lm bug)

Notes for reproducing the fix on a fresh machine. Written 2026-06-22.

## TL;DR

The default model (`mlx-community/gemma-4-e4b-it-4bit`) **won't load** with the
current latest `mlx-lm` (0.31.3). The MLX server starts and reports healthy, but
the model never loads, so the first chat message fails. It's a bug in `mlx-lm`,
not in this app. There is a one-line-ish patch to the installed `mlx-lm` that
makes it work; the real fix belongs upstream in `mlx-lm`.

## Symptom

`npm run dev` starts fine, `GET /v1/models` returns 200 ("Server is healthy"),
but the model load throws:

```
ValueError: Received 126 parameters not in model:
language_model.model.layers.24.self_attn.k_norm.weight,
language_model.model.layers.24.self_attn.k_proj.biases,
language_model.model.layers.24.self_attn.k_proj.scales,
language_model.model.layers.24.self_attn.k_proj.weight,
language_model.model.layers.24.self_attn.v_proj.biases,
... (layers 24 through 41)
```

Environment when first seen:
- mlx-lm **0.31.3** (latest release at the time — no newer version exists)
- mlx **0.31.2**, Python 3.13
- Model: `mlx-community/gemma-4-e4b-it-4bit`

## Root cause

Gemma 4 E4B has **42 layers** and `num_kv_shared_layers = 18` (see the model's
`config.json`, `model_type = gemma4_text`). The last 18 layers (indices 24–41)
**share** keys/values from an earlier layer, so the model is built *without*
their own `k_proj` / `v_proj` / `k_norm` / `v_norm`:

```
# mlx_lm/models/gemma4_text.py
self.has_kv = layer_idx < config.num_hidden_layers - config.num_kv_shared_layers
...
if self.has_kv:
    self.k_proj = ...
    self.v_proj = ...   # only created for non-shared layers
```

But the published quant **still ships** those tensors (7 per layer × 18 layers =
exactly **126**). `mlx_lm`'s `sanitize()` for gemma4 merges MoE experts but
**forgets to drop the shared-layer K/V weights**, so `load_weights(strict=True)`
fails on the 126 extras.

This is a real `mlx-lm` bug: `gemma3n.py`'s `sanitize()` already strips these
weights for the equivalent Gemma 3n architecture; `gemma4_text.py` just missed it.

Dropping those weights is **lossless** — the shared layers never use them.

## The manual fix (what we did)

Patch `sanitize()` in the installed `mlx_lm` inside the app's managed venv.

### 1. Locate the file (machine-independent)

```bash
# Use the venv python the app created. On this machine it was:
#   "/Users/<you>/Library/Application Support/gemma-chat/mlx/venv/bin/python3"
"<venv>/bin/python3" -c "import mlx_lm.models.gemma4_text as m; print(m.__file__)"
```

That prints the path to `gemma4_text.py` to edit. (App data lives at
`~/Library/Application Support/gemma-chat/mlx/` — `venv/` for the venv,
`models/` for the HF cache i.e. `HF_HOME`.)

### 2. Add `import re` near the top of the file

```python
import re
from dataclasses import dataclass
```

### 3. In `Model.sanitize()`, drop K/V weights for KV-shared layers

Add this block right after the existing `rotary_emb / input_max / ...` skip,
**before** the `if k.endswith(".experts.gate_up_proj"):` branch:

```python
    def sanitize(self, weights):
        # KV-shared layers reuse an earlier layer's keys/values, so the model has
        # no k_proj/v_proj/k_norm/v_norm for them (see has_kv). Some quants still
        # ship those weights; drop them so load_weights(strict=True) doesn't fail.
        first_kv_shared = self.args.num_hidden_layers - self.args.num_kv_shared_layers
        sanitized = {}
        for k, v in weights.items():
            if any(s in k for s in (
                "self_attn.rotary_emb", "input_max", "input_min",
                "output_max", "output_min",
            )):
                continue

            m = re.search(
                r"\.layers\.(\d+)\.self_attn\.(?:k_proj|v_proj|k_norm|v_norm)\b", k
            )
            if m and int(m.group(1)) >= first_kv_shared:
                continue

            # ... existing experts/switch_glu handling and `sanitized[k] = v` ...
```

This is generic across Gemma 4 sizes (E2B / E4B / 26B / 31B) because it reads
`num_kv_shared_layers` from the model args.

### 4. Restart the app

The running `python -m mlx_lm.server` process has the old module cached in
memory. Kill the dev server and re-run `npm run dev` so a fresh Python process
imports the patched code.

## How to verify

```bash
HF_HOME="<app data>/mlx/models" HF_HUB_OFFLINE=1 \
  "<venv>/bin/python3" -c "
from mlx_lm import load, generate
model, tok = load('mlx-community/gemma-4-e4b-it-4bit')
msgs=[{'role':'user','content':'In one sentence, what is the capital of France?'}]
p = tok.apply_chat_template(msgs, add_generation_prompt=True)
print(generate(model, tok, prompt=p, max_tokens=40, verbose=False))
"
# Expected: "The capital of France is Paris."
```

## ⚠️ Why this is not permanent

The patch lives in `site-packages` inside the app-managed venv. It is **wiped**
whenever the venv is recreated or `mlx-lm` is reinstalled/upgraded — and **every
fresh install of this app hits the same bug**, because 0.31.3 is the latest
`mlx-lm` and the default model is Gemma 4.

## The real fix (pick one for the repo)

Install flow lives in `src/main/mlx.ts` (`pip install` at the `mlx-lm>=0.24.0`
line; server launch at `python -m mlx_lm.server`).

- **A. Version pin (correct long-term).** Once the fix is in an `mlx-lm` release,
  bump `mlx-lm>=0.24.0` → e.g. `mlx-lm>=0.31.4`. Until then you can pin a git ref
  that has the fix: `mlx-lm @ git+https://github.com/ml-explore/mlx-lm@<sha>`.
  No local patching, nothing to clean up.

- **B. Runtime monkeypatch via `sitecustomize` (best local fix).** Ship a
  `sitecustomize.py` in the repo that wraps `gemma4_text.Model.sanitize`, and add
  its directory to `PYTHONPATH` in the server-launch `env`. Python auto-imports
  `sitecustomize` before `mlx_lm.server`. Survives venv recreation and mlx-lm
  upgrades, doesn't touch site-packages, and auto-no-ops once upstream is fixed.

- **C. Post-install text patch (simplest, brittle).** After the `pip install`
  step, run a script that applies steps 2–3 above to the installed file. Must be
  idempotent; breaks if upstream edits that file.

Recommendation: **B** now, switch to **A** and delete B once the upstream fix
ships.

## Upstream

File / find a PR against `ml-explore/mlx-lm`. Suggested title:

> Fix Gemma 4 load failure: drop KV-shared layers' k/v weights in sanitize

Put the verbatim `ValueError: Received 126 parameters not in model: ...` in the
PR body so others with the same traceback can find it.

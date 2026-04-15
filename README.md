# Gemma 4 E2B — refusal removal (Uncensoring)

![Removing refusal directions from transformer weights — pipeline overview](notebooks/image.png)

This repository is a **hands-on study** of how to **reduce model refusals** (“uncensor” in the sense of weight edits, not retraining) for **Gemma 4 E2B IT** by estimating a per-layer refusal-related direction from activations and **projecting it out of** `o_proj` and `down_proj` while **preserving row norms**. The method follows the **norm-preserving biprojected abliteration** line of work documented upstream.

> **Safety:** Edited weights can comply with requests the base model refused. Use only on hardware you control, for research, and in line with licenses and applicable policies.


## Try it in the browser (slow)

A hosted **Gradio** demo compares **baseline vs. abliterated** replies side by side:

**[Gemma 4 E2B — Refusal Removal Study (Hugging Face Space)](https://huggingface.co/spaces/kvsnoufal/gemma-abliteration-study)**

Spaces are convenient but often **cold-start slowly**, queue under load, and run on modest hardware. For a responsive experience and full control over the checkpoint, **run the Gradio app locally** (see below).

## What to look at in this repo

| Focus | Path | Role |
|--------|------|------|
| **Notebook** | [`notebooks/abliterate_gemma4b.ipynb`](notebooks/abliterate_gemma4b.ipynb) | Step-by-step walkthrough: load model, harvest last-token activations, winsorize, centroid difference + orthogonalization, project directions out of weights, optional plots and generations. Uses the same infographic asset as the top of this README. |
| **Gradio app** | [`scripts/gradio/app.py`](scripts/gradio/app.py) | One-shot load: keeps **original** and **abliterated** weight snapshots in memory and swaps them per query so you can compare **Original** vs **Restrictions removed** in the UI. Listens on **port 7860**. |

Supporting pipeline code (harvest, directions, apply, eval) lives under [`scripts/`](scripts/); see [`scripts/README.md`](scripts/README.md) for CLI usage (`run.py`, caches, `output/`).

## Examples (before / after)

Side-by-side captures from the demo are in `docs/`:

| Example | Preview |
|--------|---------|
| Example 1 | ![Example 1 — baseline vs abliterated](docs/EX1.png) |
| Example 2 | ![Example 2 — baseline vs abliterated](docs/EX2.png) |

The Gradio UI also ships with **built-in example prompts** (same family as in `scripts/config.py` — e.g. instructions that often trigger refusals in the IT model). After abliteration, many of these show **compliance** where the baseline showed **refusal** or heavy disclaimers.

## Run the Gradio app locally

1. Place **Gemma 4 E2B IT** weights at `models/gemma-4-E2B-it/` (see [google/gemma-4-E2B-it](https://huggingface.co/google/gemma-4-E2B-it) on Hugging Face; gated).
2. Install dependencies (e.g. from [`scripts/gradio/requirements.txt`](scripts/gradio/requirements.txt) plus your PyTorch build for MPS/CUDA).
3. From the repo root:

```bash
conda activate nemoenv
cd scripts/gradio
python app.py
```

Open **http://127.0.0.1:7860**. The first load compiles the refusal directions and applies weight edits; subsequent queries only swap weights and run generation.

## Run the notebook locally

```bash
conda activate nemoenv
# Start Jupyter or VS Code, then open:
# notebooks/abliterate_gemma4b.ipynb
```

The notebook expects `../models/gemma-4-E2B-it` relative to the `notebooks/` folder (same layout as this repo).


## Approach

### Intuition

Instruction-tuned models learn internal representations and readout weights that steer certain inputs toward **refusal-style completions**. This pipeline does **not** fine-tune the model. Instead it treats refusal behavior as partly aligned with a **one-dimensional subspace per layer** in hidden space, estimates a proxy direction from data, then **bakes a fixed edit into selected linear layers** so their outputs are discouraged from lying along that direction.

That is **weight-space intervention**: selected `nn.Linear` weights are modified in place. It is related to, but not the same as, **activation steering** at inference (adding a vector to hidden states each forward pass). Here the intervention is **static** in the weights after a one-time edit.

### 1. Data and what is measured

Two prompt corpora drive the geometry:

- **Harmful-oriented** prompts (e.g. from `mlabonne/harmful_behaviors`).
- **Harmless** instructions (e.g. from `tatsu-lab/alpaca`).

Each prompt is formatted with the model’s **chat template**, then the model runs a **single forward pass**. Forward hooks on every **text decoder layer** read that layer’s hidden tensor; the code keeps only the **last input token** position. That position sits **immediately before** the assistant would start generating, so its state summarizes the user request—where intent to refuse or comply is encoded.

Activations are stacked into tensors of shape **`[n_prompts, 35, 1536]`** (35 layers, hidden size 1536 for E2B), kept on **CPU** after each prompt to limit device memory use.

### 2. Estimating a per-layer direction

All direction math runs in **float32 on CPU** (stable quantiles and means).

1. **Winsorization** — For each (prompt, layer) vector, coordinates are clamped using a high quantile of absolute values (default **0.995**). That dampens extreme dimensions so a few outliers do not dominate the mean.
2. **Centroids** — Per layer, take the mean of winsorized activations over harmful prompts and over harmless prompts: \(\mu^{\mathrm{harm}}_\ell\), \(\mu^{\mathrm{safe}}_\ell\).
3. **Raw difference** — \(\delta_\ell = \mu^{\mathrm{harm}}_\ell - \mu^{\mathrm{safe}}_\ell\). This points from “average harmless” toward “average harmful” at layer \(\ell\), but it can still mix refusal with **style, length, or topic** differences between datasets.
4. **Orthogonalization** — Let \(\hat{h}\) be the unit vector along the harmless mean \(\mu^{\mathrm{safe}}_\ell\). The component of \(\delta_\ell\) parallel to \(\hat{h}\) is subtracted **twice** (Gram–Schmidt style passes) for numerical stability. The idea is to keep a direction that contrasts the two classes **while reducing overlap with the average harmless representation** at that layer.
5. **Unit direction** — Normalize the result to \(\hat{r}_\ell \in \mathbb{R}^{1536}\) (unless degenerate). Each layer gets its **own** \(\hat{r}_\ell\); directions are **not** shared across layers.

The repo also computes a **layer quality** score from separation strength, cosine similarity of the two means, and how much of the raw difference survives orthogonalization—useful for debugging and for quick single-layer experiments.

### 3. Editing weights (projection + norm restore)

For each layer \(\ell\), the **same** \(\hat{r}_\ell\) is applied to **`self_attn.o_proj`** and **`mlp.down_proj`**. In PyTorch, `Linear.weight` has shape **`[out_features, in_features]`**; here the edited layers map **into** the residual stream width, so \(\hat{r}_\ell\) lives in **output space** \(d_{\mathrm{out}} = 1536\).

The update is an **output-space rank-1 projection**:

\[
P = I - \hat{r}\hat{r}^\top, \qquad W' = P W = W - \hat{r}\,(\hat{r}^\top W).
\]

So every forward pass through that layer applies **post-multiplication by \(P\)** on the linear output in the appropriate sense: the component along \(\hat{r}\) is driven toward zero. The implementation applies the rank-1 subtraction **twice** to reduce drift from half-precision rounding.

Strict projection can **shrink row norms** of \(W\). The code **rescales each row** of \(W'\) to match the **original** row Euclidean norms before the edit, then casts back to **float16** for inference. That is the **norm-preserving** part of the method: refusal signal is suppressed while keeping similar per-row magnitude statistics.

**Scope:** E2B here uses **35 layers × 2 modules = 70** weight matrices. This repo does **not** implement MoE **expert-granular** edits; for the larger Gemma 4 family and MoE details, see the upstream repository.

### 4. Caveats (short)

- Refusal behavior is **high-rank** and contextual; a **single** direction per layer is a deliberate simplification.
- \(\mu^{\mathrm{harm}} - \mu^{\mathrm{safe}}\) can reflect **dataset confounds**, not only policy/refusal; orthogonalizing against \(\mu^{\mathrm{safe}}\) only removes the part parallel to that centroid.
- Row rescaling means the map is **approximately**, not exactly, a projector in float16.
- Behavioral **eval** in the scripts uses **substring heuristics** on generated text, not logits—a coarse sanity check, not a safety guarantee.

For tensor shapes, file-to-phase mapping, and step-by-step math aligned with `harvest.py`, `directions.py`, and `abliterate.py`, see **[`scripts/explanation.md`](scripts/explanation.md)**.

## References

- **Primary method, evaluation framing, and research write-up:** [TrevorS/gemma-4-abliteration](https://github.com/TrevorS/gemma-4-abliteration) — norm-preserving refusal removal, larger Gemma 4 family, and MoE extensions (EGA) where relevant.
- **Hosted demo:** [kvsnoufal/gemma-abliteration-study](https://huggingface.co/spaces/kvsnoufal/gemma-abliteration-study) on Hugging Face Spaces.

The infographic at the top of this README matches the **eight-step** summary in that work: collect harmful/harmless prompts, hook layers for **last-token** hidden states, winsorize, centroids and difference, orthogonalize, project out of `o_proj` / `down_proj`, restore row norms, repeat per layer. The prose above is a condensed version of the same pipeline; details and notation live in [`scripts/explanation.md`](scripts/explanation.md).

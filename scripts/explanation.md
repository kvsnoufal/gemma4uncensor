# How this refusal removal pipeline works

This document explains the **steering / intervention approach**, the **exact tensors and shapes** used in this repo, and the **mathematics** behind harvesting, refusal-direction estimation, and weight modification. It is written to match the code in `harvest.py`, `directions.py`, and `remove refusals.py`.

---

## 1. What problem we are solving

Instruction-tuned models often **refuse** certain user requests. Roughly, the model’s internal representations and readout weights are aligned so that harmful or policy-violating queries push activations toward a subspace that **triggers refusal language** at the output.

**refusal removal** here means: estimate a direction in hidden space that differs between “harmful” and “harmless” prompts, treat that as a proxy for **refusal-related signal**, then **remove that direction from selected weight matrices** so the network is less able to map into the refusal mode—without retraining.

This is a form of **weight-space intervention**: we edit `nn.Linear` weights in place. It is **not** the same as runtime **activation steering** (adding a vector to hidden states during forward passes), though both ideas are related under the umbrella of “steering.”

---

## 2. Notation and model geometry

- **Hidden size** \(d = 1536\) (`HIDDEN_SIZE` in `config.py`).
- **Text layers** \(\ell = 0,\ldots,L-1\) with \(L = 35\) (`N_LAYERS`).
- **Layer output hook:** for each decoder layer \(\ell\), we read a tensor \(H_\ell \in \mathbb{R}^{B \times T \times d}\) (here \(B=1\)). We keep only the **last position** \(t = T-1\):

  \[
  h_\ell^{(p)} = H_\ell[0,\, T-1,\, :] \in \mathbb{R}^d
  \]

  for prompt index \(p\). Stacked over prompts, activations have shape `[n_prompts, L, d]` in code.

- **Linear layer in PyTorch** (`nn.Linear`): weight `weight` has shape **`[out_features, in_features]`**. Forward pass (no bias for intuition):

  \[
  y^\top = W\, x^\top
  \quad\text{equivalently}\quad
  y = x\, W^\top
  \]

  with \(x \in \mathbb{R}^{1 \times d_{\text{in}}}\), \(W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\), \(y \in \mathbb{R}^{1 \times d_{\text{out}}}\).

  The code’s refusal vector \(r\) lives in \(\mathbb{R}^{d_{\text{out}}}\) for the layers we edit (`o_proj` and `down_proj` both have \(d_{\text{out}} = d\) in this architecture, matching the residual stream width used to define directions).

---

## 3. Phase A — Harvesting activations (`harvest.py`)

### 3.1 What is measured

For each **already formatted** chat string (user message + generation prompt marker):

1. Tokenize with `max_length=MAX_SEQ_LEN`, truncation.
2. Run a **single full forward** through `model(...)` with `torch.no_grad()`.
3. On each text decoder layer module, a **forward hook** reads that layer’s **output** (post-attention + post-MLP block output for `Gemma4TextDecoderLayer`, depending on how the class defines `output`; the code treats it as hidden states of shape `[B, T, d]`).
4. Extract **last token** row and move to CPU.

So for each prompt \(p\) and layer \(\ell\) we get a vector \(h_\ell^{(p)} \in \mathbb{R}^d\).

### 3.2 Why the last token

The last input token sits **right before** the model would begin generating the assistant reply. Its hidden state summarizes the **prompt context** including the user request, which is what we want to contrast across harmful vs harmless datasets.

### 3.3 Tensor layout in code

- Per prompt after stacking layers: `[L, d]`.
- Over all prompts: **`[n_prompts, L, d]`** on CPU.

### 3.4 Memory rationale

Activations are moved to **CPU** immediately; hooks clear; `gc.collect()` and `clear_mps_cache()` run per prompt so MPS does not hold a long graph of activations.

---

## 4. Phase B — Refusal directions (`directions.py`)

All of this runs in **float32 on CPU** (winsorization uses `torch.quantile`, which is not used on MPS in this pipeline).

### 4.1 Winsorization

Input activations have shape `[N, L, d]`. The helper converts to float32 on CPU, then:

```python
t = tensor.float().cpu()
q = torch.quantile(t.abs(), quantile, dim=-1, keepdim=True)
return t.clamp_(-q, q)
```

`dim=-1` is the hidden axis \(d\), so `q` has shape `[N, L, 1]`: **one scalar per (prompt, layer)** equal to the `WINSORIZE_QUANTILE` (default 0.995) of \(|v_1|,\ldots,|v_d|\) for that vector.

For each pair \((p,\ell)\), let \(v = h_\ell^{(p)} \in \mathbb{R}^d\) be one row of the tensor. Compute \(q = Q_{q_0}(|v|)\) where \(Q_{q_0}\) is that quantile **across the \(d\) coordinates** of \(|v|\). Then replace \(v\) by \(\mathrm{clip}(v, -q, q)\) coordinate-wise.

So outliers along the **hidden dimensions** for each activation vector are damped. Winsorization is **not** applied across the prompt batch at this step (the mean in §4.2 does that aggregation separately).

### 4.2 Per-layer means

After winsorizing the full tensors `harmful_w` and `harmless_w`:

\[
\mu_\ell^{\mathrm{harm}} = \frac{1}{N_h}\sum_{p=1}^{N_h} \tilde{h}_\ell^{(p)} \in \mathbb{R}^d,
\qquad
\mu_\ell^{\mathrm{safe}} = \frac{1}{N_s}\sum_{p=1}^{N_s} \tilde{h}_\ell^{(p)} \in \mathbb{R}^d
\]

In code: `mean(dim=0)` over prompt dimension → shape **`[L, d]`**, then indexed by layer \(\ell\).

### 4.3 Raw difference direction

\[
\delta_\ell = \mu_\ell^{\mathrm{harm}} - \mu_\ell^{\mathrm{safe}} \in \mathbb{R}^d
\]

This is a **mean activation difference** vector. It can mix “refusal” with other systematic differences between the two corpora (length, style, topic). The next step removes one confounding direction.

### 4.4 Gram–Schmidt style orthogonalization w.r.t. harmless mean

Let \(h = \mu_\ell^{\mathrm{safe}}\). Define \(\hat{h} = h / \|h\|\) (or zero if degenerate; code uses `F.normalize`).

Remove the component of \(\delta_\ell\) along \(\hat{h}\), **twice** for numerical stability:

\[
r \leftarrow \delta_\ell - (\delta_\ell^\top \hat{h})\,\hat{h},
\qquad
r \leftarrow r - (r^\top \hat{h})\,\hat{h}.
\]

In code, inner products are `r @ h_hat` with 1D tensors (dot product).

**Interpretation:** we ask for a direction that contrasts harmful vs harmless **while being orthogonal to the average harmless representation** at that layer. That reduces the extent to which \(\delta_\ell\) is “just the harmless centroid shifted.”

### 4.5 Unit refusal direction used later

If \(\|r\| > 10^{-8}\), set

\[
\hat{r}_\ell = \frac{r}{\|r\|}
\]

else \(\hat{r}_\ell = r\) (degenerate). Stacked: **`refusal_dirs`** shape `[L, d]`, used as the direction for layer \(\ell\).

### 4.6 Layer quality score (for logging / `test_quick.py`)

For each layer \(\ell\), with \(\delta_\ell\) the **raw** difference before orthogonalization (code uses `raw_dirs[l]`), let:

- \(\mathrm{raw\_norm} = \|\delta_\ell\|\).
- \(\mathrm{snr} = \dfrac{\mathrm{raw\_norm}}{\max(\|\mu_\ell^{\mathrm{harm}}\|, \|\mu_\ell^{\mathrm{safe}}\|, 10^{-8})}\).
- \(\mathrm{cos\_sim} = \cos\big(\mu_\ell^{\mathrm{harm}}, \mu_\ell^{\mathrm{safe}}\big)\).
- After the two Gram–Schmidt passes, let \(r\) be the vector before final normalization; \(\mathrm{purity} = \|r\| / \max(\mathrm{raw\_norm}, 10^{-8})\).

Then

\[
\mathrm{quality} = \mathrm{snr} \cdot (1 - \mathrm{cos\_sim}) \cdot \mathrm{purity}.
\]

**Intuition:** larger raw separation, less alignment between harmful and harmless means, and more of that separation surviving orthogonalization → higher quality.

`run.py` prints the top layers; `test_quick.py` ablates only the **best** layer by this score.

---

## 5. Phase C — Weight refusal removal (`remove refusals.py`)

For each layer \(\ell\), we take **the same** hidden-space direction vector \(\hat{r}_\ell \in \mathbb{R}^d\) and apply it to **`o_proj`** and **`down_proj`** in that layer. Those matrices map **into** the residual stream width \(d\), so \(d_{\text{out}} = d\) and \(r\) lives in the same space as the **rows** of \(W\) index output coordinates.

### 5.1 Goal in linear algebra

Let \(W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\). Let \(r \in \mathbb{R}^{d_{\text{out}}}\) be a **unit** column vector. Define the **output-space** projector

\[
P = I_{d_{\text{out}}} - r r^\top.
\]

We want a new weight

\[
W' = P\, W = W - r\,(r^\top W).
\]

- \(r^\top W\) is a **row vector** of length \(d_{\text{in}}\); in code `proj_vec = r @ W` with shapes `[d_out]` @ `[d_out, d_in]` → `[d_in]`.
- Outer product \(r\,(r^\top W)\) has shape `[d_out, d_in]`; code: `r.unsqueeze(1) * proj_vec.unsqueeze(0)`.

So each **row** \(j\) of \(W\) has its component along \(r\) removed in the row’s output-coordinate sense consistent with left multiplication by \(P\).

### 5.2 Effect on the linear map

Recall \(y = x\, W^\top\). Then

\[
y' = x\, W'^\top = x\, W^\top\, P^\top = y\, P^\top = y\, P
\]

because \(P\) is symmetric. So **each forward output** is **post-multiplied** by the projector \(P\): the component of \(y\) along \(r\) is zero **for every input** \(x\):

\[
y'\, r = 0.
\]

So the intervention is: **remove the readout of weights along direction \(r\) in output space**. That is why \(r\) must be \(d_{\text{out}}\)-dimensional and why the code uses `r @ W` rather than right-multiplying by \(r\) (which would be the wrong shape for general rectangular \(W\)).

### 5.3 Double pass

The code applies the same rank-1 subtraction **twice**:

\[
W_1 = W - r(r^\top W),\qquad
W_2 = W_1 - r(r^\top W_1).
\]

In exact arithmetic one pass suffices for projection onto \(\mathrm{span}\{r\}^\perp\); the second pass reduces drift from **float32** rounding.

### 5.4 Row-norm preservation

Strict orthogonal projection can change the **Euclidean norm of each row** of \(W\). The code stores

\[
n_j^{\mathrm{old}} = \|W_{j,:}\|_2
\]

per row \(j\), computes \(W_2\), then scales each row of \(W_2\) so its norm matches \(n_j^{\mathrm{old}}\):

\[
W_{\text{final}} = D\, W_2
\]

where \(D\) is diagonal with \(D_{jj} = n_j^{\mathrm{old}} / (\|W_{2,j,:}\|_2 + \varepsilon)\), \(\varepsilon = 10^{-8}\).

So the edit is **approximately** a refusal-direction removal, then a **row-wise renorm** to keep magnitude statistics similar to before (heuristic for stability / capacity).

Finally weights are cast back to **`float16`** for inference.

### 5.5 Where it is applied

For every layer index \(i = 0,\ldots,L-1\):

- `layer.self_attn.o_proj.weight`
- `layer.mlp.down_proj.weight`

Each uses **that layer’s** direction `refusal_directions[i]`. There is **no** cross-layer sharing of directions in the current code.

---

## 6. How this relates to “steering”

| Idea | What this repo does |
|------|---------------------|
| **Activation steering** | At inference, add \(\alpha v\) to some hidden activation to push behavior. **Not implemented here.** |
| **Weight steering / refusal removal** | **Implemented:** bake a fixed projector into selected weights so outputs lose a component along \(r\). |
| **Representation engineering** | Broad term; this pipeline estimates \(r\) from data statistics then edits weights. |

The **“direction”** is estimated from **activation differences** (harmful vs harmless). The **intervention** is applied in **weight space** on matrices that **write into** that same hidden dimension. The hope is that the estimated \(r\) aligns with refusal-triggering readout, so removing it reduces refusals.

---

## 7. Evaluation hook (`evaluate.py`, used by `run.py`)

Generation uses sampling (`temperature`, `top_p`, `top_k`). **Refusal** is not from logits; it is a **string heuristic**: if the response contains substrings from `REFUSAL_MARKERS` (case-insensitive), it is classified as refused unless enough lines look like substantive non-refusal content (`disclaimer_but_complied`). This is a **coarse** behavioral metric, not a proof of safety or harmlessness.

---

## 8. Important caveats

1. **Single-direction, per-layer model:** The pipeline assumes one vector \(\hat{r}_\ell\) per layer captures the relevant difference. Real refusal behavior is higher-rank and contextual.

2. **Dataset confounding:** \(\mu^{\mathrm{harm}} - \mu^{\mathrm{safe}}\) can reflect topic or style, not only “refusal.” Orthogonalizing against \(\mu^{\mathrm{safe}}\) mitigates only the part parallel to the harmless centroid.

3. **Norm rescaling breaks strict projection:** After row rescaling, \(r^\top W_{\text{final}}\) is **small** but not necessarily **exactly** zero; the unit tests check approximate removal.

4. **`SCALE` in `config.py`:** Present as a constant but **not applied** in `remove refusals.py` in the current code; changing it has no effect unless you wire it in.

5. **Ethics:** Reducing refusals can increase **harmful compliance**. This is research tooling; treat outputs and checkpoints responsibly.

---

## 9. Quick symbol table

| Symbol | Meaning |
|--------|---------|
| \(d\) | Hidden size (1536) |
| \(L\) | Number of text layers (35) |
| \(h_\ell^{(p)}\) | Last-token hidden state at layer \(\ell\), prompt \(p\) |
| \(\tilde{h}\) | Winsorized activation |
| \(\mu_\ell^{\mathrm{harm}}, \mu_\ell^{\mathrm{safe}}\) | Per-layer mean activations |
| \(\delta_\ell\) | Raw difference \(\mu^{\mathrm{harm}} - \mu^{\mathrm{safe}}\) |
| \(\hat{r}_\ell\) | Unit direction after removing \(\hat{h}\) component and normalizing |
| \(W, W'\) | Original and modified linear weights `[out, in]` |
| \(P\) | \(I - r r^\top\) (output-space projector) |

---

## 10. File map

| File | Role |
|------|------|
| `harvest.py` | Collect \(h_\ell^{(p)}\) (last token) per layer |
| `directions.py` | Winsorize → means → \(\delta\) → remove \(\hat{h}\) → \(\hat{r}_\ell\) + quality |
| `remove refusals.py` | \(W' \approx P W\) with row-norm fixup, per `o_proj` / `down_proj` |
| `run.py` | End-to-end orchestration + optional cache and save |
| `README.md` | Operational usage |

This should be enough to read the implementation line-by-line with a clear picture of the linear algebra and the steering philosophy used here.

---

## 11. Credits and upstream resources

- **Research repo and broader context:** [TrevorS/gemma-4-refusal removal](https://github.com/TrevorS/gemma-4-refusal removal/tree/master).
- **Model weights:** [`google/gemma-4-E2B-it` on Hugging Face](https://huggingface.co/google/gemma-4-E2B-it).
- **Datasets used by this repo’s scripts:** [`mlabonne/harmful_behaviors`](https://huggingface.co/datasets/mlabonne/harmful_behaviors), [`tatsu-lab/alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca).

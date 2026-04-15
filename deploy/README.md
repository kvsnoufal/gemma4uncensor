---
title: Gemma 4 E2B — Refusal Removal Study
emoji: 🔬
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.12.0
app_file: app.py
pinned: false
license: apache-2.0
---

# Gemma 4 E2B — Refusal Removal Study

Side-by-side comparison of the original safety-trained Gemma 4 E2B model versus a version with refusal directions surgically removed from 70 weight matrices via norm-preserving rank-1 projection.

**Inspired by:** [github.com/TrevorS/gemma-4-abliteration](https://github.com/TrevorS/gemma-4-abliteration)

## How it works

1. The model is loaded once and all weights are kept in memory
2. Refusal directions are pre-computed (included as `cache/directions.pt`)
3. Per query: weights are swapped to original → generate → swap to uncensored → generate
4. Both responses are shown side-by-side with a classification badge

## Setup

Set `HF_TOKEN` as a Space secret (required — Gemma 4 is a gated model).

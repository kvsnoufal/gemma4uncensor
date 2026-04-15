"""Quick smoke test: 10 prompts, abliterate top-1 layer, verify before/after."""

import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DEVICE, DTYPE, MODEL_PATH, TEST_PROMPTS, get_text_layers, clear_mps_cache
from data import load_harmful_prompts, load_harmless_prompts, format_prompts
from harvest import harvest_activations
from directions import compute_refusal_directions
from abliterate import modify_weight_norm_preserved
from evaluate import generate_response, classify_response


def main():
    print("=" * 60)
    print("QUICK SMOKE TEST")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── Load model ───────────────────────────────────────────────────────
    print("\n1. Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto"
    )
    model.eval()
    print(f"   Loaded in {time.time() - t0:.1f}s")

    # ── Before: generate response to test prompt ─────────────────────────
    test_prompt = TEST_PROMPTS[0]
    print(f"\n2. Before refusal removal — prompt: '{test_prompt}'")
    before_response = generate_response(model, tokenizer, test_prompt, DEVICE)
    before_class = classify_response(before_response)
    print(f"   [{before_class.upper()}] {before_response[:300]}")

    # ── Harvest (10 each) ────────────────────────────────────────────────
    print("\n3. Harvesting activations (10 harmful + 10 harmless)...")
    harmful = format_prompts(tokenizer, load_harmful_prompts(10))
    harmless = format_prompts(tokenizer, load_harmless_prompts(10))

    t0 = time.time()
    harmful_acts = harvest_activations(model, tokenizer, harmful, DEVICE)
    harmless_acts = harvest_activations(model, tokenizer, harmless, DEVICE)
    print(f"   Harvested in {time.time() - t0:.1f}s")
    print(f"   Shapes: harmful={harmful_acts.shape}, harmless={harmless_acts.shape}")

    # ── Compute directions ───────────────────────────────────────────────
    print("\n4. Computing refusal directions...")
    refusal_dirs, layer_qualities = compute_refusal_directions(harmful_acts, harmless_acts)
    del harmful_acts, harmless_acts
    gc.collect()

    best_layer, best_quality = layer_qualities[0]
    print(f"   Best layer: {best_layer} (quality={best_quality:.4f})")
    print(f"   Top-5: {[(l, f'{q:.3f}') for l, q in layer_qualities[:5]]}")

    # ── Abliterate single layer ──────────────────────────────────────────
    print(f"\n5. Abliterating layer {best_layer} only...")
    layers = get_text_layers(model)
    layer = layers[best_layer]
    r = refusal_dirs[best_layer].to(DEVICE)

    # Test norm preservation on o_proj
    old_norms = layer.self_attn.o_proj.weight.data.float().cpu().norm(dim=1)

    layer.self_attn.o_proj.weight.data = modify_weight_norm_preserved(
        layer.self_attn.o_proj.weight.data, r
    )
    layer.mlp.down_proj.weight.data = modify_weight_norm_preserved(
        layer.mlp.down_proj.weight.data, r
    )

    new_norms = layer.self_attn.o_proj.weight.data.float().norm(dim=1)
    max_diff = (old_norms - new_norms.cpu()).abs().max().item()
    print(f"   o_proj row-norm max diff: {max_diff:.6f}")
    assert max_diff < 5e-3, f"Norm preservation failed: {max_diff}"
    print("   ✓ Norm preservation OK")

    clear_mps_cache()

    # ── After: generate response to same prompt ──────────────────────────
    print(f"\n6. After refusal removal — prompt: '{test_prompt}'")
    after_response = generate_response(model, tokenizer, test_prompt, DEVICE)
    after_class = classify_response(after_response)
    print(f"   [{after_class.upper()}] {after_response[:300]}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Before: [{before_class}] {before_response[:150]}...")
    print(f"After:  [{after_class}] {after_response[:150]}...")
    changed = before_response != after_response
    print(f"Response changed: {changed}")
    print(f"Norm preserved: ✓ (max diff={max_diff:.6f})")
    print("✓ Smoke test complete.")


if __name__ == "__main__":
    main()

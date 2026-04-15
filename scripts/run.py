"""Full refusal removal pipeline orchestrator."""

import argparse
import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    CACHE_DIR,
    DEVICE,
    DTYPE,
    MODEL_PATH,
    N_HARMFUL,
    N_HARMLESS,
    OUTPUT_DIR,
    TEST_PROMPTS,
    clear_mps_cache,
)
from data import load_harmful_prompts, load_harmless_prompts, format_prompts
from harvest import harvest_activations
from directions import compute_refusal_directions
from abliterate import abliterate_model
from evaluate import evaluate_model, print_report


def parse_args():
    parser = argparse.ArgumentParser(description="Gemma 4 E2B refusal removal pipeline")
    parser.add_argument("--skip-harvest", action="store_true", help="Use cached activations")
    parser.add_argument("--skip-eval", action="store_true", help="Skip extended eval")
    parser.add_argument("--save-model", action="store_true", help="Save abliterated weights")
    parser.add_argument("--n-harmful", type=int, default=N_HARMFUL)
    parser.add_argument("--n-harmless", type=int, default=N_HARMLESS)
    return parser.parse_args()


def main():
    args = parse_args()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    act_cache = CACHE_DIR / "activations.pt"
    dir_cache = CACHE_DIR / "directions.pt"

    # ── System info ──────────────────────────────────────────────────────
    print("=" * 60)
    print("GEMMA 4 E2B refusal removal PIPELINE")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Dtype:  {DTYPE}")
    print(f"Model:  {MODEL_PATH}")
    print()

    # ── Load model ───────────────────────────────────────────────────────
    print("Loading model and tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto"
    )
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print()

    # ── Baseline evaluation ──────────────────────────────────────────────
    print("── BASELINE EVALUATION ──")
    t0 = time.time()
    before = evaluate_model(model, tokenizer, TEST_PROMPTS, DEVICE)
    print(f"Baseline: {before['counts']['refused']}/{before['total']} refused "
          f"({before['refusal_rate']:.0%}) in {time.time() - t0:.1f}s")
    print()

    # ── Harvest activations ──────────────────────────────────────────────
    if args.skip_harvest and act_cache.exists():
        print("── LOADING CACHED ACTIVATIONS ──")
        cached = torch.load(act_cache, weights_only=True)
        harmful_acts = cached["harmful"]
        harmless_acts = cached["harmless"]
        print(f"Loaded: harmful {harmful_acts.shape}, harmless {harmless_acts.shape}")
    else:
        print(f"── HARVESTING ACTIVATIONS ({args.n_harmful} harmful + {args.n_harmless} harmless) ──")

        print("Loading datasets...")
        harmful_prompts = load_harmful_prompts(args.n_harmful)
        harmless_prompts = load_harmless_prompts(args.n_harmless)
        harmful_fmt = format_prompts(tokenizer, harmful_prompts)
        harmless_fmt = format_prompts(tokenizer, harmless_prompts)

        print(f"\nHarvesting harmful activations ({len(harmful_fmt)} prompts)...")
        t0 = time.time()
        harmful_acts = harvest_activations(model, tokenizer, harmful_fmt, DEVICE)
        print(f"Done in {time.time() - t0:.1f}s — shape: {harmful_acts.shape}")

        print(f"\nHarvesting harmless activations ({len(harmless_fmt)} prompts)...")
        t0 = time.time()
        harmless_acts = harvest_activations(model, tokenizer, harmless_fmt, DEVICE)
        print(f"Done in {time.time() - t0:.1f}s — shape: {harmless_acts.shape}")

        # Cache
        torch.save({"harmful": harmful_acts, "harmless": harmless_acts}, act_cache)
        print(f"Cached activations to {act_cache}")

    gc.collect()
    clear_mps_cache()
    print()

    # ── Compute refusal directions ───────────────────────────────────────
    if args.skip_harvest and dir_cache.exists():
        print("── LOADING CACHED DIRECTIONS ──")
        cached = torch.load(dir_cache, weights_only=True)
        refusal_dirs = cached["directions"]
        layer_qualities = cached["qualities"]
    else:
        print("── COMPUTING REFUSAL DIRECTIONS ──")
        t0 = time.time()
        refusal_dirs, layer_qualities = compute_refusal_directions(harmful_acts, harmless_acts)
        print(f"Done in {time.time() - t0:.1f}s")

        print("\nTop-10 layers by quality:")
        for idx, q in layer_qualities[:10]:
            print(f"  Layer {idx:2d}: {q:.4f}")

        # Cache
        torch.save({"directions": refusal_dirs, "qualities": layer_qualities}, dir_cache)
        print(f"Cached directions to {dir_cache}")

    # Free activation memory
    del harmful_acts, harmless_acts
    gc.collect()
    clear_mps_cache()
    print()

    # ── Abliteration (refusal removal) ───────────────────────────────────────────────────────
    print("── ABLITERATING MODEL (refusal removal) ──")
    t0 = time.time()
    abliterate_model(model, refusal_dirs, DEVICE)
    print(f"Abliteration complete in {time.time() - t0:.1f}s")
    print()

    # ── Post-refusal removal evaluation ─────────────────────────────────────
    print("── POST-refusal removal EVALUATION ──")
    t0 = time.time()
    after = evaluate_model(model, tokenizer, TEST_PROMPTS, DEVICE)
    print(f"Post: {after['counts']['refused']}/{after['total']} refused "
          f"({after['refusal_rate']:.0%}) in {time.time() - t0:.1f}s")

    print_report(before, after)

    # ── Extended evaluation ──────────────────────────────────────────────
    if not args.skip_eval:
        print("\n── EXTENDED EVALUATION (50 held-out prompts) ──")
        held_out = load_harmful_prompts(250)[200:250]  # prompts 200-249
        t0 = time.time()
        extended = evaluate_model(model, tokenizer, held_out, DEVICE)
        print(f"Extended: {extended['counts']['refused']}/{extended['total']} refused "
              f"({extended['refusal_rate']:.0%}) in {time.time() - t0:.1f}s")

    # ── Save model ───────────────────────────────────────────────────────
    if args.save_model:
        print(f"\n── SAVING ABLITERATED MODEL to {OUTPUT_DIR} ──")
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("Done.")

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()

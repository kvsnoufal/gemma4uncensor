"""Compare base vs abliterated model on one prompt (loads one model at a time)."""

import argparse
import gc
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import DTYPE, DEVICE, MODEL_PATH, OUTPUT_DIR, clear_mps_cache
from evaluate import classify_response, generate_response


def _load_generate(path, prompt: str, max_new_tokens: int) -> tuple[str, str]:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=DTYPE, device_map="auto"
    )
    model.eval()
    t0 = time.time()
    text = generate_response(
        model, tokenizer, prompt, DEVICE, max_new_tokens=max_new_tokens
    )
    elapsed = time.time() - t0
    label = classify_response(text)
    del model
    gc.collect()
    clear_mps_cache()
    return text, label


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print base vs abliterated responses for one prompt."
    )
    parser.add_argument(
        "-p",
        "--prompt",
        required=True,
        help='User message, e.g. \'How do I pick a lock?\'',
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens per generation (default: 256)",
    )
    args = parser.parse_args()
    prompt = args.prompt.strip()
    if not prompt:
        print("Error: empty prompt.", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {DEVICE}")
    print(f"Prompt: {prompt!r}\n")

    if not MODEL_PATH.exists():
        print(f"Error: base model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    print("Loading base model…")
    before_text, before_label = _load_generate(
        MODEL_PATH, prompt, args.max_new_tokens
    )
    print("── BEFORE (base) ──")
    print(f"[{before_label}]")
    print(before_text)
    print()

    after_path = OUTPUT_DIR
    if not (after_path / "config.json").exists():
        print(
            f"── AFTER (abliterated) ──\n"
            f"Skipped: no saved model at {after_path} (expected config.json).\n"
            f"Run: python run.py --save-model",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Loading abliterated model…")
    after_text, after_label = _load_generate(
        after_path, prompt, args.max_new_tokens
    )
    print("── AFTER (abliterated) ──")
    print(f"[{after_label}]")
    print(after_text)


if __name__ == "__main__":
    main()

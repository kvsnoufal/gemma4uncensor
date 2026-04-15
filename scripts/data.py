"""Dataset loading and prompt formatting for refusal removal pipeline.

HF datasets: https://huggingface.co/datasets/mlabonne/harmful_behaviors ,
https://huggingface.co/datasets/tatsu-lab/alpaca
Method context: https://github.com/TrevorS/gemma-4-refusal removal/tree/master
"""

from datasets import load_dataset

from config import N_HARMFUL, N_HARMLESS


def load_harmful_prompts(n: int = N_HARMFUL) -> list[str]:
    """Load harmful prompts from mlabonne/harmful_behaviors (HF: huggingface.co/datasets/mlabonne/harmful_behaviors)."""
    ds = load_dataset("mlabonne/harmful_behaviors", split="train")
    return [row["text"] for row in ds.select(range(min(n, len(ds))))]


def load_harmless_prompts(n: int = N_HARMLESS) -> list[str]:
    """Load harmless prompts from tatsu-lab/alpaca (HF: huggingface.co/datasets/tatsu-lab/alpaca; simple instructions only)."""
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    # Filter to entries with no extra input context
    simple = [row["instruction"] for row in ds if row["input"] == ""]
    return simple[:n]


def format_prompts(tokenizer, prompts: list[str]) -> list[str]:
    """Format raw prompts through the model's chat template."""
    formatted = []
    for text in prompts:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted.append(out)
    return formatted


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from config import MODEL_PATH

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("\n── Harmful prompts (first 3) ──")
    harmful = load_harmful_prompts(5)
    for p in harmful[:3]:
        print(f"  • {p[:80]}...")

    print("\n── Harmless prompts (first 3) ──")
    harmless = load_harmless_prompts(5)
    for p in harmless[:3]:
        print(f"  • {p[:80]}...")

    print("\n── Formatted example ──")
    fmt = format_prompts(tokenizer, harmful[:1])
    print(repr(fmt[0]))

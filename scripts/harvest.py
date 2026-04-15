"""Activation harvesting: collect last-token residual stream activations per layer."""

import gc

import torch

from config import DEVICE, MAX_SEQ_LEN, get_text_layers, clear_mps_cache


def harvest_activations(
    model,
    tokenizer,
    formatted_prompts: list[str],
    device: torch.device = DEVICE,
) -> torch.Tensor:
    """Run forward passes and collect last-token hidden states from every layer.

    Returns:
        Tensor of shape [n_prompts, n_layers, hidden_size] on CPU.
    """
    layers = get_text_layers(model)
    n_layers = len(layers)
    n_prompts = len(formatted_prompts)

    # Storage: list of [n_layers, hidden_size] tensors (one per prompt), kept on CPU
    all_activations = []

    for i, prompt in enumerate(formatted_prompts):
        # Per-prompt storage for hook captures
        layer_acts = {}

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                # Gemma4TextDecoderLayer returns a plain tensor, but handle tuples defensively
                out = output[0] if isinstance(output, tuple) else output
                # Last token hidden state → CPU immediately
                layer_acts[layer_idx] = out[:, -1, :].squeeze(0).detach().cpu()
            return hook_fn

        # Register hooks
        handles = []
        for li in range(n_layers):
            h = layers[li].register_forward_hook(make_hook(li))
            handles.append(h)

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding=False,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass (no grad)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)

        # Remove hooks
        for h in handles:
            h.remove()

        # Stack this prompt's activations: [n_layers, hidden_size]
        prompt_acts = torch.stack([layer_acts[li] for li in range(n_layers)])
        all_activations.append(prompt_acts)

        # Free memory
        del input_ids, attention_mask, inputs, layer_acts, prompt_acts
        gc.collect()
        clear_mps_cache()

        # Progress
        if (i + 1) % 10 == 0 or i == 0 or i == n_prompts - 1:
            print(f"  [{i + 1}/{n_prompts}]")

    # Stack all: [n_prompts, n_layers, hidden_size]
    result = torch.stack(all_activations)
    del all_activations
    gc.collect()
    return result


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from config import MODEL_PATH, DTYPE
    from data import load_harmful_prompts, format_prompts

    print("Loading model for harvest test (5 prompts)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=DTYPE, device_map="auto"
    )

    harmful = load_harmful_prompts(5)
    formatted = format_prompts(tokenizer, harmful)

    print("Harvesting...")
    acts = harvest_activations(model, tokenizer, formatted)
    print(f"Result shape: {acts.shape}")  # expect [5, 35, 1536]
    print(f"Dtype: {acts.dtype}, Device: {acts.device}")

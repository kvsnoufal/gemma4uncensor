"""Norm-preserving biprojection weight modification for refusal removal."""

import gc

import torch
import torch.nn.functional as F

from config import DEVICE, TARGET_MODULES, get_text_layers, clear_mps_cache


def modify_weight_norm_preserved(weight: torch.Tensor, refusal_dir: torch.Tensor) -> torch.Tensor:
    """Remove refusal direction from a weight matrix while preserving row norms.

    Args:
        weight: [out_features, in_features] — the linear layer weight
        refusal_dir: [out_features] — unit refusal direction in output space

    Returns:
        Modified weight with same shape and dtype, row norms preserved.
    """
    W = weight.float()
    r = F.normalize(refusal_dir.float(), dim=0)  # ensure unit vector [out_features]

    # Save original row norms
    old_row_norms = W.norm(dim=1, keepdim=True)  # [out_features, 1]

    # Project out refusal direction: W_new = (I - r r^T) @ W
    # Efficient rank-1 update: proj_vec = r^T @ W → [in_features]
    proj_vec = r @ W  # [in_features]
    W_new = W - r.unsqueeze(1) * proj_vec.unsqueeze(0)

    # Double-pass for numerical cleanup
    proj_vec2 = r @ W_new
    W_new = W_new - r.unsqueeze(1) * proj_vec2.unsqueeze(0)

    # Norm preservation: rescale each row to original magnitude
    new_row_norms = W_new.norm(dim=1, keepdim=True)
    scale = old_row_norms / (new_row_norms + 1e-8)
    W_final = W_new * scale

    return W_final.to(weight.dtype)


def abliterate_model(
    model,
    refusal_directions: torch.Tensor,
    device: torch.device = DEVICE,
):
    """Apply norm-preserving biprojection to o_proj and down_proj in every layer.

    This method removes refusal directions from weight matrices via rank-1 projection
    while preserving row norms (norm-preserving biprojection).

    Args:
        model: The loaded model (modified in-place).
        refusal_directions: [n_layers, hidden_size] float32 on CPU.
        device: Device where model weights live.
    """
    layers = get_text_layers(model)
    n_layers = len(layers)

    for i in range(n_layers):
        r = refusal_directions[i].to(device)
        layer = layers[i]

        for module_name in TARGET_MODULES:
            # Navigate to the target module (e.g., self_attn.o_proj or mlp.down_proj)
            if module_name == "o_proj":
                target = layer.self_attn.o_proj
            elif module_name == "down_proj":
                target = layer.mlp.down_proj
            else:
                raise ValueError(f"Unknown target module: {module_name}")

            target.weight.data = modify_weight_norm_preserved(target.weight.data, r)

        clear_mps_cache()

        if (i + 1) % 5 == 0 or i == 0 or i == n_layers - 1:
            print(f"  abliterated layer [{i + 1}/{n_layers}]")

    gc.collect()
    clear_mps_cache()


if __name__ == "__main__":
    # Quick test: verify norm preservation on a random matrix
    print("Testing norm preservation...")
    W = torch.randn(1536, 4096, dtype=torch.float16)
    r = F.normalize(torch.randn(1536), dim=0)

    old_norms = W.float().norm(dim=1)
    W_mod = modify_weight_norm_preserved(W, r)
    new_norms = W_mod.float().norm(dim=1)

    max_diff = (old_norms - new_norms).abs().max().item()
    print(f"Max row-norm difference: {max_diff:.6f}")
    assert max_diff < 5e-3, f"Norm preservation failed: max diff = {max_diff}"
    print("PASSED: Row norms preserved within tolerance.")

    # Verify refusal component is removed from output space
    # For W [out, in], the output-space projection is: (r^T @ W) should be near zero
    proj_before = (r @ W.float()).abs().mean().item()
    proj_after = (r @ W_mod.float()).abs().mean().item()
    print(f"Mean |output-space projection| before: {proj_before:.4f}, after: {proj_after:.6f}")

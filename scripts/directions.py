"""Compute per-layer orthogonalized refusal directions from harvested activations."""

import torch
import torch.nn.functional as F

from config import WINSORIZE_QUANTILE


def _winsorize(tensor: torch.Tensor, quantile: float = WINSORIZE_QUANTILE) -> torch.Tensor:
    """Clamp outlier values in-place. Operates on CPU (torch.quantile unsupported on MPS)."""
    t = tensor.float().cpu()
    q = torch.quantile(t.abs(), quantile, dim=-1, keepdim=True)
    return t.clamp_(-q, q)


def compute_refusal_directions(
    harmful_acts: torch.Tensor,
    harmless_acts: torch.Tensor,
) -> tuple[torch.Tensor, list[tuple[int, float]]]:
    """Compute per-layer orthogonalized refusal directions.

    Args:
        harmful_acts:  [n_harmful, n_layers, hidden_size] on CPU
        harmless_acts: [n_harmless, n_layers, hidden_size] on CPU

    Returns:
        refusal_directions: [n_layers, hidden_size] float32 on CPU (unit vectors)
        layer_qualities: list of (layer_idx, quality_score) sorted descending
    """
    n_layers = harmful_acts.shape[1]

    # Winsorize
    harmful_w = _winsorize(harmful_acts)
    harmless_w = _winsorize(harmless_acts)

    # Per-layer means (float32)
    harmful_means = harmful_w.mean(dim=0)   # [n_layers, hidden_size]
    harmless_means = harmless_w.mean(dim=0)

    # Raw refusal directions
    raw_dirs = harmful_means - harmless_means  # [n_layers, hidden_size]

    # Orthogonalize and compute quality scores
    refusal_dirs = torch.zeros_like(raw_dirs)
    layer_qualities = []

    for l in range(n_layers):
        r = raw_dirs[l]
        h = harmless_means[l]

        # Normalize harmless mean
        h_hat = F.normalize(h, dim=0)

        # Double-pass Gram-Schmidt: remove component along harmless mean
        r = r - (r @ h_hat) * h_hat  # pass 1
        r = r - (r @ h_hat) * h_hat  # pass 2

        # Normalize to unit vector
        r_norm = r.norm()
        refusal_dirs[l] = F.normalize(r, dim=0) if r_norm > 1e-8 else r

        # Quality metrics
        raw_norm = raw_dirs[l].norm().item()
        snr = raw_norm / max(harmful_means[l].norm().item(), harmless_means[l].norm().item(), 1e-8)
        cos_sim = F.cosine_similarity(
            harmful_means[l].unsqueeze(0),
            harmless_means[l].unsqueeze(0),
        ).item()
        purity = r_norm.item() / max(raw_norm, 1e-8)
        quality = snr * (1.0 - cos_sim) * purity

        layer_qualities.append((l, quality))

    # Sort by quality descending
    layer_qualities.sort(key=lambda x: x[1], reverse=True)

    return refusal_dirs, layer_qualities


if __name__ == "__main__":
    # Quick test with random data
    print("Testing directions computation with random data...")
    h_acts = torch.randn(10, 35, 1536)
    s_acts = torch.randn(10, 35, 1536)

    dirs, quals = compute_refusal_directions(h_acts, s_acts)
    print(f"Directions shape: {dirs.shape}")
    print(f"Direction norms (should be ~1.0): {dirs.norm(dim=1)[:5]}")
    print(f"\nTop-5 layers by quality:")
    for layer_idx, q in quals[:5]:
        print(f"  Layer {layer_idx:2d}: quality={q:.4f}")

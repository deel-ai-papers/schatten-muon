"""
Approximate optimal p* via randomized top-k SVD with tail correction.

Instead of full SVD, computes rank-k approximation via torch.svd_lowrank
and models the tail singular values with a power-law decay fitted from
the bottom of the captured spectrum, rescaled to match the residual
Frobenius energy exactly.

Complexity: O(mn·k) for randomized SVD,
            O(mn) for Frobenius norm (single pass),
            O(k + n_tail) per J(p) evaluation.
"""

import torch
from .common import maximize_p


def _schatten_power_sum(svs_topk, frob_sq, s, n_tail):
    """
    Approximate Σ σ_i^s from top-k singular values + Frobenius residual.

    The tail is modeled by fitting a geometric decay rate from the last
    few captured singular values, then rescaling the synthetic tail to
    match the residual Frobenius energy exactly. Falls back to uniform
    tail if the anchor values are too small.
    """
    topk_sum_s = (svs_topk**s).sum()
    tail_energy = (frob_sq - (svs_topk**2).sum()).clamp(min=0.0)

    if n_tail > 0 and tail_energy > 1e-12:
        # Fit decay rate from last few captured singular values
        n_anchor = min(3, len(svs_topk))
        anchor = svs_topk[-n_anchor:]

        if n_anchor >= 2 and anchor.min() > 1e-10:
            # Geometric mean decay per index
            ratio = (anchor[-1] / anchor[0]).pow(1.0 / (n_anchor - 1))
            # Clamp to avoid degenerate extrapolation
            ratio = ratio.clamp(min=0.01, max=0.999)

            # Synthetic tail: σ_{k+i} ≈ σ_k · ratio^i
            indices = torch.arange(
                1, n_tail + 1, dtype=svs_topk.dtype, device=svs_topk.device
            )
            tail_svs = anchor[-1] * ratio**indices

            # Rescale to match residual Frobenius energy exactly
            tail_frob_sq = (tail_svs**2).sum()
            if tail_frob_sq > 1e-12:
                tail_svs = tail_svs * (tail_energy / tail_frob_sq).sqrt()

            tail_sum_s = (tail_svs**s).sum()
        else:
            # Fallback: uniform tail
            sigma_tail = (tail_energy / n_tail).sqrt()
            tail_sum_s = n_tail * sigma_tail**s
    else:
        tail_sum_s = 0.0

    return topk_sum_s + tail_sum_s


def optimal_p(G, A, k=20, grid=None):
    """
    Approximate argmax of J(p) using rank-k randomized SVD.

    Parameters
    ----------
    G : Tensor (m1, n1)  – gradient matrix
    A : Tensor (m2, n2)  – activation matrix
    k : int              – number of top singular values to compute
    grid : list[float] or None

    Returns
    -------
    p_star : float
    log_J_star : float
    """
    k_g = min(k, min(G.shape) - 1)
    k_a = min(k, min(A.shape) - 1)

    _, Sk_g, _ = torch.svd_lowrank(G, q=k_g)
    frob_G_sq = G.pow(2).sum()
    n_tail_g = min(G.shape) - k_g

    _, Sk_a, _ = torch.svd_lowrank(A, q=k_a)
    frob_A_sq = A.pow(2).sum()
    n_tail_a = min(A.shape) - k_a

    def neg_log_J(p):
        p_t = torch.tensor(p, dtype=G.dtype, device=G.device)
        q_star = 1.0 + 1.0 / p_t
        k_p = 2.0 * (p_t + 1.0) / (p_t - 1.0)

        sum_g = _schatten_power_sum(Sk_g, frob_G_sq, q_star, n_tail_g)
        sum_a = _schatten_power_sum(Sk_a, frob_A_sq, k_p, n_tail_a)

        log_num = (p_t / (p_t + 1.0)) * sum_g.log()
        log_den = ((p_t - 1.0) / (2.0 * (p_t + 1.0))) * sum_a.log()
        return -(log_num - log_den).item()

    return maximize_p(neg_log_J, grid=grid)


if __name__ == "__main__":
    G = torch.randn(128, 512)
    A = torch.randn(256, 512)

    p_star, log_J = optimal_p(G, A, k=20)
    print(f"Top-k  p* = {p_star:.4f}  log J = {log_J:.6f}")

    p_grid, log_J_grid = optimal_p(
        G, A, k=20, grid=[1.25, 1.5, 2, 3, 5, 10, 25, 50, 100]
    )
    print(f"Grid   p* = {p_grid:<8}  log J = {log_J_grid:.6f}")

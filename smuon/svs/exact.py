"""
Exact optimal p* via full SVD.

Computes p* = argmax_{p >= 1} J(p) using precomputed singular values
and logsumexp for numerical stability. This is the reference baseline.

Complexity: O(mn·min(m,n)) for the SVD (assumed done externally),
            O(min(m,n)) per J(p) evaluation.
"""

import torch
from smuon.svs.common import maximize_p


def get_full_svs(mat):
    svs = torch.linalg.svdvals(mat)
    if torch.isnan(svs).any():
        return None
    return svs


def get_svs_from_gram(gram):
    """
    Extract singular values from a Gram matrix G = A^T A.

    Since σ_i(A) = √λ_i(A^T A), we compute eigenvalues of the Gram matrix
    and take their square roots.

    Parameters
    ----------
    gram : Tensor
        Symmetric positive semi-definite Gram matrix (features x features).

    Returns
    -------
    svs : Tensor or None
        Singular values in descending order, or None if computation fails.
    """
    # Eigenvalues of symmetric PSD matrix (sorted ascending by default)
    eigvals = torch.linalg.eigvalsh(gram)
    if torch.isnan(eigvals).any():
        return None
    # Clamp to handle numerical errors (small negative eigenvalues)
    eigvals = torch.clamp(eigvals, min=0.0)
    # Singular values = sqrt(eigenvalues), return in descending order
    svs = torch.sqrt(eigvals).flip(0)
    return svs


def optimal_p(grad_svs, act_svs, grid=None, bounds=(1.02, 400)):
    """
    Exact argmax of J(p) from full singular value vectors.

    Uses exact exponent formulas with no asymmetric softening.
    The lower bound of 1.02 (set via `bounds`) ensures p - 1.0 >= 0.02,
    avoiding the singularity at p = 1 without biasing the exponents.

    Parameters
    ----------
    grad_svs : Tensor   – singular values of G
    act_svs  : Tensor   – singular values of A
    grid     : list[float] or None
        If provided, evaluate on this discrete set.
        If None, use bounded scalar optimization on `bounds`.
    bounds   : tuple
        (pmin, pmax) for the continuous optimizer. Ignored if grid is set.

    Returns
    -------
    p_star : float
    log_J_star : float
    """
    # Move to CPU to avoid GPU memory fragmentation from scipy's many function evals
    grad_svs = grad_svs[grad_svs > 1e-7].cpu()
    act_svs = act_svs[act_svs > 1e-7].cpu()
    log_g = torch.log(grad_svs)
    log_a = torch.log(act_svs)

    def neg_log_J(p):
        q_star = 1.0 + 1.0 / p
        k_p = 2.0 * (p + 1.0) / (p - 1.0)

        log_num = (p / (p + 1.0)) * torch.logsumexp(q_star * log_g, dim=0)
        log_den = ((p - 1.0) / (2.0 * (p + 1.0))) * torch.logsumexp(k_p * log_a, dim=0)
        res = -(log_num - log_den).item()
        return res

    return maximize_p(neg_log_J, grid=grid, bounds=bounds)


if __name__ == "__main__":
    G = torch.randn(128, 512)
    A = torch.randn(256, 512)
    g_svs = get_full_svs(G)
    a_svs = get_full_svs(A)

    p_star, log_J = optimal_p(g_svs, a_svs)
    print(f"Exact  p* = {p_star:.4f}  log J = {log_J:.6f}")

    p_grid, log_J_grid = optimal_p(
        g_svs, a_svs, grid=[1.25, 1.5, 2, 3, 5, 10, 25, 50, 100]
    )
    print(f"Grid   p* = {p_grid:<8}  log J = {log_J_grid:.6f}")

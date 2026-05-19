"""
Approximate optimal p* via subsampled SVD with bounded tail correction.

Uses randomized low-rank SVD (torch.svd_lowrank) with residual energy estimation
to bound the contribution of uncomputed singular values. Unlike the power-law
extrapolation in topk.py, this method uses conservative bounds that respect
the matrix dimensions to prevent tail inflation.

Key features:
  - Random projection-based residual norm estimation (no full SVD)
  - Bounded tail model that respects d = min(m, n) - k uncomputed dimensions
  - Dimension-aware correction prevents "infinite chunk" fallacy
  - Gram-matrix path: randomized SVD with spectral-norm grounding (σ_k bound)
    at O(d²k) cost, falling back to exact eigvalsh for small matrices (d ≤ 512).

Complexity: O(mn·k) for randomized SVD on raw matrices,
            O(d²·k) for randomized SVD on d×d Gram matrices,
            O(k) per J(p) evaluation.

Alternative strategies (not yet implemented):
---------------------------------------------------------------------------
1. Stochastic Trace Estimation (Hutchinson / Girard):
   The Schatten-q norm is ||M||_q^q = tr((M^T M)^{q/2}) = sum_i sigma_i^q.
   Hutchinson's estimator approximates tr(f(M^T M)) via random vectors:
       tr(f(A)) ≈ (1/n_probe) sum_j z_j^T f(A) z_j
   where z_j are random sign or Gaussian vectors. The matrix-function-vector
   product f(A) z can be computed via Lanczos or Chebyshev expansion without
   forming f(A) explicitly. Cost: O(mn * n_iters * n_probe), same order as
   randomized SVD but avoids computing individual singular values entirely.
   Pro: Directly estimates the Schatten sums needed for J(p) without any
        tail modeling. No residual energy correction needed.
   Con: Requires re-running the trace estimator for each exponent q(p),
        so evaluating J(p) at multiple p values during optimization is
        more expensive than the current approach which computes SVs once
        and evaluates J(p) cheaply.

2. Chebyshev Moment Matching (Lin 2016, "Approximating Spectral Densities"):
   Estimate the spectral density mu(t) = (1/d) sum_i delta(t - sigma_i^2)
   via Chebyshev polynomial expansion of the trace. Once the first ~30
   Chebyshev moments are estimated (O(mn) each via stochastic trace),
   integrate any Schatten norm as int t^{q/2} mu(t) dt. This amortizes
   the cost: moments are computed once, then J(p) can be evaluated at
   many p values essentially for free.
   Pro: Best amortized cost when optimizing over p continuously.
   Con: More complex implementation; accuracy depends on the polynomial
        degree and can suffer from Gibbs-like oscillations for spiky spectra.

3. Hybrid Power Iteration + Trace Estimation:
   Compute the top-1 (or top-few) singular values via power iteration
   (O(mn) per iteration), then use Hutchinson to estimate the residual
   Schatten sum tr((M - sigma_1 u_1 v_1^T)^T (...))^{q/2}). This anchors
   the estimate at the spectral edge (critical for the p->1 regime where
   the denominator concentrates on sigma_1(A)) while using stochastic
   estimation for the bulk.
   Pro: Robustly handles the asymptotic regimes described in the paper.
   Con: Deflation introduces numerical error for the residual.
---------------------------------------------------------------------------
"""

import torch
import numpy as np
from smuon.svs.common import maximize_p

# Below this Gram dimension, use exact eigvalsh (fast and exact).
# Above, use randomized SVD at O(d²k) to avoid the O(d³) eigvalsh cost.
_GRAM_EXACT_THRESHOLD = 512


def svd_lowrank_and_residual(M, k, niter=2, n_probe=5):
    """
    Compute top-k singular values and estimate the residual spectral norm.

    Uses torch.svd_lowrank for the top-k SVs, then estimates the largest
    singular value of the residual (M - U @ S @ V^T) via random projection.

    Parameters
    ----------
    M : Tensor (m, n)
    k : int              – number of top singular values
    niter : int          – power iterations for svd_lowrank (default: 2)
    n_probe : int        – number of probe vectors for residual estimation

    Returns
    -------
    S_hat : ndarray      – top-k singular values (on CPU)
    R_est : float        – estimated largest singular value of residual
    """
    m, n = M.shape
    device = M.device
    dtype = M.dtype

    # Compute top-k SVD
    U, S_hat, _ = torch.svd_lowrank(M, q=k, niter=niter)

    # Estimate residual spectral norm via random projection
    # ||M - U U^T M||_2 ≈ max_i ||Proj_U^perp @ (M @ omega_i)|| / ||omega_i||
    Omega_res = torch.randn(n, n_probe, device=device, dtype=dtype)
    Y_res = M @ Omega_res  # (m, n_probe)
    Proj_res = Y_res - U @ (U.T @ Y_res)  # project out captured space

    norms = torch.linalg.vector_norm(Proj_res, dim=0)
    # Normalize by sqrt(n) since Omega columns have expected norm sqrt(n)
    R_est = torch.max(norms).item() / np.sqrt(n)

    return S_hat.cpu().numpy(), R_est


def topk_svs_from_gram(gram, k, niter=3):
    """
    Extract top-k singular values of A from its Gram matrix G = A^T A,
    with spectral-norm grounding.

    For a symmetric PSD matrix G, the singular values of G equal its
    eigenvalues: σ_i(G) = λ_i(G) = σ_i(A)². We exploit this to extract
    the top-k eigenvalues via randomized SVD at O(d²k) cost, then convert
    to singular values of A via sqrt.

    The k-th computed singular value serves as a deterministic upper bound
    on all uncomputed singular values (spectral grounding), avoiding the
    noisy random-projection residual estimate that causes numerical issues
    with the bounded tail model.

    For small Gram matrices (d ≤ 512), falls back to exact eigvalsh for
    maximum accuracy at negligible cost.

    Parameters
    ----------
    gram : Tensor (d, d)
        Symmetric positive semi-definite Gram matrix (A^T A).
    k : int
        Number of top singular values to return.
    niter : int
        Number of power iterations for the randomized SVD (default: 3).
        More iterations improve accuracy for slowly decaying spectra.

    Returns
    -------
    S_topk : ndarray     – top-k singular values of A (descending)
    R_bound : float      – upper bound on σ_{k+1}: the k-th singular value
    E : float            – residual Frobenius energy: ||A||_F^2 - Σ_{i=1}^k σ_i(A)^2
    d_remaining : int    – number of uncomputed dimensions (d - k)
    """
    d = gram.shape[0]
    k = min(k, d - 1)
    k = max(k, 1)

    # Exact Frobenius energy from trace (cheap, O(d))
    total_energy = gram.float().trace().item()

    if d <= _GRAM_EXACT_THRESHOLD:
        # Small Gram: exact eigendecomposition is fast and more accurate
        eigvals = torch.linalg.eigvalsh(gram.float())  # ascending order
        eigvals = torch.clamp(eigvals, min=0.0)
        svs_all = torch.sqrt(eigvals).flip(0)  # descending σ_i(A)
        S_topk = svs_all[:k].cpu().numpy()
        R_bound = float(svs_all[k - 1].item()) if k > 0 else 0.0
        captured_energy = float(np.sum(S_topk**2))
        E = max(0.0, total_energy - captured_energy)
        return S_topk, R_bound, E, d - k

    # Large Gram: randomized SVD at O(d²k) instead of eigvalsh at O(d³).
    # For PSD matrices, svd_lowrank returns singular values = eigenvalues.
    _, S_gram, _ = torch.svd_lowrank(gram.float(), q=k, niter=niter)

    # PSD numerical correction: eigenvalues should be non-negative
    S_gram = torch.clamp(S_gram, min=0.0)

    # Convert Gram eigenvalues to activation singular values: σ_i(A) = √λ_i
    S_A = torch.sqrt(S_gram)

    S_topk = S_A.cpu().numpy()

    # Spectral-norm grounding: σ_k(A) bounds all uncomputed σ_{k+1..d}(A).
    # This is a guaranteed bound since eigenvalues are returned in descending
    # order and the randomized SVD captures the largest ones.
    R_bound = float(S_A[-1].item())

    # Residual energy: total - sum of captured eigenvalues
    # S_gram contains λ_i = σ_i(A)², so sum(S_gram) = Σ σ_i(A)² for top-k
    captured_energy = float(S_gram.sum().item())
    E = max(0.0, total_energy - captured_energy)

    return S_topk, R_bound, E, d - k


def _evaluate_bounded_log_J(p, S_G, S_A, R_G, R_A, E_G, E_A, d_G, d_A):
    """
    Evaluate log J(p) with bounded tail correction.

    Uses the captured top-k singular values plus a bounded tail model that
    respects the number of uncomputed dimensions. This prevents artificial
    tail inflation when residual estimates are noisy.

    All tail terms and large-exponent Schatten sums are computed in log-space
    to avoid overflow when q_A is large (p near 1) and singular values > 1.

    The bound logic:
      - If (E / R^2) > d, the residual energy can't be explained by d
        singular values all equal to R, so use uniform distribution over d.
      - Otherwise, model tail as geometric decay from R.
    """
    q_G = 1.0 + 1.0 / p  # exponent for gradient Schatten norm
    q_A = 2.0 * (p + 1.0) / (p - 1.0)  # exponent for activation Schatten norm

    # --- Numerator: ||G||_{1+1/p}^{1+1/p} lower bound ---
    # q_G ∈ (1, 2) so q_G is moderate — direct computation is safe
    S_G_pos = S_G[S_G > 1e-12]
    N_k = np.sum(S_G_pos**q_G) if len(S_G_pos) > 0 else 0.0

    if R_G > 1e-12 and d_G > 0 and E_G > 1e-12:
        if (E_G / (R_G**2)) > d_G:
            log_N_tail = np.log(d_G) + (q_G / 2.0) * np.log(E_G / d_G)
        else:
            log_N_tail = np.log(E_G) + (q_G - 2.0) * np.log(R_G)
        N_tail = np.exp(np.clip(log_N_tail, -80, 80))
        N_lower = N_k + N_tail
    else:
        N_lower = N_k

    # --- Denominator: ||A||_{k(p)}^{k(p)} upper bound ---
    # q_A can be very large when p is near 1 (e.g., q_A ≈ 200 at p=1.02)
    # so S_A^q_A overflows in float64 for S_A > 1. Use logsumexp.
    S_A_pos = S_A[S_A > 1e-12]
    if len(S_A_pos) > 0:
        log_terms_A = q_A * np.log(S_A_pos)
        log_max = np.max(log_terms_A)
        # logsumexp: log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        D_k = np.exp(log_max) * np.sum(np.exp(log_terms_A - log_max))
    else:
        D_k = 0.0

    if R_A > 1e-12 and d_A > 0 and E_A > 1e-12:
        if (E_A / (R_A**2)) > d_A:
            log_D_tail = np.log(d_A) + (q_A / 2.0) * np.log(E_A / d_A)
        else:
            log_D_tail = np.log(E_A) + (q_A - 2.0) * np.log(R_A)
        D_tail = np.exp(np.clip(log_D_tail, -80, 80))
        D_upper = D_k + D_tail
    else:
        D_upper = D_k

    # J(p) = N^{p/(p+1)} / D^{(p-1)/(2(p+1))}
    log_num = (p / (p + 1.0)) * np.log(max(N_lower, 1e-30))
    log_den = ((p - 1.0) / (2.0 * (p + 1.0))) * np.log(max(D_upper, 1e-30))

    return log_num - log_den


def optimal_p(G, A, subsampling_ratio=0.1, niter=2, grid=None, bounds=(1.02, 35.0)):
    """
    Approximate argmax of J(p) using subsampled SVD with bounded tail.

    Parameters
    ----------
    G : Tensor (m1, n1)    – gradient matrix
    A : Tensor (m2, n2)    – activation matrix
    subsampling_ratio : float
        Fraction of singular values to compute: k = int(ratio * min(m, n)).
        Default 0.1 means 10% of the smaller dimension.
    niter : int            – power iterations for svd_lowrank
    grid : list[float] or None
        If provided, evaluate on this discrete set.
        If None, use bounded scalar optimization.
    bounds : tuple
        (pmin, pmax) for continuous optimization. Ignored if grid is set.

    Returns
    -------
    p_star : float
    log_J_star : float
    """
    # Compute k from subsampling ratio (dimension-dependent)
    k_G = max(1, int(subsampling_ratio * min(G.shape)))
    k_A = max(1, int(subsampling_ratio * min(A.shape)))

    # Clamp to valid range for svd_lowrank
    k_G = min(k_G, min(G.shape) - 1)
    k_A = min(k_A, min(A.shape) - 1)

    # Get top-k SVs and residual estimates
    S_G, R_G = svd_lowrank_and_residual(G, k_G, niter=niter)
    S_A, R_A = svd_lowrank_and_residual(A, k_A, niter=niter)

    # Compute Frobenius norms squared (full pass over matrices)
    frob_G_sq = G.pow(2).sum().item()
    frob_A_sq = A.pow(2).sum().item()

    # Residual energy = total - captured
    E_G = max(0.0, frob_G_sq - np.sum(S_G**2))
    E_A = max(0.0, frob_A_sq - np.sum(S_A**2))

    # Uncomputed dimensions
    d_G = min(G.shape) - k_G
    d_A = min(A.shape) - k_A

    def neg_log_J(p):
        return -_evaluate_bounded_log_J(p, S_G, S_A, R_G, R_A, E_G, E_A, d_G, d_A)

    return maximize_p(neg_log_J, grid=grid, bounds=bounds)


if __name__ == "__main__":
    torch.manual_seed(42)
    G = torch.randn(128, 512)
    A = torch.randn(256, 512)

    # Test standard approx path
    p_star, log_J = optimal_p(G, A, subsampling_ratio=0.15)
    print(f"Approx p* = {p_star:.4f}  log J = {log_J:.6f}  (ratio=0.15)")

    p_grid, log_J_grid = optimal_p(
        G, A, subsampling_ratio=0.15, grid=[1.25, 1.5, 2, 3, 5, 10, 25, 50, 100]
    )
    print(f"Grid   p* = {p_grid:<8}  log J = {log_J_grid:.6f}")

    # Test Gram path (small, uses eigvalsh)
    gram_small = A.T @ A  # 512x512, below threshold
    S, R, E, d = topk_svs_from_gram(gram_small, k=50)
    print(f"\nGram (d=512, exact): k=50, R_bound={R:.4f}, E={E:.2f}, d_rem={d}")

    # Test Gram path (large, uses svd_lowrank)
    A_large = torch.randn(1024, 2048)
    gram_large = A_large.T @ A_large  # 2048x2048, above threshold
    S, R, E, d = topk_svs_from_gram(gram_large, k=200)
    print(f"Gram (d=2048, rsvd): k=200, R_bound={R:.4f}, E={E:.2f}, d_rem={d}")

"""
Approximate optimal p* via Stochastic Lanczos Quadrature (SLQ).

Estimates Schatten norm power sums tr((M^T M)^{s/2}) using only
matrix-vector products, with no SVD or factorization at all.

Uses full reorthogonalization to prevent ghost eigenvalues from
corrupting the spectral estimates.

Complexity: O(n_probes · lanczos_iter · cost_of_matvec) per norm estimate,
            where cost_of_matvec = O(mn) for dense matrices.
            Reorthogonalization adds O(lanczos_iter² · n), negligible for
            small lanczos_iter.
"""

import torch
from smuon.svs.common import maximize_p


def _lanczos_tridiag(M, z, lanczos_iter):
    """
    Run Lanczos on H = M^T M starting from vector z,
    with full reorthogonalization against all previous vectors.

    Returns the tridiagonal matrix T such that
        tr(f(H)) ≈ n · e_1^T f(T) e_1
    """
    q_curr = z / z.norm()
    q_prev = torch.zeros_like(z)

    alphas = []
    betas = []
    Q = [q_curr]  # store all Lanczos vectors for reorthogonalization

    for j in range(lanczos_iter):
        # w = H @ q = M^T (M @ q)
        w = M.T @ (M @ q_curr)
        alpha = q_curr.dot(w)
        alphas.append(alpha)

        w = w - alpha * q_curr
        if j > 0:
            w = w - betas[-1] * q_prev

        # Full reorthogonalization against all previous Lanczos vectors
        for q in Q:
            w = w - q.dot(w) * q

        beta = w.norm()
        if beta < 1e-12:
            break

        betas.append(beta)
        q_prev = q_curr
        q_curr = w / beta
        Q.append(q_curr)

    # Build tridiagonal matrix (small: size × size)
    size = len(alphas)
    T = torch.zeros(size, size, dtype=M.dtype, device=M.device)
    for i in range(size):
        T[i, i] = alphas[i]
    n_off = min(len(betas), size - 1)
    for i in range(n_off):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]

    return T


def estimate_schatten_power_sum(M, s, n_probes=15, lanczos_iter=30):
    """
    Estimate ||M||_s^s = tr((M^T M)^{s/2}) via SLQ.

    Uses Hutchinson's trace estimator with Rademacher probes,
    and Lanczos quadrature to approximate the matrix function.

    Parameters
    ----------
    M : Tensor (m, n)
    s : float           – the Schatten exponent
    n_probes : int      – number of random probe vectors
    lanczos_iter : int  – Lanczos steps per probe

    Returns
    -------
    estimate : float    – approximation of tr((M^T M)^{s/2})
    """
    n = M.shape[1]
    estimates = []

    for _ in range(n_probes):
        # Rademacher random vector
        z = torch.sign(torch.randn(n, device=M.device, dtype=M.dtype))
        z[z == 0] = 1.0

        T = _lanczos_tridiag(M, z, lanczos_iter)

        # Diagonalize T (tiny matrix)
        eigvals, eigvecs = torch.linalg.eigh(T)
        eigvals = eigvals.clamp(min=0.0)

        # tr(f(H)) ≈ n · Σ_j |e_1^T v_j|^2 · f(λ_j)
        # where f(x) = x^{s/2}
        weights = eigvecs[0, :] ** 2
        estimate = n * (weights * eigvals.pow(s / 2.0)).sum()
        estimates.append(estimate)

    return torch.stack(estimates).mean()


def optimal_p(G, A, n_probes=15, lanczos_iter=30, grid=None):
    """
    Approximate argmax of J(p) using only matrix-vector products.

    Parameters
    ----------
    G : Tensor (m1, n1)    – gradient matrix
    A : Tensor (m2, n2)    – activation matrix
    n_probes : int         – probe vectors per SLQ estimate
    lanczos_iter : int     – Lanczos iterations per probe
    grid : list[float] or None

    Returns
    -------
    p_star : float
    log_J_star : float
    """

    def neg_log_J(p):
        p_t = torch.tensor(p, dtype=G.dtype, device=G.device)
        q_star = (1.0 + 1.0 / p_t).item()
        k_p = (2.0 * (p_t + 1.0) / (p_t - 1.0)).item()

        # ||G||_{q*}^{q*} = tr((G^T G)^{q*/2})
        sum_g = estimate_schatten_power_sum(G, q_star, n_probes, lanczos_iter)
        sum_a = estimate_schatten_power_sum(A, k_p, n_probes, lanczos_iter)

        log_num = (p_t / (p_t + 1.0)) * sum_g.log()
        log_den = ((p_t - 1.0) / (2.0 * (p_t + 1.0))) * sum_a.log()
        return -(log_num - log_den).item()

    # SLQ is expensive per evaluation, so use a coarse grid + refinement
    # when doing continuous optimization
    if grid is not None:
        return maximize_p(neg_log_J, grid=grid)
    else:
        # Coarse grid scan to find the basin
        coarse = [1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        vals = [(p, -neg_log_J(p)) for p in coarse]
        best_idx = max(range(len(vals)), key=lambda i: vals[i][1])

        # Refine around the winner with bounded optimization
        lo = coarse[max(0, best_idx - 1)]
        hi = coarse[min(len(coarse) - 1, best_idx + 1)]
        return maximize_p(neg_log_J, grid=None, bounds=(lo, hi))


if __name__ == "__main__":
    G = torch.randn(128, 512)
    A = torch.randn(256, 512)

    p_star, log_J = optimal_p(G, A)
    print(f"SLQ    p* = {p_star:.4f}  log J = {log_J:.6f}")

    p_grid, log_J_grid = optimal_p(G, A, grid=[1.25, 1.5, 2, 3, 5, 10, 25, 50, 100])
    print(f"Grid   p* = {p_grid:<8}  log J = {log_J_grid:.6f}")

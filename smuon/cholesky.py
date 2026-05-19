import torch


def estimate_spectral_norm(A, num_iters=15):
    """
    Estimates the largest singular value (2-norm) of A using Power Iteration.
    Extremely fast on GPU and avoids calling a full SVD.
    """
    m = A.shape[1]
    # Initialize a random vector
    v = torch.randn(m, 1, device=A.device, dtype=A.dtype)

    for _ in range(num_iters):
        u = A @ v
        v = A.T @ u
        v = v / torch.linalg.vector_norm(v)

    # Rayleigh quotient for the largest eigenvalue of A^T A
    lambda_max = torch.sum(v * (A.T @ (A @ v)))
    return torch.sqrt(lambda_max)


def batched_fractional_polar_zolotarev(A, p, zolotarev_coeffs):
    r"""
    Computes U \Sigma^p V^T efficiently using batched PyTorch operations.

    Parameters:
    -----------
    A : torch.Tensor
        Input matrix of shape (n, m) with n >= m. Should be on the target device (e.g., 'cuda').
    p : float
        The fractional power in [0, 1].
    zolotarev_coeffs : tuple
        A tuple (C0, weights, shifts) containing the Zolotarev coefficients.
        `weights` and `shifts` should be 1D torch.Tensors on the same device as A.

    Returns:
    --------
    torch.Tensor
        The approximated matrix U \Sigma^p V^T.
    """
    n, m = A.shape
    if n < m:
        raise ValueError("A must have more rows than columns (n >= m).")

    C0, weights, shifts = zolotarev_coeffs
    k = shifts.shape[0]

    # 1. Scale A using Power Iteration to avoid SVD overhead
    norm_A = estimate_spectral_norm(A)
    A_scaled = A / norm_A

    # Form the normal equations matrix (m, m)
    AtA = A_scaled.T @ A_scaled

    # 2. Create the batched shifted matrices
    # I has shape (m, m)
    I = torch.eye(m, device=A.device, dtype=A.dtype)

    # M_batched has shape (k, m, m).
    # Broadcasting expands AtA to k copies and adds the k respective scaled identities.
    M_batched = AtA.unsqueeze(0) + shifts.view(k, 1, 1) * I.unsqueeze(0)

    # 3. Batched Cholesky Factorization
    # L_batched has shape (k, m, m)
    L_batched = torch.linalg.cholesky(M_batched)

    # 4. Batched Cholesky Solve
    # We want to solve M_batched * X_batched = I
    # Expand I to shape (k, m, m) to act as the right-hand side
    B_batched = I.unsqueeze(0).expand(k, m, m)
    X_batched = torch.cholesky_solve(B_batched, L_batched)

    # 5. Accumulate the weighted sum
    # X_batched is (k, m, m), weights is (k,).
    # We multiply each (m, m) matrix by its scalar weight and sum along the batch dimension (dim=0).
    X_summed = torch.sum(X_batched * weights.view(k, 1, 1), dim=0)

    # 6. Final assembly: C0 * A_scaled + A_scaled @ X_summed
    # Factoring out A_scaled: A_scaled @ (C0 * I + X_summed)
    result = A_scaled @ (C0 * I + X_summed)

    # 7. Reverse the scaling
    return result * (norm_A**p)

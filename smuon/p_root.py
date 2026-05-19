import math
import torch
from smuon.coeffs.polar_express import optimal_composition

NS_COEFFS = optimal_composition(
    l=1e-3, num_iters=5, safety_factor_eps=1e-2, cushion=0.03
)


@torch.no_grad
def zeropower_via_newtonschulz5(G):
    """Polar factor via Newton-Schulz. Caller guarantees m <= n.
    Returns (P_bf16, s_inv) with s_inv = ||G||_{S4}^{-1}.
    """
    assert G.ndim >= 2
    assert G.size(-2) <= G.size(-1), "transpose tall inputs first"
    X = G.bfloat16()
    s_inv = None
    for i, (a, b, c) in enumerate(NS_COEFFS):
        A = X @ X.mT
        if i == 0:
            s_inv = A.norm(dim=(-2, -1), keepdim=True).rsqrt()
            X = X * s_inv
            A = A * (s_inv**2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X, s_inv


@torch.no_grad
def _polar_polish_fp32(P):
    """One fp32 Newton step: P <- (3I_m - P P^T)/2 @ P."""
    P = P.float()
    m = P.size(-2)
    I_m = torch.eye(m, dtype=P.dtype, device=P.device)
    PPt = P @ P.mT
    return (1.5 * I_m - 0.5 * PPt) @ P


@torch.no_grad
def fractional_power_from_polar(
    mat: torch.Tensor,
    p: float | torch.Tensor,
    max_extra_iters: int = 5,
    linf_tol: float = 1e-2,
    lambda_reg: float = 1e-5,
    polish: bool = True,
    eigen_floor: float = 1e-3,
    p_threshold: float = 0.0,
    normalize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate U Sigma^p V^T for mat = U Sigma V^T."""
    # --- 1. Shape: always work with m <= n ---
    transposed = mat.size(-2) > mat.size(-1)
    if transposed:
        mat = mat.mT
    m = mat.size(-2)

    # --- 2. p broadcasting ---
    if isinstance(p, torch.Tensor):
        p_view = p.view(-1, 1, 1) if p.ndim == 1 else p
        p_flat = p.to(device=mat.device, dtype=torch.float32).flatten()
    else:
        p_view = float(p)
        p_flat = torch.tensor([float(p)], dtype=torch.float32, device=mat.device)

    # --- 3. Trivial-p shortcuts ---
    if isinstance(p, torch.Tensor):
        if (p == 1.0).all():
            p_scale = mat.norm(dim=(-2, -1), keepdim=True)
            return (mat.mT if transposed else mat), p_scale
        if (p <= p_threshold).all():
            P, _ = zeropower_via_newtonschulz5(mat)
            return (P.mT if transposed else P), torch.ones(p.shape[0], 1, 1, device=mat.device, dtype=mat.dtype)
    else:
        if p == 1.0:
            p_scale = mat.norm(dim=(-2, -1), keepdim=True)
            return (mat.mT if transposed else mat), p_scale
        if p <= p_threshold:
            P, _ = zeropower_via_newtonschulz5(mat)
            return (P.mT if transposed else P), torch.tensor(1.0, device=mat.device, dtype=mat.dtype)

    # --- 4. Polar factor + fp32 polish ---
    P, s_inv = zeropower_via_newtonschulz5(mat)
    P = _polar_polish_fp32(P) if polish else P.float()

    # --- 5. Build M: symmetric + regularized, all fp32 ---
    mat_scaled = mat.float() * s_inv.float()
    M = mat_scaled @ P.mT 
    M = 0.5 * (M + M.mT)
    if lambda_reg > 0:
        M.diagonal(dim1=-2, dim2=-1).add_(lambda_reg)
    I_m = torch.eye(m, dtype=M.dtype, device=M.device)
    E = M - I_m

    # --- 6. Choose Taylor order ---
    max_k = max(1, max_extra_iters)
    best_k = max_k
    if max_k > 1:
        x = torch.logspace(math.log10(eigen_floor), 0.0, 200, dtype=torch.float32, device=mat.device)
        diff = x - 1.0
        exact = x.unsqueeze(0) ** p_flat.unsqueeze(1)
        approx = torch.ones_like(exact)
        power = torch.ones_like(exact)
        coeff = torch.ones_like(p_flat)
        errs = []
        for k in range(1, max_k + 1):
            coeff = coeff * (p_flat - k + 1) / k
            power = power * diff.unsqueeze(0)
            approx = approx + coeff.unsqueeze(1) * power
            errs.append((exact - approx).abs().max())
        
        for k, e in enumerate(torch.stack(errs).tolist(), start=1):
            if e <= linf_tol:
                best_k = k
                break

    c = [torch.ones_like(p_view)] if isinstance(p_view, torch.Tensor) else [1.0]
    for j in range(1, best_k + 1):
        c.append(c[-1] * (p_view - j + 1) / j)

    # --- 7. Horner ---
    if best_k == 1:
        S = c[1] * E
    else:
        S = c[best_k] * E + c[best_k - 1] * I_m
        for j in range(best_k - 2, 0, -1):
            S = S @ E + c[j] * I_m
        S = S @ E

    # --- 8. Final Assemble ---
    Mp_P = P + S @ P 

    if not normalize:
        scale_back = (1.0 / s_inv.float()) ** p_view
        out = scale_back * Mp_P
    else:
        out = Mp_P

    if isinstance(p, torch.Tensor):
        mask = p_view <= p_threshold
        if mask.any():
            out = torch.where(mask, P, out)

    p_scale = P.norm(dim=(-2, -1), keepdim=True)
    return (out.mT if transposed else out), p_scale

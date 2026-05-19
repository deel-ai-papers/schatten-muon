#!/usr/bin/env python3
"""
Fast Approx vs Exact p* benchmark.

Same template as experiments/approx_p.py, but uses a handful of named matrix
geometries covering the regimes where the tightness approximation is known
to behave differently:

  * flat spectrum, narrow / wide / square     -> exact stays at pmin;
                                                 approx should match to within ~0.
  * heavy-tailed M (power-law singular values) -> p* moves away from pmin;
                                                 approx tracks with single-digit error.
  * heavy-tailed M and heavy-tailed A          -> stress-test coupled tails.

Runs far fewer trials than the 250-random version but is equally diagnostic
because each case is deterministically seeded and has a known regime label,
so regressions show up in a specific row instead of getting averaged out.
"""

import argparse
import numpy as np
import torch

from smuon.svs.p_registry import create_p_approximator


def _heavy_tail_matrix(m, n, device, decay=1.0, scale=1.0):
    """sigma_i = scale * i^{-decay}, random orthogonal bases."""
    r = min(m, n)
    U, _ = torch.linalg.qr(torch.randn(m, r, device=device))
    V, _ = torch.linalg.qr(torch.randn(n, r, device=device))
    idx = torch.arange(1, r + 1, device=device, dtype=torch.float32)
    sigmas = scale * idx.pow(-decay)
    return U @ torch.diag(sigmas) @ V.T


def make_synthetic(m, n, batch_size, device, regime):
    """
    regime='noisy'      : flat-spectrum M, G dominated by Laplace noise
    regime='aligned'    : flat-spectrum M, G ~ M
    regime='heavy'      : M has power-law singular values, G = M + small noise
    regime='activation' : heavy-tailed M AND heavy-tailed A
    """
    if regime == "noisy":
        mom = torch.randn(m, n, device=device) * 0.01
        lap_scale = 0.05
    elif regime == "aligned":
        mom = torch.randn(m, n, device=device) * 0.01
        lap_scale = 0.002
    elif regime == "heavy":
        mom = _heavy_tail_matrix(m, n, device, decay=1.5, scale=0.5)
        lap_scale = 0.005
    elif regime == "activation":
        mom = _heavy_tail_matrix(m, n, device, decay=0.8, scale=0.3)
        lap_scale = 0.005
    else:
        raise ValueError(regime)

    lap = torch.distributions.Laplace(
        torch.zeros(m, n, device=device),
        torch.full((m, n), lap_scale, device=device),
    ).sample()
    grad = mom + lap

    if regime == "activation":
        r = min(n, batch_size)
        U, _ = torch.linalg.qr(torch.randn(n, r, device=device))
        V, _ = torch.linalg.qr(torch.randn(batch_size, r, device=device))
        idx = torch.arange(1, r + 1, device=device, dtype=torch.float32)
        sA = 10.0 * idx.pow(-0.7)
        act = torch.relu(U @ torch.diag(sA) @ V.T)
    else:
        act = torch.relu(torch.randn(n, batch_size, device=device))
    return grad, act, mom


def run_shape(m, n, regime, args, device, seed):
    torch.manual_seed(seed)
    exact = create_p_approximator(
        method="exact_tightness",
        pmin=args.pmin,
        pmax=args.pmax,
        subsampling_ratio=args.subsampling_ratio,
        sv_momentum=args.sv_momentum,
    )
    approx = create_p_approximator(
        method="approx_tightness",
        pmin=args.pmin,
        pmax=args.pmax,
        subsampling_ratio=args.subsampling_ratio,
        sv_momentum=args.sv_momentum,
    )
    se, sa = {}, {}
    exact.create_state(se)
    approx.create_state(sa)

    p_ex, p_ap = None, None
    for step in range(args.warmup_steps):
        g, a, mom = make_synthetic(m, n, args.batch_size, device, regime)
        kw = dict(mom_2d=mom, beta1=args.beta1, min_rank=args.min_rank)
        with torch.no_grad():
            p_ex_i, _ = exact.update_and_compute_p(
                se, g, a, use_gram=False, nesterov=True, **kw
            )
            p_ap_i, _ = approx.update_and_compute_p(
                sa, g, a, use_gram=False, nesterov=True, **kw
            )
        if step == args.warmup_steps - 1:
            p_ex, p_ap = p_ex_i, p_ap_i
    return p_ex, p_ap


def main():
    parser = argparse.ArgumentParser(description="Fast P* approximation benchmark")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--subsampling_ratio", type=float, default=0.25)
    parser.add_argument("--min_rank", type=int, default=50)
    parser.add_argument("--pmin", type=float, default=1.02)
    parser.add_argument("--pmax", type=float, default=50.0)
    parser.add_argument("--beta1", type=float, default=0.95)
    parser.add_argument("--sv_momentum", type=float, default=0.95)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    print(f"Running on {args.device.upper()}")

    # (m, n, regime, label)
    shapes = [
        (512, 512, "noisy", "flat/noisy 512x512"),
        (1024, 256, "noisy", "flat/noisy 1024x256"),
        (256, 1024, "noisy", "flat/noisy 256x1024"),
        (128, 2048, "noisy", "flat/noisy 128x2048 (FC1)"),
        (2048, 128, "noisy", "flat/noisy 2048x128 (FC2)"),
        (512, 512, "aligned", "flat/aligned 512x512"),
        (512, 512, "heavy", "heavy-M 512x512"),
        (256, 1024, "heavy", "heavy-M 256x1024 (wide)"),
        (1024, 256, "heavy", "heavy-M 1024x256 (tall)"),
        (128, 2048, "heavy", "heavy-M 128x2048"),
        (512, 512, "activation", "heavy-MA 512x512"),
        (256, 1024, "activation", "heavy-MA 256x1024"),
    ]

    print("=" * 72)
    print(" FAST P* APPROXIMATION BENCHMARK (named shapes)")
    print("=" * 72)
    header = f"{'case':>32s} | {'exact':>7s} | {'approx':>7s} | {'|err|':>7s}"
    print(header)
    print("-" * len(header))

    errs, ex_vals, ap_vals = [], [], []
    for m, n, regime, label in shapes:
        p_ex, p_ap = run_shape(m, n, regime, args, args.device, seed=0)
        err = abs(p_ex - p_ap)
        errs.append(err)
        ex_vals.append(p_ex)
        ap_vals.append(p_ap)
        print(f"{label:>32s} | {p_ex:7.3f} | {p_ap:7.3f} | {err:7.3f}")

    errs = np.array(errs)
    print("-" * len(header))
    print(f"Shapes tested:     {len(shapes)}")
    print(f"Subsampling ratio: {args.subsampling_ratio}")
    print(f"Min rank floor:    {args.min_rank}")
    print(f"Mean |err|:        {errs.mean():.4f}")
    print(f"Median |err|:      {np.median(errs):.4f}")
    print(f"Max |err|:         {errs.max():.4f}")
    print(f"95th pct |err|:    {np.percentile(errs, 95):.4f}")
    print(f"Avg Exact p*:      {np.mean(ex_vals):.4f}")
    print(f"Avg Approx p*:     {np.mean(ap_vals):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()

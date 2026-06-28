"""Spectral gap and edge density of states of the FULL Cayley Laplacian of
Heis_3(Z/qZ), to decide the spectral dimension d_s that controls the QFI
scaling exponent.

The single winding block w = 2 (Harper, flux 2/q) was found to be marginal
(d_s = 2, a = 1). Here we test the full substrate, whose homogeneous (growth)
dimension is D = 4 (Bass-Guivarch). The question is whether the edge density
of states near lambda = 0 is governed by

  - the abelian/torus bottom (d_s = 2, a = 1, marginal QFI), or
  - the nilpotent growth dimension (d_s = 4, a = 2, power-law QFI),

or shows a crossover between the two.

Graph: vertices (a,b,c) in (Z/qZ)^3 with the Heisenberg law
(a,b,c)*(a',b',c') = (a+a', b+b', c+c'+a b'). Generators X=(1,0,0),
Y=(0,1,0) and inverses give a connected 4-regular graph; L = I - A/4 is the
normalised Laplacian with spectrum in [0,2].

For each prime q:
  - gap lambda_2(q) via shift-invert eigsh (smallest positive eigenvalue);
  - edge spectral dimension from the integrated counting function
    N(lambda) ~ lambda^{d_s/2} near lambda -> 0^+, obtained from the exact
    spectrum (dense eigvalsh) when q^3 <= DENSE_MAX, otherwise from a
    Kernel Polynomial Method (KPM) estimate of the density of states.

A sliding log-log slope of N(lambda) is reported so that any 2 -> 4 crossover
in d_s is visible rather than averaged away.

Output: a PDF figure and a printed summary. Spectra/moments are cached for
resumable runs.
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "heis_full_cache"
FIG_PATH = "heis_full_edge_dos.pdf"
DENSE_MAX = 8000      # use dense eigvalsh when q^3 <= DENSE_MAX
KPM_MOMENTS = 2500    # Chebyshev moments for the KPM density of states
KPM_VECTORS = 24      # stochastic (Hutchinson) probe vectors
EDGE_FRACTION = 0.06  # lowest fraction of the spectrum used for the edge fit


def primes_in(lo, hi):
    out = []
    for n in range(lo, hi + 1):
        if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)):
            out.append(n)
    return out


def build_laplacian(q):
    """Normalised Laplacian L = I - A/4 of Cay(Heis_3(Z/qZ), {X,Y,X^-1,Y^-1})."""
    a, b, c = np.meshgrid(np.arange(q), np.arange(q), np.arange(q), indexing="ij")
    a, b, c = a.ravel(), b.ravel(), c.ravel()

    def enc(A, B, C):
        return (A % q) + (B % q) * q + (C % q) * q * q

    src = enc(a, b, c)
    cols = np.concatenate(
        [enc(a + 1, b, c), enc(a - 1, b, c), enc(a, b + 1, c + a), enc(a, b - 1, c - a)]
    )
    rows = np.tile(src, 4)
    n = q ** 3
    adj = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    return sp.identity(n, format="csr") - adj / 4.0


def gap(lap):
    """Smallest positive eigenvalue of L via shift-invert just below zero."""
    vals = eigsh(lap.tocsc(), k=6, sigma=-1e-3, which="LM", return_eigenvectors=False)
    vals = np.sort(vals)
    return float(vals[vals > 1e-9][0])


def dense_spectrum(lap):
    return np.sort(np.linalg.eigvalsh(lap.toarray()))


def kpm_dos(lap, n_moments, n_vectors, n_points=4000, seed=0):
    """KPM estimate of the integrated counting function N(lambda).

    L has spectrum in [0,2]; rescale to xtil in [-1,1] via H = L - I.
    Returns (lambda_grid, N_cumulative) with N in units of eigenvalue count.
    """
    rng = np.random.default_rng(seed)
    n = lap.shape[0]
    h = (lap - sp.identity(n, format="csr")).tocsr()  # spectrum in [-1,1]

    mu = np.zeros(n_moments)
    for _ in range(n_vectors):
        z = rng.choice([-1.0, 1.0], size=n)
        t_prev, t_cur = z.copy(), h @ z
        mu[0] += z @ t_prev
        mu[1] += z @ t_cur
        for m in range(2, n_moments):
            t_next = 2.0 * (h @ t_cur) - t_prev
            mu[m] += z @ t_next
            t_prev, t_cur = t_cur, t_next
    mu /= (n_vectors * n)

    m = np.arange(n_moments)  # Jackson kernel
    nm = n_moments
    g = ((nm - m + 1) * np.cos(np.pi * m / (nm + 1))
         + np.sin(np.pi * m / (nm + 1)) / np.tan(np.pi / (nm + 1))) / (nm + 1)

    x = np.linspace(-0.999, 0.999, n_points)
    theta = np.arccos(x)
    dens = g[0] * mu[0] + 2.0 * np.sum(
        (g[1:] * mu[1:])[:, None] * np.cos(np.outer(m[1:], theta)), axis=0
    )
    dens = dens / (np.pi * np.sqrt(1.0 - x ** 2))
    dens = np.clip(dens, 0.0, None)

    lam = x + 1.0  # back to [0,2]
    dlam = lam[1] - lam[0]
    cumulative = np.cumsum(dens) * dlam * n
    return lam, cumulative


def counting_from_spectrum(eigs):
    pos = np.sort(eigs[eigs > 1e-9])
    return pos, np.arange(1, pos.size + 1, dtype=float)


def slope_profile(lam, count, window=0.5):
    """Local log-log slope d log N / d log lambda over a sliding window
    (in decades). Returns (lambda_centres, slopes)."""
    mask = (lam > 0) & (count > 0)
    lx, ly = np.log(lam[mask]), np.log(count[mask])
    centres, slopes = [], []
    for i in range(lx.size):
        sel = (lx >= lx[i] - window) & (lx <= lx[i] + window)
        if sel.sum() >= 5:
            s = np.polyfit(lx[sel], ly[sel], 1)[0]
            centres.append(np.exp(lx[i]))
            slopes.append(s)
    return np.array(centres), np.array(slopes)


def process(q):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"q{q}.npz")
    lap = build_laplacian(q)
    n = q ** 3
    if os.path.exists(cache):
        d = np.load(cache)
        return q, float(d["gap"]), d["lam"], d["count"], str(d["method"])

    g = gap(lap)
    if n <= DENSE_MAX:
        eigs = dense_spectrum(lap)
        lam, count = counting_from_spectrum(eigs)
        method = "dense"
    else:
        lam, count = kpm_dos(lap, KPM_MOMENTS, KPM_VECTORS)
        method = "kpm"
    np.savez(cache, gap=g, lam=lam, count=count, method=method)
    return q, g, lam, count, method


def main():
    qs = primes_in(5, int(sys.argv[1]) if len(sys.argv) > 1 else 31)
    print(f"Full Heisenberg Cayley Laplacian, primes q in {qs}\n")

    results = {}
    t0 = time.time()
    for i, q in enumerate(qs, 1):
        q, g, lam, count, method = process(q)
        results[q] = (g, lam, count, method)
        el = time.time() - t0
        print(f"[{i}/{len(qs)}] q={q:3d} N={q**3:6d} method={method:5s} "
              f"gap={g:.5e}  elapsed={el:6.1f}s", flush=True)

    qs_sorted = sorted(results)
    gaps = np.array([results[q][0] for q in qs_sorted])
    beta = np.polyfit(np.log(qs_sorted), np.log(gaps), 1)[0]
    print(f"\nGlobal gap closure: lambda_2(q) ~ q^({beta:.3f})")

    # Edge exponent from the largest exact (dense) spectrum available.
    dense_qs = [q for q in qs_sorted if results[q][3] == "dense"]
    q_edge = dense_qs[-1]
    lam, count = results[q_edge][1], results[q_edge][2]
    pos = lam[lam > 1e-9]
    cnt = np.arange(1, pos.size + 1, dtype=float)
    win = (pos >= 0.05) & (pos < 0.20)  # nilpotent edge window, above the abelian bottom
    p_edge = np.polyfit(np.log(pos[win]), np.log(cnt[win]), 1)[0]
    n_edge = int(win.sum())
    print(f"Edge exponent (q={q_edge}, dense, lambda in [0.05,0.20)): "
          f"N ~ lambda^{p_edge:.3f}  => d_s = {2 * p_edge:.3f}, a = {p_edge:.3f}")
    print(f"=> QFI scaling f_Q ~ T^(1-a) = T^({1 - p_edge:.3f})  "
          f"({'power-law growth' if p_edge > 1.05 else 'marginal' if p_edge > 0.9 else 'no growth'})")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    ax1.loglog(qs_sorted, gaps, "o-", ms=5)
    qf = np.array([qs_sorted[0], qs_sorted[-1]], float)
    ax1.loglog(qf, gaps[0] * (qf / qs_sorted[0]) ** beta, "k--",
               label=f"$q^{{{beta:.2f}}}$")
    ax1.set_xlabel("$q$")
    ax1.set_ylabel(r"$\lambda_2(q)$")
    ax1.set_title("Global gap closure")
    ax1.legend()

    for q in dense_qs:
        lq, cq = results[q][1], results[q][2]
        ax2.loglog(lq, cq, "-", lw=1, label=f"q={q}")
    ax2.loglog(pos[win], cnt[win], "k.", ms=4, label="fit window")
    xref = np.array([0.05, 0.20])
    c0 = cnt[win][0]
    ax2.loglog(xref, c0 * (xref / 0.05) ** 1.0, "b:", label=r"slope 1 ($d_s{=}2$)")
    ax2.loglog(xref, c0 * (xref / 0.05) ** 2.0, "r:", label=r"slope 2 ($d_s{=}4$)")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"$N(\lambda)$")
    ax2.set_title("Edge integrated DOS")
    ax2.legend(fontsize=8)

    cc, ss = slope_profile(results[q_edge][1], results[q_edge][2])
    ax3.semilogx(cc, ss, "-")
    ax3.axhline(1.0, color="b", ls=":", label="$d_s=2$")
    ax3.axhline(2.0, color="r", ls=":", label="$d_s=4$")
    ax3.set_xlabel(r"$\lambda$")
    ax3.set_ylabel(r"local slope $d\log N/d\log\lambda$")
    ax3.set_title(f"Sliding spectral dimension (q={q_edge})")
    ax3.set_ylim(0, 3)
    ax3.legend()

    fig.tight_layout()
    fig.savefig(FIG_PATH)
    print(f"\nFigure written to {FIG_PATH}")


if __name__ == "__main__":
    main()

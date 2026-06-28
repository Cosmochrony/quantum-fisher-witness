"""Edge density of states and gap closure of the w = c Weil block of the
Heisenberg Cayley Laplacian, used to test the QFI scaling derivation.

Context
-------
The projective stability operator restricted to the winding-c sector,
L_Pi|_{w=c}, is the normalised Laplacian L = I - A/4 of the Cayley graph of
Heis_3(Z/qZ) (generators X, Y) projected onto the Weil block of central
character c. In that block X, Y act as shift and clock operators, and the
adjacency reduces exactly to the Harper / almost-Mathieu operator at flux
alpha = c/q:

    (A f)(j) = f(j-1) + f(j+1) + 2 cos(2 pi c j / q) f(j),   j in Z/qZ.

We extract two quantities as functions of q (prime):

  1. Gap closure    Delta_Pi(q) = smallest positive eigenvalue of L^{(c)};
     fitted to Delta_Pi ~ q^{-beta}. Closure (beta > 0) is condition (i)
     of the derivation (its thermal analogue Delta_Pi(T) ~ T is a separate
     dynamical input, not tested here).

  2. Edge exponent  integrated counting N(lambda) = #{eigenvalues <= lambda}
     near lambda -> 0^+, fitted to N(lambda) ~ lambda^{p}. With a flat
     matrix-element weight this gives spectral dimension d_s = 2p, effective
     spectral-weight exponent W(lambda) ~ lambda^{p-1}, hence the dynamical
     susceptibility exponent a = p (chi'' ~ omega^{a} at T = 0), and the QFI
     scaling f_Q(T) ~ T^{1-a}. Scale-free growth requires a > 1.

The matrix-element weight w(lambda) of the physical witness operator is set
to 1 here; a non-flat weight shifts a and is flagged as an additional input.

Output: a PDF figure and a printed summary table. Intermediate spectra are
cached so the sweep can be resumed without recomputation.
"""

import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "qfi_edge_cache"
FIG_PATH = "qfi_edge_dos.pdf"
CENTRAL_CHARACTER = 2  # w = 2 sector
EDGE_FRACTION = 0.10   # lowest 10% of the spectrum used for the edge fit


def primes_up_to(limit):
    """Return the list of primes p with 5 <= p <= limit (sieve)."""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return [int(p) for p in np.nonzero(sieve)[0] if p >= 5]


def weil_block_laplacian_eigenvalues(q, c):
    """Eigenvalues of the normalised Laplacian L = I - A/4 of the Harper
    operator (Weil block of central character c, flux c/q) on Z/qZ.

    The matrix is real symmetric and q x q, so a dense solver is used.
    """
    j = np.arange(q)
    diag = 2.0 * np.cos(2.0 * np.pi * c * j / q)
    a = np.diag(diag)
    off = np.ones(q - 1)
    a += np.diag(off, 1) + np.diag(off, -1)
    a[0, q - 1] += 1.0  # periodic wrap of the shift on the q-cycle
    a[q - 1, 0] += 1.0
    adj_eigs = np.linalg.eigvalsh(a)
    lap_eigs = 1.0 - adj_eigs / 4.0
    lap_eigs.sort()
    return lap_eigs


def cache_file(q, c):
    return os.path.join(CACHE_DIR, f"weil_q{q}_c{c}.npy")


def compute_one(args):
    """Worker: compute (or load from cache) the spectrum for a single q."""
    q, c = args
    path = cache_file(q, c)
    if os.path.exists(path):
        eigs = np.load(path)
    else:
        eigs = weil_block_laplacian_eigenvalues(q, c)
        np.save(path, eigs)
    positive = eigs[eigs > 1e-12]
    delta_pi = float(positive[0]) if positive.size else float("nan")
    return q, delta_pi, eigs


def fit_powerlaw(x, y):
    """Least-squares fit of log y = slope * log x + intercept.
    Returns (slope, intercept)."""
    lx, ly = np.log(np.asarray(x)), np.log(np.asarray(y))
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def edge_exponent(eigs, edge_fraction):
    """Fit the integrated counting function N(lambda) ~ lambda^p over the
    lowest edge_fraction of positive eigenvalues. Returns (p, lam, counts)."""
    positive = np.sort(eigs[eigs > 1e-12])
    n_edge = max(8, int(edge_fraction * positive.size))
    lam = positive[:n_edge]
    counts = np.arange(1, n_edge + 1, dtype=float)
    p, _ = fit_powerlaw(lam, counts)
    return p, lam, counts


def main():
    c = CENTRAL_CHARACTER
    q_max = int(sys.argv[1]) if len(sys.argv) > 1 else 1009
    os.makedirs(CACHE_DIR, exist_ok=True)

    qs = primes_up_to(q_max)
    print(f"Sweeping {len(qs)} primes q in [{qs[0]}, {qs[-1]}], sector w = {c}")
    print(f"Using {cpu_count()} cores.\n")

    results = {}
    t0 = time.time()
    with Pool(processes=cpu_count()) as pool:
        for i, (q, delta_pi, eigs) in enumerate(
            pool.imap_unordered(compute_one, [(q, c) for q in qs]), start=1
        ):
            results[q] = (delta_pi, eigs)
            elapsed = time.time() - t0
            eta = elapsed / i * (len(qs) - i)
            print(
                f"[{i:3d}/{len(qs)}] q={q:5d}  Delta_Pi={delta_pi:.6e}  "
                f"elapsed={elapsed:6.1f}s  ETA={eta:6.1f}s",
                flush=True,
            )

    qs_sorted = sorted(results)
    delta = np.array([results[q][0] for q in qs_sorted])

    # Condition (i): gap closure Delta_Pi(q) ~ q^{-beta}.
    beta, b0 = fit_powerlaw(qs_sorted, delta)
    print(f"\nGap closure fit:   Delta_Pi(q) ~ q^(-beta),  beta = {-beta:.4f}")

    # Condition (ii): edge exponent from the largest available q.
    q_big = qs_sorted[-1]
    p_edge, lam, counts = edge_exponent(results[q_big][1], EDGE_FRACTION)
    a_exp = p_edge  # under a flat matrix-element weight
    print(f"Edge fit (q={q_big}): N(lambda) ~ lambda^p,  p = {p_edge:.4f}")
    print(f"  => spectral dimension d_s = {2 * p_edge:.3f}")
    print(f"  => susceptibility exponent a = {a_exp:.3f}  (flat weight)")
    print(f"  => QFI scaling f_Q(T) ~ T^(1-a) = T^({1 - a_exp:.3f})")
    print(f"  growth (a > 1)? {'YES' if a_exp > 1 else 'NO / marginal'}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(qs_sorted, delta, "o", ms=4, label="data")
    qfit = np.array([qs_sorted[0], qs_sorted[-1]], dtype=float)
    ax1.loglog(qfit, np.exp(b0) * qfit ** beta, "-",
               label=f"fit $q^{{{beta:.3f}}}$")
    ax1.set_xlabel("$q$")
    ax1.set_ylabel(r"$\Delta_\Pi(q)=\lambda_{\min}^{+}(L^{(2)})$")
    ax1.set_title("Gap closure, $w=2$ Weil block")
    ax1.legend()

    for q in [q for q in qs_sorted if q in (qs_sorted[len(qs_sorted)//3],
                                            qs_sorted[2*len(qs_sorted)//3],
                                            q_big)]:
        _, lq, cq = edge_exponent(results[q][1], EDGE_FRACTION)
        ax2.loglog(lq, cq, ".", ms=3, label=f"q={q}")
    ax2.loglog(lam, np.exp(np.polyfit(np.log(lam), np.log(counts), 1)[1])
               * lam ** p_edge, "k-", label=f"fit $\\lambda^{{{p_edge:.3f}}}$")
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel(r"$N(\lambda)$ (integrated count)")
    ax2.set_title("Edge integrated DOS")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(FIG_PATH)
    print(f"\nFigure written to {FIG_PATH}")


if __name__ == "__main__":
    main()

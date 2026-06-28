"""Refinement of the QFI spectral-witness analysis, exploiting the exact
block decomposition of the Heisenberg Cayley Laplacian.

The full normalised Laplacian L = I - A/4 of Cay(Heis_3(Z/qZ)) has spectrum

    spec(L) = {abelian torus modes, q^2 of them}
              U  union over c=1..q-1 of  spec(L^{(c)}) each with multiplicity q,

where L^{(c)} = I - A^{(c)}/4 and A^{(c)} is the Harper operator at flux c/q.
This identity (verified to machine precision) lets us assemble the exact full
spectrum at cost O(q^4) instead of diagonalising a q^3 x q^3 matrix, reaching
much larger q.

Three refinements are produced:

  (1) q-convergence of the edge exponent p(q) (N(lambda) ~ lambda^p) of the
      full substrate, extrapolated to p_inf = d_s/2;
  (2) the temperature curve f_Q(T) for the single sector w=2 (marginal) and
      for the full substrate (power-law), from the modal relaxational form of
      chi''(omega);
  (3) the band-stacking mechanism: the per-block gaps Delta(c) and the
      stacked density of states that turns many 2d band edges into d_s = 4.

Output: a four-panel PDF figure and a printed summary. Spectra are cached.
"""

import os
import sys
import time
import numpy as np
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "qfi_refine_cache"
FIG_PATH = "qfi_refine.pdf"
FIT_LO, FIT_HI = 0.05, 0.20   # lambda window for the nilpotent edge exponent


def primes_in(lo, hi):
    return [n for n in range(lo, hi + 1)
            if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))]


def harper_eigs(q, c):
    """Eigenvalues of L^{(c)} = I - A^{(c)}/4, Harper operator at flux c/q."""
    j = np.arange(q)
    a = np.diag(2.0 * np.cos(2.0 * np.pi * c * j / q))
    a += np.diag(np.ones(q - 1), 1) + np.diag(np.ones(q - 1), -1)
    a[0, q - 1] += 1.0
    a[q - 1, 0] += 1.0
    return 1.0 - np.linalg.eigvalsh(a) / 4.0


def abelian_eigs(q):
    """The q^2 torus modes from the abelianisation (Z/qZ)^2."""
    k1, k2 = np.meshgrid(np.arange(q), np.arange(q), indexing="ij")
    adj = 2.0 * np.cos(2.0 * np.pi * k1 / q) + 2.0 * np.cos(2.0 * np.pi * k2 / q)
    return (1.0 - adj.ravel() / 4.0)


def assemble(q):
    """Return (sorted full spectrum, per-block gaps Delta(c))."""
    parts = [abelian_eigs(q)]
    gaps = np.empty(q - 1)
    for c in range(1, q):
        e = harper_eigs(q, c)
        parts.append(np.repeat(e, q))  # multiplicity q
        pos = e[e > 1e-12]
        gaps[c - 1] = pos.min() if pos.size else np.nan
    return np.sort(np.concatenate(parts)), gaps


def edge_exponent(spec, lo=FIT_LO, hi=FIT_HI):
    pos = spec[spec > 1e-12]
    cnt = np.arange(1, pos.size + 1, dtype=float)
    m = (pos >= lo) & (pos < hi)
    if m.sum() < 6:  # small single block: fall back to the lowest 40% of modes
        k = max(6, int(0.40 * pos.size))
        return float(np.polyfit(np.log(pos[:k]), np.log(cnt[:k]), 1)[0])
    return float(np.polyfit(np.log(pos[m]), np.log(cnt[m]), 1)[0])


def chi_double_prime(spec, omega, n_modes):
    """Modal relaxational response chi''(omega) = (1/N) sum_n omega lam_n /
    (omega^2 + lam_n^2), flat matrix-element weight, per mode."""
    lam = spec[spec > 1e-12]
    out = np.empty_like(omega)
    for i, w in enumerate(omega):
        out[i] = np.sum(w * lam / (w * w + lam * lam))
    return out / n_modes


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def fq_curve(spec, temps, omega):
    """f_Q(T) = (4/pi) int_0^wmax tanh(omega/2T) chi''(omega) domega."""
    chi = chi_double_prime(spec, omega, spec.size)
    fq = np.empty_like(temps)
    for i, t in enumerate(temps):
        fq[i] = (4.0 / np.pi) * _trapz(np.tanh(omega / (2.0 * t)) * chi, omega)
    return fq, chi


def process(q):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"q{q}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return q, d["spec"], d["gaps"]
    spec, gaps = assemble(q)
    np.savez_compressed(path, spec=spec, gaps=gaps)
    return q, spec, gaps


def main():
    q_max = int(sys.argv[1]) if len(sys.argv) > 1 else 101
    qs = primes_in(11, q_max)
    print(f"Assembling exact full spectra for primes q in [{qs[0]}, {qs[-1]}], "
          f"{cpu_count()} cores.\n")

    specs, gaps = {}, {}
    t0 = time.time()
    with Pool(cpu_count()) as pool:
        for i, (q, sp, g) in enumerate(pool.imap_unordered(process, qs), 1):
            specs[q], gaps[q] = sp, g
            el = time.time() - t0
            print(f"[{i:2d}/{len(qs)}] q={q:4d} N={q**3:8d} "
                  f"elapsed={el:6.1f}s ETA={el/i*(len(qs)-i):6.1f}s", flush=True)

    qs_sorted = sorted(specs)

    # (1) q-convergence of the edge exponent, full substrate and single w=2 block.
    p_full = np.array([edge_exponent(specs[q]) for q in qs_sorted])
    p_w2 = np.array([edge_exponent(np.sort(harper_eigs(q, 2))) for q in qs_sorted])
    inv_q = 1.0 / np.array(qs_sorted, float)
    _, pf_inf = np.polyfit(inv_q, p_full, 1)
    _, pw_inf = np.polyfit(inv_q, p_w2, 1)
    print(f"\n(1) edge exponent extrapolation (p_inf = d_s/2 = a):")
    print(f"    full substrate: p(q={qs_sorted[-1]})={p_full[-1]:.3f}, "
          f"p_inf={pf_inf:.3f}  => d_s={2*pf_inf:.2f}, a={pf_inf:.2f}")
    print(f"    single w=2:     p(q={qs_sorted[-1]})={p_w2[-1]:.3f}, "
          f"p_inf={pw_inf:.3f}  => d_s={2*pw_inf:.2f}, a={pw_inf:.2f}")

    q_big = qs_sorted[-1]
    spec_full = specs[q_big]
    gq = gaps[q_big]
    cc = np.arange(1, q_big)
    spec_block = np.sort(harper_eigs(q_big, 2))

    # (2) static-spectrum f_Q(T): sum-rule plateau, i.e. no growth without the
    # thermal bridge. The static d_s only fixes the exponent a that the bridge
    # would carry.
    omega = np.logspace(-4.5, np.log10(2.0), 600)
    temps = np.logspace(-3.5, -0.2, 60)
    fq_full, _ = fq_curve(spec_full, temps, omega)
    print(f"\n(2) static f_Q(T) is flat-to-decreasing (sum rule): "
          f"f_Q low/high = {fq_full[0]/fq_full[-1]:.2f}")
    print("    => scale-free GROWTH requires the thermal bridge Delta_Pi(T) ~ T (open).")

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    ax[0, 0].plot(qs_sorted, p_full, "o-", ms=5, label="full substrate")
    ax[0, 0].plot(qs_sorted, p_w2, "s-", ms=5, label="single $w=2$")
    ax[0, 0].axhline(2.0, color="r", ls=":", label="$d_s=4$ ($a=2$)")
    ax[0, 0].axhline(1.0, color="b", ls=":", label="$d_s=2$ ($a=1$)")
    ax[0, 0].set_xlabel("$q$")
    ax[0, 0].set_ylabel("edge exponent $p(q)=d_s/2=a$")
    ax[0, 0].set_title(f"(1) q-convergence: full $a\\to{pf_inf:.2f}$, $w{{=}}2$ $a\\to{pw_inf:.2f}$")
    ax[0, 0].legend(fontsize=8)

    ax[0, 1].plot(cc / q_big, gq, ".", ms=4)
    ax[0, 1].set_xlabel("flux $c/q$")
    ax[0, 1].set_ylabel(r"block gap $\Delta(c)$")
    ax[0, 1].set_title(f"(3a) staggered Weil-block gaps ($q={q_big}$)")

    posb = np.sort(spec_block[spec_block > 1e-12])
    posf = np.sort(spec_full[spec_full > 1e-12])
    ax[1, 0].loglog(posb, np.arange(1, posb.size + 1) / posb.size, "-",
                    label="single block ($d_s{=}2$)")
    ax[1, 0].loglog(posf, np.arange(1, posf.size + 1) / posf.size, "-",
                    label="stacked / full ($d_s{\\approx}4$)")
    xr = np.array([0.03, 0.4])
    ax[1, 0].loglog(xr, 0.03 * (xr / xr[0]) ** 1, "b:", label="slope 1")
    ax[1, 0].loglog(xr, 0.012 * (xr / xr[0]) ** 2, "r:", label="slope 2")
    ax[1, 0].set_xlabel(r"$\lambda$")
    ax[1, 0].set_ylabel("normalised $N(\\lambda)$")
    ax[1, 0].set_title("(3b) band stacking: 2d edges $\\to$ 4d")
    ax[1, 0].legend(fontsize=8)

    ax[1, 1].loglog(temps, fq_full / fq_full[-1], "-", label="static $f_Q(T)$ (full)")
    tref = np.array([2e-3, 5e-2])
    ax[1, 1].loglog(tref, 6.0 * (tref / tref[-1]) ** (-1.0), "r:",
                    label=r"$T^{1-a}$, $a{=}2$ (needs bridge)")
    ax[1, 1].set_xlabel("$T$ (spectral units)")
    ax[1, 1].set_ylabel("$f_Q(T)$ (normalised)")
    ax[1, 1].set_title("(2) static $f_Q$: no growth; bridge gates it")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_PATH)
    print(f"\nFigure written to {FIG_PATH}")


if __name__ == "__main__":
    main()

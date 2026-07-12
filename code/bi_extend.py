"""Born--Infeld nonlinearity: q-extension and crossing stabilisation.

Self-contained (builds the Heisenberg Cayley graph itself). Runs the bounded-flux
BI relaxation flow

    phi'_i = - sqrt(max(0, 1 - S_i)) * (L_Pi phi)_i,
    S_i    = (1/c_BI^2) sum_{j ~ i} (phi_i - phi_j)^2,   L_Pi = I - A/4,

on Cay(Heis_3(Z/qZ)) for several system sizes, and reads the effective QFI
exponent alpha_eff from the relaxation return C(t) = <phi(0), phi(t)> / ||phi(0)||^2
via u(t) = -d log C / dt, with u(t) t -> d_s^eff/2 and alpha_eff = d_s^eff/2 - 1.

KEY POINT (crossing stabilisation): the exponent is read on a FIXED ABSOLUTE
temperature window WINDOW = (t_lo, t_hi), the SAME for every q. A q-scaled window
mixes the clean scaling plateau with the finite-size rise (u t -> lambda_2 t) and
produces a spurious drift of the crossing; a fixed window removes it. The exponent
runs weakly with temperature, so the crossing is reported as a function of the
window centre as well.

Outputs:
  - alpha_eff vs mean saturation <S>, overlaid for all q (left panel);
  - the alpha_eff = alpha_exp crossing <S>_c vs q, to confirm q-stability (right).

Parallel over (q, target, seed); spectra/trajectories are cached so the run
resumes after interruption. Progress and ETA are printed. Figures are PDF.

Usage:
    python3 bi_extend.py                 # default QS, TARGETS, SEEDS
    python3 bi_extend.py 13 17 19 23     # explicit prime sizes
Tune QS / TARGETS / SEEDS / N_CORES below. Primes only (Weil-block substrate).
"""

import os
import sys
import time
import numpy as np
import scipy.sparse as sp
from multiprocessing import Pool, cpu_count

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- parameters ---------------------------------
QS = [13, 17, 19, 23]              # prime system sizes (override via argv)
TARGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]   # target mean saturation <S>
SEEDS = list(range(8))             # ensemble size per (q, target)
C_CHI = 1.0                        # Born--Infeld saturation scale
DT = 0.05                          # time step (linear stability: dt < 1)
T_MAX_FAC = 0.18                   # integrate to t_max = T_MAX_FAC * q^2
WINDOW = (6.0, 12.0)               # FIXED absolute window for the exponent
ALPHA_EXP = 0.7                    # observed strange-metal exponent
N_CORES = cpu_count()              # set to a fixed number to throttle
CACHE_DIR = "bi_extend_cache"
FIG_PATH = "bi_extend.pdf"
# --------------------------------------------------------------------------


def build_AL(q):
    """Adjacency A and normalised Laplacian L = I - A/4 of Cay(Heis_3(Z/qZ))."""
    a, b, c = np.meshgrid(np.arange(q), np.arange(q), np.arange(q), indexing="ij")
    a, b, c = a.ravel(), b.ravel(), c.ravel()

    def enc(A, B, C):
        return (A % q) + (B % q) * q + (C % q) * q * q

    src = enc(a, b, c)
    cols = np.concatenate([enc(a + 1, b, c), enc(a - 1, b, c),
                           enc(a, b + 1, c + a), enc(a, b - 1, c - a)])
    rows = np.tile(src, 4)
    n = q ** 3
    A = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n)).tocsr()
    L = sp.identity(n, format="csr") - A / 4.0
    return A, L, n


def stress(phi, A, deg=4):
    """S_i = (1/c^2)[deg phi^2 - 2 phi (A phi) + A(phi^2)]."""
    return (deg * phi * phi - 2.0 * phi * (A @ phi) + A @ (phi * phi)) / (C_CHI * C_CHI)


def rhs(phi, L, A, linear):
    Lphi = L @ phi
    if linear:
        return -Lphi
    return -np.sqrt(np.clip(1.0 - stress(phi, A), 0.0, None)) * Lphi


def rk4_step(phi, L, A, linear, dt):
    k1 = rhs(phi, L, A, linear)
    k2 = rhs(phi + 0.5 * dt * k1, L, A, linear)
    k3 = rhs(phi + 0.5 * dt * k2, L, A, linear)
    k4 = rhs(phi + dt * k3, L, A, linear)
    return phi + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def run_one(args):
    q, target, seed, linear = args
    tag = "lin" if linear else f"s{target:.2f}"
    path = os.path.join(CACHE_DIR, f"q{q}_{tag}_seed{seed}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return (q, target, seed, linear, d["t"], d["C"])

    A, L, n = build_AL(q)
    rng = np.random.default_rng(seed)
    phi0 = rng.standard_normal(n)
    phi0 -= phi0.mean()
    phi0 /= np.linalg.norm(phi0)
    if not linear:
        s_unit = stress(phi0, A).mean()
        phi0 *= np.sqrt(target / max(s_unit, 1e-12))

    norm0 = float(phi0 @ phi0)
    phi = phi0.copy()
    nsteps = int(T_MAX_FAC * q * q / DT)
    rec_every = max(1, nsteps // 400)
    ts, Cs = [], []
    for k in range(nsteps + 1):
        if k % rec_every == 0:
            ts.append(k * DT)
            Cs.append(float(phi0 @ phi) / norm0)
        phi = rk4_step(phi, L, A, linear, DT)
    t, C = np.array(ts), np.array(Cs)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, t=t, C=C)
    return (q, target, seed, linear, t, C)


def ut_median(t, C, win):
    lo = (C > 1e-7) & (t > 0)
    t, C = t[lo], C[lo]
    ut = -t * np.gradient(np.log(C), t)
    m = (t >= win[0]) & (t <= win[1])
    return float(np.median(ut[m])) if m.any() else np.nan


def main():
    global QS
    if len(sys.argv) > 1:
        QS = [int(x) for x in sys.argv[1:]]

    os.makedirs(CACHE_DIR, exist_ok=True)
    jobs = []
    for q in QS:
        jobs += [(q, None, s, True) for s in SEEDS]
        jobs += [(q, tg, s, False) for tg in TARGETS for s in SEEDS]

    print(f"BI extension: q in {QS}, {len(jobs)} trajectories, {N_CORES} cores, "
          f"window {WINDOW}.")
    t0 = time.time()
    store = {}
    with Pool(N_CORES) as pool:
        for i, (q, target, seed, linear, t, C) in enumerate(
                pool.imap_unordered(run_one, jobs), 1):
            key = (q, "lin" if linear else f"s{target:.2f}")
            store.setdefault(key, []).append((t, C))
            el = time.time() - t0
            print(f"[{i:3d}/{len(jobs)}] q={q} done  elapsed={el:7.1f}s "
                  f"ETA={el/i*(len(jobs)-i):7.1f}s", flush=True)

    def mean_curve(q, tag):
        rec = store.get((q, tag))
        if not rec:
            return None, None
        t = rec[0][0]
        return t, np.mean([C for _, C in rec], axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    cross_q = []
    for q in QS:
        tb, Cb = mean_curve(q, "lin")
        base = ut_median(tb, Cb, WINDOW)
        sat, aeff = [], []
        for tg in TARGETS:
            t, C = mean_curve(q, f"s{tg:.2f}")
            sat.append(tg)
            aeff.append(2.0 * ut_median(t, C, WINDOW) / base - 1.0)
        sat, aeff = np.array(sat), np.array(aeff)
        ax1.plot(sat, aeff, "o-", ms=4, label=f"q={q}")
        sc = (np.interp(ALPHA_EXP, aeff[::-1], sat[::-1])
              if aeff.min() <= ALPHA_EXP <= aeff.max() else np.nan)
        cross_q.append(sc)
        print(f"q={q}: alpha_eff={ALPHA_EXP} crossing at <S> ~ {sc:.3f}")

    ax1.axhline(1.0, color="r", ls=":", label=r"linear $\alpha=1$")
    ax1.axhline(ALPHA_EXP, color="g", ls=":", label=rf"$\alpha_{{\rm exp}}={ALPHA_EXP}$")
    ax1.set_xlabel(r"mean saturation $\langle S\rangle$")
    ax1.set_ylabel(r"$\alpha_{\rm eff}$")
    ax1.set_title(f"BI exponent vs saturation (window t={WINDOW})")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(QS, cross_q, "ks-")
    ax2.set_xlabel("system size $q$")
    ax2.set_ylabel(r"crossing $\langle S\rangle_c$")
    ax2.set_title(r"q-stability of the $\alpha=0.7$ crossing")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH)
    print(f"\nCrossing <S>_c by q: " +
          ", ".join(f"q{q}={c:.3f}" for q, c in zip(QS, cross_q)))
    print(f"Figure written to {FIG_PATH}")


if __name__ == "__main__":
    main()

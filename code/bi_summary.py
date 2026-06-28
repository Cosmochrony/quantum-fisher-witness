"""Stabilised summary of the Born--Infeld nonlinearity scan.

The effective QFI exponent is read from the cached relaxation returns C(t) as
u(t) t = -t d log C / dt, evaluated on a FIXED ABSOLUTE temperature window
(the same t-range for every system size), not a q-scaled window. This removes
the spurious q-drift of the crossing: the q^2-scaled window used previously
mixed the clean scaling plateau with the finite-size rise (u t -> lambda_2 t)
differently at each q.

Left: alpha_eff vs mean saturation <S>, for q=13 and q=17 on the clean window
t in [6,12]; the curves overlap and the alpha=0.7 crossing is q-stable at
<S> ~ 0.19. Right: the crossing <S>_c as a function of the absolute window,
showing q-stability at fixed window and a mild decrease with the window centre
-- the physical running of the effective exponent with temperature (the BI flow
does not produce a pure power law).
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "bi_nonlinear_cache"
FIG_PATH = "bi_summary.pdf"
TARGETS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.45]


def load_mean(q, tag):
    files = sorted(glob.glob(os.path.join(CACHE_DIR, f"q{q}_{tag}_seed*.npz")))
    Cs, t = [], None
    for f in files:
        d = np.load(f)
        t = d["t"]
        Cs.append(d["C"])
    return t, np.mean(Cs, axis=0)


def ut_median(q, tag, win):
    t, C = load_mean(q, tag)
    lo = (C > 1e-7) & (t > 0)
    t, C = t[lo], C[lo]
    ut = -t * np.gradient(np.log(C), t)
    m = (t >= win[0]) & (t <= win[1])
    return float(np.median(ut[m]))


def alpha_curve(q, win):
    base = ut_median(q, "lin", win)
    sat, aeff = [], []
    for tg in TARGETS:
        sat.append(tg)
        aeff.append(2.0 * ut_median(q, f"s{tg:.2f}", win) / base - 1.0)
    return np.array(sat), np.array(aeff)


def crossing(q, win):
    sat, aeff = alpha_curve(q, win)
    return float(np.interp(0.7, aeff[::-1], sat[::-1]))


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    win = (6, 12)
    for q, mk in [(13, "o-"), (17, "s-")]:
        sat, aeff = alpha_curve(q, win)
        ax1.plot(sat, aeff, mk, label=f"q={q}")
        sc = crossing(q, win)
        ax1.plot([sc], [0.7], "k*", ms=13, zorder=5)
        print(f"q={q}, window {win}: alpha=0.7 crossing at <S> ~ {sc:.3f}")
    ax1.axhline(1.0, color="r", ls=":", label=r"linear $\alpha=1$ (derived)")
    ax1.axhline(0.7, color="g", ls=":", label=r"$\alpha_{\rm exp}\approx0.7$")
    ax1.set_xlabel(r"mean substrate saturation $\langle S\rangle$")
    ax1.set_ylabel(r"effective exponent $\alpha_{\rm eff}$")
    ax1.set_title(r"BI bends $\alpha\!:1\!\to\!0.7$ at $\langle S\rangle\!\approx\!0.19$ (window $t\in[6,12]$)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    windows = [(4, 8), (5, 10), (6, 12), (8, 14), (8, 16)]
    centres = [np.mean(w) for w in windows]
    for q, mk in [(13, "o-"), (17, "s-")]:
        cr = [crossing(q, w) for w in windows]
        ax2.plot(centres, cr, mk, label=f"q={q}")
    ax2.set_xlabel("absolute window centre $t$")
    ax2.set_ylabel(r"crossing $\langle S\rangle_c$ ($\alpha_{\rm eff}=0.7$)")
    ax2.set_title("q-stability and running of the exponent")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_PATH)
    print(f"Figure written to {FIG_PATH}")


if __name__ == "__main__":
    main()

"""Diagnostic for the q-drift of the BI crossing.

Reads the cached relaxation trajectories and plots the local effective exponent
u(t) t = -t d log C / dt against the ABSOLUTE scale t = 1/T (not the q-scaled
window), overlaying q=13 and q=17 at fixed mean saturation. If the curves
overlap against t, the apparent drift of the crossing is a measurement-window
artefact (the window grows as q^2), and the exponent must be quoted at a fixed
physical temperature rather than as a window median.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = "bi_nonlinear_cache"
FIG_PATH = "bi_diagnostic.pdf"


def load_mean(q, tag):
    files = sorted(glob.glob(os.path.join(CACHE_DIR, f"q{q}_{tag}_seed*.npz")))
    if not files:
        return None, None
    Cs, t = [], None
    for f in files:
        d = np.load(f)
        t = d["t"]
        Cs.append(d["C"])
    return t, np.mean(Cs, axis=0)


def u_times_t(t, C):
    lo = (C > 1e-7) & (t > 0)
    t, C = t[lo], C[lo]
    dlog = np.gradient(np.log(C), t)
    # light smoothing
    ut = -t * dlog
    k = np.ones(3) / 3.0
    ut = np.convolve(ut, k, mode="same")
    return t, ut


def main():
    sats = ["lin", "s0.15", "s0.20", "s0.30"]
    labels = ["linear", "<S>=0.15", "<S>=0.20", "<S>=0.30"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(sats)))

    for tag, lab, col in zip(sats, labels, colors):
        for q, ls in [(13, "-"), (17, "--")]:
            t, C = load_mean(q, tag)
            if t is None:
                continue
            tu, ut = u_times_t(t, C)
            m = tu > 1.0
            ax.semilogx(tu[m], ut[m], ls, color=col,
                        label=f"{lab}, q={q}")

    ax.axhline(2.0, color="r", ls=":", lw=1)
    ax.axhline(1.7, color="g", ls=":", lw=1)
    ax.text(1.2, 2.05, r"$\alpha=1$", color="r", fontsize=9)
    ax.text(1.2, 1.55, r"$\alpha_{\rm exp}\approx0.7$", color="g", fontsize=9)
    ax.set_xlabel("$t = 1/T$ (absolute heat-kernel scale)")
    ax.set_ylabel(r"local $u(t)\,t = \Delta_\Pi(T)/T$")
    ax.set_title("Local exponent vs absolute scale: q=13 (solid) vs q=17 (dashed)")
    ax.set_ylim(0, 3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_PATH)

    # quantitative overlap check at a few fixed t
    print("Local u(t) t at fixed absolute t (q=13 vs q=17):")
    for tag, lab in zip(sats, labels):
        t13, C13 = load_mean(13, tag)
        t17, C17 = load_mean(17, tag)
        if t13 is None or t17 is None:
            continue
        tu13, ut13 = u_times_t(t13, C13)
        tu17, ut17 = u_times_t(t17, C17)
        for tref in [5.0, 10.0, 20.0]:
            v13 = np.interp(tref, tu13, ut13)
            v17 = np.interp(tref, tu17, ut17)
            print(f"  {lab:10s} t={tref:5.1f}: q13={v13:.3f}  q17={v17:.3f}  "
                  f"diff={abs(v13-v17):.3f}")
    print(f"\nFigure written to {FIG_PATH}")


if __name__ == "__main__":
    main()

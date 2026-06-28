"""Thermal closure of the projective stability gap from the heat-kernel flow.

The framework identifies the heat-kernel diffusion scale t with inverse
temperature: the local spectral energy density u(x;t) = -d/dt log K(x,x;t) is
the analogue of E = -d/dbeta log Z, so t plays the role of beta = 1/T
(Thermodynamics, Section 2). The linearised relaxation dynamics is the
heat-kernel flow exp(-L_Pi t).

The trace return is P(t) = (1/N) Tr exp(-L_Pi t) = (1/N) sum_n exp(-lambda_n t),
and the effective spectral gap at scale t is

    u(t) = -d/dt log P(t) = <lambda>_t = sum lam e^{-lam t} / sum e^{-lam t}.

For an edge density of states rho(lambda) ~ lambda^{d_s/2 - 1} one has
P(t) ~ t^{-d_s/2}, hence u(t) = (d_s/2)/t. Identifying T = 1/t gives the
Planckian closure

    Delta_Pi(T) := u(1/T) = (d_s/2) T,

linear in temperature, with slope d_s/2. This closes condition (i) of the QFI
derivation structurally rather than as an assumption: combined with the
power-law density of states it yields chi''(omega,T) = T^{-a} Phi(omega/T)
with a = d_s/2, and f_Q(T) ~ T^{1-a}.

This script verifies u(t) t -> d_s/2 on the exact assembled spectra and plots
the linear closure Delta_Pi(T) ~ (d_s/2) T. The BI nonlinearity of the full
App.-D dynamics is a correction to the linear (heat-kernel) flow used here.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qfi_refine import harper_eigs, abelian_eigs

FIG_PATH = "thermal_closure.pdf"


def assemble(q):
    parts = [abelian_eigs(q)]
    for c in range(1, q):
        parts.append(np.repeat(harper_eigs(q, c), q))
    return np.sort(np.concatenate(parts))


def u_of_t(spec, t):
    """u(t) = sum lam e^{-lam t} / sum e^{-lam t}, vectorised over t."""
    lam = spec[:, None]
    e = np.exp(-lam * t[None, :])
    return (lam * e).sum(0) / e.sum(0)


def main():
    qs = [61, 101, 151]
    specs = {q: assemble(q) for q in qs}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    print("Thermal closure  u(t) t -> d_s/2  (full substrate; expect 2 for d_s=4)")
    for q in qs:
        t = np.logspace(0.3, np.log10(0.06 * q * q), 40)
        u = u_of_t(specs[q], t)
        ax1.semilogx(t, u * t, "-", label=f"q={q}")
        m = (t > 4) & (t < 0.03 * q * q)
        plateau = np.median((u * t)[m])
        print(f"  q={q:4d}: median u*t = {plateau:.3f}  => d_s = {2*plateau:.2f}")

    ax1.axhline(2.0, color="r", ls=":", label="$d_s/2=2$ ($d_s=4$)")
    ax1.set_xlabel("$t = 1/T$ (heat-kernel scale)")
    ax1.set_ylabel(r"$u(t)\,t = \Delta_\Pi(T)/T$")
    ax1.set_title("Planckian closure: $u(t)\\,t\\to d_s/2$")
    ax1.set_ylim(0, 3)
    ax1.legend(fontsize=8)

    # Delta_Pi(T) = u(1/T) vs T, showing linearity.
    q = qs[-1]
    Tg = np.logspace(np.log10(1.0 / (0.06 * q * q)), -0.7, 40)
    dpi = u_of_t(specs[q], 1.0 / Tg)
    ax2.loglog(Tg, dpi, "o", ms=4, label=f"$\\Delta_\\Pi(T)=u(1/T)$, q={q}")
    ax2.loglog(Tg, 2.0 * Tg, "r-", label="$2\\,T$ (slope 1, $d_s=4$)")
    ax2.set_xlabel("$T = 1/t$")
    ax2.set_ylabel(r"$\Delta_\Pi(T)$")
    ax2.set_title("Linear thermal closure $\\Delta_\\Pi(T)\\sim 2T$")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_PATH)
    print(f"\nDerived: Delta_Pi(T) ~ (d_s/2) T linear; full substrate slope 2 (d_s=4).")
    print(f"=> with power-law DOS, chi''=T^-a Phi(w/T), a=d_s/2=2, f_Q ~ T^(1-a)=T^-1.")
    print(f"Figure written to {FIG_PATH}")


if __name__ == "__main__":
    main()

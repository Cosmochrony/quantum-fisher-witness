# Quantum Fisher Information as a Spectral Entanglement Witness

Source of the note *Quantum Fisher Information as a Spectral Entanglement Witness:
the Projective Stability Gap and the Origin of Scale-Free Growth* (J. Beau, 2026).

This note is part of the **condensed-matter** track of the Cosmochrony programme, alongside the structural
superconductivity paper (paper G). It is a companion to that paper but scoped separately: it asks whether the
non-injective projection framework reproduces the scale-free growth of the quantum Fisher information (QFI) density
recently reported in the heavy-fermion strange metal Ce3Pd20Si6, and isolates the exact structural condition that
controls it.

## Result in one paragraph

Using linear response, the QFI density reduces to a weighted integral of the imaginary part of the dynamical
susceptibility, whose infrared behaviour is fixed by the projective stability gap and the edge spectral weight of
the projective stability operator. This gives a dichotomy: a finite gap forces saturation, while a gap that closes
with edge spectral dimension d_s > 2 forces power-law growth f_Q(T) ~ T^(1-a) with a = d_s/2. Exact diagonalisation
then separates two cases. The single winding sector w=2 reduces to the critical almost-Mathieu (Harper) operator at
flux 2/q and is marginal (d_s -> 2, a -> 1, logarithmic). The full Cayley substrate of Heis_3(Z/qZ), of homogeneous
growth dimension D=4, has d_s -> 4, a -> 2, and a genuine power law f_Q(T) ~ T^(-1). The required thermal closure
Delta_Pi(T) = (d_s/2) T is derived from the framework's heat-kernel identification of the diffusion scale with
inverse temperature, not assumed. The derived exponent is steeper than observed; the Born-Infeld nonlinearity of the
relaxation dynamics softens it from alpha_th = 1 towards alpha_exp ~ 0.7, with a size-stable crossing.

## Repository layout

```
quantum-fisher-witness/
├── tex/
│   ├── quantum_fisher_witness.tex     # main source (single file)
│   ├── quantum_fisher_witness.bib     # bibliography
│   └── figures/                       # the five figures included in the note
├── code/                              # numerical scripts that produce the figures
├── out/                              # compiled PDF (build output)
├── compile.sh
├── zenodo.json
└── README.md
```

## Build

```bash
bash compile.sh        # -> out/quantum_fisher_witness.pdf
```

## Reproducibility map (script -> figure)

| Script                      | Figure                   | Role                                                |
|-----------------------------|--------------------------|-----------------------------------------------------|
| `spectral_qfi_edge_dos.py`  | `qfi_edge_dos.pdf`       | single sector w=2: marginal almost-Mathieu (a -> 1) |
| `heis_full_edge_dos.py`     | `heis_full_edge_dos.pdf` | full substrate edge DOS, d_s ≈ 4                     |
| `qfi_refine.py`             | `qfi_refine.pdf`         | q-convergence of the exponent and band stacking     |
| `thermal_closure.py`        | `thermal_closure.pdf`    | Planckian closure Delta_Pi(T) = (d_s/2) T           |
| `bi_extend.py`              | `bi_extend.pdf`          | Born-Infeld softening, four sizes (final figure)    |

Auxiliary scripts kept for provenance (not included in the note): `bi_nonlinear.py` (initial Born-Infeld flow),
`bi_diagnostic.py` (drift diagnostic), `bi_summary.py` (two-size Born-Infeld summary, superseded by `bi_extend.py`).

`qfi_refine.py` defines `harper_eigs(q, c)` and `abelian_eigs(q)`, the exact Weil-block and abelian-torus spectra of
the block decomposition; `thermal_closure.py` imports them, so run the scripts from `code/`.

```bash
cd code
python3 qfi_refine.py            # full block assembly to q = 101 (caches to qfi_refine_cache/)
python3 thermal_closure.py       # imports qfi_refine; q = 61, 101, 151
python3 bi_extend.py             # uses bi_extend_cache/; replots in seconds
python3 spectral_qfi_edge_dos.py # default sweep to q = 1009 (~1 min)
python3 heis_full_edge_dos.py 23 # full edge-DOS panels up to q = 23 (dense to q = 19)
```

Dependencies: `numpy`, `scipy`, `matplotlib`. Several scripts use multiprocessing and a local `*_cache/` directory
so that interrupted runs resume.

## Status

Draft / preprint in preparation. Not yet deposited on Zenodo (see `zenodo.json`). The thermal closure is derived;
the Born-Infeld softening is numerically confirmed and size-stable, with its asymptotic exponent open.

## Acknowledgements

Portions of editorial refinement and numerical checking benefited from iterative interactions with large language
models, used strictly as analytical assistants. All scientific interpretations, derivations, and conclusions remain
the sole responsibility of the author.

# Appendix A -- Projects

Companion notebooks: `../../LectureNotes/fam.ipynb` (Project 1),
`../../LectureNotes/effective3body.ipynb` (Project 2),
`../../LectureNotes/atoms.ipynb` (Project 3),
`../../LectureNotes/ccdchannels.ipynb` (Project 4) and
`../../LectureNotes/nqs.ipynb` (Project 5)

Figures generated from those notebooks: `../../BookFigures/appendixA/`

| Program | What it does | Runtime |
|---|---|---|
| `fam.py` | Project 1: the finite amplitude method of Nakatsukasa, Inakura and Yabana (PRC 76, 024318 (2007)) on the pairing-plus-particle-hole model, built on `../chapter07/rpa.py`: the mean field for independent bra and ket orbitals, the matrix linear response from the chapter-7 `A` and `B`, the finite-difference induced field (forward, central, exact), GMRES/Bi-CGSTAB/plain-iteration solvers, the exact response of the N-particle system, Thouless' sum rule as a double commutator, and contour integrals for poles, moments and the RPA correlation energy | about fifteen seconds; needs `scipy.linalg`, `scipy.sparse` and `scipy.sparse.linalg` |
| `effective3b.py` | Project 2: effective interactions and induced three-body forces from full configuration interaction, built on the bit-string routines of `../chapter05/fci.py`: the 1D oscillator basis with a Gaussian interaction and a shell gap, determinant bases as ordered tuples and bit strings, `KBodyOperator` with the embedding of a k-body operator in any N-particle space, the Lee-Suzuki-Okubo transformation (`LeeSuzuki`, Hermitian and non-Hermitian forms, `lowest`/`overlap` eigenvector selection, condition numbers) sector by sector, and `ClusterHierarchy`, which extracts V2_eff, V3_eff and V4_eff and compares model-space spectra of N-particle systems with the exact ones | a few seconds; needs `scipy.linalg` |
| `atoms.py` | Project 3: helium and beryllium in the hydrogenic 1s-2s-3s basis, built on `../chapter03/wick.py` (`FockSpace`, the 64 x 64 matrices of the creation and annihilation operators) and `../chapter06/hartreefock.py` (`SelfConsistentField`, optional): the closed-form Coulomb table with a quadrature check, the reference energy alpha Z^2 + beta Z, the CIS matrix from the Condon-Slater rules and from the Fock-space matrix, full CI in the M_S = 0 sector, restricted Hartree-Fock with Brillouin's theorem, the TDA and RPA A and B matrices in closed form and as double commutators, the RPA correlation energy, and the helium- and beryllium-like isoelectronic sequences | two seconds; `numpy` only |
| `ccdchannels.py` | Project 4: CCD for the two-dimensional quantum dot three ways, built on `../chapter11/quantumdot.py` and `../chapter10/coupledcluster.py`: `hartree_fock_blocks` (Fock matrix diagonalised per (m, s), so the orbitals keep their quantum numbers), `ccd_naive` (one loop per index), `ChannelCCD` (pair channels (M, S) for the ladders, cross channels for the rings, one-body intermediates for the rest; agrees with the dense solver to 1e-16 and is 25-40 times faster at 42 orbitals), `ring_ccd` (the Riccati equation B + AT + TA + TBT = 0, whose energy equals the RPA correlation energy), `rpa_matrices`/`solve_rpa`, `two_electron_spectrum`, and `run(N, shells, hw)` for one complete calculation | seconds at 42 orbitals; the notebook with up to 72 orbitals a few minutes; `numpy` only |
| `nqs.py` | Project 5: neural quantum states for the two-electron dot, built on `../chapter17/rbm.py` (`NeuralQuantumState`) and `../chapter14/vmcoptimise.py` (`blocking`): `ProjectNQS` with a flat parameter vector and the logarithmic derivatives of every parameter (checked against finite differences), brute-force and importance-sampled Metropolis returning energy, gradient, metric and the series for blocking, `train(method='sgd'|'sr')`, the width as a variational parameter, the Pade-Jastrow factor that puts the cusp into the ansatz (3.000031(18) with four hidden units), and Taut's exact state as a local-energy test | two minutes for the demo, ten for the notebook; `numpy`, `scipy.stats` for blocking |

`fam.py` imports `rpa.py` and `effective3b.py` imports `fci.py`, so run them with the
relevant chapter directory on the path:

```bash
cd BookPrograms/appendixA
PYTHONPATH=../chapter07 python3 fam.py
PYTHONPATH=../chapter05 python3 effective3b.py
PYTHONPATH=../chapter03:../chapter06 python3 atoms.py
PYTHONPATH=../chapter10:../chapter11 python3 ccdchannels.py
PYTHONPATH=../chapter13:../chapter14:../chapter17 python3 nqs.py
```

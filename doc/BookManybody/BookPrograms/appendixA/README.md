# Appendix A -- Projects

Companion notebook: `../../LectureNotes/fam.ipynb` (Project 1)

Figures generated from that notebook: `../../BookFigures/appendixA/`

| Program | What it does | Runtime |
|---|---|---|
| `fam.py` | Project 1: the finite amplitude method of Nakatsukasa, Inakura and Yabana (PRC 76, 024318 (2007)) on the pairing-plus-particle-hole model, built on `../chapter07/rpa.py`: the mean field for independent bra and ket orbitals, the matrix linear response from the chapter-7 `A` and `B`, the finite-difference induced field (forward, central, exact), GMRES/Bi-CGSTAB/plain-iteration solvers, the exact response of the N-particle system, Thouless' sum rule as a double commutator, and contour integrals for poles, moments and the RPA correlation energy | about fifteen seconds; needs `scipy.linalg`, `scipy.sparse` and `scipy.sparse.linalg` |

`fam.py` imports `rpa.py`, so run it with the chapter-7 directory on the path:

```bash
cd BookPrograms/appendixA
PYTHONPATH=../chapter07 python3 fam.py
```

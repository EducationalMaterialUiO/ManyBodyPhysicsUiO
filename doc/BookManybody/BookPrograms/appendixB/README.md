# Appendix B -- Effective three-body interactions from exact diagonalisation

Companion notebook: `../../LectureNotes/effective3body.ipynb`

Figures generated from that notebook: `../../BookFigures/appendixB/`

| Program | What it does | Runtime |
|---|---|---|
| `effective3b.py` | The a-body cluster construction of effective interactions on a toy model (spinless fermions in a 1D oscillator with a Gaussian interaction and a schematic shell gap): oscillator orbitals and two-body matrix elements on a grid, Slater determinants as ordered tuples/bit strings, k-body operators stored by their k-particle matrix elements and embedded in the N-particle space, the Lee-Suzuki-Okubo effective Hamiltonian from exact eigenpairs (sector by sector, with "lowest" or "largest P-overlap" eigenvector selection), and the extraction of `V2_eff`, the induced `V3_eff` and `V4_eff`, tested on three- and four-particle spectra | a few seconds; needs `scipy.linalg` |

Run it as a script for the summary tables:

```bash
cd BookPrograms/appendixB
python3 effective3b.py
```

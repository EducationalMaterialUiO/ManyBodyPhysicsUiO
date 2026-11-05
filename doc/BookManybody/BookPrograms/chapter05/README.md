# Chapter 5 -- Full configuration interaction theory

Companion notebook: `../../LectureNotes/fullci.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter05/`

| Program | What it does | Runtime |
|---|---|---|
| `fci.py` | determinants as bit strings, full configuration interaction for the pairing and Hubbard models, truncations, size consistency and the growth of the Hilbert space | a few seconds |
| `fciequations.py` | the FCI equation taken apart: the characteristic polynomial of the pairing matrices and its conditioning, steepest descent on the Rayleigh quotient, the energy/singles/doubles equations checked term by term with the exact eigenvector of a random Hamiltonian, and the iterative solution of the amplitude equations (imports `fci.py`) | about a second |

Run any of them as a script:

```bash
cd BookPrograms/chapter05
python3 <program>.py
```

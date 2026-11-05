# Chapter 1 -- Linear algebra and eigenvalue problems

Companion notebook: `../../LectureNotes/linearalgebra.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter01/`

| Program | What it does | Runtime |
|---|---|---|
| `spectral.py` | the spectral decomposition of a Hermitian operator and its physical readings (Section 1.11): outcomes, projectors and probabilities, functions of an observable, the two-level time evolution, the oscillator projectors, thermal and reduced density matrices with their entropies, and the basis independence of a degenerate projector | about a second |
| `direct_solvers.py` | Gaussian elimination with partial pivoting, the Doolittle LU factorisation, Cholesky and the tridiagonal algorithm | a few seconds |
| `iterative_solvers.py` | Jacobi, Gauss--Seidel, successive over-relaxation and the conjugate gradient method | a few seconds |
| `householder.py` | Householder tridiagonalisation, the QL algorithm with implicit shifts (`tqli}) and the power method | a few seconds |
| `lanczos.py` | the Lanczos algorithm with reorthogonalisation, applied to the pairing model | a few seconds |
| `svd.py` | the singular value decomposition, the pseudoinverse, the Schmidt decomposition and the factorisation of a two-body matrix | a few seconds |
| `schrodinger_diagonalization.py` | the one-dimensional Schrodinger equation solved by diagonalisation on a grid | a few seconds |

Run any of them as a script:

```bash
cd BookPrograms/chapter01
python3 <program>.py
```

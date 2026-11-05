# Chapter 8 -- Density functional theory

Companion notebook: `../../LectureNotes/dft.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter08/`

| Program | What it does | Runtime |
|---|---|---|
| `dft.py` | the local density approximation built from the uniform gas, the exchange hole, a Kohn--Sham solver in the chapter-6 basis, and the self-interaction and locality errors; the 1D exchange functional is tabulated in Fourier space (Bessel $K_0$) and interpolated with a cubic Hermite spline through energy and derivative | a few seconds; needs `scipy.linalg`, `scipy.integrate`, `scipy.interpolate`, `scipy.special` |
| `densitymatrices.py` | one- and two-body density matrices of a determinant on the grid with every identity of section 8.2 checked (traces, contraction, idempotency, exchange-hole sum rule, three forms of the Hartree and exchange energies); kinetic energy from the off-diagonal density matrix and the phase example; the four-orbital pair-transfer states; the exact two-particle state, its cumulant and the decomposition E = T + E_H + E_x + E_lambda. Imports `dft.py` | a second |
| `kohnsham1d.py` | a second, verified Kohn--Sham solver: quadrature-orthogonalised basis, direct energy on the output state, eigenvalue identity, orbital/density residuals, representation tests, the Jacobian of the output-density map with its parity-resolved spectrum against the measured mixing rates, and the functional-derivative check. Imports `dft.py` | a few seconds |

Run any of them as a script:

```bash
cd BookPrograms/chapter08
python3 <program>.py
```

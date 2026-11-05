# Chapter 19 -- The quantum Fourier transform and phase estimation

Companion notebook: `../../LectureNotes/quantumfourier.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter19/`

| Program | What it does | Runtime |
|---|---|---|
| `qpe.py` | a statevector simulator: gate application by tensor contraction, the quantum Fourier transform built gate by gate and checked against the Fourier matrix, gate counts and the approximate transform, `phase_estimation} for an arbitrary unitary, the model Hamiltonians with their Pauli terms and commuting groups, `choose_time} from the Pauli 1-norm, and exact against Trotterised evolution | about a minute; needs `scipy.linalg.expm} |

Run any of them as a script:

```bash
cd BookPrograms/chapter19
python3 <program>.py
```

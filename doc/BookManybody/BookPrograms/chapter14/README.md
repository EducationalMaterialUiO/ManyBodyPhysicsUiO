# Chapter 14 -- Optimization and resampling

Companion notebook: `../../LectureNotes/optimisation.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter14/`

| Program | What it does | Runtime |
|---|---|---|
| `vmcoptimise.py` | the parameter gradient; gradient descent, momentum and stochastic reconfiguration; the Broyden, BFGS and DFP updates with `check_quasi_newton_algebra}; `broyden_root} and `newton_root}; line searches; `golden_section} and `powell_minimise}; a vectorised many-walker production run; the autocovariance, the summation window and the exact variance of a correlated mean; blocking; and the block bootstrap and jackknife | about nine minutes; needs `scipy.stats} for the blocking test |

Run any of them as a script:

```bash
cd BookPrograms/chapter14
python3 <program>.py
```

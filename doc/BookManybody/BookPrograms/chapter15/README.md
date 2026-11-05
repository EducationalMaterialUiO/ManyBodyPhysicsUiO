# Chapter 15 -- Diffusion Monte Carlo

Companion notebook: `../../LectureNotes/diffusionmc.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter15/`

| Program | What it does | Runtime |
|---|---|---|
| `dmc.py` | importance-sampled diffusion Monte Carlo for the two-electron dot: drift-diffusion with accept/reject, trapezoidal branching, the Umrigar effective time step, population control, the mixed and growth estimators, time-step extrapolation, and a bare branching walk on the one-dimensional oscillator for contrast. Imports `vmc.py} and `vmcoptimise.py} | about ninety seconds |

Run any of them as a script:

```bash
cd BookPrograms/chapter15
python3 <program>.py
```

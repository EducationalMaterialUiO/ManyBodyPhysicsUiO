# Chapter 17 -- Boltzmann machines and generative models

Companion notebook: `../../LectureNotes/boltzmannmachines.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter17/`

| Program | What it does | Runtime |
|---|---|---|
| `rbm.py` | Gibbs sampling checked against a Cholesky draw and a transfer matrix; `BinaryBinaryRBM} and `GaussianBinaryRBM} with `check_identities} verifying every marginal and conditional; contrastive divergence on bars-and-stripes with the exact log-likelihood and Kullback--Leibler divergence; and `NeuralQuantumState}, a Gaussian--binary machine used as a trial wave function for the two-electron dot | about five minutes |

Run any of them as a script:

```bash
cd BookPrograms/chapter17
python3 <program>.py
```

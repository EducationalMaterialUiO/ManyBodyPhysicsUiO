# Chapter 20 -- The variational quantum eigensolver

Companion notebook: `../../LectureNotes/vqe.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter20/`

| Program | What it does | Runtime |
|---|---|---|
| `vqe.py` | built on the simulator of `qpe.py}: the model Hamiltonians in Pauli form, `rotate_to_z_basis} and `expectation_sampled} with the three measurement circuits and shot sampling, the marginal-readout counterexample and the deliberate VQE failure it causes, `parameter_shift_gradient} with its trigonometric structure test and its failure for three eigenvalues, the hardware-efficient ansatz and the gradient-descent loop, and both Lipkin encodings | about six minutes |

Run any of them as a script:

```bash
cd BookPrograms/chapter20
python3 <program>.py
```

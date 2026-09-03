# Chapter 3 -- Second quantization and Wick's theorem

Companion notebook: `../../LectureNotes/wicktheorem.ipynb and jordanwigner.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter03/`

| Program | What it does | Runtime |
|---|---|---|
| `slater2fock.py` | the first-to-second-quantization correspondence checked numerically  --  determinant amplitudes recovered from field operators, unitarity of the correspondence, and the exterior-power matrix of minors between two Slater-determinant bases | a few seconds |
| `wick.py` | vacuum expectation values evaluated three ways  --  by repeated anticommutation, by Wick contractions and by the generalised theorem  --  with the generalised theorem also checked as an operator identity in a small Fock space | a few seconds |
| `jordanwigner.py` | the Jordan--Wigner operators as matrices, the exact Pauli decomposition by traces, the anticommutator check, the hopping, pair and two-body identities, and the pairing and pairing-plus-particle-hole Hamiltonians in both the 2L-qubit and the compact L-qubit encodings | a few seconds |

Run any of them as a script:

```bash
cd BookPrograms/chapter03
python3 <program>.py
```

# Chapter 2 -- From linear algebra to many-body physics

Companion notebook: `../../LectureNotes/manybodybasics.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter02/`

| Program | What it does | Runtime |
|---|---|---|
| `slaterdeterminant.py` | Slater determinants, the energy functional and a minimal self-consistent field iteration; `PermutationIntegrals` evaluates every one of the N! terms of <Phi|H|Phi> for three particles and shows which permutations survive (Section 2.8.1) | a few seconds |
| `slater_update.py` | the ratio R of determinants, Sherman--Morrison updating and the stability of the nodal surface | a few seconds |

Run any of them as a script:

```bash
cd BookPrograms/chapter02
python3 <program>.py
```

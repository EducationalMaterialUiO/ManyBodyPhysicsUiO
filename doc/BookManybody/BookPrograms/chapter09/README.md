# Chapter 9 -- Many-body perturbation theory

Companion notebook: `../../LectureNotes/mbptheory.ipynb`

Figures generated from that notebook: `../../BookFigures/chapter09/`

| Program | What it does | Runtime |
|---|---|---|
| `mbpt.py` | Rayleigh--Schrodinger perturbation theory to any order by recursion, with orders one to four also written out in resolvent form (`rs_explicit`, `rs_fourth_order` returning the principal and the three renormalisation terms separately); Brillouin--Wigner solved self-consistently and its exact resummation; the two-level benchmark (`two_level_partition`, `two_level_exact`, `two_level_closed_form`) with every order in closed form and the series to thirtieth order on both sides of its radius of convergence; four model systems, size extensivity and convergence; `reduced_resolvent_demo`, the check that Q/(omega - H_0) is diagonal in the eigenbasis of H_0. Imports `fci.py`, `hartreefock.py` and `rpa.py` and reuses their model builders | a few seconds |
| `diagrams.py` | The diagram rules put to the test: `GeneralHamiltonian` (arbitrary eps, u, vbar in the full determinant space, with the RS recursion), `second_order_diagrams` and `third_order_diagrams` (the two second-order and the fourteen third-order Goldstone diagrams evaluated from the orbital-level elements with the four rules, against the determinant-space E3), `third_order_pieces` (principal, renormalisation and <WRWRW>), `fourth_order_unlinked` (the unlinked fourth-order diagrams of two non-interacting copies against the folded term); builders for the pairing, pairing + particle-hole and a random two-body Hamiltonian. Self-contained, numpy only | a few seconds |

Run any of them as a script:

```bash
cd BookPrograms/chapter09
python3 <program>.py
```

# FYS4480/FYS9480 – Week 37 (September 10–11, 2026)

## Particle-hole formalism, the normal-ordered Hamiltonian, and full configuration interaction

**Lecturer:** Morten Hjorth-Jensen, Department of Physics and Center for Computing in Science Education, University of Oslo

**Thursday September 10 (2 × 45 min):** Wick's generalised theorem and the Condon-Slater rules; the particle-hole formalism with examples; the normal-ordered Hamiltonian $\hat{H}=E_{\mathrm{ref}}+\hat{F}_N+\hat{V}_N$.
**Friday September 11 (2 × 45 min):** Slater determinants as bit patterns; full configuration interaction theory (chapter 5); the pairing model by FCI; truncations and size consistency; exercises.

### Reading assignments

- These slides: `WeeklySlides/week37.pdf`
- Lecture notes, chapter 3, sections 3.4–3.5 and 3.13–3.16, and chapter 5
- Szabo and Ostlund, chapters 2 and 4
- Shavitt and Bartlett, chapters 3–4
- Not covered this week: section 5.7 (the Hubbard ring by FCI) – part of next week's examples

### Code

- Companion notebook: `WeeklySlides/week37.ipynb`
- Companion programs: `BookPrograms/chapter03/wick.py` and `BookPrograms/chapter05/fci.py`

## Plan for week 37

### Thursday September 10, 2 × 45 min

- Where week 36 stopped: Wick's theorem was proved, the rest was not reached. Short reminder (5 min)
- Wick's generalised theorem for products of normal-ordered groups, proof, and what it buys (sections 3.13–3.14)
- Applications: matrix elements between two-particle states – the Condon-Slater rules (section 3.15)
- The particle-hole formalism: the Fermi sea as vacuum, quasiparticle operators, $\hat{H}_0$ and $\hat{H}_I$ in particle-hole form, with worked examples (section 3.4)
- The normal-ordered Hamiltonian: a proper derivation of $\hat{H}=E_{\mathrm{ref}}+\hat{F}_N+\hat{V}_N$ with $i,j,k,\dots$ below and $a,b,c,\dots$ above the Fermi level (section 3.5)

### Friday September 11, 2 × 45 min

- From the formalism to a running code: Slater determinants as bit patterns, $\hat{H}|\Phi\rangle$ by bit operations (section 3.16)
- Full configuration interaction theory: determinants as a basis, the CI expansion, the eigenvalue problem, the structure of the Hamiltonian matrix, the exponential wall (chapter 5, sections 5.1–5.5)
- FCI for the pairing model, truncated CI, size consistency, the other many-body methods as truncations of FCI, practical considerations (sections 5.6–5.11)
- Exercise session: chapter 3, exercise 4 (normal ordering on a Fermi sea) and chapter 5, exercises 1–3 (counting determinants, which blocks vanish, full CI for the pairing model) – full answers in the lecture notes

## Next week (week 38)

Full configuration interaction continued, with worked examples: the FCI code on the pairing model beyond four levels (six levels, six particles, 924 determinants, CISD and CISDTQ against FCI), the Hubbard ring in the momentum basis (section 5.7) and its strong-correlation regime, the amplitude equations solved by iteration, and the counting exercises done in class. At the end of the week we begin Hartree-Fock theory (chapter 6): the reference determinant as a variational ansatz, the energy functional $E_{\mathrm{ref}}$ of this week and its minimisation.

**Reading for week 38:** chapter 5 (all sections) and the first sections of chapter 6 of the lecture notes; Szabo and Ostlund chapters 3–4.

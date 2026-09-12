# FYS4480/FYS9480 – Week 38 (September 17–18, 2026)

## Full configuration interaction: variational principle, Hamiltonian matrix, diagrams, iteration – and a first step into Hartree-Fock theory

**Lecturer:** Morten Hjorth-Jensen, Department of Physics and Center for Computing in Science Education, University of Oslo

**Thursday September 17 (2 × 45 min):** Slater determinants as bit patterns; the pairing model as our first FCI model; the FCI equation from variational calculus (Ritz), the eigenvalue problem and the characteristic polynomial; the structure of the Hamiltonian matrix and the Condon-Slater rules; truncations and size consistency.
**Friday September 18 (2 × 45 min):** the projected FCI equations, their diagrams, the "non-practical" iterative solution and its link to Hartree-Fock; Hartree-Fock theory (chapter 6): mean field, variational calculus, the Hartree-Fock equations, the self-consistent field.

### Reading assignments

- These slides: `WeeklySlides/week38.pdf`
- Lecture notes, chapter 3, section 3.16 (bit representation); chapter 5, all sections, in particular 5.3–5.4 and 5.8–5.10; chapter 6, sections 6.1–6.5
- Szabo and Ostlund, chapters 3–4
- Shavitt and Bartlett, chapters 4 and 9
- Not covered this week: section 5.7 (the Hubbard ring by FCI) – left as a computational exercise (chapter 5, exercise 5)

### Code

- Companion notebook: `WeeklySlides/week38.ipynb`
- Companion programs: `BookPrograms/chapter05/fci.py` and `BookPrograms/chapter06/hartreefock.py`

## Plan for week 38

### Thursday September 17, 2 × 45 min

- Where week 37 stopped (5 min): the normal-ordered Hamiltonian and the first FCI ideas were covered; bit patterns, truncations and size consistency were not
- Slater determinants as bit patterns, $\hat{H}|\Phi\rangle$ by bit operations – the inner loop of every FCI code (section 3.16); the pairing model as our first FCI model, 70 determinants against 6 (section 5.6)
- The FCI equation properly derived: variational calculus with a Lagrange multiplier, Ritz's method as an optimisation of the coefficients, the eigenvalue problem, $\det(\mathbf{H}-E\mathbf{1})=0$ and its characteristic polynomial (section 5.3)
- The structure of the Hamiltonian matrix in particle-hole language and the Condon-Slater rules of weeks 36–37 (section 5.4); truncations and size consistency (sections 5.8–5.9)

### Friday September 18, 2 × 45 min

- The FCI equation projected: energy, singles and doubles equations, and their diagrammatic representation
- The non-practical way of solving the FCI equations: the amplitude equations by iteration, and what that teaches about perturbation theory, coupled cluster and Hartree-Fock (section 5.10, `fci.do.txt`)
- Hartree-Fock theory (chapter 6): mean field, variational calculus, determinants under a change of basis, the Hartree-Fock equations, the density matrix and the self-consistent field, Brillouin's theorem (sections 6.1–6.5)
- Exercise session: suggested set for week 38 (last frames of the slides) – bit patterns; the FCI equation from the variational principle (Rayleigh quotient, the 2 × 2 secular equation, the characteristic polynomial, interlacing); chapter 5, exercise 4 (truncations and size consistency); the projected equations and their diagrams; chapter 6, exercises 1–2 (first parts, Hartree-Fock); for the ambitious, chapter 5, exercise 5 (the Hubbard ring)

## Next week (week 39)

Hartree-Fock theory continued (chapter 6): the self-consistent field in practice with `hartreefock.py`; Hartree-Fock in second quantization – varying the determinant instead of the orbitals – and Thouless' theorem on what a nearby determinant is, with the Baker-Campbell-Hausdorff expansion it rests on; stability of the Hartree-Fock solution: stationary versus minimal, and the Lipkin model as an explicit instability; and, if time permits, the one system where the whole programme is analytic, the homogeneous electron gas.

**Reading for week 39:** chapter 6, sections 6.6–6.11 of the lecture notes; Szabo and Ostlund chapter 3.

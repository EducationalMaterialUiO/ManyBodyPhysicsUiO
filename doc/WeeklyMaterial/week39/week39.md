# FYS4480/FYS9480 – Week 39 (September 24–25, 2026)

## Hartree-Fock theory: from the variational principle to ⟨a|f̂|i⟩ = 0, Thouless' theorem, and the stability of the mean field

**Lecturer:** Morten Hjorth-Jensen, Department of Physics and Center for Computing in Science Education, University of Oslo

**Thursday September 24 (2 × 45 min):** why a mean field; variational calculus and Lagrange multipliers; determinants under a change of basis; the Hartree-Fock equations by varying the coefficients; the density matrix and the self-consistent field; Brillouin's theorem; Hartree-Fock in second quantization – varying one orbital, $|\Phi_0\rangle+\eta\,a_a^\dagger a_i|\Phi_0\rangle$, and the equation $\langle a|\hat f|i\rangle=0$.
**Friday September 25 (2 × 45 min):** Thouless' theorem – what a nearby determinant is, and its proof; the stability of the Hartree-Fock solution – the energy to second order, the stability matrix, stationary versus minimal, and the Lipkin model as an explicit instability.

### Reading assignments

- These slides: `WeeklySlides/week39.pdf`
- Lecture notes, chapter 6, sections 6.1–6.7 and 6.10–6.11
- Szabo and Ostlund, chapter 3
- Shavitt and Bartlett, chapter 4

### Code

- Companion notebook: `WeeklySlides/week39.ipynb`
- Companion program: `BookPrograms/chapter06/hartreefock.py`

## Plan for week 39

### Thursday September 24, 2 × 45 min

- Why a mean field; the two ingredients of Hartree-Fock theory; $E^{\rm HF}\ge E_0$ (section 6.1)
- Variational calculus with Lagrange multipliers; determinants under a linear change of basis and the invariance under unitary mixing of the occupied orbitals (sections 6.2–6.3)
- The Hartree-Fock equations by varying the expansion coefficients; the Fock matrix; a nonlinear eigenvalue problem (section 6.4)
- The density matrix and the self-consistent field; convergence of the energy versus convergence of the orbitals; Brillouin's theorem (section 6.5)
- Hartree-Fock in second quantization: varying one orbital is adding a $1p$-$1h$ determinant; the energy to first order with complex $\eta$; the normal-ordered Hamiltonian gives $\langle a|\hat f|i\rangle=0$, and how that becomes the eigenvalue problem (section 6.6)

### Friday September 25, 2 × 45 min

- Thouless' theorem: every determinant not orthogonal to $|c\rangle$ is $e^{\hat T}|c\rangle$ with $\hat T$ a sum of $1p$-$1h$ operators; the three steps of the proof; what it buys us (section 6.7)
- Stability: stationary is not minimal; the energy to second order; the stability matrix $\hat M$ with blocks $\Delta$, $A$, $B$; the criterion; $\hat M$ is the RPA matrix (section 6.10); the Lipkin model as an explicit instability, the pairing model as stable but useless (section 6.11)
- Exercises: suggested set for week 39 (varying one orbital; single-particle energies and Thouless' theorem; stability, with the Lipkin and pairing models), and answers to the week-38 set (last frames of the slides)

## Next week (week 40)

What remains of chapter 6: the Baker-Campbell-Hausdorff expansion and the Trotter-Suzuki splitting that Thouless' theorem let us skip (sections 6.8–6.9), Hartree-Fock drawn as diagrams – the Fock operator as a bubble insertion and self-consistency as an infinite sum of bubbles (section 6.13) – and the one system where the whole programme is analytic, the homogeneous electron gas (section 6.12). Then the stability matrix becomes an eigenvalue problem of its own: the Tamm-Dancoff and random-phase approximations of chapter 7.

**Reading for week 40:** chapter 6, sections 6.8–6.9 and 6.12–6.13, and chapter 7 of the lecture notes; Szabo and Ostlund chapter 3 (the electron gas); Ring and Schuck on the random-phase approximation.

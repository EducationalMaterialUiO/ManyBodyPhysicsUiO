# FYS4480/FYS9480 – Week 48, November 26–27, 2026

**Coupled cluster with singles and doubles, and a summary of the course**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday November 26: coupled cluster with singles and doubles (CCSD), derived analytically and diagrammatically
- Friday November 27: summary of the course, and the final oral exam

Reading: the slides; lecture notes chapter 10, sections 10.7 (the CCSD equations), 10.10 (when singles matter), 10.13 (beyond CCSD) and 10.14 (the diagrammatic derivation). Shavitt and Bartlett, chapters 9–10. Code: `BookPrograms/chapter10/coupledcluster.py`, `BookPrograms/chapter10/ccsd_diagrams.py`; notebook `week48.ipynb` (three routes to the CCSD residuals – brute force, the 3 + 14 + 31 diagrams, and CCD with a T₁-dressed Hamiltonian – and a short CCSD solver).

Exercises: none this week. This is the last week of lectures.

## Plan for week 48

### Thursday November 26, 2 × 45 min: coupled cluster with singles and doubles
- Lecture 1, analytically: from CCD to CCSD, T = T₁ + T₂ – what CCD left alone; how long the equations are (excitation-level counting: 3 energy terms, 14 for the singles, 31 for the doubles) and the energy ΔE = Σ f_ia t_i^a + ¼ Σ ⟨ij‖ab⟩ τ_ij^ab; what e^(T₁) does – it rotates the orbitals (Thouless' theorem), and the Baker–Campbell–Hausdorff series for a single creation or annihilation operator stops after one commutator; the T₁-dressed Hamiltonian, again a one- plus two-body operator with integrals h̃ = X h Yᵀ and ṽ = X X v Yᵀ Yᵀ, X = 1 − t₁, Y = 1 + t₁ᵀ; since [T₁, T₂] = 0, H̄ = e^(−T₂) H̃ e^(T₂): CCSD is the CCD equation of week 47 with dressed integrals, plus a four-term singles equation; with T₂ = 0 the singles equation is Hartree–Fock, with T₁ = 0 the doubles equation is CCD; unfolding the dressing – which of the 14 + 31 terms each dressed vertex contains; reading the equations (first iteration is MP2, singles start at second order in a Hartree–Fock basis, insensitivity to the choice of orbitals, cost n⁶)
- Lecture 2, diagrammatically: the rules with one more element (the T₁ bar); the three energy diagrams; the fourteen diagrams of the singles equation and the equation read off; the thirty-one diagrams of the doubles equation and the equation read off; reading a diagram and finding its CCD relative and its place in the dressed equation; from diagrams to intermediates (the Stanton–Gauss–Watts–Bartlett form); when singles matter – the pairing model with a pair-breaking particle-hole term; beyond CCSD – triples and CCSD(T), excited states and properties, unitary coupled cluster, where single-reference coupled cluster ends

### Friday November 27, 2 × 45 min: summary of the course, and the final oral exam
- Lecture 3, the semester in review, part 1: the one-sentence summary and the route through the fourteen weeks; second quantization, Wick's theorem and the normal-ordered Hamiltonian; full configuration interaction, truncations and size consistency; Hartree–Fock theory, Thouless' theorem and stability; matrix product states and the DMRG; the electron gas and density functional theory – each with a list of what you should be able to do
- Lecture 4, part 2 and the exam: many-body perturbation theory; diagrams and the linked-diagram theorem; coupled-cluster theory; one model, every method – the pairing Hamiltonian by Hartree–Fock, perturbation theory, CID, RPA, CCD and FCI; how the methods relate (what is truncated, upper bound, size extensivity, improvability, cost); true-or-false questions as an oral-exam warm-up; what we did not do and where to go next; the final oral exam – practicalities, and a list of topics to prepare

## The final oral exam

- The two midterms count 25 % each and the final oral exam 50 % of the final grade.
- Dates, time slots, duration and format are announced in the Friday lecture.
- Topics to prepare: second quantization and Wick's theorem; full configuration interaction; Hartree–Fock theory (including the electron gas); density functional theory; matrix product states and the DMRG; many-body perturbation theory; diagrams and the linked-diagram theorem; coupled-cluster theory.

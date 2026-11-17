# FYS4480/FYS9480 – Week 47, November 19–20, 2026

**Coupled-cluster doubles, step by step**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday November 19: coupled-cluster theory once more (a repetition of the lectures of Friday November 13) – the exponential ansatz, the similarity-transformed Hamiltonian, why the Baker–Campbell–Hausdorff expansion stops – and the coupled-cluster doubles (CCD) amplitude equation derived term by term with Wick's theorem
- Friday November 20: the same equation derived with diagrams, and CCD for the pairing model in code

Reading: the slides; lecture notes chapter 10, sections 10.1–10.6 (from the exponential ansatz to the CCD equations), 10.9 (the pairing model) and the CCD diagrams of section 10.14. Shavitt and Bartlett, chapter 9, sections 9.1–9.4 and chapter 10, sections 10.1–10.4. Code: `BookPrograms/chapter10/coupledcluster.py`, `BookPrograms/chapter10/ccsd_diagrams.py`; notebook `week47.ipynb` (the classes `PairingModel`, `CCD`, `DeterminantSpace` and `PairCCD`).

Exercises: none this week – the second midterm is due Monday November 23 at midnight. Short answers to the week-46 exercises are on the last slides.

## Plan for week 47

### Thursday November 19, 2 × 45 min: from the ansatz to the CCD equation, analytically
- Lecture 1 (repetition of November 13): three defects of the methods so far and one change of ansatz, |Ψ⟩ = e^T |Φ₀⟩; the cluster operator; commuting excitations and T = log(1 + C); why exponentiate – products of low-rank clusters replace independent high-rank ones; size consistency for two non-interacting subsystems; the similarity-transformed Hamiltonian and the projected equations; why the Baker–Campbell–Hausdorff series stops, H̄_N = (H_N e^T)_c; coupled-cluster doubles, T = T₂: operator counting limits the doubles equation to T₂²; why CCD is not exact although a similarity transformation preserves the spectrum; the road map – ten terms to derive
- Lecture 2: the toolbox (contractions relative to |Φ₀⟩, the generalised Wick theorem, the sign of a contraction) and a projection lemma, ⟨Φ_ij^ab| Σ X_kl^cd {a†_c a†_d a_l a_k} |Φ₀⟩ = P(ij)P(ab) X_ij^ab; the energy ΔE = ¼ Σ ⟨ij‖ab⟩ t_ij^ab and the driving term ⟨ab‖ij⟩ as warm-ups; the two Fock terms, P(ab) f_bc t_ij^ac − P(ij) f_kj t_ik^ab, and why their diagonal part is the energy denominator; counting contractions of V_N T₂ – the particle-particle ladder ½⟨ab‖cd⟩t_ij^cd, the hole-hole ladder ½⟨kl‖ij⟩t_kl^ab and the ring P(ij)P(ab)⟨kb‖cj⟩t_ik^ac, with the weights obtained by counting equal contraction patterns; two T₂: the five ways of sharing the four lines of ⟨kl‖cd⟩ between two amplitudes, the four quadratic terms as the linear terms with dressed vertices (intermediates), and the fifth, disconnected distribution ΔE·t that cancels – configuration interaction versus coupled cluster in one line each; CCD is exact for two particles and for any number of independent pairs; the complete equation, checked against e^(−T) H e^(T) in the space of determinants

### Friday November 20, 2 × 45 min: the same equation with diagrams, and in code
- Lecture 3: from contraction patterns to pictures; the fragments of H_N and their excitation level; bookkeeping by excitation level (ten diagrams at most, before anything is drawn); the rules for drawing and reading – sign (−1)^(h+l), ½ for each pair of equivalent lines or vertices, P(ij) and P(ab) for inequivalent external lines, no denominators; the energy, the driving term, the two Fock diagrams, the two ladders and the ring read one by one; the four diagrams with two bars; two pictures that are not drawn (a bar attached to nothing, a line between two bars); the CCD equation in pictures; what iterating the equation sums (iteration 1 is MP2, iteration 2 adds third order and the quadruples part of fourth order); which parts of the fourth-order energy CCD contains; intermediates and cost
- Lecture 4: the pairing model in spin orbitals – matrix elements, Fock matrix, reference energy; what the pairing interaction does to the ten terms (no ring term, no singles, pair amplitudes only); the program – four small classes; the ten terms as one `einsum` each; the solver (damped Jacobi iteration on the residual) and the referee (brute-force e^(−T) H e^(T) on bit strings); results for four levels and four particles (CCD four parts in a million from the exact energy at g = 0.25, below the exact energy and 0.12 off at g = 2); what each diagram contributes; switching the diagrams on group by group (MP2, linear CCD, CID, CCD, exact); the reduction to four pair amplitudes and its ten-line residual; what CCD misses – the connected quadruple T₄, seen by comparing with the exact wave function; why a coupled-cluster energy can lie below the exact one; convergence, mixing, vanishing denominators and the need to keep the amplitudes antisymmetric; a larger model space (ten levels, ten particles) and why intermediates matter

## Next week (week 48, the last week of the semester)

- Thursday November 26: coupled cluster with singles and doubles (CCSD), derived analytically (the T₁-dressed Hamiltonian) and diagrammatically (three energy diagrams, fourteen for the singles equation, thirty-one for the doubles equation); intermediates; when singles matter; beyond CCSD.
- Friday November 27: summary of the course and discussion of the final oral exam.
- Reading: chapter 10, sections 10.7, 10.10, 10.13 and 10.14; Shavitt and Bartlett, chapters 9–10. No exercises.

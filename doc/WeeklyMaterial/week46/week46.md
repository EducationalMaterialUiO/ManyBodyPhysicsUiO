# FYS4480/FYS9480 – Week 46, November 12–13, 2026

**The linked-diagram theorem, and coupled-cluster theory**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday November 12 (double lecture): linked and unlinked diagrams – the wave operator drawn at second and third order, how open diagrams close into energy diagrams, the renormalisation terms as unlinked diagrams with a minus sign, the factorisation of denominators (Frantz–Mills), the linked-diagram theorem and its proof in imaginary time, exclusion-principle-violating diagrams and the two-level benchmark
- Friday November 13: coupled-cluster theory – the exponential ansatz and why e^T is the linked-diagram theorem built into the wave function, the similarity-transformed Hamiltonian, why the Baker–Campbell–Hausdorff expansion terminates, the CCD equations and their diagrams, coupled cluster as a resummation of perturbation theory, the pairing model

Reading: the slides; lecture notes chapter 9, section 9.15 from "Unlinked diagrams and the linked-diagram theorem" to the end (9.15.7–9.15.9); chapter 10, sections 10.1–10.7. Shavitt and Bartlett, chapters 5–6 and 9. Code: `BookPrograms/chapter09/diagrams.py`, `BookPrograms/chapter10/coupledcluster.py`; notebook `week46.ipynb` (unlinked pieces, two copies, the two-level count, a forty-line CCD solver).

Exercises: the week-46 set is on the last slide; answers on the week-47 slides. The answers to the week-45 exercises (`Exercises/ExercisesWeek45.tex`) are on the slides before it.

## Plan for week 46

### Thursday November 12, 2 × 45 min: the linked-diagram theorem
- Lecture 1: the wave operator drawn with bare vertices – connected, disconnected-but-linked (products of amplitudes: the quadruple of Ψ⁽²⁾ is t_m t_n) and unlinked (a closed first-order piece floating beside an open one) diagrams at second and third order; closing with a vertex gives connected and unlinked energy diagrams; every renormalisation term is an unlinked drawing with a minus sign (pairing model: principal = ladders + hole bubbles + unlinked, and the unlinked piece is −0.0125868 at g = 0.5)
- Normal ordering removes the one-vertex closed pieces; the first surviving unlinked diagrams appear at fourth order, and the factorisation 1/(D_m(D_m+D_n)D_n) + 1/(D_m(D_m+D_n)D_m) = 1/(D_m² D_n) turns them into the folded term −E⁽²⁾⟨W R₀² W⟩; two non-interacting copies as a check
- Lecture 2: the factorisation theorem for two open histories by induction; the theorem for energy and wave operator; the proof in imaginary time (Z(T), Dyson series, connected pieces exponentiate, the marked vertex, intervals become denominators, vacuum pieces cancel in the wave operator); Pauli exclusion at an intermediate cut, direct and exchanged hole lines at fourth order, why the exclusion-principle-violating linked terms are kept; the two-level benchmark counted (+g⁴/Δ³ from EPV diagrams alone); Brillouin–Wigner denominators do not factorise; consequences for independent systems

### Friday November 13, 2 × 45 min: coupled-cluster theory, chapter 10
- Lecture 3: three defects and one change of ansatz, |Ψ⟩ = e^T |Φ₀⟩; the cluster operator (commuting, excitations cannot be undone); why exponentiate – C₂ = T₂ + ½T₁², C₄ ∋ ½T₂², products of low-rank amplitudes replace independent high-rank ones; e^(T_A + T_B) = e^(T_A) e^(T_B); the similarity-transformed Hamiltonian (same spectrum, non-Hermitian); why BCH stops at the fourfold commutator: (H_N e^T)_c; the projected equations and the energy ΔE = Σ f_ia t_i^a + ¼ Σ ⟨ij‖ab⟩ t_ij^ab + ½ Σ ⟨ij‖ab⟩ t_i^a t_j^b
- Lecture 4: the CCD equations term by term (driving term, Fock terms, pp ladder, hh ladder, ring, four quadratic terms) and as diagrams with amplitude bars; intermediates and the O(n⁶) cost; iteration; the pairing model (CCD 3.6 × 10⁻⁶ off at g = 0.25, over-correlating at strong g – missing connected quadruples); coupled cluster as a resummation (order by order = Møller–Plesset in the Fock partition; the partition warning); size extensivity in one line; everything against everything

## Exercises, week 46

1. Linked and unlinked at third order: write out all unlinked drawings in ⟨Φ_m|RVRV RV|Φ₀⟩ with bare vertices, evaluate them with the rules, show their sum is minus the three renormalisation terms of Ω⁽³⁾, then close every drawing with a top vertex and identify the four pieces of E⁽⁴⁾.
2. The factorisation theorem: prove S_rs = Π A_u⁻¹ Π B_v⁻¹ by induction and derive the closed-piece form; check it for r = 3, s = 2 by listing the five allowed interleavings and identify the fifth-order renormalisation terms.
3. Exclusion-principle-violating diagrams: list the four complete contractions of the fourth-order four-cross diagram of the two-level model with loops and signs, and show that the linked exchange terms alone give +g⁴/Δ³; redo the count for the pairing model with one pair in two levels.
4. The exponential ansatz (chapter 10, exercise 1): [T_m, T_n] = 0; C₁ … C₄ in terms of the T's; factorisation of e^T for two non-interacting subsystems and its failure for 1 + C₂; why the series terminates.
5. CCD for the pairing model: show that the ring term and t_i^a vanish, reduce the CCD equation to pair amplitudes t_p^q, solve it by iteration for four levels and two pairs at g = 0.5, and compare with the lecture-4 table (`week46.ipynb` has a general solver to check against).

## Next week (week 47)

- Coupled cluster continued (chapter 10, sections 10.8–10.12): the CCSD equations with intermediates; when singles matter – the pairing plus particle-hole model on the bare and on the Hartree–Fock reference; size extensivity verified; beyond CCSD (triples); a diagrammatic derivation of the amplitude equations from the rules of weeks 45–46; a first look at unitary coupled cluster.
- Reading: chapter 10, sections 10.8–10.12; Shavitt and Bartlett, chapters 9–10. The week-46 exercises are discussed in the exercise session and answered on the week-47 slides.

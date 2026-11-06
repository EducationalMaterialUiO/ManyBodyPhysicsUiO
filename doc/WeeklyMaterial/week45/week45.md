# FYS4480/FYS9480 – Week 45, November 5–6, 2026

**Many-body perturbation theory: diagram rules and the third order**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday November 5: every contribution to the energy through fourth order and to the wave operator through third (the list from week 44), then the diagrams: contractions are lines, cuts are denominators, and the four rules – sums, denominators, ½ per pair of equivalent lines, (−1)^(h+l) – derived from Wick's theorem
- Friday November 6: the fourteen third-order diagrams, Møller–Plesset third order, the pairing model diagram by diagram, one class of diagrams summed to all orders (the unfolded pair chains and a restricted Bethe–Salpeter equation), and a look ahead at the unlinked diagrams of fourth order

Reading: the slides; lecture notes chapter 9, section 9.15 (diagrammatic approach and the linked diagram theorem) and the section "Pair ladders to all orders: a restricted Bethe–Salpeter equation"; Shavitt and Bartlett, chapters 2–5. Code: `BookPrograms/chapter09/diagrams.py` and `mbpt.py` (`pair_ladder_orders`, `pair_ladder_sum`, `pair_ladder_closed_form`, `pair_ladder_pp_closed_form`, `pairing_branch_points`); notebooks `week45.ipynb` (section 6) and `mbptheory.ipynb` (section 13).

Exercises: `Exercises/ExercisesWeek45.tex` (reproduced on the last two slides); answers on the week-46 slides. The answers to the week-44 exercises are on the slides before them (and in `Exercises/AnswersWeek44.tex`).

## Plan for week 45

### Thursday November 5, 2 × 45 min: the lists, and the rules derived
- From week 44: the recursion, Ω⁽¹⁾, Ω⁽²⁾, Ω⁽³⁾ and E⁽¹⁾ … E⁽⁴⁾ in resolvent form and in the determinant basis
- The scalar separation H_I = E⁽¹⁾ I + W absorbs the explicit E⁽¹⁾ insertions in the recursion: E⁽³⁾ = ⟨W R₀ W R₀ W⟩, while ε₀ + E⁽¹⁾ stays in the energy and the folded term −E⁽²⁾‖Ψ⁽¹⁾‖² survives at fourth order; Hartree–Fock and the choice of reference energy
- Contractions are lines; cuts are denominators ([H₀, a†_p] = ε_p a†_p); what Q removes; Pauli-forbidden label assignments in determinant sums versus unrestricted diagram sums
- The weight (4^(n_V) relabellings, equivalent lines) and the sign (vertex ends, the permutation π, cycles are loops, (−1)^(K−l)(−1)^p = (−1)^(h+l)); open diagrams and quasi-loops; a fixed doubles amplitude versus the ¼ of the summed wave function
- First and second order re-read with the rules

### Friday November 6, 2 × 45 min: the third order, enumerated and evaluated
- Lecture 3: third order by intermediate states – fourteen connected diagrams, three of them MP3 (pp ladder, hh ladder, ph ring), eleven with crosses; what each family means and where it is summed to all orders (ladders: Brueckner; rings: RPA; both: coupled cluster); diagrams 1–7 read off the drawings
- Lecture 4: diagrams 8–14 and two sign checks; the pairing model diagram by diagram, −53/576 − 53/576 + 29/288 = −1/12 in units of g³, and where the "renormalisation term" went; reading a diagram into code
- Summing one class of diagrams to all orders: in the pairing model (four levels, four particles) the unfolded pair chains – one pair promoted, scattered within the four 2p2h determinants and returned, every propagator at the fixed reference energy – live in a four-state pair space with W_DD = −g/2 on the edges of a square and R₀ = diag(1/D_m); the chain at order n is wᵀR₀(W_DD R₀)ⁿ⁻²w; second, third and fourth order (−7/24 g², −1/12 g³, −7/288 g⁴) by hand, the sixteen two-step walks on the square; what the restriction leaves out at fourth order (the 4p4h chain −49/4608 g⁴ and the folded term +203/6912 g⁴, which partially cancel; total −77/13824 g⁴, the chain alone 48/11 ≈ 4.36 times the full fourth order); the Bethe–Salpeter-type equation t = t⁽¹⁾ + Kt with t⁽¹⁾ = R₀w and K = R₀W_DD (associativity), t_m = [w_m + Σₙ W_mn t_n]/D_m, ΔE_lad = H_0D(E_ref − H_DD)⁻¹H_D0; K²t⁽¹⁾ = (g²/12)t⁽¹⁾ makes the even and odd orders separately geometric, closed form ΔE_lad = −g²(7+2g)/(2(12−g²)), poles at ±2√3; the comparison with the pp-only chains, CID, CCD and FCI; frozen denominators versus CID (the folded terms are the expansion of the CID resolvent in ΔE, first fold −E⁽²⁾wᵀR₀²w) and the convergence radii: the full Rayleigh–Schrödinger series has R_RS ≃ 1.52052 from the branch points g ≃ 0.679965 ± 1.360006 i of the quartic p(y,g) = y⁴ − (20+3g²)y² + 2g³y + 64 + 20g², y = E + g − 6, while the chain series converges to 2√3 and is wrong there (−2.75 vs −1.49 at g = 2); pause to think #5 (Brillouin–Wigner in the reference-plus-doubles space; the pp-only energy is above FCI at g = 1.5 and below at g = 2, so no variational bound; "pp only" means the pp moves of the zero-diagonal W_DD, not the conventional Brueckner G-matrix, which iterates the interaction with a Pauli-projected two-particle propagator)
- A look ahead: at fourth order a complete contraction can fall into two closed pieces overlapping in time – the unlinked diagrams, and the linked-diagram theorem, are the subject of week 46

## Exercises, week 45

1. a) Show that for any ω, E₀ = ε₀ + Σₙ ⟨Φ₀|H_I [Q/(ω − H₀) (ω − E₀ + H_I)]ⁿ|Φ₀⟩. b) Discuss ω = E₀ (Brillouin–Wigner) and ω = ε₀ (Rayleigh–Schrödinger); compare the first few terms. c) Show that H₀ = Σ ⟨p|h₀|q⟩ a†_p a_q = Σ ⟨p|h₀|q⟩ {a†_p a_q} + Σᵢ ⟨i|h₀|i⟩ and that the two-body H_I = ¼ Σ ⟨pq|v|rs⟩ a†_p a†_q a_s a_r = ¼ Σ ⟨pq|v|rs⟩ {a†_p a†_q a_s a_r} + Σ_{pqi} ⟨pi|v|qi⟩ {a†_p a_q} + ½ Σ_{ij} ⟨ij|v|ij⟩; explain the symbols, name the reference vacuum, and draw all these terms. d) Set up all diagrams (antisymmetrised Goldstone or Hugenholtz) to second order, including the reference energy, that contribute to E₀ in Rayleigh–Schrödinger theory, and write down their closed-form expressions with the diagram rules.
2. A one-particle system with H₀ = Σ ε_p a†_p a_p, H_I = g Σ_{pq} a†_p a_q, two states with ε₁ < ε₂ (p = 1 the model space, p = 2 the excluded space). e) Write the Hamiltonian in normal-ordered form and set up all diagrams (an X for H_I). f) With |Φ₀⟩ = a†₁|0⟩ and |Φ_i^a⟩ = a†₂ a₁|Φ₀⟩ set up the 2 × 2 matrix, find the exact energy and expand it in g. g) Find the ground-state energy to third order in Rayleigh–Schrödinger theory, compare with the exact expansion, draw all contributing diagrams and comment.

## Next week (week 46)

- Thursday (double lecture): the linked-diagram theorem. The wave operator drawn – linked, disconnected-but-linked and unlinked open diagrams at second and third order, and how closing them with a vertex turns them into connected and unlinked energy diagrams; the renormalisation terms as the unlinked diagrams with a minus sign; fourth order with normal-ordered vertices and the factorisation of denominators (Frantz–Mills); the theorem and its proof in imaginary time; exclusion-principle-violating diagrams and the two-level benchmark; consequences (size extensivity, Brillouin–Wigner, summation by topology).
- Friday: coupled-cluster theory, chapter 10 – the exponential ansatz and why e^T is the linked-diagram theorem built into the wave function, the similarity-transformed Hamiltonian, why the expansion terminates, the CCD equations, coupled cluster as a resummation of perturbation theory, and the pairing model.
- Reading: chapter 9, section 9.15 from "Unlinked diagrams and the linked-diagram theorem" to the end; chapter 10, sections 10.1–10.7; Shavitt and Bartlett, chapters 5–6 and 9. The week-45 exercises are discussed in the exercise session and answered on the week-46 slides.

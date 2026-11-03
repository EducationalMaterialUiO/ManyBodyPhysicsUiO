# FYS4480/FYS9480 – Week 45, November 5–6, 2026

**Many-body perturbation theory: diagram rules, third order and the linked-diagram theorem**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday November 5: every contribution to the energy through fourth order and to the wave operator through third (the list from week 44), then the diagrams: contractions are lines, cuts are denominators, and the four rules – sums, denominators, ½ per pair of equivalent lines, (−1)^(h+l) – derived from Wick's theorem
- Friday November 6: the fourteen third-order diagrams, MP3 and the pairing model diagram by diagram; unlinked diagrams at fourth order, the factorisation of denominators, and the linked-diagram theorem

Reading: the slides; lecture notes chapter 9, section 9.15 (diagrammatic approach and the linked diagram theorem); Shavitt and Bartlett, chapters 4–5. Code: `BookPrograms/chapter09/diagrams.py`; notebook `week45.ipynb`.

Exercises: `Exercises/ExercisesWeek45.tex` (reproduced on the last two slides); answers on the week-46 slides. The answers to the week-44 exercises are on the slides before them (and in `Exercises/AnswersWeek44.tex`).

## Plan for week 45

### Thursday November 5, 2 × 45 min: the lists, and the rules derived
- From week 44: the recursion, Ω⁽¹⁾, Ω⁽²⁾, Ω⁽³⁾ and E⁽¹⁾ … E⁽⁴⁾ in resolvent form and in the determinant basis
- Normal ordering the vertices, H_I = E⁽¹⁾ + W, absorbs every renormalisation term with a factor E⁽¹⁾: E⁽³⁾ = ⟨W R₀ W R₀ W⟩ with nothing to subtract
- Contractions are lines; cuts are denominators ([H₀, a†_p] = ε_p a†_p); what Q removes
- The weight (4^(n_V) relabellings, equivalent lines) and the sign (vertex ends, the permutation π, cycles are loops, (−1)^(K−l)(−1)^p = (−1)^(h+l)); open diagrams and quasi-loops
- First and second order re-read with the rules

### Friday November 6, 2 × 45 min: third order and the linked-diagram theorem
- Third order by intermediate states: fourteen connected diagrams, three of them MP3 (pp ladder, hh ladder, ph ring); the pairing model diagram by diagram, −53/576 − 53/576 + 29/288 = −1/12 in units of g³; where the "renormalisation term" went
- Fourth order: the unlinked diagrams, the factorisation of their denominators, and the folded term −E⁽²⁾⟨W R₀² W⟩ that removes them
- The linked-diagram theorem: energy = connected closed diagrams, wave operator = linked diagrams; size extensivity, Brillouin–Wigner, partial summations. The proof: week 46

## Exercises, week 45

1. a) Show that for any ω, E₀ = ε₀ + Σₙ ⟨Φ₀|H_I [Q/(ω − H₀) (ω − E₀ + H_I)]ⁿ|Φ₀⟩. b) Discuss ω = E₀ (Brillouin–Wigner) and ω = ε₀ (Rayleigh–Schrödinger); compare the first few terms. c) Show that H₀ = Σ ⟨p|h₀|q⟩ a†_p a_q = Σ ⟨p|h₀|q⟩ {a†_p a_q} + Σᵢ ⟨i|h₀|i⟩ and that the two-body H_I = ¼ Σ ⟨pq|v|rs⟩ a†_p a†_q a_s a_r = ¼ Σ ⟨pq|v|rs⟩ {a†_p a†_q a_s a_r} + Σ_{pqi} ⟨pi|v|qi⟩ {a†_p a_q} + ½ Σ_{ij} ⟨ij|v|ij⟩; explain the symbols, name the reference vacuum, and draw all these terms. d) Set up all diagrams (antisymmetrised Goldstone or Hugenholtz) to second order, including the reference energy, that contribute to E₀ in Rayleigh–Schrödinger theory, and write down their closed-form expressions with the diagram rules.
2. A one-particle system with H₀ = Σ ε_p a†_p a_p, H_I = g Σ_{pq} a†_p a_q, two states with ε₁ < ε₂ (p = 1 the model space, p = 2 the excluded space). e) Write the Hamiltonian in normal-ordered form and set up all diagrams (an X for H_I). f) With |Φ₀⟩ = a†₁|0⟩ and |Φ_i^a⟩ = a†₂ a₁|Φ₀⟩ set up the 2 × 2 matrix, find the exact energy and expand it in g. g) Find the ground-state energy to third order in Rayleigh–Schrödinger theory, compare with the exact expansion, draw all contributing diagrams and comment.

## Next week (week 46)

- The proof of the linked-diagram theorem: the factorisation theorem of Frantz and Mills in general (the sum over all relative time orderings of two pieces equals the product of their denominator products), proved by induction; the exclusion-principle-violating diagrams and why they must be summed; the proof by induction on the order, for the energy and for the wave operator; the wave-operator form and the disconnected-but-linked diagrams of Ω; the time-dependent (Gell-Mann–Low) route as a second proof; then ladders, rings and the step to coupled-cluster theory (chapter 10).
- Reading: chapter 9, section 9.15; Shavitt and Bartlett, chapters 4–5 (and 6 for the time-dependent proof). The week-45 exercises are discussed in the exercise session and answered on the week-46 slides.

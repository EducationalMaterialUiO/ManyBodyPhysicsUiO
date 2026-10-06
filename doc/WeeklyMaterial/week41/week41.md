# FYS4480/FYS9480 – Week 41, October 8–9, 2026

**Matrix product operators, and finding the ground state with DMRG**

- Thursday October 8: matrix product operators – what their bond dimension counts, where the nonzero entries sit, the operator Schmidt rank, and fermions on a chain
- Friday October 9: the DMRG – the variational principle on the MPS manifold, environments and the effective Hamiltonian, the two-site update, the sweep, and the discarded weight as an error indicator

Reading: the slides; lecture notes chapter 11, sections 11.3 and 11.5–11.7; Schollwöck, Annals of Physics 326, 96 (2011), sections 4 and 6. Code: `BookPrograms/chapterdmrg/dmrg.py`; notebook `week41.ipynb`.

## Plan for week 41

### Thursday October 8, 2 × 45 min: matrix product operators
- The MPO, and why its matrices are not d × d: a D × D matrix whose entries are d × d operators, shape (D, D, d, d) (section 11.3)
- Automata: Heisenberg D = 5, pairing D = 4 with every level coupled
- Where the nonzero entries sit – first row starts a term, last column finishes it, the bulk carries it – and what a *bulk* entry means: range-2 couplings, exponential decay, the number penalty λ(N − N₀)²
- How many channels must cross a cut: the operator Schmidt rank, and the staircase shape as a gauge
- Fermions: Jordan–Wigner strings cost no bond dimension but cost locality; the three Ẑ's of the Hubbard MPO and the ordering trap of the parity operator F̂ (section 11.7)

### Friday October 9, 2 × 45 min: the DMRG algorithm
- Variational principle on the MPS manifold; the norm matrix G, its condition number, and why canonical form turns the generalised eigenproblem into an ordinary one (section 11.5)
- Environments and the effective Hamiltonian; a cut severs three lines; why the cost is χ³
- The two-site update, and exactly how the centre moves; one sweep watched there and back; one site or two, and the subspace expansion
- The discarded weight as an error indicator; extrapolation; monotonicity is not a theorem; four convergence checks; White's blocks and superblocks (section 11.6)
- Exercises: the week-41 set (an MPS for the W state, right- and mixed-canonical forms and cheap overlaps, how many parameters an MPS really has) and answers to the week-40 set (the stability matrix from Wick's theorem; positive semidefinite matrices, Lipkin and pairing)

## Next week (week 42)

- Thursday: back to the mean field – the infinite homogeneous electron gas in Hartree-Fock: plane waves as the self-consistent orbitals, the direct term against the background, the exchange energy in closed form, the single-particle energies and their logarithmic singularity at k_F, the energy per particle in r_s.
- Friday: density functional theory, part I – the electronic Hamiltonian and reduced density matrices, the Hohenberg–Kohn theorems and what they do not give, and a first functional built from the electron gas; week 43 continues with Kohn–Sham and the local density approximation.
- Reading: chapter 6, section 6.14; chapter 8, sections 8.1–8.4; Szabo–Ostlund, chapter 3 for the plane-wave Hartree-Fock. No exercise sessions in weeks 42 and 43 (first midterm); worked answers to the week-41 exercises are in the lecture notes, chapter 11, exercises 1 and 7.

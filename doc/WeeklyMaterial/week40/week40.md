# FYS4480/FYS9480 – Week 40, October 1–2, 2026

**Matrix product states: the Schmidt decomposition at every bond**

- Thursday October 1: ten minutes of repetition – Thouless' theorem and the stability matrix; then entanglement across a cut, area laws, and the matrix product state from L−1 singular value decompositions
- Friday October 2: canonical forms, the transfer matrix and the cost of contractions; matrix product operators as finite-state automata; adding, applying and compressing; fermions on a chain

Reading: the slides; lecture notes chapter 11, sections 11.1–11.4 and 11.7 (chapter 6, sections 6.7 and 6.10 for the repetition); chapter 1, sections 1.28–1.31. Code: `BookPrograms/chapterdmrg/dmrg.py`.

## Plan for week 40

### Thursday October 1, 2 × 45 min: from entanglement to matrix product states
- Repetition (10 min): Thouless' theorem, the energy to second order, the stability matrix M and its blocks Δ, A, B; Lipkin unstable, pairing stable and useless (sections 6.7, 6.10–6.11)
- A third strategy: restrict the *entanglement*, not the basis; the methods in one table (chapter 11, introduction)
- A bipartite state is a matrix; Schmidt decomposition = SVD; isometries; determinants are entangled (section 11.1)
- Area laws: S ~ ℓ^(D−1), what is a theorem, the logarithm at criticality, which states are compressible (section 11.1)
- The MPS from L−1 SVDs; the bond index; GHZ, W and AKLT by hand; truncation and the discarded weight (section 11.2)

### Friday October 2, 2 × 45 min: MPS technology and operators
- Gauge freedom and canonical forms; why the block states are orthonormal; norms, expectation values and the transfer matrix; the cost is in the order (section 11.2)
- Matrix product operators as finite-state automata: Heisenberg D = 5, pairing D = 4, the systematic construction (section 11.3); adding, applying, compressing (section 11.4)
- Fermions on a chain: Jordan–Wigner, free in D, expensive in locality (section 11.7)
- Exercises: suggested set for week 40 (the stability matrix by Wick's theorem, positive semidefinite matrices) and answers to the week-39 set (last frames)

## Next week (week 41)

- Thursday: the variational principle on the MPS manifold; environments and the effective Hamiltonian; the two-site update, Lanczos and SVD; extrapolation in the discarded weight.
- Friday: the pairing model, where the DMRG *discovers* seniority zero; the Hubbard chain past exact diagonalisation; TEBD, where Trotter–Suzuki returns; the limits in 2D and in chemistry.
- Reading: chapter 11, sections 11.5–11.6 (the sweep, and White's original blocks and superblocks), 11.8–11.12; Schollwöck, Annals of Physics 326, 96 (2011), sections 2 and 6.

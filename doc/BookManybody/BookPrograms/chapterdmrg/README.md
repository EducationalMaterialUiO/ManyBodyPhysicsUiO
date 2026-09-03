# The density matrix renormalisation group

Companion notebook: `../../LectureNotes/dmrg.ipynb`

Figures generated from that notebook: `../../BookFigures/chapterdmrg/`

The chapter is `chapterdmrg.tex`, which `book.tex` inputs after
`chapter11.tex`; its programs live here rather than in a numbered
`chapterNN/` directory, and the `chapter*` pattern that the notebooks put on
`sys.path` matches this directory as well.

| Program | What it does | Runtime |
|---|---|---|
| `dmrg.py` | The pairing, pairing plus particle-hole and Hubbard Hamiltonians as lists of operator strings on the 2L modes of an L-site chain (one doubly-degenerate level or one lattice site per site, d = 4); exact diagonalisation in a fixed sector built from the same list; the matrix product operator assembled from the strings with the Jordan-Wigner parities and compressed to its minimal bond dimension by the SVD (pairing 4, pairing + particle-hole 14, open Hubbard 6, Hubbard ring 10, number penalty 3); matrix product states with random and product initialisation, right-canonical form, dense vector, Schmidt values across any bond, expectation values of an MPO or of a term list; a Lanczos solver with full reorthogonalisation and residual-based early exit; the two-site DMRG with left and right environments, a matrix-free effective Hamiltonian, SVD truncation with discarded weight and entanglement entropy per bond; the number penalty lambda (N - N_0)^2 that selects the particle-number sector; and the Lieb-Wu integral for the infinite half-filled Hubbard chain | about four minutes; needs `scipy.linalg`, `scipy.sparse`, `scipy.sparse.linalg`, `scipy.integrate` and `scipy.special` |

Run it as a script to print the four demonstrations of the chapter -- the
MPO bond dimensions and the MPO/exact-diagonalisation check, the pairing
model (Table 4.2 for four levels, and the twelve-level chain against the
seniority-zero diagonalisation and the Schmidt spectrum of chapter 1), the
pairing plus particle-hole model (the couplings of chapter 7, and the
eight-level convergence in chi), and the Hubbard model (the four-site ring
of Table 4.3, then open chains up to twenty sites against the Bethe ansatz):

```bash
cd BookPrograms/chapterdmrg
python3 dmrg.py
```

The main entry point for other programs is

```python
from dmrg import pairing_terms, particle_hole_terms, hubbard_terms, add_terms
from dmrg import dmrg_ground_state, exact_ground_state
terms = add_terms(pairing_terms(8, g=1.0), particle_hole_terms(8, f=0.5))
solver = dmrg_ground_state(terms, L=8, n_particles=8, chi_max=32)
solver.energy, solver.discarded[-1], solver.entropies   # energy, discarded weight, S per bond
exact_ground_state(terms, 8, 8, n_a=4)                  # the S_z = 0 sector, 4900 states
```

Conventions: level p = 1..L of the pairing model is site p-1, its two
time-reversed states are the modes a (spin +) and b (spin -) of that site;
the Hubbard site i carries spin up as mode a and spin down as mode b.  The
Hamiltonians use the couplings of chapter 4 (`pairing_terms` is
xi sum (p-1) n - g/2 sum P+ P-, so g = 1 here is delta = 1, g = 0.5 in the
convention of chapter 1).

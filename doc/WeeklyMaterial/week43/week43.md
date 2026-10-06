# FYS4480/FYS9480 – Week 43, October 22–23, 2026

**Kohn and Sham: orbitals for the kinetic energy, the electron gas for the rest**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday October 22: the Kohn–Sham construction, and the local density approximation – derived from the electron gas, not quoted
- Friday October 23: Kohn–Sham in practice – a functional built numerically, three methods on one Hamiltonian, the self-interaction error, how local is too local

Reading: the slides; lecture notes chapter 8, sections 8.5–8.8; Kohn and Sham, Phys. Rev. 140, A1133 (1965). Code: `BookPrograms/chapter08/dft.py`; notebook `week43.ipynb`.

**No exercise session this week: the time is for the first midterm.**

## Plan for week 43

### Thursday October 22, 2 × 45 min: the Kohn–Sham construction and the local density approximation (sections 8.5–8.6)
- Orbitals back in for the one term a local functional cannot handle: T_s[n], and E_xc redefined to absorb the rest
- The Kohn–Sham equations: a Hartree-Fock-like self-consistent problem with a *local* potential v_xc = δE_xc/δn; the total energy and its double counting
- Kohn–Sham against Hartree-Fock, side by side; what the Kohn–Sham orbitals and eigenvalues are – and are not
- The local density approximation: ε_x(n) = −(3/4)(3/π)^(1/3) n^(1/3) checked against −0.916/r_s; v_x = (4/3) ε_x; correlation from quantum Monte Carlo
- The exchange hole, g(0) = 1/2, and the sum rule that explains why a local functional works at all

### Friday October 23, 2 × 45 min: Kohn–Sham in practice (sections 8.7–8.8)
- Beyond LDA in one frame: gradients, meta-GGAs, hybrids
- A functional built numerically for the one-dimensional trap; Kohn–Sham, Hartree-Fock and the exact answer on one Hamiltonian
- Densities and the self-consistent loop; the self-interaction error, one number that explains a list of failures
- How local is too local: the 0.078/N law; where density functional theory sits among the methods of this course; when to use what

## Next week (week 44)

- Many-body perturbation theory (chapter 9), to be confirmed: an exact starting point, projection operators and the resolvent, Brillouin–Wigner versus Rayleigh–Schrödinger perturbation theory, the wave operator and its link to configuration interaction, the pairing model order by order, and how far a series can be trusted.
- Reading: chapter 9, sections 9.1–9.7. Exercise sessions resume after the midterm.

# FYS4480/FYS9480 – Week 42, October 15–16, 2026

**The electron gas, and the density as the variable**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday October 15: the infinite homogeneous electron gas in Hartree-Fock – the one realistic system solved in closed form
- Friday October 16: density functional theory, part I – densities, the Hohenberg–Kohn theorems, and a first functional from the gas

Reading: the slides; lecture notes chapter 6, section 6.14; chapter 8, sections 8.1–8.4; Szabo–Ostlund, chapter 3. Code: `BookPrograms/chapter06/hartreefock.py` (electron-gas part), `BookPrograms/chapter08/dft.py`; notebook `week42.ipynb`.

**No exercise sessions in weeks 42 and 43: the time is for the first midterm.**

## Plan for week 42

### Thursday October 15, 2 × 45 min: the electron gas (chapter 6, section 6.14)
- The model: electrons, a uniform background, neutrality; plane waves are the Hartree-Fock orbitals for free (f_ai = 0 by momentum conservation)
- Three divergent pieces and one cancellation; the interaction in momentum space, 4πe²/q²
- The reference energy: 3/5 ε_F per particle, direct term gone, exchange left
- The single-particle energy in closed form; F(x), the band width 1 + 0.33 r_s, the effective mass that goes to zero logarithmically, and why (unscreened 1/q², cured by RPA)
- The energy per electron 2.21/r_s² − 0.916/r_s Ry: exchange alone binds a metal at r_s = 4.83; correlation moves the minimum

### Friday October 16, 2 × 45 min: density functional theory, part I (chapter 8, sections 8.1–8.4)
- From 3N coordinates to three: the electronic Hamiltonian, and what distinguishes one material from another (only v_ext)
- One- and two-body densities; the energy needs only ρ⁽¹⁾ and ρ⁽²⁾; the factorisation of ρ⁽²⁾ for a determinant is the boundary of mean-field theory
- The Hohenberg–Kohn theorems, their four-line proof, and what they do *not* give (existence only, v-representability and the Levy–Lieb search, ground states only)
- The variational equation δE/δn = μ; Thomas–Fermi built from the electron gas, and why it is not enough (1% in the largest term, no shell structure)

## Next week (week 43)

- Thursday: the Kohn–Sham construction (section 8.5) – a fictitious non-interacting system with the same density, orbitals back in for the kinetic energy, the Kohn–Sham equations as a Hartree-Fock-like self-consistent problem with a local potential v_xc = δE_xc/δn.
- Friday: the local density approximation (section 8.6), derived rather than quoted – the exchange hole of the Fermi sea, −0.916/r_s recovered from ρ⁽¹⁾, Ceperley–Alder correlation – and Kohn–Sham in practice (section 8.7): the trap system of week 38 solved by Hartree-Fock, Kohn–Sham and exactly; self-interaction; what the Kohn–Sham eigenvalues mean.
- Reading: chapter 8, sections 8.5–8.7. Code: `dft.py`. No exercise session in week 43 either: first midterm.

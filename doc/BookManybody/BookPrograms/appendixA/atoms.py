"""
Helium and beryllium in a hydrogenic s-wave basis: from the reference
determinant to Hartree-Fock, configuration interaction, TDA and RPA.

Companion code to Project 3 of appendix A of *Quantum mechanics for
Many-particle Systems*.

The single-particle basis is the 1s, 2s and 3s hydrogenic orbitals of
nuclear charge Z, each with two spin projections: six spin-orbitals in all,
indexed p = 2 (n - 1) + s with s = 0 for spin up and s = 1 for spin down.
The one-body operator is diagonal, <p|h_0|p> = -Z^2 / (2 n_p^2), and the
Coulomb matrix elements are the closed-form radial integrals of the table
below, multiplied by the spin deltas of the direct and exchange terms.

Because the space is so small -- 2^6 = 64 Fock states in all -- everything
in the project can be done both ways: with the closed-form expressions of
the text (the reference energy, the CIS matrix, the Fock matrix, the A and
B matrices of the RPA) and, as an independent check, by brute force with the
explicit 64 x 64 matrices of the creation and annihilation operators of
Section 3.2, built by the class FockSpace of wick.py.  The full
configuration-interaction energies of chapter 5 come from the N-particle
block of that matrix and are the exact answers in this basis; the
Hartree-Fock solution reuses the SelfConsistentField class of chapter 6; the
Tamm-Dancoff and random-phase approximations follow chapter 7.

Everything is numpy only.  The chapter programs wick.py (chapter03) and
hartreefock.py (chapter06) must be on the path; the notebook shows how.

Author: Morten Hjorth-Jensen
"""

from itertools import combinations
from math import sqrt
import re

import numpy as np

from wick import FockSpace                           # chapter 3
try:                                                 # chapter 6 (needs scipy)
    from hartreefock import SelfConsistentField
except ImportError:                                  # pragma: no cover
    SelfConsistentField = None


# ---------------------------------------------------------------------------
#  The Coulomb radial integrals  <n_a n_b | 1/r_12 | n_c n_d>  for s waves
# ---------------------------------------------------------------------------
#  <ab|V|cd> = int r1^2 dr1 int r2^2 dr2 R_a(r1) R_b(r2) (1/r_>) R_c(r1) R_d(r2)
#  in atomic units, for hydrogenic orbitals of charge Z.  Every entry is a
#  rational number, possibly times a square root, times Z.  The table is the
#  one handed out with the FYS4480 midterm; radial_quadrature() below checks
#  it by direct numerical integration.
RADIAL_TABLE = """
<11|V|11> = (5*Z)/8
<11|V|12> = (4096*Sqrt[2]*Z)/64827
<11|V|13> = (1269*Sqrt[3]*Z)/50000
<11|V|21> = (4096*Sqrt[2]*Z)/64827
<11|V|22> = (16*Z)/729
<11|V|23> = (110592*Sqrt[6]*Z)/24137569
<11|V|31> = (1269*Sqrt[3]*Z)/50000
<11|V|32> = (110592*Sqrt[6]*Z)/24137569
<11|V|33> = (189*Z)/32768
<12|V|11> = (4096*Sqrt[2]*Z)/64827
<12|V|12> = (17*Z)/81
<12|V|13> = (1555918848*Sqrt[6]*Z)/75429903125
<12|V|21> = (16*Z)/729
<12|V|22> = (512*Sqrt[2]*Z)/84375
<12|V|23> = (2160*Sqrt[3]*Z)/823543
<12|V|31> = (110592*Sqrt[6]*Z)/24137569
<12|V|32> = (29943*Sqrt[3]*Z)/13176688
<12|V|33> = (1216512*Sqrt[2]*Z)/815730721
<13|V|11> = (1269*Sqrt[3]*Z)/50000
<13|V|12> = (1555918848*Sqrt[6]*Z)/75429903125
<13|V|13> = (815*Z)/8192
<13|V|21> = (110592*Sqrt[6]*Z)/24137569
<13|V|22> = (2160*Sqrt[3]*Z)/823543
<13|V|23> = (37826560*Sqrt[2]*Z)/22024729467
<13|V|31> = (189*Z)/32768
<13|V|32> = (1216512*Sqrt[2]*Z)/815730721
<13|V|33> = (617*Z)/(314928*Sqrt[3])
<21|V|11> = (4096*Sqrt[2]*Z)/64827
<21|V|12> = (16*Z)/729
<21|V|13> = (110592*Sqrt[6]*Z)/24137569
<21|V|21> = (17*Z)/81
<21|V|22> = (512*Sqrt[2]*Z)/84375
<21|V|23> = (29943*Sqrt[3]*Z)/13176688
<21|V|31> = (1555918848*Sqrt[6]*Z)/75429903125
<21|V|32> = (2160*Sqrt[3]*Z)/823543
<21|V|33> = (1216512*Sqrt[2]*Z)/815730721
<22|V|11> = (16*Z)/729
<22|V|12> = (512*Sqrt[2]*Z)/84375
<22|V|13> = (2160*Sqrt[3]*Z)/823543
<22|V|21> = (512*Sqrt[2]*Z)/84375
<22|V|22> = (77*Z)/512
<22|V|23> = (5870679552*Sqrt[6]*Z)/669871503125
<22|V|31> = (2160*Sqrt[3]*Z)/823543
<22|V|32> = (5870679552*Sqrt[6]*Z)/669871503125
<22|V|33> = (73008*Z)/9765625
<23|V|11> = (110592*Sqrt[6]*Z)/24137569
<23|V|12> = (2160*Sqrt[3]*Z)/823543
<23|V|13> = (37826560*Sqrt[2]*Z)/22024729467
<23|V|21> = (29943*Sqrt[3]*Z)/13176688
<23|V|22> = (5870679552*Sqrt[6]*Z)/669871503125
<23|V|23> = (32857*Z)/390625
<23|V|31> = (1216512*Sqrt[2]*Z)/815730721
<23|V|32> = (73008*Z)/9765625
<23|V|33> = (6890942464*Sqrt[2/3]*Z)/1210689028125
<31|V|11> = (1269*Sqrt[3]*Z)/50000
<31|V|12> = (110592*Sqrt[6]*Z)/24137569
<31|V|13> = (189*Z)/32768
<31|V|21> = (1555918848*Sqrt[6]*Z)/75429903125
<31|V|22> = (2160*Sqrt[3]*Z)/823543
<31|V|23> = (1216512*Sqrt[2]*Z)/815730721
<31|V|31> = (815*Z)/8192
<31|V|32> = (37826560*Sqrt[2]*Z)/22024729467
<31|V|33> = (617*Z)/(314928*Sqrt[3])
<32|V|11> = (110592*Sqrt[6]*Z)/24137569
<32|V|12> = (29943*Sqrt[3]*Z)/13176688
<32|V|13> = (1216512*Sqrt[2]*Z)/815730721
<32|V|21> = (2160*Sqrt[3]*Z)/823543
<32|V|22> = (5870679552*Sqrt[6]*Z)/669871503125
<32|V|23> = (73008*Z)/9765625
<32|V|31> = (37826560*Sqrt[2]*Z)/22024729467
<32|V|32> = (32857*Z)/390625
<32|V|33> = (6890942464*Sqrt[2/3]*Z)/1210689028125
<33|V|11> = (189*Z)/32768
<33|V|12> = (1216512*Sqrt[2]*Z)/815730721
<33|V|13> = (617*Z)/(314928*Sqrt[3])
<33|V|21> = (1216512*Sqrt[2]*Z)/815730721
<33|V|22> = (73008*Z)/9765625
<33|V|23> = (6890942464*Sqrt[2/3]*Z)/1210689028125
<33|V|31> = (617*Z)/(314928*Sqrt[3])
<33|V|32> = (6890942464*Sqrt[2/3]*Z)/1210689028125
<33|V|33> = (17*Z)/256
"""


def parse_radial_table(text=RADIAL_TABLE):
    """The table as {(na, nb, nc, nd): coefficient}, <ab|V|cd> = coefficient * Z."""
    table = {}
    pattern = re.compile(r"<(\d)(\d)\|V\|(\d)(\d)>\s*=\s*(.+)")
    for line in text.strip().splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        a, b, c, d = (int(match.group(k)) for k in range(1, 5))
        expression = match.group(5).replace("Sqrt[", "sqrt(").replace("]", ")")
        expression = expression.replace("Z", "1.0")
        table[(a, b, c, d)] = eval(expression, {"sqrt": sqrt, "__builtins__": {}})
    return table


RADIAL = parse_radial_table()


# ---------------------------------------------------------------------------
#  The hydrogenic s-wave radial functions and a quadrature check of the table
# ---------------------------------------------------------------------------
def radial_function(n, Z, r):
    """R_{n0}(r) for n = 1, 2, 3, normalised as int r^2 R^2 dr = 1."""
    x = Z * r
    if n == 1:
        return 2.0 * Z ** 1.5 * np.exp(-x)
    if n == 2:
        return (Z / 2.0) ** 1.5 * (2.0 - x) * np.exp(-x / 2.0)
    if n == 3:
        return 2.0 * (Z / 3.0) ** 1.5 * (1.0 - 2.0 * x / 3.0
                                        + 2.0 * x ** 2 / 27.0) * np.exp(-x / 3.0)
    raise ValueError("only n = 1, 2, 3 are tabulated")


def radial_quadrature(a, b, c, d, Z=1.0, points=3000, rmax=None):
    """<ab|V|cd> by direct integration of 1/r_> over (r_1, r_2)."""
    if rmax is None:
        rmax = 60.0 / Z
    r = np.linspace(0.0, rmax, points + 1)[1:]
    dr = r[1] - r[0]
    inner = radial_function(a, Z, r) * radial_function(c, Z, r) * r ** 2
    outer = radial_function(b, Z, r) * radial_function(d, Z, r) * r ** 2
    r_greater = np.maximum.outer(r, r)
    return dr * dr * inner @ (1.0 / r_greater) @ outer


# ---------------------------------------------------------------------------
#  The atom: spin-orbitals, matrix elements, reference determinant
# ---------------------------------------------------------------------------
class Atom:
    """N electrons of an atom with nuclear charge Z in the 1s-2s-3s basis.

    Spin-orbital p = 2 (n - 1) + s, so orbitals 0, 1 are 1s up/down, 2, 3
    are 2s up/down and 4, 5 are 3s up/down.  The reference determinant fills
    the N lowest spin-orbitals: 1s^2 for helium, 1s^2 2s^2 for beryllium.
    """

    def __init__(self, Z, N, n_max=3):
        self.Z = float(Z)
        self.N = int(N)
        self.n_max = n_max
        self.norb = 2 * n_max
        self.n_of = np.array([p // 2 + 1 for p in range(self.norb)])
        self.spin_of = np.array([p % 2 for p in range(self.norb)])
        self.h0 = np.diag([-self.Z ** 2 / (2.0 * n ** 2) for n in self.n_of])
        self.v = self._two_body()                    # <pq|v|rs>, not antisymmetrised
        self.v_as = self.v - self.v.transpose(0, 1, 3, 2)
        self.reference = (1 << self.N) - 1           # bit pattern of |c>
        self.holes = list(range(self.N))
        self.particles = list(range(self.N, self.norb))

    def _two_body(self):
        n, s = self.n_of, self.spin_of
        v = np.zeros((self.norb,) * 4)
        for p in range(self.norb):
            for q in range(self.norb):
                for r in range(self.norb):
                    for t in range(self.norb):
                        if s[p] == s[r] and s[q] == s[t]:
                            v[p, q, r, t] = self.Z * RADIAL[(n[p], n[q], n[r], n[t])]
        return v

    def label(self, p):
        return f"{self.n_of[p]}s{'+' if self.spin_of[p] == 0 else '-'}"

    # ------------------------------------------------------------------
    #  The reference energy, Eq. (2-70d) / (6-hfenergy)
    # ------------------------------------------------------------------
    def reference_energy(self):
        occ = self.holes
        one = sum(self.h0[i, i] for i in occ)
        two = 0.5 * sum(self.v_as[i, j, i, j] for i in occ for j in occ)
        return one + two

    def reference_energy_coefficients(self):
        """E[Phi_0] = alpha Z^2 + beta Z: the two coefficients."""
        occ = self.holes
        alpha = sum(-1.0 / (2.0 * self.n_of[i] ** 2) for i in occ)
        beta = 0.5 * sum(self.v_as[i, j, i, j] for i in occ for j in occ) / self.Z
        return alpha, beta

    # ------------------------------------------------------------------
    #  The Hamiltonian as a 64 x 64 matrix in Fock space (Section 3.2)
    # ------------------------------------------------------------------
    def fock_space_hamiltonian(self, h=None, v_as=None):
        """H = sum h_pq a+_p a_q + 1/4 sum <pq|v|rs>_AS a+_p a+_q a_s a_r."""
        h = self.h0 if h is None else h
        v_as = self.v_as if v_as is None else v_as
        space = FockSpace(self.norb)
        a = [space.annihilate(p) for p in range(self.norb)]
        c = [m.T for m in a]
        H = np.zeros((space.dim, space.dim))
        for p in range(self.norb):
            for q in range(self.norb):
                if h[p, q] != 0.0:
                    H += h[p, q] * (c[p] @ a[q])
        for p in range(self.norb):
            for q in range(self.norb):
                cc = c[p] @ c[q]
                for r in range(self.norb):
                    for t in range(self.norb):
                        if v_as[p, q, r, t] != 0.0:
                            H += 0.25 * v_as[p, q, r, t] * (cc @ a[t] @ a[r])
        return H

    def sector(self, n_particles, ms2=0):
        """Bit patterns with n_particles electrons and 2 M_S = ms2, sorted."""
        states = []
        for occ in combinations(range(self.norb), n_particles):
            bits = sum(1 << p for p in occ)
            ups = sum(1 for p in occ if self.spin_of[p] == 0)
            if 2 * ups - n_particles == ms2:
                states.append(bits)
        return sorted(states)

    def determinant_label(self, bits):
        return "|" + " ".join(self.label(p) for p in range(self.norb)
                              if (bits >> p) & 1) + ">"

    # ------------------------------------------------------------------
    #  Full configuration interaction (chapter 5): the exact answer here
    # ------------------------------------------------------------------
    def fci(self, H=None, ms2=0):
        H = self.fock_space_hamiltonian() if H is None else H
        states = self.sector(self.N, ms2)
        block = H[np.ix_(states, states)]
        energies, vectors = np.linalg.eigh(block)
        return energies, vectors, states

    # ------------------------------------------------------------------
    #  Configuration interaction with singles, in the given basis
    # ------------------------------------------------------------------
    def singles(self, ms2=0):
        """The reference and its spin-conserving 1p-1h excitations (M_S = 0)."""
        states = [self.reference]
        pairs = []
        for i in self.holes:
            for m in self.particles:
                if self.spin_of[i] == self.spin_of[m]:
                    states.append((self.reference ^ (1 << i)) | (1 << m))
                    pairs.append((m, i))
        return states, pairs

    def cis(self, H=None):
        """The Hamiltonian in the space of |c> and a+_m a_i |c>, with the signs
        of the operator strings (not of the bit patterns) kept."""
        H = self.fock_space_hamiltonian() if H is None else H
        states, pairs = self.singles()
        space = FockSpace(self.norb)
        a = [space.annihilate(p) for p in range(self.norb)]
        ref = np.zeros(space.dim)
        ref[self.reference] = 1.0
        vectors = [ref] + [a[m].T @ a[i] @ ref for m, i in pairs]
        V = np.array(vectors).T
        block = V.T @ H @ V
        energies, coefficients = np.linalg.eigh(block)
        return energies, coefficients, block, states, pairs

    def cis_closed_form(self):
        """The CIS matrix from the Condon-Slater rules, Eqs. (3-onebodyresult)
        and (3-twobodyresult), without any matrix in Fock space."""
        h, v, occ = self.h0, self.v_as, self.holes
        states, pairs = self.singles()
        dim = len(states)
        M = np.zeros((dim, dim))
        E0 = self.reference_energy()
        M[0, 0] = E0
        for k, (m, i) in enumerate(pairs, start=1):
            # <c|H|Phi_i^m> = h_im + sum_j <ij|v|mj>_AS = f_im
            M[0, k] = M[k, 0] = h[i, m] + sum(v[i, j, m, j] for j in occ)
            for l, (n, j) in enumerate(pairs, start=1):
                value = (E0 * (m == n) * (i == j)
                         + (i == j) * (h[m, n] + sum(v[m, k2, n, k2] for k2 in occ))
                         - (m == n) * (h[j, i] + sum(v[j, k2, i, k2] for k2 in occ))
                         + v[m, j, i, n])
                M[k, l] = value
        return M, states, pairs

    # ------------------------------------------------------------------
    #  Hartree-Fock (chapter 6)
    # ------------------------------------------------------------------
    def hartree_fock(self, tol=1e-12, max_iter=200):
        """Restricted HF on the spatial 1s-2s-3s problem, expanded to spin-orbitals.

        For a closed shell the Fock matrix is block diagonal in spin, so we
        solve the 3 x 3 spatial problem with the closed-shell Fock matrix
        F_ab = h_ab + sum_gd rho_gd [2 (ag|bd) - (ag|db)] and expand the
        spatial orbitals to spin-orbitals afterwards; SelfConsistentField of
        chapter 6 solves the same problem on the 6 x 6 spin-orbital matrices
        and is used as a check.
        """
        n_spatial = self.n_max
        n_occ = self.N // 2
        h = np.diag([-self.Z ** 2 / (2.0 * n ** 2) for n in range(1, n_spatial + 1)])
        R = np.zeros((n_spatial,) * 4)
        for a in range(n_spatial):
            for b in range(n_spatial):
                for c in range(n_spatial):
                    for d in range(n_spatial):
                        R[a, b, c, d] = self.Z * RADIAL[(a + 1, b + 1, c + 1, d + 1)]
        C = np.eye(n_spatial)
        eps_old = np.zeros(n_spatial)
        history = []
        for iteration in range(1, max_iter + 1):
            occ = C[:, :n_occ]
            rho = occ @ occ.T
            F = h + np.einsum("gd,agbd->ab", rho, 2.0 * R - R.transpose(0, 1, 3, 2))
            eps, C = np.linalg.eigh(F)
            occ = C[:, :n_occ]
            rho = occ @ occ.T
            # E = 2 tr(rho h) + sum rho_ag rho_bd [2 <ab|v|gd> - <ab|v|dg>]
            energy = (2.0 * np.einsum("ab,ab->", rho, h)
                      + np.einsum("ag,bd,abgd->", rho, rho,
                                  2.0 * R - R.transpose(0, 1, 3, 2)))
            history.append((iteration, energy, eps.copy()))
            if np.abs(eps - eps_old).max() < tol:
                break
            eps_old = eps
        # expand to spin-orbitals: p = 2 (n - 1) + s
        C_spin = np.kron(C, np.eye(2))
        eps_spin = np.repeat(eps, 2)
        return dict(energy=energy, eps=eps_spin, C=C_spin, C_spatial=C,
                    eps_spatial=eps, iterations=iteration, history=history)

    def hartree_fock_chapter6(self, tol=1e-12):
        """The same problem solved with SelfConsistentField of chapter 6."""
        if SelfConsistentField is None:
            return None
        scf = SelfConsistentField(self.h0, self.v_as, self.N)
        E, eps, C = scf.run(tol=tol)
        return dict(energy=E, eps=eps, C=C, brillouin=scf.brillouin(),
                    iterations=scf.iterations)

    def transform(self, C):
        """h and <pq|v|rs>_AS in the orbital basis given by the columns of C."""
        h = C.T @ self.h0 @ C
        v = np.einsum("ap,bq,cr,ds,abcd->pqrs", C, C, C, C, self.v_as)
        return h, v

    # ------------------------------------------------------------------
    #  TDA and RPA on the Hartree-Fock determinant (chapter 7)
    # ------------------------------------------------------------------
    def ph_pairs(self, spin_conserving=True):
        pairs = []
        for i in self.holes:
            for m in self.particles:
                if (not spin_conserving) or self.spin_of[i] == self.spin_of[m]:
                    pairs.append((m, i))
        return pairs

    def tda_rpa(self, hf=None, spin_conserving=True):
        """A and B from Eqs. (7-Amatrix) and (7-Bmatrix) in the HF basis."""
        hf = self.hartree_fock() if hf is None else hf
        h, v = self.transform(hf["C"])
        eps = hf["eps"]
        pairs = self.ph_pairs(spin_conserving)
        n = len(pairs)
        A = np.zeros((n, n))
        B = np.zeros((n, n))
        for k, (m, i) in enumerate(pairs):
            for l, (nn, j) in enumerate(pairs):
                A[k, l] = (eps[m] - eps[i]) * (m == nn) * (i == j) + v[m, j, i, nn]
                B[k, l] = v[m, nn, i, j]
        tda = np.linalg.eigvalsh(A)
        M = np.block([[A, B], [-B, -A]])
        w = np.linalg.eigvals(M)
        n_imag = int(np.sum(np.abs(w.imag) > 1e-8))
        real = np.sort(w[np.abs(w.imag) <= 1e-8].real)
        rpa = real[real > 1e-8]
        stability = np.linalg.eigvalsh(A - B).min(), np.linalg.eigvalsh(A + B).min()
        ecorr = 0.5 * (rpa.sum() - np.trace(A)) if n_imag == 0 else np.nan
        return dict(A=A, B=B, tda=tda, rpa=rpa, n_imag=n_imag, pairs=pairs,
                    ecorr=ecorr, stability=stability, h=h, v=v, eps=eps)

    def double_commutators(self, hf, pairs):
        """A and B from the double commutators of Eqs. (7-Amatrix)/(7-Bmatrix),
        with everything as explicit matrices in the 64-dimensional Fock space."""
        h, v = self.transform(hf["C"])
        H = self.fock_space_hamiltonian(h, v)          # in the HF basis
        space = FockSpace(self.norb)
        a = [space.annihilate(p) for p in range(self.norb)]
        c = [m.T for m in a]
        ref = np.zeros(space.dim)
        ref[self.reference] = 1.0
        ops = [c[m] @ a[i] for m, i in pairs]
        n = len(pairs)
        A = np.zeros((n, n))
        B = np.zeros((n, n))
        comm = lambda X, Y: X @ Y - Y @ X
        for k in range(n):
            for l in range(n):
                A[k, l] = ref @ comm(ops[k].T, comm(H, ops[l])) @ ref
                B[k, l] = -ref @ comm(ops[k].T, comm(H, ops[l].T)) @ ref
        return A, B, H


# ---------------------------------------------------------------------------
#  Demo
# ---------------------------------------------------------------------------
def _line(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def _demo():
    _line("1. The Coulomb table against direct quadrature (Z = 2)")
    for key in ((1, 1, 1, 1), (1, 2, 1, 2), (1, 1, 2, 2), (2, 3, 3, 2), (3, 3, 3, 3)):
        exact = 2.0 * RADIAL[key]
        numeric = radial_quadrature(*key, Z=2.0)
        print(f"  <{key[0]}{key[1]}|V|{key[2]}{key[3]}> = {exact:.8f}  "
              f"quadrature {numeric:.8f}  difference {abs(exact - numeric):.1e}")

    for name, Z, N, exact_full in (("helium", 2, 2, -2.9037), ("beryllium", 4, 4, -14.6674)):
        atom = Atom(Z, N)
        _line(f"2. {name}: Z = {Z}, N = {N}, {atom.norb} spin-orbitals")
        alpha, beta = atom.reference_energy_coefficients()
        print(f"  reference determinant {atom.determinant_label(atom.reference)}")
        print(f"  E[Phi_0] = {alpha:+.6f} Z^2 {beta:+.6f} Z = {atom.reference_energy():.6f}")
        H = atom.fock_space_hamiltonian()
        print(f"  Fock-space Hamiltonian: {H.shape[0]} x {H.shape[1]}, "
              f"symmetric: {np.allclose(H, H.T)}, "
              f"<c|H|c> = {H[atom.reference, atom.reference]:.6f}")

        fci_E, _, states = atom.fci(H)
        print(f"  FCI, M_S = 0 sector of dimension {len(states)}: "
              f"E_0 = {fci_E[0]:.6f}, first excitations "
              + ", ".join(f"{e - fci_E[0]:.4f}" for e in fci_E[1:4]))
        print(f"  (the exact energy with the full Hamiltonian is {exact_full})")

        cis_E, cis_V, M, cstates, pairs = atom.cis(H)
        M2, _, _ = atom.cis_closed_form()
        print(f"  CIS in the hydrogenic basis, {len(cstates)} x {len(cstates)}: "
              f"E_0 = {cis_E[0]:.6f}; closed form vs Fock-space block: "
              f"{np.abs(M - M2).max():.1e}")
        print("  <c|H|Phi_i^m> = " + ", ".join(f"{M[0, k]:+.5f}" for k in range(1, len(cstates))))

        hf = atom.hartree_fock()
        hf6 = atom.hartree_fock_chapter6()
        print(f"  HF after 1 iteration: E = {hf['history'][0][1]:.6f}, "
              f"eps = {np.round(hf['history'][0][2], 6)}")
        print(f"  HF converged in {hf['iterations']} iterations: E = {hf['energy']:.6f}, "
              f"eps = {np.round(hf['eps_spatial'], 6)}")
        if hf6 is not None:
            print(f"  chapter-6 SelfConsistentField: E = {hf6['energy']:.6f}, "
                  f"max |f_ai| = {hf6['brillouin']:.1e}")

        # CIS in the HF basis: Brillouin's theorem empties the first row
        h, v = atom.transform(hf["C"])
        H_hf = atom.fock_space_hamiltonian(h, v)
        cisHF_E, _, M_hf, _, _ = atom.cis(H_hf)
        print(f"  CIS in the HF basis: max |<c|H|Phi_i^m>| = "
              f"{np.abs(M_hf[0, 1:]).max():.1e}, E_0 = {cisHF_E[0]:.6f} (= E_HF)")
        fciHF_E, _, _ = atom.fci(H_hf)
        print(f"  FCI in the HF basis: E_0 = {fciHF_E[0]:.6f} (basis independent)")

        res = atom.tda_rpa(hf)
        A_dc, B_dc, _ = atom.double_commutators(hf, res["pairs"])
        print(f"  TDA/RPA on {len(res['pairs'])} spin-conserving ph pairs: "
              f"closed-form A, B vs double commutators: "
              f"{np.abs(res['A'] - A_dc).max():.1e}, {np.abs(res['B'] - B_dc).max():.1e}")
        print(f"  stability: min eig(A - B) = {res['stability'][0]:.4f}, "
              f"min eig(A + B) = {res['stability'][1]:.4f}, imaginary RPA roots: {res['n_imag']}")
        print("  excitation energies (hartree):")
        print(f"    {'TDA':>10s} {'RPA':>10s} {'FCI':>10s} {'CIS(hydrogenic)':>16s}")
        n_show = min(len(res["tda"]), len(fci_E) - 1)
        for k in range(n_show):
            cis_exc = cis_E[k + 1] - cis_E[0] if k + 1 < len(cis_E) else float("nan")
            print(f"    {res['tda'][k]:10.5f} {res['rpa'][k]:10.5f} "
                  f"{fci_E[k + 1] - fci_E[0]:10.5f} {cis_exc:16.5f}")
        print(f"  RPA correlation energy: {res['ecorr']:.6f}; "
              f"E_HF + E_corr = {hf['energy'] + res['ecorr']:.6f}, FCI: {fci_E[0]:.6f}")
        full = atom.tda_rpa(hf, spin_conserving=False)
        print(f"  with all {len(full['pairs'])} pairs (spin flips included), TDA roots: "
              + ", ".join(f"{w:.5f}" for w in full["tda"]))

    _line("3. The isoelectronic sequences")
    print(f"  {'Z':>3s} {'E_ref':>11s} {'E_HF':>11s} {'E_CIS':>11s} {'E_FCI':>11s} "
          f"{'E_HF+RPA':>11s}   (helium-like)")
    for Z in range(2, 11):
        atom = Atom(Z, 2)
        H = atom.fock_space_hamiltonian()
        hf = atom.hartree_fock()
        res = atom.tda_rpa(hf)
        print(f"  {Z:3d} {atom.reference_energy():11.5f} {hf['energy']:11.5f} "
              f"{atom.cis(H)[0][0]:11.5f} {atom.fci(H)[0][0]:11.5f} "
              f"{hf['energy'] + res['ecorr']:11.5f}")
    print(f"  {'Z':>3s} {'E_ref':>11s} {'E_HF':>11s} {'E_CIS':>11s} {'E_FCI':>11s} "
          f"{'E_HF+RPA':>11s}   (beryllium-like)")
    for Z in range(3, 13):
        atom = Atom(Z, 4)
        H = atom.fock_space_hamiltonian()
        hf = atom.hartree_fock()
        res = atom.tda_rpa(hf)
        print(f"  {Z:3d} {atom.reference_energy():11.5f} {hf['energy']:11.5f} "
              f"{atom.cis(H)[0][0]:11.5f} {atom.fci(H)[0][0]:11.5f} "
              f"{hf['energy'] + res['ecorr']:11.5f}")


if __name__ == "__main__":
    _demo()

"""
One- and two-body reduced density matrices in coordinate space.

Companion code to chapter 8 of *Quantum mechanics for Many-particle Systems*,
section "Reduced density matrices".

The chapter derives the one-body density matrix gamma(x; x') and the
ordered-pair density P(x, y) of an N-fermion state, shows that the energy is
an exact functional of the two, and that for a single Slater determinant the
pair density factorises into a direct and an exchange term,

    P_Phi(x, y) = n(x) n(y) - |gamma_Phi(x; y)|^2 .

Everything in that section is checked here on the one-dimensional trap of
chapters 2 and 6 -- N spinless fermions in a harmonic well with the softened
Coulomb repulsion, eight oscillator orbitals, a grid of 401 points:

  * `DeterminantDensities` builds gamma, n and P of a determinant on the
    grid and verifies the traces, the contraction int P dy = (N - 1) n, the
    vanishing of P on the diagonal, the idempotency gamma^2 = gamma, the
    exchange-hole sum rule int n_x(x, y) dy = -1 at every x, and the three
    ways of writing the Hartree and exchange energies (from n and gamma,
    from the Coulomb and exchange integrals J_ij and K_ij, and from the
    antisymmetrised matrix elements of chapter 6);
  * `kinetic_from_gamma` shows that the kinetic energy needs the
    off-diagonal density matrix: a phase theta(x) multiplying an orbital
    leaves n unchanged and changes T by int n (theta')^2 / 2;
  * `pair_transfer_example` is the two-fermion, four-orbital state
    (|12> +- |34>)/sqrt 2 whose one-body density matrix is the same for
    both signs while the pair-transfer amplitude <a3+ a4+ a2 a1> is not: a
    one-body density matrix does not fix the two-body one;
  * `CorrelatedTwoParticle` takes the exact two-particle state of the trap
    (full diagonalisation in the eight-orbital basis) and evaluates the
    cumulant lambda = P - P_Phi[gamma], its contraction (gamma^2 - gamma),
    and the exact energy decomposition E = T + E_H + E_x + E_lambda, which
    must reproduce the full-CI energy to machine precision.

    DeterminantDensities, CorrelatedTwoParticle, kinetic_from_gamma,
    pair_transfer_example, demo_*

Author: Morten Hjorth-Jensen
"""

from itertools import combinations
from math import pi

import numpy as np

import dft


# ---------------------------------------------------------------------------
#  A single determinant on the grid
# ---------------------------------------------------------------------------
class DeterminantDensities:
    """gamma, n and P of a Slater determinant of orbitals on a grid.

    `orbitals` has one occupied orbital per row, sampled on `grid`; the
    quadrature is the trapezoidal rule with spacing dx, which is what the
    rest of the chapter uses for this grid.
    """

    def __init__(self, orbitals, grid, strength=1.0, softening=0.5):
        self.phi = np.asarray(orbitals, dtype=float)
        self.N = self.phi.shape[0]
        self.x = np.asarray(grid, dtype=float)
        self.dx = self.x[1] - self.x[0]
        self.v = strength / np.sqrt(
            (self.x[:, None] - self.x[None, :]) ** 2 + softening ** 2)
        # gamma(x; x') = sum_i phi_i(x) phi_i(x')
        self.gamma = self.phi.T @ self.phi
        self.n = np.diag(self.gamma).copy()
        # ordered-pair density, P(x, y) = n(x) n(y) - |gamma(x; y)|^2
        self.P = np.outer(self.n, self.n) - self.gamma ** 2

    # ---- identities ---------------------------------------------------
    def checks(self):
        dx = self.dx
        gamma2 = self.gamma @ self.gamma * dx
        hole = -self.gamma ** 2 / np.where(self.n > 1e-12, self.n, np.inf)[:, None]
        hole_sum = hole.sum(axis=1) * dx
        inside = self.n > 1e-6
        return dict(
            trace_gamma=float(self.n.sum() * dx),
            trace_P=float(self.P.sum() * dx * dx),
            contraction=float(np.abs(self.P.sum(axis=1) * dx
                                     - (self.N - 1) * self.n).max()),
            diagonal=float(np.abs(np.diag(self.P)).max()),
            idempotency=float(np.abs(gamma2 - self.gamma).max()),
            hole_sum_rule=float(np.abs(hole_sum[inside] + 1.0).max()),
        )

    def exchange_hole(self, x0):
        """n_x(x0, y) = -|gamma(x0; y)|^2 / n(x0) on the grid."""
        j = int(np.argmin(np.abs(self.x - x0)))
        return self.x[j], -self.gamma[j] ** 2 / self.n[j]

    def conditional_density(self, x0):
        """n_cond(y | x0) = P(x0, y) / n(x0)."""
        j = int(np.argmin(np.abs(self.x - x0)))
        return self.x[j], self.P[j] / self.n[j]

    # ---- energies -----------------------------------------------------
    def hartree_energy(self):
        """E_H = (1/2) int n v n."""
        return 0.5 * float(self.n @ self.v @ self.n) * self.dx ** 2

    def exchange_energy(self):
        """E_x = -(1/2) int v |gamma|^2."""
        return -0.5 * float(np.sum(self.v * self.gamma ** 2)) * self.dx ** 2

    def coulomb_and_exchange_integrals(self):
        """J_ij = <ij|v|ij>, K_ij = <ij|v|ji> for the occupied orbitals."""
        dens = self.phi ** 2                                    # (N, grid)
        J = dens @ self.v @ dens.T * self.dx ** 2
        K = np.einsum("ix,jx,xy,jy,iy->ij", self.phi, self.phi, self.v,
                      self.phi, self.phi, optimize=True) * self.dx ** 2
        return J, K

    def interaction_energy_from_vbar(self, C_occ, vbar):
        """(1/2) sum_ij <ij|v|ij>_AS in the oscillator basis of chapter 6."""
        return 0.5 * float(np.einsum("ai,bj,abcd,ci,dj->", C_occ, C_occ,
                                     vbar, C_occ, C_occ, optimize=True))


# ---------------------------------------------------------------------------
#  The kinetic energy needs the off-diagonal density matrix
# ---------------------------------------------------------------------------
def kinetic_from_gamma(gamma, dx):
    """T = (1/2) int [d/dx d/dx' gamma(x; x')]_{x'=x} dx by fourth-order
    central differences on the grid (gamma may be complex; the density
    matrix vanishes at the edges of the box, so zero padding is exact
    there)."""
    def derivative(f, axis):
        g = np.moveaxis(f, axis, 0)
        p = np.concatenate([np.zeros((2,) + g.shape[1:]), g,
                            np.zeros((2,) + g.shape[1:])])
        d = (-p[4:] + 8 * p[3:-1] - 8 * p[1:-3] + p[:-4]) / (12.0 * dx)
        return np.moveaxis(d, 0, axis)
    dd = derivative(derivative(gamma, 0), 1)
    return 0.5 * float(np.real(np.trace(dd)) * dx)


def phase_example(grid, orbital, slope=1.0):
    """Multiply a real orbital by exp(i theta(x)) with theta = slope x:
    the density is unchanged, the kinetic energy rises by slope^2/2."""
    dx = grid[1] - grid[0]
    psi = orbital * np.exp(1j * slope * grid)
    gamma0 = np.outer(orbital, orbital)
    gamma1 = np.outer(psi, np.conj(psi))
    return dict(density_change=float(np.abs(np.abs(psi) ** 2
                                            - orbital ** 2).max()),
                T_real=kinetic_from_gamma(gamma0, dx),
                T_phase=kinetic_from_gamma(gamma1, dx),
                predicted_shift=0.5 * slope ** 2 * float(orbital @ orbital) * dx)


# ---------------------------------------------------------------------------
#  Fock space of four orbitals: the pair-transfer example
# ---------------------------------------------------------------------------
def _ladder_operators(n_orbitals):
    """Annihilation operators as 2^M x 2^M matrices (Jordan-Wigner)."""
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    z = np.diag([1.0, -1.0])
    eye = np.eye(2)
    ops = []
    for p in range(n_orbitals):
        factors = [z] * p + [a] + [eye] * (n_orbitals - p - 1)
        m = factors[0]
        for f in factors[1:]:
            m = np.kron(m, f)
        ops.append(m)
    return ops


def pair_transfer_example():
    """|Psi_pm> = (|12> pm |34>)/sqrt 2 for two fermions in four orbitals:
    identical one-body density matrices, opposite pair-transfer
    amplitudes <a3+ a4+ a2 a1>."""
    a = _ladder_operators(4)
    vac = np.zeros(16)
    vac[0] = 1.0
    ket12 = a[0].T @ a[1].T @ vac
    ket34 = a[2].T @ a[3].T @ vac
    out = {}
    for sign, label in ((+1, "+"), (-1, "-")):
        psi = (ket12 + sign * ket34) / np.sqrt(2.0)
        D = np.array([[psi @ a[q].T @ a[p] @ psi for q in range(4)]
                      for p in range(4)])
        transfer = psi @ a[2].T @ a[3].T @ a[1] @ a[0] @ psi
        out[label] = dict(D=D, transfer=float(transfer))
    out["D_difference"] = float(np.abs(out["+"]["D"] - out["-"]["D"]).max())
    return out


# ---------------------------------------------------------------------------
#  The exact two-particle state and its cumulant
# ---------------------------------------------------------------------------
class CorrelatedTwoParticle:
    """Full diagonalisation for two spinless fermions in the eight-orbital
    basis of the trap, and the reduced density matrices of the result."""

    def __init__(self, n_orbitals=8, strength=1.0, softening=0.5):
        self.grid, self.phi = dft.oscillator_basis(n_orbitals)
        self.dx = self.grid[1] - self.grid[0]
        self.vbar = dft.trap_two_body(n_orbitals, strength, softening)
        self.h0 = np.diag(np.arange(n_orbitals) + 0.5)
        self.v = strength / np.sqrt(
            (self.grid[:, None] - self.grid[None, :]) ** 2 + softening ** 2)
        pairs = list(combinations(range(n_orbitals), 2))
        dim = len(pairs)
        H = np.zeros((dim, dim))
        for a, (i, j) in enumerate(pairs):
            H[a, a] += self.h0[i, i] + self.h0[j, j]
            for b, (k, l) in enumerate(pairs):
                H[a, b] += self.vbar[i, j, k, l]
        energies, vectors = np.linalg.eigh(H)
        self.energy = float(energies[0])
        self.pairs = pairs
        self.c = vectors[:, 0]
        # one-body density matrix D_pq = <a_q^+ a_p> from the pair amplitudes
        M = n_orbitals
        D = np.zeros((M, M))
        for a, (i, j) in enumerate(pairs):
            for b, (k, l) in enumerate(pairs):
                # <a_q^+ a_p> connects |kl> to |ij> if they share one orbital
                if j == l:
                    D[i, k] += self.c[a] * self.c[b]
                if i == k:
                    D[j, l] += self.c[a] * self.c[b]
                if i == l:
                    D[j, k] -= self.c[a] * self.c[b]
                if j == k:
                    D[i, l] -= self.c[a] * self.c[b]
        self.D = D
        self.occupations = np.sort(np.linalg.eigvalsh(D))[::-1]
        # coordinate-space objects
        self.gamma = self.phi.T @ D @ self.phi
        self.n = np.diag(self.gamma).copy()
        # Psi(x, y) = sum_{i<j} c_ij [phi_i(x) phi_j(y) - phi_j(x) phi_i(y)] / sqrt 2
        psi = np.zeros((self.grid.size, self.grid.size))
        for a, (i, j) in enumerate(pairs):
            psi += self.c[a] * (np.outer(self.phi[i], self.phi[j])
                                - np.outer(self.phi[j], self.phi[i]))
        psi /= np.sqrt(2.0)
        self.P = 2.0 * psi ** 2                     # N (N - 1) |Psi|^2
        self.P_factorised = np.outer(self.n, self.n) - self.gamma ** 2
        self.cumulant = self.P - self.P_factorised

    def decomposition(self):
        dx = self.dx
        one_body = float(np.sum(self.D * self.h0))
        hartree = 0.5 * float(self.n @ self.v @ self.n) * dx ** 2
        exchange = -0.5 * float(np.sum(self.v * self.gamma ** 2)) * dx ** 2
        correlation = 0.5 * float(np.sum(self.v * self.cumulant)) * dx ** 2
        gamma2 = self.gamma @ self.gamma * dx
        return dict(one_body=one_body, hartree=hartree, exchange=exchange,
                    cumulant=correlation,
                    total=one_body + hartree + exchange + correlation,
                    fci=self.energy,
                    trace_P=float(self.P.sum() * dx * dx),
                    cumulant_contraction=float(np.abs(
                        self.cumulant.sum(axis=1) * dx
                        - (np.diag(gamma2) - self.n)).max()),
                    cumulant_trace=float(self.cumulant.sum() * dx * dx),
                    gamma2_minus_gamma_trace=float((np.trace(gamma2)
                                                    - self.n.sum()) * dx))


# ---------------------------------------------------------------------------
#  Demonstrations
# ---------------------------------------------------------------------------
def demo_determinant():
    print("=" * 74)
    print("1. The densities of a determinant, and their identities")
    print("=" * 74)
    print("The Hartree-Fock determinant of chapter 6 for N = 4 fermions in the")
    print("trap, on the 401-point grid.  Every identity of the chapter is a")
    print("number that should vanish, or a trace that should be N or N(N-1):")
    print()
    hf = dft.hartree_fock(4)
    grid, phi = dft.oscillator_basis(8)
    occ = (phi.T @ hf["C"][:, :4]).T
    dd = DeterminantDensities(occ, grid)
    c = dd.checks()
    print(f"   int n dx                      = {c['trace_gamma']:.12f}   (N = 4)")
    print(f"   int int P dx dy               = {c['trace_P']:.12f}   (N(N-1) = 12)")
    print(f"   max |int P(x,y) dy - (N-1) n(x)| = {c['contraction']:.1e}")
    print(f"   max |P(x,x)|                  = {c['diagonal']:.1e}")
    print(f"   max |gamma^2 - gamma|         = {c['idempotency']:.1e}")
    print(f"   max |int n_x(x,y) dy + 1|     = {c['hole_sum_rule']:.1e}"
          "   (exchange-hole sum rule, every x with n > 1e-6)")
    print()
    print("Three ways of writing the same interaction energy:")
    E_H, E_x = dd.hartree_energy(), dd.exchange_energy()
    J, K = dd.coulomb_and_exchange_integrals()
    from_vbar = dd.interaction_energy_from_vbar(hf["C"][:, :4], hf["vbar"])
    print(f"   from n and gamma:   E_H = {E_H:.10f}   E_x = {E_x:.10f}"
          f"   sum = {E_H + E_x:.10f}")
    print(f"   from J and K:       E_H = {0.5 * J.sum():.10f}   "
          f"E_x = {-0.5 * K.sum():.10f}   sum = {0.5 * (J - K).sum():.10f}")
    print(f"   from <ij|v|ij>_AS of chapter 6:                       "
          f"sum = {from_vbar:.10f}")
    print(f"   max |J_ii - K_ii| = {np.abs(np.diag(J) - np.diag(K)).max():.1e}"
          "   (every self-pair cancels)")
    print()
    print("The exchange hole around a particle at the centre and at x = 1.5:")
    for x0 in (0.0, 1.5):
        xj, hole = dd.exchange_hole(x0)
        print(f"   x = {xj:4.1f}: n(x) = {dd.n[np.argmin(np.abs(grid - xj))]:.4f},"
              f"  depth n_x(x,x) = {hole[np.argmin(np.abs(grid - xj))]:.4f},"
              f"  int n_x dy = {hole.sum() * dd.dx:.6f}")
    print("At coincidence the hole depth is exactly -n(x): the particle")
    print("excludes itself, and the sum rule says it excludes exactly one.")


def demo_kinetic():
    print("=" * 74)
    print("2. The kinetic energy needs the off-diagonal density matrix")
    print("=" * 74)
    grid, phi = dft.oscillator_basis(8)
    dx = grid[1] - grid[0]
    N = 4
    dd = DeterminantDensities(phi[:N], grid)
    T = kinetic_from_gamma(dd.gamma, dx)
    exact = sum(k + 0.5 for k in range(N)) - 0.5 * float(dd.n @ grid ** 2) * dx
    print(f"Non-interacting N = 4: T from [d_x d_x' gamma]_(x'=x) = {T:.6f},")
    print(f"                       T from the oscillator spectrum  = {exact:.6f}")
    print()
    ex = phase_example(grid, phi[0], slope=1.0)
    print("Multiply the ground orbital by exp(i x):")
    print(f"   change in the density        {ex['density_change']:.1e}")
    print(f"   T before                     {ex['T_real']:.6f}")
    print(f"   T after                      {ex['T_phase']:.6f}")
    print(f"   predicted shift (1/2) int n  {ex['predicted_shift']:.6f}")
    print("The density cannot tell the two states apart; the density matrix")
    print("can, and so can the energy.  A functional of n alone that")
    print("returned T for every state is therefore impossible -- which is")
    print("why the Hohenberg-Kohn theorem is a statement about ground states.")


def demo_pair_transfer():
    print("=" * 74)
    print("3. A one-body density matrix does not fix the two-body one")
    print("=" * 74)
    out = pair_transfer_example()
    print("Two fermions in four orbitals, |Psi_pm> = (|12> pm |34>)/sqrt 2:")
    print(f"   D_pq for Psi_+ : diag {np.diag(out['+']['D'])}")
    print(f"   D_pq for Psi_- : diag {np.diag(out['-']['D'])}")
    print(f"   max |D_+ - D_-|                 = {out['D_difference']:.1e}")
    print(f"   <a3+ a4+ a2 a1> for Psi_+       = {out['+']['transfer']:+.4f}")
    print(f"   <a3+ a4+ a2 a1> for Psi_-       = {out['-']['transfer']:+.4f}")
    print("A one-body operator cannot connect determinants that differ in")
    print("two occupations, so the two states have the same gamma and the")
    print("same density; an interaction that transfers the pair gives them")
    print("different energies.  The missing information is the cumulant.")


def demo_cumulant():
    print("=" * 74)
    print("4. The exact two-particle state: the cumulant and the energy")
    print("=" * 74)
    ct = CorrelatedTwoParticle()
    d = ct.decomposition()
    print("Full diagonalisation for N = 2 in the eight-orbital basis, then")
    print("gamma, n, P and lambda = P - [n n - |gamma|^2] on the grid:")
    print()
    print(f"   natural occupations           {np.array2string(ct.occupations[:4], precision=6)} ...")
    print(f"   int int P                     = {d['trace_P']:.10f}   (N(N-1) = 2)")
    print(f"   max |int lambda dy - (gamma^2 - gamma)(x;x)| = "
          f"{d['cumulant_contraction']:.1e}")
    print(f"   int int lambda                = {d['cumulant_trace']:.10f}"
          f"   = tr(gamma^2 - gamma) = {d['gamma2_minus_gamma_trace']:.10f}")
    print()
    print("The exact energy decomposed as T + E_H + E_x + E_lambda:")
    print(f"   <h_0>  (from D)               = {d['one_body']:.10f}")
    print(f"   E_H[n]                        = {d['hartree']:.10f}")
    print(f"   E_x[gamma]                    = {d['exchange']:.10f}")
    print(f"   E_lambda                      = {d['cumulant']:.10f}")
    print(f"   sum                           = {d['total']:.10f}")
    print(f"   full CI                       = {d['fci']:.10f}")
    print()
    hf = dft.hartree_fock(2)
    print(f"   Hartree-Fock                  = {hf['energy']:.10f}")
    print(f"   E_exact - E_HF                = {d['fci'] - hf['energy']:.10f}")
    print("E_lambda is not the correlation energy: the exact gamma differs")
    print("from the Hartree-Fock one, so <h_0>, E_H and E_x all move too,")
    print("and only their sum with E_lambda is the exact answer.")


def _demo():
    for f in (demo_determinant, demo_kinetic, demo_pair_transfer,
              demo_cumulant):
        f()
        print()


if __name__ == "__main__":
    _demo()

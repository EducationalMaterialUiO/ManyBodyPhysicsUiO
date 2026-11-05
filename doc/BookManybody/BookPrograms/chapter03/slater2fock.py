"""
From first to second quantization: the correspondence, checked numerically.

Companion code to chapter 3 of *Quantum mechanics for Many-particle Systems*.

There is a unitary isomorphism between the antisymmetric N-particle Hilbert
space of first quantization and the N-particle sector of the fermionic Fock
space.  This program turns every statement of the corresponding section of
the text into a numerical check:

  1. the position-space amplitudes of a_{p_1}^+ ... a_{p_N}^+ |0>, recovered
     through the field operators as <0| psi(x_N)...psi(x_1) |state>/sqrt(N!),
     reproduce the normalised Slater determinant
     det[phi_{p_i}(x_j)]/sqrt(N!) at every grid point -- and the same holds
     for a general linear combination of determinants, with *identical*
     expansion coefficients on both sides;

  2. the correspondence is unitary: the inner product of two antisymmetric
     wave functions equals the inner product of the corresponding Fock
     states;

  3. under a change of single-particle basis phi~_alpha = sum_p phi_p
     U_{p alpha}, the transformation between the two Slater-determinant
     bases is the N-th exterior power of U,
         <p_1...p_N | alpha~_1...alpha~_N> = det[ U_{p_i alpha_j} ],
     the matrix of all N x N minors of U, which is again unitary; a U that
     mixes only the occupied orbitals reproduces the det(U) phase of the
     rotated determinant found in chapter 2.

The single-particle "position" space is a grid of L points, so an orbital is
a column of an L x n matrix Phi with orthonormal columns and integrals are
sums over the grid.  A Fock-space state is a dictionary {bit pattern:
amplitude} with orbital p occupying bit p, and the sign
(-1)^(number of occupied orbitals below p) keeps the anticommutation
relations exact, as in wick.py.  Everything is complex, so that the
conjugation conventions are exercised as well.
"""

from itertools import combinations, product
from math import factorial, sqrt

import numpy as np


# ------------------------------------------------------------- Fock states

def create(p, state):
    """Apply a_p^dagger to a state given as {bit pattern: amplitude}."""
    out = {}
    for bits, amp in state.items():
        if bits & (1 << p):                      # Pauli principle
            continue
        sign = (-1) ** bin(bits & ((1 << p) - 1)).count("1")
        new = bits | (1 << p)
        out[new] = out.get(new, 0.0) + sign * amp
    return out


def annihilate(p, state):
    """Apply a_p to a state given as {bit pattern: amplitude}."""
    out = {}
    for bits, amp in state.items():
        if not bits & (1 << p):
            continue
        sign = (-1) ** bin(bits & ((1 << p) - 1)).count("1")
        new = bits ^ (1 << p)
        out[new] = out.get(new, 0.0) + sign * amp
    return out


def determinant_state(occupied, coeff=1.0):
    """a_{p_1}^+ a_{p_2}^+ ... a_{p_N}^+ |0> for p_1 < p_2 < ... < p_N."""
    state = {0: coeff}
    for p in sorted(occupied, reverse=True):     # rightmost operator first
        state = create(p, state)
    return state


def add(state, other):
    out = dict(state)
    for bits, amp in other.items():
        out[bits] = out.get(bits, 0.0) + amp
    return out


def overlap(bra, ket):
    """<bra|ket> for two states given as dictionaries."""
    return sum(np.conj(amp) * ket[bits]
               for bits, amp in bra.items() if bits in ket)


# ------------------------------------------- first quantization on a grid

def orbitals(L, n, rng):
    """An L x n matrix Phi with orthonormal columns: n orbitals on L points."""
    Phi, _ = np.linalg.qr(rng.standard_normal((L, n))
                          + 1j * rng.standard_normal((L, n)))
    return Phi


def slater_amplitude(Phi, occupied, xs):
    """The normalised Slater determinant det[phi_{p_i}(x_j)]/sqrt(N!)
    evaluated at the grid points xs = (x_1, ..., x_N)."""
    A = Phi[np.ix_(list(xs), list(occupied))].T   # rows: orbitals; cols: particles
    return np.linalg.det(A) / sqrt(factorial(len(occupied)))


def field_annihilate(phi_row, state):
    """Apply the field operator psi(x) = sum_p phi_p(x) a_p; phi_row = Phi[x]."""
    out = {}
    for p, coeff in enumerate(phi_row):
        if coeff == 0.0:
            continue
        out = add(out, {bits: coeff * amp
                        for bits, amp in annihilate(p, state).items()})
    return out


def field_amplitude(Phi, state, xs):
    """Psi(x_1,...,x_N) = <0| psi(x_N) ... psi(x_1) |state> / sqrt(N!)."""
    for x in xs:                                  # psi(x_1) acts first
        state = field_annihilate(Phi[x], state)
    return state.get(0, 0.0) / sqrt(factorial(len(xs)))


def first_quantized_inner(Phi, state1, state2, N):
    """<Psi_1|Psi_2> as a sum over all N-tuples of grid points."""
    L = Phi.shape[0]
    return sum(np.conj(field_amplitude(Phi, state1, xs))
               * field_amplitude(Phi, state2, xs)
               for xs in product(range(L), repeat=N))


# --------------------------------------------------- change of orbital basis

def tilde_create(alpha, U, state):
    """Apply a~_alpha^+ = sum_p U_{p alpha} a_p^+."""
    out = {}
    for p in range(U.shape[0]):
        out = add(out, {bits: U[p, alpha] * amp
                        for bits, amp in create(p, state).items()})
    return out


def tilde_determinant_state(alphas, U):
    """a~_{alpha_1}^+ ... a~_{alpha_N}^+ |0> for alpha_1 < ... < alpha_N."""
    state = {0: 1.0}
    for alpha in sorted(alphas, reverse=True):
        state = tilde_create(alpha, U, state)
    return state


def exterior_power(U, N):
    """U^{wedge N}: all N x N minors of U, rows and columns ordered
    lexicographically over the N-element subsets."""
    n = U.shape[0]
    sets = list(combinations(range(n), N))
    return np.array([[np.linalg.det(U[np.ix_(rows, cols)]) for cols in sets]
                     for rows in sets]), sets


# ----------------------------------------------------------------- the demo

def _demo():
    rng = np.random.default_rng(2026)
    L, n, N = 7, 5, 3                 # grid points, orbitals, particles
    Phi = orbitals(L, n, rng)

    print("=" * 74)
    print("1. Same orbital basis: the transformation is the identity")
    print("=" * 74)
    print(f"{n} orbitals on {L} grid points, {N} particles.")
    print()

    occupied = (0, 2, 3)
    state = determinant_state(occupied)
    worst = max(abs(slater_amplitude(Phi, occupied, xs)
                    - field_amplitude(Phi, state, xs))
                for xs in product(range(L), repeat=N))
    print(f"  det[phi_p_i(x_j)]/sqrt(N!)  vs  <0|psi(x_N)..psi(x_1)|p1..pN>/sqrt(N!)")
    print(f"  occupied orbitals {occupied}: "
          f"max difference over all {L}^{N} grid tuples = {worst:.1e}")

    # a general state: same coefficients in both languages
    dets = list(combinations(range(n), N))
    C = rng.standard_normal(len(dets)) + 1j * rng.standard_normal(len(dets))
    C /= np.linalg.norm(C)
    psi = {}
    for coeff, det in zip(C, dets):
        psi = add(psi, determinant_state(det, coeff))
    worst = max(abs(sum(coeff * slater_amplitude(Phi, det, xs)
                        for coeff, det in zip(C, dets))
                    - field_amplitude(Phi, psi, xs))
                for xs in product(range(L), repeat=N))
    print(f"  random {len(dets)}-determinant state, identical coefficients: "
          f"max difference = {worst:.1e}")

    print()
    print("=" * 74)
    print("2. The correspondence is unitary")
    print("=" * 74)
    bra = determinant_state((0, 1, 4))
    pairs = [(bra, bra), (bra, state), (psi, psi), (psi, state)]
    labels = ["<014|014>", "<014|023>", "<Psi|Psi>", "<Psi|023>"]
    print("  first-quantized integral  vs  Fock-space overlap")
    for (s1, s2), label in zip(pairs, labels):
        fq = first_quantized_inner(Phi, s1, s2, N)
        sq = overlap(s1, s2)
        print(f"  {label:>10s}:  {fq: .10f}   {sq: .10f}   "
              f"|difference| = {abs(fq - sq):.1e}")

    print()
    print("=" * 74)
    print("3. Two orbital bases: the exterior power U^(wedge N)")
    print("=" * 74)
    U, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    W, sets = exterior_power(U, N)
    G = np.array([[overlap(determinant_state(rows),
                           tilde_determinant_state(cols, U))
                   for cols in sets] for rows in sets])
    print(f"  random unitary U on the {n} orbitals; "
          f"determinant bases of dimension {len(sets)}")
    print(f"  max |<p1..pN|a~1..a~N> - det[U_p_i,a_j]|      = "
          f"{np.max(np.abs(G - W)):.1e}")
    print(f"  max |(U^wN)^+ U^wN - 1|  (unitarity of U^wN)  = "
          f"{np.max(np.abs(W.conj().T @ W - np.eye(len(sets)))):.1e}")

    # a rotation among the occupied orbitals alone: a pure phase det(U_occ)
    U_occ, _ = np.linalg.qr(rng.standard_normal((N, N))
                            + 1j * rng.standard_normal((N, N)))
    U_block = np.eye(n, dtype=complex)
    U_block[np.ix_(occupied, occupied)] = U_occ
    rotated = tilde_determinant_state(occupied, U_block)
    phase = overlap(determinant_state(occupied), rotated)
    print(f"  U mixing only the occupied orbitals {occupied}:")
    print(f"  |<p1..pN|rotated> - det(U_occ)| = "
          f"{abs(phase - np.linalg.det(U_occ)):.1e}, "
          f"|det(U_occ)| = {abs(np.linalg.det(U_occ)):.10f}")


if __name__ == "__main__":
    _demo()

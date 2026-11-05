"""
The full CI equation taken apart: the secular determinant, the projected
amplitude equations and their iterative solution.

Companion code to chapter 5 of *Quantum mechanics for Many-particle Systems*.

Three demonstrations, each one paragraph of the text:

    demo_secular       -- the characteristic polynomial of the seniority-zero
                          pairing matrix and of the full 70 x 70 matrix, why
                          one never computes eigenvalues that way, and
                          steepest descent on the Rayleigh quotient
    demo_projected     -- the energy, singles and doubles equations evaluated
                          term by term with the exact FCI eigenvector of a
                          random two-body Hamiltonian (n = 8, N = 4), so that
                          every sign and factor of the text is checked
    demo_iteration     -- the "non-practical" solution of the amplitude
                          equations by iteration for the pairing model, whose
                          first iterate is second-order perturbation theory

The general Hamiltonian is built with the bit-string machinery of fci.py.

Author: Morten Hjorth-Jensen
"""

from itertools import combinations, permutations

import numpy as np

from fci import PairingFCI, SlaterBasis, apply_string, popcount


# ---------------------------------------------------------------------------
#  A general two-body Hamiltonian in a determinant basis
# ---------------------------------------------------------------------------
def apply_hamiltonian(state, h, v_as):
    """H|state> as {new_state: amplitude}, H = sum h_pq a+_p a_q
    + 1/4 sum <pq|v|rs>_AS a+_p a+_q a_s a_r, by bit operations only."""
    n = h.shape[0]
    out = {}
    occ = [p for p in range(n) if (state >> p) & 1]
    for q in occ:
        for p in range(n):
            if h[p, q] != 0.0:
                s, new = apply_string(state, [(p, True), (q, False)])
                if s:
                    out[new] = out.get(new, 0.0) + s * h[p, q]
    for r in occ:
        for s_ in occ:
            if s_ == r:
                continue
            for p in range(n):
                for q in range(n):
                    if v_as[p, q, r, s_] != 0.0:
                        s, new = apply_string(
                            state, [(p, True), (q, True), (s_, False), (r, False)])
                        if s:
                            out[new] = out.get(new, 0.0) + 0.25 * s * v_as[p, q, r, s_]
    return out


class GeneralFCI:
    """Full CI for arbitrary one- and antisymmetrised two-body matrix elements."""

    def __init__(self, h, v_as, n_particles):
        self.h, self.v = h, v_as
        self.n = h.shape[0]
        self.N = n_particles
        self.basis = SlaterBasis(self.n, n_particles)
        self.reference = (1 << n_particles) - 1
        self.levels_of = self.basis.excitation_levels(self.reference)
        self.holes = list(range(n_particles))
        self.parts = list(range(n_particles, self.n))

    def matrix(self, max_excitation=None):
        keep = [s for s, k in zip(self.basis.states, self.levels_of)
                if max_excitation is None or k <= max_excitation]
        idx = {s: i for i, s in enumerate(keep)}
        H = np.zeros((len(keep), len(keep)))
        for s in keep:
            for new, amp in apply_hamiltonian(s, self.h, self.v).items():
                if new in idx:
                    H[idx[new], idx[s]] += amp
        return H, keep

    def reference_energy(self):
        i = self.holes
        return sum(self.h[p, p] for p in i) + 0.5 * sum(self.v[p, q, p, q] for p in i for q in i)

    def fock_matrix(self):
        return self.h + sum(self.v[:, i, :, i] for i in self.holes)

    def determinant(self, parts, holes):
        """Bit pattern and phase of a+_a a+_b ... a_j a_i |Phi_0>."""
        ops = [(a, True) for a in parts] + [(i, False) for i in reversed(holes)]
        return apply_string(self.reference, ops)


def random_hamiltonian(n, seed=2038, scale=0.15):
    """A 'shell structure' plus random mixing, and a random antisymmetrised interaction."""
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(n, n))
    h = np.diag(np.arange(n, dtype=float)) + 0.2 * (h + h.T) / 2
    v = rng.normal(size=(n, n, n, n))
    v = v + v.transpose(1, 0, 3, 2)
    v = v + v.transpose(2, 3, 0, 1)
    v = scale * (v - v.transpose(0, 1, 3, 2))
    return h, v


# ---------------------------------------------------------------------------
#  1. The secular determinant and the characteristic polynomial
# ---------------------------------------------------------------------------
def seniority_zero_matrix(g, levels=4, xi=1.0):
    pairs = list(combinations(range(1, levels + 1), 2))
    M = np.zeros((len(pairs), len(pairs)))
    for A, (p, q) in enumerate(pairs):
        M[A, A] = 2 * xi * ((p - 1) + (q - 1)) - g
        for B, (r, s) in enumerate(pairs):
            if A != B and len({p, q} & {r, s}) == 1:
                M[A, B] = -g / 2
    return M


def demo_secular():
    print("=" * 74)
    print("1. The secular equation det(H - E 1) = 0 as a polynomial")
    print("=" * 74)
    g, xi = 1.0, 1.0
    M2 = np.array([[-g / 2, -g / 2], [-g / 2, 2 * xi - g / 2]])
    print("two levels, one pair: roots xi - g/2 -/+ sqrt(xi^2 + g^2/4) =",
          xi - g / 2 - np.sqrt(xi**2 + g**2 / 4), xi - g / 2 + np.sqrt(xi**2 + g**2 / 4))
    print("                      eigh                                 =", np.linalg.eigvalsh(M2))
    H6 = seniority_zero_matrix(g)
    coeffs = np.poly(H6)
    print("\nfour levels, seniority zero, g = 1:")
    print("  characteristic polynomial, coefficients of E^6 ... E^0:", np.round(coeffs, 6))
    print("  trace =", np.trace(H6), " determinant =", round(np.linalg.det(H6), 6))
    roots = np.sort(np.roots(coeffs).real)
    eig = np.linalg.eigvalsh(H6)
    print("  roots       :", roots)
    print("  eigenvalues :", eig)
    print(f"  max |root - eigenvalue| = {np.abs(roots - eig).max():.1e}")
    # steepest descent on the Rayleigh quotient finds the lowest eigenpair
    rng = np.random.default_rng(0)
    c = rng.normal(size=6)
    for _ in range(500):
        E_c = (c @ H6 @ c) / (c @ c)
        c = c - 0.1 * 2 * (H6 @ c - E_c * c) / (c @ c)
        c /= np.linalg.norm(c)
    print(f"  steepest descent on E[C], 500 steps of 0.1: E = {(c @ H6 @ c):.8f}, "
          f"overlap with the eigenvector {abs(c @ np.linalg.eigh(H6)[1][:, 0]):.10f}")
    model = PairingFCI(levels=4, n_particles=4, g=g)
    H70 = model.matrix()
    E70 = np.linalg.eigvalsh(H70)
    c70 = np.poly(H70)
    r70 = np.sort(np.roots(c70).real)
    rng = np.random.default_rng(1)
    r70p = np.sort(np.roots(c70 * (1 + 1e-14 * rng.normal(size=c70.size))).real)
    Hp = H70 + 1e-14 * np.abs(H70).max() * rng.normal(size=H70.shape)
    Hp = (Hp + Hp.T) / 2
    print("\nthe full 70 x 70 pairing matrix:")
    print(f"  largest polynomial coefficient            {np.abs(c70).max():.2e}")
    print(f"  max |root - eigenvalue|, exact coefficients {np.abs(r70 - E70).max():.2e}")
    print(f"  ... after 1e-14 relative coefficient noise  {np.abs(r70p - E70).max():.2e}")
    print(f"  eigenvalue shift for 1e-14 matrix noise     {np.abs(np.linalg.eigvalsh(Hp) - E70).max():.2e}")
    print("\nThe polynomial defines the eigenvalue problem; it is never the algorithm.")


# ---------------------------------------------------------------------------
#  2. The projected equations, term by term
# ---------------------------------------------------------------------------
def perm_sign(p):
    s = 1
    p = list(p)
    for i in range(len(p)):
        for j in range(i + 1, len(p)):
            if p[i] > p[j]:
                s = -s
    return s


def amplitudes(model, vec, keep):
    """Antisymmetric C1, C2, C3, C4 tensors from an intermediate-normalised eigenvector."""
    idx = {s: k for k, s in enumerate(keep)}
    n = model.n
    P, Hh = model.parts, model.holes

    def amp(parts, holes):
        s, st = model.determinant(list(parts), list(holes))
        return s * vec[idx[st]]

    C1 = np.zeros((n, n))
    C2 = np.zeros((n,) * 4)
    C3 = np.zeros((n,) * 6)
    C4 = np.zeros((n,) * 8)
    for a in P:
        for i in Hh:
            C1[a, i] = amp([a], [i])
    for k, Ck in ((2, C2), (3, C3), (4, C4)):
        for pa in combinations(P, k):
            for ho in combinations(Hh, k):
                x = amp(pa, ho)
                for pp in permutations(range(k)):
                    for ph in permutations(range(k)):
                        Ck[tuple(pa[m] for m in pp) + tuple(ho[m] for m in ph)] = \
                            perm_sign(pp) * perm_sign(ph) * x
    return C1, C2, C3, C4


def singles_terms(f, v, C1, C2, C3, P, Hh, a, i):
    return {
        "f_ai": f[a, i],
        "sum_b f_ab C_i^b": sum(f[a, b] * C1[b, i] for b in P),
        "-sum_j f_ji C_j^a": -sum(f[j, i] * C1[a, j] for j in Hh),
        "sum_jb <aj|v|ib> C_j^b": sum(v[a, j, i, b] * C1[b, j] for j in Hh for b in P),
        "sum_jb f_jb C_ij^ab": sum(f[j, b] * C2[a, b, i, j] for j in Hh for b in P),
        "1/2 sum <aj|v|bc> C_ij^bc": 0.5 * sum(v[a, j, b, c] * C2[b, c, i, j]
                                              for j in Hh for b in P for c in P),
        "-1/2 sum <jk|v|ib> C_jk^ab": -0.5 * sum(v[j, k, i, b] * C2[a, b, j, k]
                                                for j in Hh for k in Hh for b in P),
        "1/4 sum <jk|v|bc> C_ijk^abc": 0.25 * sum(v[j, k, b, c] * C3[a, b, c, i, j, k]
                                                 for j in Hh for k in Hh for b in P for c in P),
    }


def doubles_terms(f, v, C1, C2, C3, C4, P, Hh, a, b, i, j):
    Pab = lambda X: X(a, b) - X(b, a)
    Pij = lambda X: X(i, j) - X(j, i)
    T = {}
    T["<ab|v|ij>"] = v[a, b, i, j]
    T["P(ab) sum_c f_bc C_ij^ac"] = Pab(lambda a_, b_: sum(f[b_, c] * C2[a_, c, i, j] for c in P))
    T["-P(ij) sum_k f_kj C_ik^ab"] = -Pij(lambda i_, j_: sum(f[k, j_] * C2[a, b, i_, k] for k in Hh))
    T["1/2 sum <ab|v|cd> C_ij^cd"] = 0.5 * sum(v[a, b, c, d] * C2[c, d, i, j] for c in P for d in P)
    T["1/2 sum <kl|v|ij> C_kl^ab"] = 0.5 * sum(v[k, l, i, j] * C2[a, b, k, l] for k in Hh for l in Hh)
    T["P(ij)P(ab) sum <kb|v|cj> C_ik^ac"] = sum(
        s1 * s2 * sum(v[k, b_, c, j_] * C2[a_, c, i_, k] for k in Hh for c in P)
        for s1, (i_, j_) in ((1, (i, j)), (-1, (j, i)))
        for s2, (a_, b_) in ((1, (a, b)), (-1, (b, a))))
    T["P(ij) sum_c <ab|v|cj> C_i^c"] = Pij(lambda i_, j_: sum(v[a, b, c, j_] * C1[c, i_] for c in P))
    T["-P(ab) sum_k <kb|v|ij> C_k^a"] = -Pab(lambda a_, b_: sum(v[k, b_, i, j] * C1[a_, k] for k in Hh))
    T["P(ij)P(ab) f_ai C_j^b (disconnected)"] = (f[a, i] * C1[b, j] - f[a, j] * C1[b, i]
                                                 - f[b, i] * C1[a, j] + f[b, j] * C1[a, i])
    T["sum_kc f_kc C_ijk^abc"] = sum(f[k, c] * C3[a, b, c, i, j, k] for k in Hh for c in P)
    T["1/2 P(ab) sum <bk|v|cd> C_ijk^acd"] = 0.5 * Pab(
        lambda a_, b_: sum(v[b_, k, c, d] * C3[a_, c, d, i, j, k] for k in Hh for c in P for d in P))
    T["-1/2 P(ij) sum <kl|v|jc> C_ikl^abc"] = -0.5 * Pij(
        lambda i_, j_: sum(v[k, l, j_, c] * C3[a, b, c, i_, k, l] for k in Hh for l in Hh for c in P))
    T["1/4 sum <kl|v|cd> C_ijkl^abcd"] = 0.25 * sum(
        v[k, l, c, d] * C4[a, b, c, d, i, j, k, l] for k in Hh for l in Hh for c in P for d in P)
    return T


def demo_projected(n=8, n_particles=4):
    print("=" * 74)
    print("2. The projected FCI equations, term by term")
    print("=" * 74)
    h, v = random_hamiltonian(n)
    model = GeneralFCI(h, v, n_particles)
    H, keep = model.matrix()
    E, C = np.linalg.eigh(H)
    vec = C[:, 0] / C[0, 0]                      # intermediate normalisation
    E_ref = model.reference_energy()
    dE = E[0] - E_ref
    f = model.fock_matrix()
    P, Hh = model.parts, model.holes
    C1, C2, C3, C4 = amplitudes(model, vec, keep)
    print(f"random two-body Hamiltonian, n = {n}, N = {n_particles}: dimension {len(keep)},")
    print(f"E_ref = {E_ref:.8f}, E_0 = {E[0]:.8f}, Delta E = {dE:.10f}")
    e_s = sum(f[i, a] * C1[a, i] for i in Hh for a in P)
    e_d = 0.25 * sum(v[i, j, a, b] * C2[a, b, i, j] for i in Hh for j in Hh for a in P for b in P)
    print(f"energy equation: singles {e_s:.8f} + doubles {e_d:.8f} = {e_s + e_d:.10f}")
    worst = max(abs(sum(singles_terms(f, v, C1, C2, C3, P, Hh, a, i).values()) - dE * C1[a, i])
                for a in P for i in Hh)
    print(f"singles equations, all {len(P)*len(Hh)} pairs (a,i): max residual {worst:.1e}")
    t = singles_terms(f, v, C1, C2, C3, P, Hh, P[0], Hh[0])
    print(f"  the eight terms for a = {P[0]}, i = {Hh[0]}:")
    for k, x in t.items():
        print(f"    {k:<36} {x:>12.6f}")
    print(f"    {'sum':<36} {sum(t.values()):>12.6f}   Delta E C_i^a = {dE * C1[P[0], Hh[0]]:.6f}")
    worst = 0.0
    for a, b in combinations(P, 2):
        for i, j in combinations(Hh, 2):
            T = doubles_terms(f, v, C1, C2, C3, C4, P, Hh, a, b, i, j)
            worst = max(worst, abs(sum(T.values()) - dE * C2[a, b, i, j]))
    print(f"doubles equations, all 36 (ab,ij): max residual {worst:.1e}")
    T = doubles_terms(f, v, C1, C2, C3, C4, P, Hh, P[0], P[1], Hh[0], Hh[1])
    print(f"  the thirteen terms for ab = {P[0]}{P[1]}, ij = {Hh[0]}{Hh[1]}:")
    for k, x in T.items():
        print(f"    {k:<40} {x:>12.6f}")
    print(f"    {'sum':<40} {sum(T.values()):>12.6f}   Delta E C_ij^ab = {dE * C2[P[0], P[1], Hh[0], Hh[1]]:.6f}")


# ---------------------------------------------------------------------------
#  3. The non-practical way: iterate the amplitude equations
# ---------------------------------------------------------------------------
def iterate_amplitudes(H, n_iterations=60):
    """Jacobi-type iteration of the projected equations; row 0 is the reference."""
    dim = H.shape[0]
    diag = np.diag(H)
    C = np.zeros(dim)
    C[0] = 1.0
    for K in range(1, dim):                     # first-order guess
        C[K] = H[K, 0] / (H[0, 0] - diag[K]) if abs(H[0, 0] - diag[K]) > 1e-12 else 0.0
    history = []
    for _ in range(n_iterations):
        E = H[0, 0] + H[0, 1:] @ C[1:]          # energy equation
        new = C.copy()
        for K in range(1, dim):
            rhs = H[K, 0] + H[K, 1:] @ C[1:] - diag[K] * C[K]
            new[K] = rhs / (E - diag[K])
        C = new
        history.append(E)
    return history, C


def demo_iteration():
    print("=" * 74)
    print("3. Solving the amplitude equations by iteration (pairing model)")
    print("=" * 74)
    print(f"{'g':>6} {'iterate 1':>12} {'E_ref-7/24 g^2':>15} {'iterate 5':>12} "
          f"{'iterate 20':>12} {'iterate 60':>12} {'FCI':>12}")
    for g in (0.25, 0.5, 1.0):
        model = PairingFCI(levels=4, n_particles=4, g=g)
        H = model.matrix()
        # put the reference determinant first
        order = np.argsort(model.levels_of, kind="stable")
        H = H[np.ix_(order, order)]
        hist, _ = iterate_amplitudes(H)
        E_fci = np.linalg.eigvalsh(H)[0]
        print(f"{g:>6.2f} {hist[0]:>12.8f} {H[0,0] - 7/24*g**2:>15.8f} {hist[4]:>12.8f} "
              f"{hist[19]:>12.8f} {hist[-1]:>12.8f} {E_fci:>12.8f}")
    print("\nThe first iterate is second-order perturbation theory; the sequence")
    print("converges to the FCI energy, but the loop runs over every amplitude.")


def _demo():
    for f in (demo_secular, demo_projected, demo_iteration):
        f()
        print()


if __name__ == "__main__":
    _demo()

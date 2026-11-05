"""
Goldstone diagrams for Rayleigh-Schrodinger perturbation theory, evaluated
from the orbital-level matrix elements and checked against the determinant
space.  Companion code to chapter 9, section "Diagrammatic approach and the
linked diagram theorem".

The perturbation is written in normal-ordered form relative to the
reference determinant,

    H_I = Delta E^(1) + sum_pq ftilde_pq {a+_p a_q} + 1/4 sum vbar_pqrs {a+ a+ a a},
    ftilde_pq = u_pq + U_pq,   U_pq = sum_{j occupied} vbar_pjqj,

so that a one-body vertex carries ftilde (the cross plus its bubble) and a
two-body vertex the antisymmetrised element.  The functions below

    GeneralHamiltonian     -- a Hamiltonian with arbitrary eps_p, u_pq and
                              vbar_pqrs in the full determinant space, with
                              the Rayleigh-Schrodinger recursion
    third_order_diagrams   -- the fourteen connected third-order energy
                              diagrams, each evaluated with the diagram rules
    second_order_diagrams  -- the two second-order diagrams (one-body and
                              two-body vertices), for comparison
    fourth_order_unlinked  -- for two non-interacting copies: the part of the
                              principal fourth-order term that connects the
                              copies, against the renormalisation term that
                              removes it

Author: Morten Hjorth-Jensen
"""

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
#  bit-string determinants
# ---------------------------------------------------------------------------
def popcount(x):
    return bin(x).count("1")


def annihilate(state, p):
    if not (state >> p) & 1:
        return 0, 0
    sign = -1 if popcount(state & ((1 << p) - 1)) & 1 else 1
    return sign, state ^ (1 << p)


def create(state, p):
    if (state >> p) & 1:
        return 0, 0
    sign = -1 if popcount(state & ((1 << p) - 1)) & 1 else 1
    return sign, state | (1 << p)


def apply_string(state, operators):
    """Apply [(index, dagger), ...], rightmost operator first."""
    sign = 1
    for p, dagger in reversed(operators):
        s, state = (create(state, p) if dagger else annihilate(state, p))
        if s == 0:
            return 0, 0
        sign *= s
    return sign, state


class GeneralHamiltonian:
    """H = sum_p eps_p a+_p a_p + sum_pq u_pq a+_p a_q + 1/4 sum vbar a+a+aa.

    The reference determinant fills the first n_particles orbitals; the
    orbital energies eps must therefore be ordered so that these are the
    lowest.  H_0 = sum eps_p a+_p a_p, H_I = the rest.
    """

    def __init__(self, eps, u, vbar, n_particles):
        self.eps = np.asarray(eps, dtype=float)
        self.u = np.asarray(u, dtype=float)
        self.vbar = np.asarray(vbar, dtype=float)
        self.n = len(self.eps)
        self.N = n_particles
        self.occ = list(range(n_particles))
        self.vir = list(range(n_particles, self.n))
        self.states = []
        for occ in combinations(range(self.n), n_particles):
            bits = 0
            for p in occ:
                bits |= 1 << p
            self.states.append(bits)
        self.states.sort()
        self.index = {s: i for i, s in enumerate(self.states)}
        self.dim = len(self.states)
        self.reference = (1 << n_particles) - 1
        self.ref = self.index[self.reference]
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        n, dim = self.n, self.dim
        H0 = np.zeros(dim)
        V = np.zeros((dim, dim))
        for col, state in enumerate(self.states):
            H0[col] = sum(self.eps[p] for p in range(n) if (state >> p) & 1)
            for p in range(n):
                for q in range(n):
                    if self.u[p, q] == 0.0:
                        continue
                    sign, new = apply_string(state, [(p, True), (q, False)])
                    if sign:
                        V[self.index[new], col] += sign * self.u[p, q]
            for p in range(n):
                for q in range(n):
                    for r in range(n):
                        for s in range(n):
                            v = self.vbar[p, q, r, s]
                            if v == 0.0:
                                continue
                            sign, new = apply_string(
                                state, [(p, True), (q, True), (s, False), (r, False)])
                            if sign:
                                V[self.index[new], col] += 0.25 * sign * v
        self.w = H0
        self.V = V
        self.H = np.diag(H0) + V
        self.w0 = H0[self.ref]
        D = self.w0 - H0
        self.R = np.zeros(dim)
        mask = np.arange(dim) != self.ref
        self.R[mask] = 1.0 / D[mask]
        self.D = D

    # ------------------------------------------------------------------
    def rayleigh_schrodinger(self, order=4):
        """Delta E^(1) ... Delta E^(order) by the standard recursion."""
        V, R, ref = self.V, self.R, self.ref
        psi0 = np.zeros(self.dim)
        psi0[ref] = 1.0
        energies = [float(psi0 @ (V @ psi0))]
        psis = []
        for k in range(1, order):
            previous = psis[-1] if psis else psi0
            rhs = V @ previous
            for m in range(1, k):
                rhs = rhs - energies[m - 1] * psis[k - m - 1]
            psi = R * rhs
            psis.append(psi)
            energies.append(float(psi0 @ (V @ psi)))
        return energies

    def third_order_pieces(self):
        """Principal term, renormalisation term, and <W R W R W> with the
        normal-ordered vertex W = V - V_00."""
        V, R, ref = self.V, self.R, self.ref
        v0 = V[:, ref]
        principal = float(v0 @ (R * (V @ (R * v0))))
        renorm = float(V[ref, ref] * np.sum(R * R * v0 * v0))
        W = V - V[ref, ref] * np.eye(self.dim)
        w0 = W[:, ref]
        normal = float(w0 @ (R * (W @ (R * w0))))
        return principal, renorm, normal

    def exact(self):
        return float(np.linalg.eigvalsh(self.H)[0])

    # ------------------------------------------------------------------
    def ftilde(self):
        """The normal-ordered one-body vertex u + U, U_pq = sum_j vbar_pjqj."""
        U = sum(self.vbar[:, j, :, j] for j in self.occ)
        return self.u + U


# ---------------------------------------------------------------------------
#  the diagrams
# ---------------------------------------------------------------------------
def second_order_diagrams(ham):
    """The two second-order diagrams with normal-ordered vertices."""
    eps, vb = ham.eps, ham.vbar
    o, v = ham.occ, ham.vir
    f = ham.ftilde()
    e_o, e_v = eps[o], eps[v]
    Dai = e_o[:, None] - e_v[None, :]                       # [i, a]
    fov = f[np.ix_(o, v)]                                  # f_ia
    fvo = f[np.ix_(v, o)]                                  # f_ai
    singles = float(np.sum(fov * fvo.T / Dai))
    Dijab = (e_o[:, None, None, None] + e_o[None, :, None, None]
             - e_v[None, None, :, None] - e_v[None, None, None, :])
    v_oovv = vb[np.ix_(o, o, v, v)]
    v_vvoo = vb[np.ix_(v, v, o, o)]
    doubles = 0.25 * float(np.einsum("ijab,abij,ijab->", v_oovv, v_vvoo, 1.0 / Dijab))
    return {"(F,F) singles": singles, "(V,V) doubles": doubles}


def third_order_diagrams(ham):
    """The fourteen connected third-order energy diagrams.

    Vertices are read from the bottom (first to act) to the top, F is the
    normal-ordered one-body vertex ftilde and V the interaction.  Each value
    is obtained with the rules: sums over labels, one denominator per cut,
    1/2 per pair of equivalent lines, sign (-1)^(h+l).
    """
    eps, vb = ham.eps, ham.vbar
    o, v = ham.occ, ham.vir
    f = ham.ftilde()
    e_o, e_v = eps[o], eps[v]
    foo = f[np.ix_(o, o)]       # f_ij
    fvv = f[np.ix_(v, v)]       # f_ab
    fov = f[np.ix_(o, v)]       # f_ia
    fvo = f[np.ix_(v, o)]       # f_ai
    D1 = e_o[:, None] - e_v[None, :]                               # D[i,a]
    D2 = (e_o[:, None, None, None] + e_o[None, :, None, None]
          - e_v[None, None, :, None] - e_v[None, None, None, :])   # D[i,j,a,b]
    v_oovv = vb[np.ix_(o, o, v, v)]   # <ij|v|ab>
    v_vvoo = vb[np.ix_(v, v, o, o)]   # <ab|v|ij>
    v_ovov = vb[np.ix_(o, v, o, v)]   # <ib|v|ja>
    v_vvvo = vb[np.ix_(v, v, v, o)]   # <bc|v|aj>
    v_ovoo = vb[np.ix_(o, v, o, o)]   # <ib|v|jk>
    v_vvvv = vb[np.ix_(v, v, v, v)]   # <cd|v|ab>
    v_oooo = vb[np.ix_(o, o, o, o)]   # <ij|v|kl>
    v_vovo = vb[np.ix_(v, o, v, o)]   # <ci|v|ak>
    v_ovvv = vb[np.ix_(o, v, v, v)]   # <ic|v|ab>
    v_oovo = vb[np.ix_(o, o, v, o)]   # <ij|v|ak>

    d = {}
    # (F,F,F)
    d["1  (F,F,F) particle insertion"] = float(np.einsum(
        "ib,ba,ai,ia,ib->", fov, fvv, fvo, 1 / D1, 1 / D1))
    d["2  (F,F,F) hole insertion"] = -float(np.einsum(
        "ja,ij,ai,ia,ja->", fov, foo, fvo, 1 / D1, 1 / D1))
    # (F,F,V)
    d["3  (F,F,V) two crosses closed by V"] = float(np.einsum(
        "ijab,bj,ai,ia,ijab->", v_oovv, fvo, fvo, 1 / D1, 1 / D2))
    # (F,V,F)
    d["4  (F,V,F) particle-hole scattering"] = -float(np.einsum(
        "jb,ibja,ai,ia,jb->", fov, v_ovov, fvo, 1 / D1, 1 / D1))
    # (F,V,V)
    d["5  (F,V,V) particle line into V"] = 0.5 * float(np.einsum(
        "ijbc,bcaj,ai,ia,ijbc->", v_oovv, v_vvvo, fvo, 1 / D1, 1 / D2))
    d["6  (F,V,V) hole line into V"] = -0.5 * float(np.einsum(
        "jkab,ibjk,ai,ia,jkab->", v_oovv, v_ovoo, fvo, 1 / D1, 1 / D2))
    # (V,F,V)
    d["7  (V,F,V) particle insertion"] = 0.5 * float(np.einsum(
        "ijcb,ca,abij,ijab,ijcb->", v_oovv, fvv, v_vvoo, 1 / D2, 1 / D2))
    d["8  (V,F,V) hole insertion"] = -0.5 * float(np.einsum(
        "kjab,ik,abij,ijab,kjab->", v_oovv, foo, v_vvoo, 1 / D2, 1 / D2))
    # (V,F,F)
    d["9  (V,F,F) V closed by two crosses"] = float(np.einsum(
        "jb,ia,abij,ijab,jb->", fov, fov, v_vvoo, 1 / D2, 1 / D1))
    # (V,V,V)
    d["10 (V,V,V) particle-particle ladder"] = 0.125 * float(np.einsum(
        "ijcd,cdab,abij,ijab,ijcd->", v_oovv, v_vvvv, v_vvoo, 1 / D2, 1 / D2))
    d["11 (V,V,V) hole-hole ladder"] = 0.125 * float(np.einsum(
        "klab,ijkl,abij,ijab,klab->", v_oovv, v_oooo, v_vvoo, 1 / D2, 1 / D2))
    d["12 (V,V,V) particle-hole ring"] = -float(np.einsum(
        "kjcb,ciak,abij,ijab,kjcb->", v_oovv, v_vovo, v_vvoo, 1 / D2, 1 / D2))
    # (V,V,F)
    d["13 (V,V,F) V to single via particles"] = 0.5 * float(np.einsum(
        "jc,icab,abij,ijab,jc->", fov, v_ovvv, v_vvoo, 1 / D2, 1 / D1))
    d["14 (V,V,F) V to single via holes"] = -0.5 * float(np.einsum(
        "kb,ijak,abij,ijab,kb->", fov, v_oovo, v_vvoo, 1 / D2, 1 / D1))
    return d


# ---------------------------------------------------------------------------
#  model builders
# ---------------------------------------------------------------------------
def random_hamiltonian(n=8, n_particles=4, seed=1, scale_u=0.1, scale_v=0.2):
    """A random Hermitian two-body Hamiltonian with every diagram class
    non-zero: eps increasing, u symmetric and off-diagonal, vbar
    antisymmetrised and Hermitian."""
    rng = np.random.default_rng(seed)
    eps = np.sort(rng.uniform(0.0, 1.0, n))
    eps = eps - eps[0] + np.linspace(0, 2, n)             # well separated
    u = scale_u * rng.normal(size=(n, n))
    u = 0.5 * (u + u.T)
    np.fill_diagonal(u, 0.0)
    w = scale_v * rng.normal(size=(n, n, n, n))
    # Hermitian: <pq|v|rs> = <rs|v|pq>
    w = 0.5 * (w + w.transpose(2, 3, 0, 1))
    # antisymmetrise in (p,q) and (r,s)
    vbar = w - w.transpose(1, 0, 2, 3)
    vbar = vbar - vbar.transpose(0, 1, 3, 2)
    vbar *= 0.5
    return GeneralHamiltonian(eps, u, vbar, n_particles)


def add_two_body(vbar, i, j, k, l, c):
    """Add c * a+_i a+_j a_k a_l to the interaction 1/4 sum vbar_pqrs a+_p a+_q a_s a_r,
    i.e. vbar_{ijlk} += c with its three antisymmetric partners."""
    vbar[i, j, l, k] += c
    vbar[j, i, l, k] -= c
    vbar[i, j, k, l] -= c
    vbar[j, i, k, l] += c


def pairing_hamiltonian(levels=4, particles=4, g=1.0, xi=1.0):
    """The pairing model: spin-orbital 2p = (p,+), 2p+1 = (p,-), levels
    p = 0..levels-1 with energy xi*p, interaction -g/2 sum_pq a+_{p+} a+_{p-} a_{q-} a_{q+}."""
    n = 2 * levels
    eps = np.array([xi * (o // 2) for o in range(n)], dtype=float)
    u = np.zeros((n, n))
    vbar = np.zeros((n, n, n, n))
    for p in range(levels):
        for q in range(levels):
            add_two_body(vbar, 2 * p, 2 * p + 1, 2 * q + 1, 2 * q, -0.5 * g)
    return GeneralHamiltonian(eps, u, vbar, particles)


def pairing_ph_hamiltonian(levels=4, particles=4, g=1.0, f=0.0, xi=1.0):
    """The pairing plus particle-hole model of chapter 7 (rpa.py):
    pairing as above plus
        -f/2 sum_{pqr} a+_{p+} a+_{p-} a_{q-} a_{r+}  + h.c."""
    ham = pairing_hamiltonian(levels, particles, g, xi)
    vbar = ham.vbar.copy()
    for p in range(levels):
        for q in range(levels):
            for r in range(levels):
                add_two_body(vbar, 2 * p, 2 * p + 1, 2 * q + 1, 2 * r, -0.5 * f)
                add_two_body(vbar, 2 * r, 2 * q + 1, 2 * p + 1, 2 * p, -0.5 * f)
    return GeneralHamiltonian(ham.eps, ham.u, vbar, particles)


def two_copies(ham_builder, **kwargs):
    """Two non-interacting copies of a Hamiltonian, as one Hamiltonian."""
    one = ham_builder(**kwargs)
    n, N = one.n, one.N
    # orbitals of copy 2 are shifted by n; the reference must fill the lowest
    # 2N orbitals, so interleave: copy-1 occupied, copy-2 occupied, then the rest
    order = list(range(N)) + list(range(n, n + N)) + list(range(N, n)) + list(range(n + N, 2 * n))
    eps2 = np.concatenate([one.eps, one.eps])[order]
    u2 = np.zeros((2 * n, 2 * n)); v2 = np.zeros((2 * n,) * 4)
    u_full = np.zeros((2 * n, 2 * n)); u_full[:n, :n] = one.u; u_full[n:, n:] = one.u
    v_full = np.zeros((2 * n,) * 4); v_full[:n, :n, :n, :n] = one.vbar; v_full[n:, n:, n:, n:] = one.vbar
    u2 = u_full[np.ix_(order, order)]
    v2 = v_full[np.ix_(order, order, order, order)]
    return one, GeneralHamiltonian(eps2, u2, v2, 2 * N)


def fourth_order_unlinked(builder, **kwargs):
    """Two non-interacting copies A and B.  With normal-ordered vertices W,
    E4 = <WRWRWRW> - E2 <WR^2W>.  The part of the principal term in which the
    intermediate states visit A, then A and B together, then B (or A, AB, A,
    and the two with the roles exchanged) is the sum of the unlinked
    fourth-order diagrams; it must equal the cross-copy part of the
    renormalisation term, 2 E2(one copy) <WR^2W>(one copy)."""
    one, two = two_copies(builder, **kwargs)
    n, N = one.n, one.N
    # which copy is excited in each determinant of the two-copy system
    # orbitals 0..N-1: A occupied, N..2N-1: B occupied, 2N..N+n-1: A empty, rest B empty
    def excited(state):
        a = sum(1 for p in range(2 * N, N + n) if (state >> p) & 1)
        b = sum(1 for p in range(N + n, 2 * n) if (state >> p) & 1)
        return a > 0, b > 0
    flags = [excited(s) for s in two.states]
    PA = np.array([a and not b for a, b in flags], float)
    PB = np.array([b and not a for a, b in flags], float)
    PAB = np.array([a and b for a, b in flags], float)
    W = two.V - two.V[two.ref, two.ref] * np.eye(two.dim)
    R = two.R
    w0 = W[:, two.ref]
    def chain(P1, P2, P3):
        x = R * P1 * w0
        x = R * P2 * (W @ x)
        x = R * P3 * (W @ x)
        return float(w0 @ x)
    unlinked = chain(PA, PAB, PB) + chain(PB, PAB, PA) + chain(PA, PAB, PA) + chain(PB, PAB, PB)
    W1 = one.V - one.V[one.ref, one.ref] * np.eye(one.dim)
    w1 = W1[:, one.ref]
    e2_one = float(np.sum(one.R * w1 * w1))
    r2_one = float(np.sum(one.R ** 2 * w1 * w1))
    e_one = one.rayleigh_schrodinger(4)
    e_two = two.rayleigh_schrodinger(4)
    return {"unlinked diagrams (four time orderings)": unlinked,
            "2 E2(A) <WR^2W>(A)": 2 * e2_one * r2_one,
            "E4 two copies": e_two[3], "2 x E4 one copy": 2 * e_one[3],
            "A,AB,B": chain(PA, PAB, PB), "A,AB,A": chain(PA, PAB, PA)}


def fourth_order_pieces(ham):
    """E4 = principal - E1[mixed] + E1^2 cubed - E2 folded, in resolvent form."""
    V, R, ref = ham.V, ham.R, ham.ref
    v0 = V[:, ref]
    e1 = V[ref, ref]
    e2 = float(np.sum(R * v0 * v0))
    r1 = R * v0                       # R V |0>
    principal = float(v0 @ (R * (V @ (R * (V @ r1)))))
    mixed = float(v0 @ (R * R * (V @ r1))) + float(v0 @ (R * (V @ (R * r1))))
    cubed = float(np.sum(R ** 3 * v0 * v0))
    folded = float(np.sum(R * R * v0 * v0))
    return principal - e1 * mixed + e1 ** 2 * cubed - e2 * folded, principal, mixed, cubed, folded


if __name__ == "__main__":
    print(fourth_order_unlinked(pairing_hamiltonian, levels=3, particles=2, g=0.5))

    np.set_printoptions(precision=10)
    for name, ham in [("random n=8 N=4", random_hamiltonian()),
                      ("pairing g=0.5", pairing_hamiltonian(g=0.5)),
                      ("pairing+ph g=0.5 f=0.025", pairing_ph_hamiltonian(g=0.5, f=0.025))]:
        print("=" * 70); print(name, "dim", ham.dim)
        e = ham.rayleigh_schrodinger(4)
        print("E1..E4 (recursion):", e)
        pr, rn, nm = ham.third_order_pieces()
        print(f"E3 principal {pr:.12f} renorm {rn:.12f} diff {pr-rn:.12f} <WRWRW> {nm:.12f}")
        d2 = second_order_diagrams(ham)
        print("second order diagrams:", d2, "sum", sum(d2.values()))
        d3 = third_order_diagrams(ham)
        for k, val in d3.items():
            print(f"  {k:45s} {val:16.12f}")
        print(f"  sum of 14 diagrams {sum(d3.values()):.12f}   E3 {e[2]:.12f}")

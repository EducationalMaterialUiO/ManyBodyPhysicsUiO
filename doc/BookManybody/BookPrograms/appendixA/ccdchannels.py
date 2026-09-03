"""
Coupled-cluster doubles for two-dimensional quantum dots, three ways.

Companion code to Project 4 of appendix A of *Quantum mechanics for
Many-particle Systems*.

The system is the quantum dot of chapter 11 -- N electrons in a parabolic
two-dimensional trap with the Coulomb repulsion -- for the closed shells
N = 2, 6, 12 and 20, and the single-particle basis is that of the
two-dimensional harmonic oscillator, supplied by quantumdot.py.  The CCD
equations of chapter 10 are then solved in three different ways:

  1. naively, by encoding the amplitude equation as it stands, with one
     Python loop per index -- correct, transparent, and unusably slow beyond
     a dozen orbitals;

  2. as dense tensor contractions, which is what coupledcluster.py of
     chapter 10 does with the Stanton-Gauss-Watts-Bartlett intermediates;

  3. as block-diagonal matrix-matrix multiplications, after sorting the
     two-particle configurations into channels of conserved total angular
     momentum projection M and total spin projection S -- the way a
     production code is organised, and the way the ring and ladder
     structure of the equations becomes visible.

All three give the same energy to machine precision, which is the check.
The Hartree-Fock reference is recomputed here with the Fock matrix
diagonalised block by block in (m, s), so that every orbital keeps its
quantum numbers and the channels are exact; the reference energy of
chapter 6 and the transformed matrix elements are verified against those
of coupledcluster.py.  Finally the ring approximation to CCD is solved on
its own and shown to reproduce the RPA correlation energy of chapter 7 --
the RPA is a doubles theory in disguise -- and for two electrons the
two-electron full CI of chapter 5 provides the exact answer.

Author: Morten Hjorth-Jensen
"""

import time
from itertools import combinations

import numpy as np

from quantumdot import QuantumDot, matrices                  # chapter 11
from coupledcluster import ccd as ccd_dense, mp2_energy       # chapter 10
from coupledcluster import reference_energy


# ---------------------------------------------------------------------------
#  Hartree-Fock with the symmetries kept (chapter 6)
# ---------------------------------------------------------------------------
def hartree_fock_blocks(dot, h, v, n_occ, tol=1e-12, max_iter=500):
    """Solve the HF equations with the Fock matrix diagonalised per (m, s).

    The Fock matrix of a closed-shell dot is block diagonal in the angular
    momentum projection m and the spin s, so we diagonalise the blocks
    separately.  The orbitals then carry exact quantum numbers, which a
    diagonalisation of the full matrix does not guarantee for degenerate
    orbitals.  Returns the transformed h and v, the orbital energies, the
    labels (n, m, s) sorted by energy, the coefficient matrix and the
    number of iterations.
    """
    labels = [(m, s) for (_, m, s) in dot.orbitals]
    blocks = {}
    for p, lab in enumerate(labels):
        blocks.setdefault(lab, []).append(p)
    n = h.shape[0]
    C = np.eye(n)
    order = np.arange(n)
    previous = None
    for iteration in range(1, max_iter + 1):
        rho = C[:, :n_occ] @ C[:, :n_occ].T
        F = h + np.einsum("prqs,sr->pq", v, rho)
        eps = np.zeros(n)
        Cnew = np.zeros((n, n))
        for lab, idx in blocks.items():
            idx = np.array(idx)
            e, U = np.linalg.eigh(F[np.ix_(idx, idx)])
            eps[idx] = e
            Cnew[np.ix_(idx, idx)] = U
        order = np.argsort(eps, kind="stable")
        C, eps = Cnew[:, order], eps[order]
        if previous is not None and abs(eps.sum() - previous) < tol:
            break
        previous = eps.sum()
    new_labels = [(dot.orbitals[q][0], labels[q][0], labels[q][1]) for q in order]
    h_new = C.T @ h @ C
    v_new = np.einsum("pi,qj,pqrs,rk,sl->ijkl", C, C, v, C, C, optimize=True)
    return h_new, v_new, eps, new_labels, C, iteration


# ---------------------------------------------------------------------------
#  Way 1: the CCD equations as they stand
# ---------------------------------------------------------------------------
def ccd_naive(eps, v, n_occ, max_iter=200, tol=1e-10):
    """The amplitude equation of the text, one loop per index.

    t is stored as t[a, b, i, j] with a, b counted from the first particle
    orbital.  Only usable for a handful of orbitals: every iteration costs
    O(n_h^4 n_p^4) Python operations for the quadratic terms alone.
    """
    n = v.shape[0]
    holes = range(n_occ)
    parts = range(n_occ, n)
    n_p = n - n_occ
    t = np.zeros((n_p, n_p, n_occ, n_occ))
    # first guess: MP2
    for a in parts:
        for b in parts:
            for i in holes:
                for j in holes:
                    t[a - n_occ, b - n_occ, i, j] = (
                        v[a, b, i, j] / (eps[i] + eps[j] - eps[a] - eps[b]))
    energy = _naive_energy(t, v, n_occ)
    for iteration in range(1, max_iter + 1):
        t_new = np.zeros_like(t)
        for a in parts:
            for b in parts:
                A, B = a - n_occ, b - n_occ
                for i in holes:
                    for j in holes:
                        r = v[a, b, i, j]
                        for c in parts:
                            for d in parts:
                                r += 0.5 * v[a, b, c, d] * t[c - n_occ, d - n_occ, i, j]
                        for k in holes:
                            for l in holes:
                                r += 0.5 * v[k, l, i, j] * t[A, B, k, l]
                        for k in holes:
                            for c in parts:
                                Cc = c - n_occ
                                r += (v[k, b, c, j] * t[A, Cc, i, k]
                                      - v[k, b, c, i] * t[A, Cc, j, k]
                                      - v[k, a, c, j] * t[B, Cc, i, k]
                                      + v[k, a, c, i] * t[B, Cc, j, k])
                        for k in holes:
                            for l in holes:
                                for c in parts:
                                    for d in parts:
                                        Cc, D = c - n_occ, d - n_occ
                                        w = v[k, l, c, d]
                                        r += 0.25 * w * t[Cc, D, i, j] * t[A, B, k, l]
                                        r += w * (t[A, Cc, i, k] * t[B, D, j, l]
                                                  - t[A, Cc, j, k] * t[B, D, i, l])
                                        r -= 0.5 * w * (t[D, Cc, i, k] * t[A, B, l, j]
                                                        - t[D, Cc, j, k] * t[A, B, l, i])
                                        r -= 0.5 * w * (t[A, Cc, l, k] * t[D, B, i, j]
                                                        - t[B, Cc, l, k] * t[D, A, i, j])
                        t_new[A, B, i, j] = -r / (eps[a] + eps[b] - eps[i] - eps[j])
        t = t_new
        new_energy = _naive_energy(t, v, n_occ)
        if abs(new_energy - energy) < tol:
            energy = new_energy
            break
        energy = new_energy
    return dict(energy=energy, iterations=iteration, t2=t)


def _naive_energy(t, v, n_occ):
    n = v.shape[0]
    e = 0.0
    for a in range(n_occ, n):
        for b in range(n_occ, n):
            for i in range(n_occ):
                for j in range(n_occ):
                    e += 0.25 * v[i, j, a, b] * t[a - n_occ, b - n_occ, i, j]
    return e


# ---------------------------------------------------------------------------
#  Way 3: channels
# ---------------------------------------------------------------------------
class ChannelCCD:
    """CCD with the two-particle configurations sorted into channels.

    A pair (p, q) of orbitals with quantum numbers (m_p, s_p), (m_q, s_q)
    belongs to the pair channel (m_p + m_q, s_p + s_q); a particle-hole pair
    (a, i) to the cross channel (m_a - m_i, s_a - s_i).  The interaction
    conserves both, so every matrix in the CCD equations is block diagonal
    when its rows and columns are sorted by channel, and every contraction
    is a sum of small matrix-matrix products.

      ladders:  sum_{c<d} <ab||cd> t_ij^cd          -> V_pp[k] @ T[k]
                sum_{k<l} t_kl^ab <kl||ij>          -> T[k] @ V_hh[k]
                sum_{k<l} t_kl^ab sum_{c<d} <kl||cd> t_ij^cd
                                                    -> T[k] @ (V_hhpp[k] @ T[k])
      rings:    sum_{kc} t_ik^ac <kb||cj>            -> Tx[k] @ W[k]
                sum_{klcd} <kl||cd> t_ik^ac t_jl^bd  -> Tx[k] @ X[k] @ Tx[-k]^T

    with T[k] the amplitude block of pair channel k (rows a<b, columns i<j)
    and Tx[k] the cross-coupled block, rows (a,i) of cross channel k and
    columns (c,k) of cross channel -k.  The two remaining quadratic terms are
    products with the one-body intermediates F_li and G_ad and are formed
    with dense einsums, since they cost only O(n_h^3 n_p^2) and O(n_h^2 n_p^3).
    """

    def __init__(self, eps, v, n_occ, labels):
        self.eps = np.asarray(eps, dtype=float)
        self.n = len(eps)
        self.n_occ = n_occ
        self.holes = list(range(n_occ))
        self.parts = list(range(n_occ, self.n))
        self.m = np.array([m for (_, m, _) in labels])
        self.s = np.array([s for (_, _, s) in labels])
        self.v_hhpp_dense = v[:n_occ, :n_occ, n_occ:, n_occ:].copy()
        self._build_channels(v)

    # -- bookkeeping -----------------------------------------------------
    def _pair_key(self, p, q):
        return (int(self.m[p] + self.m[q]), int(self.s[p] + self.s[q]))

    def _cross_key(self, a, i):
        return (int(self.m[a] - self.m[i]), int(self.s[a] - self.s[i]))

    def _build_channels(self, v):
        no = self.n_occ
        # pair channels
        self.pp = {}
        for a, b in combinations(self.parts, 2):
            self.pp.setdefault(self._pair_key(a, b), []).append((a, b))
        self.hh = {}
        for i, j in combinations(self.holes, 2):
            self.hh.setdefault(self._pair_key(i, j), []).append((i, j))
        self.pair_keys = [k for k in self.pp if k in self.hh]
        self.V_pp, self.V_hh, self.V_pphh, self.V_hhpp = {}, {}, {}, {}
        for k in self.pair_keys:
            P = np.array(self.pp[k])
            H = np.array(self.hh[k])
            self.V_pp[k] = v[P[:, 0][:, None], P[:, 1][:, None], P[:, 0][None, :], P[:, 1][None, :]]
            self.V_hh[k] = v[H[:, 0][:, None], H[:, 1][:, None], H[:, 0][None, :], H[:, 1][None, :]]
            self.V_pphh[k] = v[P[:, 0][:, None], P[:, 1][:, None], H[:, 0][None, :], H[:, 1][None, :]]
            self.V_hhpp[k] = v[H[:, 0][:, None], H[:, 1][:, None], P[:, 0][None, :], P[:, 1][None, :]]
        # cross channels
        self.ph = {}
        for a in self.parts:
            for i in self.holes:
                self.ph.setdefault(self._cross_key(a, i), []).append((a, i))
        self.cross_keys = [k for k in self.ph if (-k[0], -k[1]) in self.ph]
        self.W, self.X = {}, {}
        for k in self.cross_keys:
            mk = (-k[0], -k[1])
            Bk = np.array(self.ph[mk])            # (c, k) pairs, key -k
            Ak = np.array(self.ph[k])             # (d, l) pairs, key  k
            c, kk = Bk[:, 0], Bk[:, 1]
            # W(ck, bj) = <kb||cj>  with (bj) of key -k, i.e. the same list Bk
            b, j = Bk[:, 0], Bk[:, 1]
            self.W[k] = v[kk[:, None], b[None, :], c[:, None], j[None, :]]
            # X(ck, dl) = <kl||cd>  with (dl) of key +k
            d, l = Ak[:, 0], Ak[:, 1]
            self.X[k] = v[kk[:, None], l[None, :], c[:, None], d[None, :]]
        self.block_sizes = {k: (len(self.pp[k]), len(self.hh[k])) for k in self.pair_keys}

    # -- gather and scatter ----------------------------------------------
    def _pair_blocks(self, t):
        """t[a,b,i,j] (particle indices counted from n_occ) -> {key: block}."""
        no = self.n_occ
        out = {}
        for k in self.pair_keys:
            P = np.array(self.pp[k]) - no
            H = np.array(self.hh[k])
            out[k] = t[P[:, 0][:, None], P[:, 1][:, None], H[:, 0][None, :], H[:, 1][None, :]]
        return out

    def _cross_blocks(self, t):
        """Tx[k](ai, ck) = t_ik^ac, rows of key k, columns of key -k."""
        no = self.n_occ
        out = {}
        for k in self.cross_keys:
            mk = (-k[0], -k[1])
            A = np.array(self.ph[k])
            B = np.array(self.ph[mk])
            a, i = A[:, 0] - no, A[:, 1]
            c, kk = B[:, 0] - no, B[:, 1]
            out[k] = t[a[:, None], c[None, :], i[:, None], kk[None, :]]
        return out

    # -- one iteration -----------------------------------------------------
    def residual(self, t):
        """R_ij^ab = <ab||ij> + all t-dependent terms except the diagonal one."""
        no, n_p = self.n_occ, self.n - self.n_occ
        R = np.zeros_like(t)
        T = self._pair_blocks(t)
        # ladders and the quadratic ladder, in pair channels
        for k in self.pair_keys:
            P = np.array(self.pp[k]) - no
            H = np.array(self.hh[k])
            block = (self.V_pphh[k] + self.V_pp[k] @ T[k] + T[k] @ self.V_hh[k]
                     + T[k] @ (self.V_hhpp[k] @ T[k]))
            a, b = P[:, 0], P[:, 1]
            i, j = H[:, 0], H[:, 1]
            # scatter with the antisymmetry of the amplitudes
            R[a[:, None], b[:, None], i[None, :], j[None, :]] += block
            R[b[:, None], a[:, None], i[None, :], j[None, :]] -= block
            R[a[:, None], b[:, None], j[None, :], i[None, :]] -= block
            R[b[:, None], a[:, None], j[None, :], i[None, :]] += block
        # rings and the quadratic ring, in cross channels
        Tx = self._cross_blocks(t)
        lin = np.zeros((n_p, no, n_p, no))           # lin(a, i, b, j)
        quad = np.zeros((n_p, no, n_p, no))
        for k in self.cross_keys:
            mk = (-k[0], -k[1])
            A = np.array(self.ph[k])
            B = np.array(self.ph[mk])
            a, i = A[:, 0] - no, A[:, 1]
            b, j = B[:, 0] - no, B[:, 1]
            lin[a[:, None], i[:, None], b[None, :], j[None, :]] += Tx[k] @ self.W[k]
            quad[a[:, None], i[:, None], b[None, :], j[None, :]] += Tx[k] @ self.X[k] @ Tx[mk].T
        # P(ij|ab) on the linear ring; P(ij) only on the quadratic one, which
        # is already symmetric under (ai) <-> (bj)
        lin_abij = lin.transpose(0, 2, 1, 3)          # (a, b, i, j)
        quad_abij = quad.transpose(0, 2, 1, 3)
        R += (lin_abij - lin_abij.transpose(0, 1, 3, 2)
              - lin_abij.transpose(1, 0, 2, 3) + lin_abij.transpose(1, 0, 3, 2))
        R += quad_abij - quad_abij.transpose(0, 1, 3, 2)
        # the one-body intermediates
        vh = self.v_hhpp_dense                          # <kl||cd>
        F = 0.5 * np.einsum("klcd,cdik->li", vh, t)     # F_li
        G = 0.5 * np.einsum("klcd,ackl->ad", vh, t)     # G_ad
        term5 = np.einsum("li,ablj->abij", F, t)
        term6 = np.einsum("ad,dbij->abij", G, t)
        R += term5 - term5.transpose(0, 1, 3, 2)
        R += term6 - term6.transpose(1, 0, 2, 3)
        return R

    def energy(self, t):
        return 0.25 * np.einsum("ijab,abij->", self.v_hhpp_dense, t)

    def solve(self, max_iter=400, tol=1e-12, mixing=0.0, verbose=False):
        no, n_p = self.n_occ, self.n - self.n_occ
        e_h, e_p = self.eps[:no], self.eps[no:]
        denominator = (e_p[:, None, None, None] + e_p[None, :, None, None]
                       - e_h[None, None, :, None] - e_h[None, None, None, :])
        t = -self.v_hhpp_dense.transpose(2, 3, 0, 1) / denominator     # MP2 start
        energy = self.energy(t)
        history = [energy]
        for iteration in range(1, max_iter + 1):
            t_new = -self.residual(t) / denominator
            t = (1.0 - mixing) * t_new + mixing * t
            new_energy = self.energy(t)
            history.append(new_energy)
            if verbose:
                print(f"   {iteration:3d}  E_corr = {new_energy:.12f}")
            if abs(new_energy - energy) < tol:
                energy = new_energy
                break
            energy = new_energy
        return dict(energy=energy, iterations=iteration, t2=t, history=history)


# ---------------------------------------------------------------------------
#  Rings, the RPA and TDA (chapter 7)
# ---------------------------------------------------------------------------
def rpa_matrices(eps, v, n_occ):
    """A_{ai,bj} = (e_a - e_i) d d + <aj||ib>,  B_{ai,bj} = <ab||ij>,
    over all particle-hole pairs (a, i) in the HF basis."""
    n = len(eps)
    pairs = [(a, i) for a in range(n_occ, n) for i in range(n_occ)]
    a = np.array([p[0] for p in pairs])
    i = np.array([p[1] for p in pairs])
    A = v[a[:, None], i[None, :], i[:, None], a[None, :]]        # <aj||ib>
    A = A + np.diag(eps[a] - eps[i])
    B = v[a[:, None], a[None, :], i[:, None], i[None, :]]        # <ab||ij>
    return A, B, pairs


def solve_rpa(A, B, tol=1e-8):
    M = np.block([[A, B], [-B, -A]])
    w, U = np.linalg.eig(M)
    n_imag = int(np.sum(np.abs(w.imag) > 1e-7))
    real = w[np.abs(w.imag) <= 1e-7].real
    positive = np.sort(real[real > tol])
    # amplitudes of the positive roots for T = Y X^{-1}
    n = A.shape[0]
    sel = np.where((np.abs(w.imag) <= 1e-7) & (w.real > tol))[0]
    X = U[:n, sel].real
    Y = U[n:, sel].real
    return positive, n_imag, X, Y


def ring_ccd(eps, v, n_occ, max_iter=500, tol=1e-12, mixing=0.3):
    """Ring-CCD from the Riccati equation  B + A T + T A + T B T = 0,
    iterated with the diagonal of A moved to the left-hand side.

    E = (1/2) Tr(B T) is the RPA correlation energy, Eq. (7-rpacorrelation),
    when the equation is solved with the same antisymmetrised matrix elements.
    """
    A, B, pairs = rpa_matrices(eps, v, n_occ)
    d = np.diag(A).copy()
    A_off = A - np.diag(d)
    D = d[:, None] + d[None, :]
    T = -B / D
    energy = 0.5 * np.trace(B @ T)
    converged = False
    with np.errstate(over="ignore", invalid="ignore"):
        for iteration in range(1, max_iter + 1):
            T_new = -(B + A_off @ T + T @ A_off + T @ B @ T) / D
            T = (1.0 - mixing) * T_new + mixing * T
            new_energy = 0.5 * np.trace(B @ T)
            if not np.isfinite(new_energy):        # the ring series diverges
                break
            if abs(new_energy - energy) < tol:
                energy = new_energy
                converged = True
                break
            energy = new_energy
    if not converged:                              # an unstable reference: no solution
        return dict(energy=np.nan, iterations=iteration, T=None, riccati=np.nan,
                    A=A, B=B, converged=False)
    riccati = np.abs(B + A @ T + T @ A + T @ B @ T).max()
    return dict(energy=energy, iterations=iteration, T=T, riccati=riccati, A=A, B=B,
                converged=True)


# ---------------------------------------------------------------------------
#  Two electrons: the exact answer (chapter 5)
# ---------------------------------------------------------------------------
def two_electron_spectrum(h, v, n_states=6):
    """Lowest eigenvalues of two electrons in the basis, from the pair matrix
    of quantumdot.two_electron_fci."""
    n = h.shape[0]
    pairs = list(combinations(range(n), 2))
    p = np.array([a for a, _ in pairs])
    q = np.array([b for _, b in pairs])
    H = v[p[:, None], q[:, None], p[None, :], q[None, :]].copy()
    H += np.where(q[:, None] == q[None, :], h[p[:, None], p[None, :]], 0.0)
    H += np.where(p[:, None] == p[None, :], h[q[:, None], q[None, :]], 0.0)
    H -= np.where(p[:, None] == q[None, :], h[q[:, None], p[None, :]], 0.0)
    H -= np.where(q[:, None] == p[None, :], h[p[:, None], q[None, :]], 0.0)
    return np.linalg.eigvalsh(H)[:n_states]


# ---------------------------------------------------------------------------
#  One complete calculation
# ---------------------------------------------------------------------------
def run(particles, shells, hw=1.0, dense=True, verbose=False):
    """HF, MP2, channel CCD (and dense CCD), ring-CCD and RPA for one dot."""
    out = dict(particles=particles, shells=shells, hw=hw)
    dot = QuantumDot(particles, shells, hw)
    h, v = matrices(shells, hw)
    out["n_orbitals"] = h.shape[0]
    out["E0"] = float(np.sum(dot.energies()[:particles]))
    out["E_ref"] = reference_energy(h, v, particles)
    t0 = time.time()
    h_hf, v_hf, eps, labels, C, it = hartree_fock_blocks(dot, h, v, particles)
    out["t_hf"] = time.time() - t0
    out["E_HF"] = reference_energy(h_hf, v_hf, particles)
    out["hf_iterations"] = it
    f = np.diag(eps)
    out["brillouin"] = float(np.abs(h_hf + np.einsum("piqi->pq", v_hf[:, :particles, :, :particles])
                                    - f)[particles:, :particles].max())
    out["E_MP2"] = mp2_energy(f, v_hf, particles)
    t0 = time.time()
    solver = ChannelCCD(eps, v_hf, particles, labels)
    out["t_channels_setup"] = time.time() - t0
    t0 = time.time()
    res = solver.solve()
    out["t_channels"] = time.time() - t0
    out["E_CCD"] = res["energy"]
    out["ccd_iterations"] = res["iterations"]
    out["channels"] = solver
    out["ccd_history"] = res["history"]
    if dense:
        t0 = time.time()
        dres = ccd_dense(f, v_hf, particles)
        out["t_dense"] = time.time() - t0
        out["E_CCD_dense"] = dres["energy"]
    rc = ring_ccd(eps, v_hf, particles)
    out["E_rCCD"] = rc["energy"]
    out["riccati"] = rc["riccati"]
    omegas, n_imag, X, Y = solve_rpa(rc["A"], rc["B"])
    out["E_RPA"] = 0.5 * (omegas.sum() - np.trace(rc["A"])) if n_imag == 0 else np.nan
    out["rpa_imag"] = n_imag
    out["omega_rpa"] = omegas[:6]
    out["omega_tda"] = np.linalg.eigvalsh(rc["A"])[:6]
    out["stability"] = (np.linalg.eigvalsh(rc["A"] - rc["B"]).min(),
                        np.linalg.eigvalsh(rc["A"] + rc["B"]).min())
    if particles == 2:
        out["exact"] = two_electron_spectrum(h, v)
    out["eps"] = eps
    return out


def _demo():
    print("=" * 74)
    print("1. Two electrons, hw = 1: three CCD codes and the exact answer")
    print("=" * 74)
    for shells in (1, 2):
        dot = QuantumDot(2, shells)
        h, v = matrices(shells)
        h_hf, v_hf, eps, labels, C, it = hartree_fock_blocks(dot, h, v, 2)
        f = np.diag(eps)
        t0 = time.time(); naive = ccd_naive(eps, v_hf, 2); t_naive = time.time() - t0
        t0 = time.time(); dense = ccd_dense(f, v_hf, 2); t_dense = time.time() - t0
        solver = ChannelCCD(eps, v_hf, 2, labels)
        t0 = time.time(); chan = solver.solve(); t_chan = time.time() - t0
        exact = two_electron_spectrum(h, v)[0]
        E_hf = reference_energy(h_hf, v_hf, 2)
        print(f"shells 0-{shells}, {h.shape[0]} orbitals: E_HF = {E_hf:.8f}")
        print(f"   naive    E_corr = {naive['energy']:.12f}  ({t_naive:7.2f} s, {naive['iterations']} it.)")
        print(f"   dense    E_corr = {dense['energy']:.12f}  ({t_dense:7.2f} s)")
        print(f"   channels E_corr = {chan['energy']:.12f}  ({t_chan:7.2f} s)")
        print(f"   FCI (chapter 5): E = {exact:.8f}, E_HF + E_CCD = {E_hf + chan['energy']:.8f}, "
              f"difference {E_hf + chan['energy'] - exact:.1e}")

    print()
    print("=" * 74)
    print("2. Closed shells at hw = 1: HF, MP2, CCD, ring-CCD and RPA")
    print("=" * 74)
    print(f"{'N':>3s} {'shells':>6s} {'norb':>5s} {'E_HF':>13s} {'E_MP2':>11s} {'E_CCD':>11s} "
          f"{'E_rCCD':>11s} {'E_RPA':>11s} {'|dense-chan|':>12s} {'t_chan':>7s} {'t_dense':>8s}")
    for N, shells in ((2, 5), (6, 5), (12, 5), (20, 5)):
        r = run(N, shells)
        print(f"{N:3d} {shells:6d} {r['n_orbitals']:5d} {r['E_HF']:13.8f} {r['E_MP2']:11.8f} "
              f"{r['E_CCD']:11.8f} {r['E_rCCD']:11.8f} {r['E_RPA']:11.8f} "
              f"{abs(r['E_CCD'] - r['E_CCD_dense']):12.1e} {r['t_channels']:7.2f} {r['t_dense']:8.2f}")
        print(f"      Riccati residual {r['riccati']:.1e}, imaginary RPA roots {r['rpa_imag']}, "
              f"min eig(A-B), (A+B) = {r['stability'][0]:.4f}, {r['stability'][1]:.4f}")
        sizes = sorted(r["channels"].block_sizes.values(), reverse=True)
        print(f"      {len(sizes)} pair channels, largest blocks (pp x hh): {sizes[:3]}")


if __name__ == "__main__":
    _demo()

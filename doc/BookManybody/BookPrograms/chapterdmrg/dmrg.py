"""
The density matrix renormalisation group for the models of this book.

Companion code to the chapter on the density matrix renormalisation group
of *Quantum mechanics for Many-particle Systems*.

The chain
---------
Every model is written on a chain of L sites, each site carrying two
fermionic modes -- the two spin projections of a level in the pairing
models, or the two spin orientations of a lattice site in the Hubbard chain
-- so that the local dimension is d = 4 with the basis |0>, |a>, |b>, |ab>.
The Hamiltonian is a list of *terms*, each a coefficient times an ordered
product of creation and annihilation operators on the 2L modes,

    (coefficient, [(mode, dagger), (mode, dagger), ...]) ,

and this one list feeds both solvers: the exact diagonalisation in the
occupation basis (bit strings, as in chapter 5) and the matrix product
operator of the DMRG.  Nothing is transcribed twice.

What is here
------------
    pairing_terms, particle_hole_terms, hubbard_terms, number_penalty_terms
        -- the models of chapter 4 as term lists
    exact_ground_state   -- sparse exact diagonalisation in a fixed sector
    build_mpo            -- the matrix product operator of a term list,
                            Jordan-Wigner strings included, compressed
                            bond by bond with the SVD to its minimal bond
                            dimension
    MPS                  -- a matrix product state, with canonical forms,
                            Schmidt spectra and expectation values
    DMRG                 -- the two-site algorithm: environments, the
                            effective Hamiltonian applied matrix-free to a
                            Lanczos solver, SVD truncation, sweeps
    lieb_wu_energy       -- the Bethe-ansatz energy per site of the
                            infinite half-filled Hubbard chain

The particle number is fixed by a penalty lambda (N - N_0)^2 added to the
Hamiltonian, which is the simplest thing that works and costs one extra
bond in the MPO; a production code would use the U(1) block structure
instead.

Author: Morten Hjorth-Jensen
"""

from itertools import combinations

import numpy as np
from scipy.linalg import eigh, svd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

# ---------------------------------------------------------------------------
#  One site: two fermionic modes, local dimension four
# ---------------------------------------------------------------------------
D_LOCAL = 4
_S = np.array([[0.0, 1.0], [0.0, 0.0]])          # annihilation on one mode
_Z = np.diag([1.0, -1.0])                        # (-1)^n on one mode
_I2 = np.eye(2)

# mode a is the first mode of the site, mode b the second; inside the site
# the Jordan-Wigner string of mode b is the parity of mode a
C_A = np.kron(_S, _I2)
C_B = np.kron(_Z, _S)
CDAG_A, CDAG_B = C_A.T.copy(), C_B.T.copy()
N_A, N_B = CDAG_A @ C_A, CDAG_B @ C_B
PARITY = np.kron(_Z, _Z)                         # (-1)^(n_a + n_b)
IDENTITY = np.eye(D_LOCAL)


def local_operator(dagger, which):
    if which == 0:
        return CDAG_A if dagger else C_A
    return CDAG_B if dagger else C_B


def mode(site, which):
    """Mode index of (site, which) with which = 0 (a) or 1 (b); sites 0..L-1."""
    return 2 * site + which


# ---------------------------------------------------------------------------
#  The models as term lists
# ---------------------------------------------------------------------------
def pairing_terms(L, g, xi=1.0):
    """H = xi sum_{p sigma} (p-1) n_{p sigma} - (g/2) sum_{pq} P+_p P-_q.

    Level p = 1..L is site p-1; spin + is mode a, spin - is mode b.
    """
    terms = []
    for p in range(L):
        for s in (0, 1):
            m = mode(p, s)
            terms.append((xi * p, [(m, True), (m, False)]))
    for p in range(L):
        for q in range(L):
            terms.append((-0.5 * g, [(mode(p, 0), True), (mode(p, 1), True),
                                     (mode(q, 1), False), (mode(q, 0), False)]))
    return terms


def particle_hole_terms(L, f):
    """V_ph = -(f/2) sum_{pqr} ( a+_{p+} a+_{p-} a_{q-} a_{r+} + h.c. )."""
    terms = []
    for p in range(L):
        for q in range(L):
            for r in range(L):
                ops = [(mode(p, 0), True), (mode(p, 1), True),
                       (mode(q, 1), False), (mode(r, 0), False)]
                terms.append((-0.5 * f, ops))
                # the Hermitian conjugate: reverse the order, flip daggers
                terms.append((-0.5 * f, [(m, not dg) for m, dg in ops[::-1]]))
    return terms


def hubbard_terms(L, t=1.0, U=4.0, pbc=False):
    """H = -t sum_{<ij> sigma} (c+_{i sigma} c_{j sigma} + h.c.) + U sum_i n_up n_down.

    Site i carries spin up as mode a and spin down as mode b.
    """
    terms = []
    bonds = [(i, i + 1) for i in range(L - 1)]
    if pbc and L > 2:
        bonds.append((L - 1, 0))
    for i, j in bonds:
        for s in (0, 1):
            terms.append((-t, [(mode(i, s), True), (mode(j, s), False)]))
            terms.append((-t, [(mode(j, s), True), (mode(i, s), False)]))
    for i in range(L):
        terms.append((U, [(mode(i, 0), True), (mode(i, 0), False),
                          (mode(i, 1), True), (mode(i, 1), False)]))
    return terms


def number_penalty_terms(L, n_target, lam):
    """lambda (N - N_0)^2 = lambda ( sum_mm' n_m n_m' - 2 N_0 sum_m n_m + N_0^2 )."""
    terms = []
    modes = range(2 * L)
    for m in modes:
        for mp in modes:
            terms.append((lam, [(m, True), (m, False), (mp, True), (mp, False)]))
    for m in modes:
        terms.append((-2.0 * lam * n_target, [(m, True), (m, False)]))
    terms.append((lam * n_target**2, []))
    return terms


def number_terms(L):
    return [(1.0, [(m, True), (m, False)]) for m in range(2 * L)]


def occupation_terms(site, which):
    m = mode(site, which)
    return [(1.0, [(m, True), (m, False)])]


# ---------------------------------------------------------------------------
#  Exact diagonalisation in the occupation basis, from the same term list
# ---------------------------------------------------------------------------
def popcount(x):
    return bin(x).count("1")


def apply_mode_string(state, ops):
    """Apply an ordered product of mode operators to a bit string.

    ``ops`` is listed as written, leftmost first; the rightmost operator acts
    first.  The Jordan-Wigner sign is the parity of the occupied modes below
    the one acted on.  Returns (sign, new state) or (0, None).
    """
    sign = 1
    for m, dagger in reversed(ops):
        occupied = (state >> m) & 1
        if dagger == bool(occupied):
            return 0, None
        if popcount(state & ((1 << m) - 1)) & 1:
            sign = -sign
        state ^= 1 << m
    return sign, state


def sector_states(L, n_particles, n_a=None):
    """Bit strings with n_particles among the 2L modes (optionally n_a in mode a)."""
    out = []
    for occ in combinations(range(2 * L), n_particles):
        if n_a is not None and sum(1 for m in occ if m % 2 == 0) != n_a:
            continue
        bits = 0
        for m in occ:
            bits |= 1 << m
        out.append(bits)
    return out


def sparse_hamiltonian(terms, L, states):
    index = {s: i for i, s in enumerate(states)}
    rows, cols, data = [], [], []
    for col, state in enumerate(states):
        for coeff, ops in terms:
            sign, new = apply_mode_string(state, ops)
            if sign and new in index:
                rows.append(index[new])
                cols.append(col)
                data.append(coeff * sign)
    return csr_matrix((data, (rows, cols)), shape=(len(states), len(states)))


def exact_ground_state(terms, L, n_particles, n_a=None, k=1):
    """Lowest k energies of the term list in a fixed particle-number sector."""
    states = sector_states(L, n_particles, n_a)
    H = sparse_hamiltonian(terms, L, states)
    if H.shape[0] <= 400:
        w = np.linalg.eigvalsh(H.toarray())[:k]
    else:
        w = np.sort(eigsh(H, k=k, which="SA")[0])
    return w if k > 1 else float(w[0])


# ---------------------------------------------------------------------------
#  Matrix product operators from operator strings
# ---------------------------------------------------------------------------
def site_operators(ops, L):
    """The per-site 4x4 matrices of one operator string, strings included.

    A mode operator on site q carries the Jordan-Wigner parity of every site
    to its left.  Writing the string as a product, leftmost first, of factors
    that are tensor products of site matrices, the matrix on site s is the
    product of that site's factors in the same order: the parity operator
    for every fermionic factor to its right on the chain, the local operator
    for a factor on the site itself, and the identity otherwise.
    """
    mats = [IDENTITY.copy() for _ in range(L)]
    for m, dagger in ops:
        q, which = divmod(m, 2)
        o = local_operator(dagger, which)
        for s in range(q):
            mats[s] = mats[s] @ PARITY
        mats[q] = mats[q] @ o
    return mats


def build_mpo(terms, L, tol=1e-12):
    """The MPO W[0..L-1] of a term list, W[i] of shape (Dl, Dr, d, d').

    The sum over operator strings is compressed bond by bond.  At bond s
    the terms are grouped by their operator string to the *right* of the
    bond, so that terms sharing a right string are added together; the
    matrix whose rows are (left index, s, s') and whose columns are the
    distinct right strings is then factorised by the SVD.  The left factor
    becomes the site tensor and the right factor carries the coefficients
    of each right string to the next site.  A final right-to-left sweep
    removes any remaining redundancy, so the bond dimensions are the ranks
    of the bipartitions of the operator -- the minimal MPO of the sum.
    """
    K = len(terms)
    d = D_LOCAL
    per_site = [site_operators(ops, L) for _, ops in terms]
    # a hashable label of every site matrix, to recognise equal strings
    labels = [[np.round(m, 10).tobytes() for m in mats] for mats in per_site]
    coeffs = np.array([c for c, _ in terms], dtype=float)
    # columns of the current left factor: one per distinct right string
    # (from site s onwards); ``columns`` maps that string to a column index
    groups = {}
    for k in range(K):
        groups.setdefault(tuple(labels[k]), []).append(k)
    keys = list(groups)
    left = np.zeros((1, len(keys)))
    for j, key in enumerate(keys):
        left[0, j] = coeffs[groups[key]].sum()
    W = []
    for s in range(L):
        Dl = left.shape[0]
        # the operator on site s of every current column, and the right
        # string beyond site s, which defines the next grouping
        new_groups = {}
        for j, key in enumerate(keys):
            new_groups.setdefault(key[1:], []).append(j)
        new_keys = list(new_groups)
        M = np.zeros((Dl, d, d, len(new_keys)))
        for jn, nkey in enumerate(new_keys):
            for j in new_groups[nkey]:
                op = np.frombuffer(keys[j][0], dtype=float).reshape(d, d)
                M[:, :, :, jn] += left[:, j, None, None] * op
        M = M.reshape(Dl * d * d, len(new_keys))
        if s == L - 1:
            W.append(M.sum(axis=1).reshape(Dl, d, d, 1).transpose(0, 3, 1, 2))
            break
        U, S, Vt = svd(M, full_matrices=False)
        keep = S > tol * S[0]
        r = int(keep.sum())
        W.append(U[:, :r].reshape(Dl, d, d, r).transpose(0, 3, 1, 2))
        left = S[:r, None] * Vt[:r]
        keys = new_keys
    return compress_mpo(W, tol)


def compress_mpo(W, tol=1e-12):
    """Right-to-left SVD compression of an MPO to its minimal bond dimensions.

    The left-to-right construction in ``build_mpo`` counts every term that
    is still distinct on the right as a separate column, which overestimates
    the rank when several terms share the same operator string to the right
    of a bond.  Sweeping back with the SVD -- the MPO treated as an MPS with
    a d^2-dimensional physical index -- removes that redundancy.
    """
    W = [w.copy() for w in W]
    d = W[0].shape[2]
    for s in range(len(W) - 1, 0, -1):
        Dl, Dr = W[s].shape[0], W[s].shape[1]
        M = W[s].transpose(0, 2, 3, 1).reshape(Dl, d * d * Dr)
        U, S, Vt = svd(M, full_matrices=False)
        keep = S > tol * S[0]
        r = int(keep.sum())
        W[s] = Vt[:r].reshape(r, d, d, Dr).transpose(0, 3, 1, 2)
        W[s - 1] = np.einsum("abst,br->arst", W[s - 1], U[:, :r] * S[:r])
    return W


def mpo_bond_dimensions(W):
    return [w.shape[1] for w in W[:-1]]


def add_terms(*lists):
    out = []
    for t in lists:
        out.extend(t)
    return out


# ---------------------------------------------------------------------------
#  Matrix product states
# ---------------------------------------------------------------------------
class MPS:
    """A matrix product state on L sites, tensors of shape (chi_l, d, chi_r)."""

    def __init__(self, tensors):
        self.A = list(tensors)
        self.L = len(self.A)

    @classmethod
    def random(cls, L, chi, d=D_LOCAL, seed=1):
        rng = np.random.default_rng(seed)
        dims = [1] + [min(chi, d**min(i, L - i)) for i in range(1, L)] + [1]
        A = [rng.normal(size=(dims[i], d, dims[i + 1])) for i in range(L)]
        mps = cls(A)
        mps.right_canonicalize()
        return mps

    @classmethod
    def product_state(cls, occupations):
        """A product state from a list of local basis indices 0..3."""
        A = []
        for n in occupations:
            t = np.zeros((1, D_LOCAL, 1))
            t[0, n, 0] = 1.0
            A.append(t)
        return cls(A)

    # ------------------------------------------------------------------
    def right_canonicalize(self):
        """Bring every tensor into right-canonical form, normalising the state."""
        for i in range(self.L - 1, 0, -1):
            chi_l, d, chi_r = self.A[i].shape
            M = self.A[i].reshape(chi_l, d * chi_r)
            Q, R = np.linalg.qr(M.T)                 # M = R^T Q^T
            self.A[i] = Q.T.reshape(-1, d, chi_r)
            self.A[i - 1] = np.tensordot(self.A[i - 1], R.T, axes=(2, 0))
        self.A[0] /= np.linalg.norm(self.A[0])

    def norm(self):
        E = np.ones((1, 1))
        for A in self.A:
            E = np.einsum("ab,asc,bsd->cd", E, A, A)
        return float(np.sqrt(E[0, 0]))

    def dense(self):
        """The full state vector, for small chains only."""
        psi = self.A[0]
        for A in self.A[1:]:
            psi = np.tensordot(psi, A, axes=(-1, 0))
        return psi.reshape(-1)

    def bond_dimensions(self):
        return [A.shape[2] for A in self.A[:-1]]

    # ------------------------------------------------------------------
    def expectation(self, W):
        """<psi|O|psi> for an MPO O of the same length."""
        E = np.ones((1, 1, 1))                       # (chi, D, chi)
        for A, w in zip(self.A, W):
            E = np.einsum("abc,asd->bcsd", E, A)        # contract ket
            E = np.einsum("bcsd,bets->cetd", E, w)      # contract MPO
            E = np.einsum("cetd,ctf->dfe", E, A).transpose(0, 2, 1)
        return float(E[0, 0, 0])

    def expectation_terms(self, terms):
        return self.expectation(build_mpo(terms, self.L))

    def schmidt_values(self, bond):
        """Singular values across the bond between sites bond-1 and bond."""
        psi = self.A[0]
        for A in self.A[1:bond]:
            psi = np.tensordot(psi, A, axes=(-1, 0))
        left = psi.reshape(-1, psi.shape[-1])
        right = self.A[bond]
        for A in self.A[bond + 1:]:
            right = np.tensordot(right, A, axes=(-1, 0))
        right = right.reshape(right.shape[0], -1)
        # the state across the bond is left @ right; use the QR of both
        Ql, Rl = np.linalg.qr(left)
        Qr, Rr = np.linalg.qr(right.T)
        return svd(Rl @ Rr.T, compute_uv=False)


def entanglement_entropy(singular_values):
    p = singular_values**2
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


# ---------------------------------------------------------------------------
#  The two-site DMRG
# ---------------------------------------------------------------------------
def lanczos_lowest(matvec, v0, m=40, tol=1e-9, restarts=3):
    """The lowest eigenpair by the Lanczos algorithm of chapter 1.

    A Krylov space is built from v0 with full reorthogonalisation and the
    tridiagonal matrix is diagonalised after every step; the iteration
    stops as soon as the residual estimate beta_j |s_j| of the lowest Ritz
    pair -- s_j being the last component of the Ritz vector -- falls below
    ``tol``, or after m vectors.  Because the DMRG supplies a good starting
    vector after the first sweep, a handful of iterations usually suffice;
    if the residual is still large the process is restarted from the Ritz
    vector, at most ``restarts`` times.
    """
    from scipy.linalg import eigh_tridiagonal
    v = v0 / np.linalg.norm(v0)
    for _ in range(restarts):
        V = [v]
        alpha, beta = [], []
        w = matvec(v)
        converged = False
        for j in range(m):
            a = float(np.dot(V[j], w))
            alpha.append(a)
            w = w - a * V[j] - (beta[-1] * V[j - 1] if j > 0 else 0.0)
            for u in V:                                # reorthogonalise
                w = w - np.dot(u, w) * u
            b = float(np.linalg.norm(w))
            ev, evec = eigh_tridiagonal(alpha, beta, select="i",
                                        select_range=(0, 0))
            if b < 1e-12 or b * abs(evec[-1, 0]) < tol:
                converged = True
                break
            if j == m - 1:
                break
            beta.append(b)
            V.append(w / b)
            w = matvec(V[-1])
        v = sum(c * u for c, u in zip(evec[:, 0], V))
        v /= np.linalg.norm(v)
        if converged:
            break
    return float(ev[0]), v


class DMRG:
    """Two-site DMRG for an MPO W, keeping at most chi_max Schmidt states.

    The state is kept in mixed-canonical form around the two sites being
    optimised.  The left environments LE[i] and right environments RE[i]
    are three-index tensors (chi, D, chi) that summarise everything to the
    left and right; the effective Hamiltonian is never formed, only applied
    to a two-site tensor inside a Lanczos solver.
    """

    def __init__(self, W, chi_max=32, mps=None, seed=1):
        self.W = W
        self.L = len(W)
        self.chi_max = chi_max
        self.mps = mps if mps is not None else MPS.random(self.L, min(chi_max, 8),
                                                          seed=seed)
        self.mps.right_canonicalize()
        self.LE = [None] * (self.L + 1)
        self.RE = [None] * (self.L + 1)
        self.LE[0] = np.ones((1, 1, 1))
        self.RE[self.L] = np.ones((1, 1, 1))
        for i in range(self.L - 1, 0, -1):
            self.RE[i] = self._contract_right(self.RE[i + 1], self.mps.A[i], self.W[i])
        self.energies = []
        self.discarded = []
        self.entropies = np.zeros(self.L - 1)
        self.schmidt = [None] * (self.L - 1)

    # ------------------------------------------------------------------
    @staticmethod
    def _contract_left(E, A, w):
        """E'[c,e,f] = sum E[a,b,d] A[a,s,c] w[b,e,s,t] A[d,t,f]."""
        T = np.einsum("abd,asc->bdsc", E, A)
        T = np.einsum("bdsc,bets->dtce", T, w)
        return np.einsum("dtce,dtf->cef", T, A)

    @staticmethod
    def _contract_right(E, A, w):
        """E'[a,b,d] = sum A[a,s,c] w[b,e,s,t] A[d,t,f] E[c,e,f]."""
        T = np.einsum("asc,cef->asef", A, E)
        T = np.einsum("asef,bets->abtf", T, w)
        return np.einsum("abtf,dtf->abd", T, A)

    def _effective(self, i, theta_shape):
        LE, RE = self.LE[i], self.RE[i + 2]
        w1, w2 = self.W[i], self.W[i + 1]

        def matvec(v):
            # environments carry (ket bond, MPO bond, bra bond); the MPO
            # tensors carry (Dl, Dr, out, in), so the ket indices of theta
            # are contracted with the "in" indices and the result is
            # indexed by the bra bonds and the "out" physical indices
            th = v.reshape(theta_shape)
            T = np.tensordot(LE, th, axes=(0, 0))          # (b, d, s, t, f)
            T = np.tensordot(T, w1, axes=((0, 2), (0, 3)))  # (d, t, f, c, u)
            T = np.tensordot(T, w2, axes=((3, 1), (0, 3)))  # (d, f, u, e, v)
            T = np.tensordot(T, RE, axes=((1, 3), (0, 1)))  # (d, u, v, g)
            return T.reshape(-1)

        dim = int(np.prod(theta_shape))
        return LinearOperator((dim, dim), matvec=matvec, dtype=float), matvec

    def _optimise(self, i, theta):
        shape = theta.shape
        op, matvec = self._effective(i, shape)
        dim = op.shape[0]
        if dim <= 1000:
            LE, RE = self.LE[i], self.RE[i + 2]
            Hd = np.einsum("abd,bcus,cevt,feg->duvgastf", LE, self.W[i],
                           self.W[i + 1], RE, optimize=True).reshape(dim, dim)
            Hd = 0.5 * (Hd + Hd.T)
            w, v = eigh(Hd, subset_by_index=[0, 0])
            return float(w[0]), v[:, 0].reshape(shape)
        E, v = lanczos_lowest(matvec, theta.reshape(-1))
        return E, v.reshape(shape)

    def _split(self, i, theta, going_right):
        chi_l, d, _, chi_r = theta.shape
        U, S, Vt = svd(theta.reshape(chi_l * d, d * chi_r), full_matrices=False)
        keep = min(self.chi_max, int(np.sum(S > 1e-12)))
        weight = float(np.sum(S[keep:]**2))
        S_k = S[:keep] / np.linalg.norm(S[:keep])
        self.discarded_sweep = max(self.discarded_sweep, weight)
        self.entropies[i] = entanglement_entropy(S_k)
        self.schmidt[i] = S_k
        A = U[:, :keep].reshape(chi_l, d, keep)
        B = Vt[:keep].reshape(keep, d, chi_r)
        if going_right:
            self.mps.A[i] = A
            self.mps.A[i + 1] = np.tensordot(np.diag(S_k), B, axes=(1, 0))
            self.LE[i + 1] = self._contract_left(self.LE[i], A, self.W[i])
        else:
            self.mps.A[i + 1] = B
            self.mps.A[i] = np.tensordot(A, np.diag(S_k), axes=(2, 0))
            self.RE[i + 1] = self._contract_right(self.RE[i + 2], B, self.W[i + 1])

    def sweep(self):
        """One left-to-right and one right-to-left pass; returns the energy."""
        self.discarded_sweep = 0.0
        energy = np.nan
        for i in range(self.L - 1):
            theta = np.tensordot(self.mps.A[i], self.mps.A[i + 1], axes=(2, 0))
            energy, theta = self._optimise(i, theta)
            self._split(i, theta, going_right=True)
        for i in range(self.L - 2, -1, -1):
            theta = np.tensordot(self.mps.A[i], self.mps.A[i + 1], axes=(2, 0))
            energy, theta = self._optimise(i, theta)
            self._split(i, theta, going_right=False)
        self.energies.append(energy)
        self.discarded.append(self.discarded_sweep)
        return energy

    def run(self, max_sweeps=12, tol=1e-8, verbose=False):
        previous = np.inf
        for n in range(1, max_sweeps + 1):
            E = self.sweep()
            if verbose:
                print(f"  sweep {n:2d}  E = {E:.12f}  "
                      f"max discarded weight = {self.discarded_sweep:.1e}  "
                      f"chi = {max(self.mps.bond_dimensions())}")
            if abs(E - previous) < tol:
                break
            previous = E
        self.mps.right_canonicalize()
        return E


def dmrg_ground_state(terms, L, n_particles, chi_max=32, lam=10.0,
                      max_sweeps=12, tol=1e-8, seed=1, verbose=False):
    """Ground state of a term list in the sector with n_particles particles.

    Returns the DMRG object; its energy is the energy of the Hamiltonian
    alone (the penalty vanishes in the target sector).
    """
    full = add_terms(terms, number_penalty_terms(L, n_particles, lam))
    solver = DMRG(build_mpo(full, L), chi_max=chi_max, seed=seed)
    solver.run(max_sweeps=max_sweeps, tol=tol, verbose=verbose)
    solver.energy = solver.mps.expectation_terms(terms)
    solver.particles = solver.mps.expectation_terms(number_terms(L))
    return solver


# ---------------------------------------------------------------------------
#  The Bethe-ansatz reference for the Hubbard chain
# ---------------------------------------------------------------------------
def lieb_wu_energy(U, t=1.0):
    """Ground-state energy per site of the infinite half-filled Hubbard chain,

        e(U) = -4t int_0^inf J_0(w) J_1(w) / ( w (1 + exp(w U / 2t)) ) dw .
    """
    from scipy.integrate import quad
    from scipy.special import j0, j1
    f = lambda w: j0(w) * j1(w) / (w * (1.0 + np.exp(0.5 * w * U / t)))
    value, _ = quad(f, 1e-12, 200.0, limit=400)
    return -4.0 * t * value


# ---------------------------------------------------------------------------
#  Demonstrations
# ---------------------------------------------------------------------------
def demo_mpo():
    print("=" * 74)
    print("1. Matrix product operators of the three models")
    print("=" * 74)
    L = 6
    for name, terms in (("pairing, g = 1", pairing_terms(L, 1.0)),
                        ("pairing + particle-hole, g = 1, f = 0.5",
                         add_terms(pairing_terms(L, 1.0), particle_hole_terms(L, 0.5))),
                        ("Hubbard chain, open, U = 4", hubbard_terms(L, 1.0, 4.0)),
                        ("Hubbard ring, periodic, U = 4", hubbard_terms(L, 1.0, 4.0, pbc=True)),
                        ("number penalty (N - 6)^2", number_penalty_terms(L, 6, 1.0))):
        W = build_mpo(terms, L)
        print(f"  {name:42s} terms = {len(terms):5d}   "
              f"MPO bond dimensions {mpo_bond_dimensions(W)}")
    print()
    print("The pairing interaction is a product of two sums, sum_p P+_p times")
    print("sum_q P-_q, and needs bond dimension 4 (the four automaton states:")
    print("nothing placed, P+ placed, P- placed, done); the particle-hole")
    print("term carries Jordan-Wigner strings and needs more.  A periodic")
    print("bond joins the first and last site and is carried along the whole")
    print("chain, which is why the ring costs more than the open chain.")
    print()
    print("The MPO reproduces the Hamiltonian: for L = 3 the dense matrix of")
    print("the pairing + particle-hole MPO against exact diagonalisation,")
    L = 3
    terms = add_terms(pairing_terms(L, 1.0), particle_hole_terms(L, 0.5))
    H = mpo_dense(build_mpo(terms, L))
    states = sector_states(L, 3)
    Hs = sparse_hamiltonian(terms, L, states).toarray()
    idx = np.array([chain_index(st, L) for st in states])
    print(f"  max |H_MPO - H_ED| in the N = 3 sector = "
          f"{np.abs(H[np.ix_(idx, idx)] - Hs).max():.1e}")
    print(f"  Hermitian: max |H - H^T| = {np.abs(H - H.T).max():.1e}")


def mpo_dense(W):
    """The dense 4^L x 4^L matrix of an MPO, for tiny chains only.

    The row and column index is the chain index of ``chain_index``: site 0
    is the most significant digit, each site contributing 2 n_a + n_b.
    """
    H = W[0][0]                                        # (D, d, d)
    for w in W[1:]:
        m, n = H.shape[1], H.shape[2]
        H = np.einsum("bst,bcuv->csutv", H, w).reshape(w.shape[1], m * 4, n * 4)
    return H[0]


def chain_index(state, L):
    """The dense-MPO index of an occupation bit string of the 2L modes."""
    idx = 0
    for k in range(L):
        n_a, n_b = (state >> (2 * k)) & 1, (state >> (2 * k + 1)) & 1
        idx = 4 * idx + 2 * n_a + n_b
    return idx


def demo_pairing():
    print("=" * 74)
    print("2. The pairing model: Table 4.2 by DMRG, and the entropy of chapter 1")
    print("=" * 74)
    L, N = 4, 4
    print(f"{'g':>6s} {'E_exact (chapter 4)':>20s} {'E_DMRG':>14s} {'error':>10s} "
          f"{'<N>':>8s} {'chi':>4s}")
    reference = {0.5: 1.41677428, 1.0: 0.63554847}
    for g in (0.5, 1.0):
        terms = pairing_terms(L, g)
        solver = dmrg_ground_state(terms, L, N, chi_max=16)
        e_ed = exact_ground_state(terms, L, N)
        print(f"{g:6.2f} {reference[g]:20.8f} {solver.energy:14.8f} "
              f"{solver.energy - e_ed:10.1e} {solver.particles:8.4f} "
              f"{max(solver.mps.bond_dimensions()):4d}")
    print("With chi_max = 16 = 4^(L/2) an MPS can represent any state of four")
    print("sites exactly; the DMRG reproduces the seniority-zero energies of")
    print("Table 4.2 with a bond dimension of only 4, while working in the")
    print("full Fock space of 4^4 = 256 states without being told that the")
    print("ground state has seniority zero.")
    print()
    L, N = 12, 12
    g = 1.0
    print(f"Twelve levels, six pairs, g = {g} in the convention of chapter 4")
    print("(delta = 1, g = 0.5 in the convention of chapter 1):")
    from itertools import combinations as comb
    # seniority-zero reference, as in models.PairingModel
    basis = list(comb(range(L), 6))
    index = {c: i for i, c in enumerate(basis)}
    Hp = np.zeros((len(basis), len(basis)))
    for c, i in index.items():
        occ = set(c)
        Hp[i, i] += 2.0 * sum(c) - 0.5 * g * 6
        for q in c:
            for p in range(L):
                if p not in occ:
                    Hp[index[tuple(sorted(occ - {q} | {p}))], i] -= 0.5 * g
    e_ref = np.linalg.eigvalsh(Hp)[0]
    for chi in (8, 16, 32):
        solver = dmrg_ground_state(pairing_terms(L, g), L, N, chi_max=chi,
                                   max_sweeps=10)
        S_mid = solver.entropies[L // 2 - 1]
        print(f"  chi = {chi:3d}:  E = {solver.energy:.10f}   error "
              f"{solver.energy - e_ref:9.1e}   discarded weight "
              f"{solver.discarded[-1]:8.1e}   S(middle cut) = {S_mid:.4f}")
    print(f"  seniority-zero diagonalisation (924 states): E = {e_ref:.10f}")
    print("  chapter 1, Schmidt spectrum of the pairing model: entropy of the middle cut 0.9238")
    print("The Schmidt spectrum of chapter 1 was computed from the exact state;")
    print("the DMRG never had it, and finds the same entropy to four digits.")


def demo_particle_hole():
    print("=" * 74)
    print("3. The pairing plus particle-hole model")
    print("=" * 74)
    print("Four levels, four particles, the couplings of chapter 7:")
    print(f"{'g':>6s} {'f':>6s} {'E_exact':>14s} {'E_DMRG':>14s} {'error':>10s}")
    L, N = 4, 4
    for g, f in ((0.5, 0.025), (0.7, 0.05), (1.0, 0.05), (1.0, 0.5)):
        terms = add_terms(pairing_terms(L, g), particle_hole_terms(L, f))
        e_ed = exact_ground_state(terms, L, N)
        solver = dmrg_ground_state(terms, L, N, chi_max=16)
        print(f"{g:6.2f} {f:6.3f} {e_ed:14.8f} {solver.energy:14.8f} "
              f"{solver.energy - e_ed:10.1e}")
    print()
    L, N = 8, 8
    g, f = 1.0, 0.5
    terms = add_terms(pairing_terms(L, g), particle_hole_terms(L, f))
    print(f"Eight levels, eight particles, g = {g}, f = {f}: the full N = 8")
    print(f"sector has {len(sector_states(L, N))} determinants, the S_z = 0")
    print(f"sector {len(sector_states(L, N, N // 2))}.  Exact diagonalisation "
          "against DMRG:")
    e_ed = exact_ground_state(terms, L, N, n_a=N // 2)
    print(f"  exact: E = {e_ed:.10f}")
    print(f"  {'chi':>4s} {'E_DMRG':>16s} {'error':>10s} {'discarded':>10s} "
          f"{'S(middle)':>10s}")
    for chi in (4, 8, 16, 32, 64):
        solver = dmrg_ground_state(terms, L, N, chi_max=chi, max_sweeps=10)
        print(f"  {chi:4d} {solver.energy:16.10f} {solver.energy - e_ed:10.1e} "
              f"{solver.discarded[-1]:10.1e} {solver.entropies[L // 2 - 1]:10.4f}")
    print("The error is variational -- always positive -- and tracks the")
    print("discarded weight, which is what makes the extrapolation to zero")
    print("discarded weight the standard way of quoting a DMRG energy.")


def demo_hubbard():
    print("=" * 74)
    print("4. The Hubbard chain")
    print("=" * 74)
    print("The half-filled four-site ring of Table 4.3, periodic:")
    print(f"{'U/t':>5s} {'E_0 (table)':>14s} {'E_DMRG':>14s} {'error':>10s} "
          f"{'double occ.':>12s}")
    table = {2: -2.82842712, 4: -2.10274848, 8: -1.32023496, 16: -0.72286432}
    L, N = 4, 4
    for U in (2, 4, 8, 16):
        terms = hubbard_terms(L, 1.0, float(U), pbc=True)
        solver = dmrg_ground_state(terms, L, N, chi_max=16)
        docc = sum(solver.mps.expectation_terms(
            [(1.0, [(mode(i, 0), True), (mode(i, 0), False),
                    (mode(i, 1), True), (mode(i, 1), False)])]) for i in range(L))
        print(f"{U:5d} {table[U]:14.8f} {solver.energy:14.8f} "
              f"{solver.energy - table[U]:10.1e} {docc:12.6f}")
    print()
    print("Open chains at U = 4t, half filling, against exact diagonalisation")
    print("where it is possible and against the Bethe-ansatz energy per site")
    print(f"of the infinite chain, e = {lieb_wu_energy(4.0):.6f}:")
    print(f"{'L':>4s} {'E_exact':>14s} {'E_DMRG':>14s} {'E/L':>10s} "
          f"{'chi':>4s} {'S(middle)':>10s}")
    for L, chi in ((4, 16), (8, 32), (12, 32), (16, 32), (20, 32)):
        terms = hubbard_terms(L, 1.0, 4.0)
        solver = dmrg_ground_state(terms, L, L, chi_max=chi, max_sweeps=10)
        if L <= 8:
            e_ed = exact_ground_state(terms, L, L, n_a=L // 2)
            ed = f"{e_ed:14.8f}"
        else:
            ed = f"{'--':>14s}"
        print(f"{L:4d} {ed} {solver.energy:14.8f} {solver.energy / L:10.6f} "
              f"{max(solver.mps.bond_dimensions()):4d} "
              f"{solver.entropies[L // 2 - 1]:10.4f}")
    print("The energy per site of the open chain approaches the Lieb-Wu value")
    print("from above with a 1/L correction from the two ends.  The entropy of")
    print("the middle cut keeps growing slowly with L -- logarithmically, the")
    print("signature of the gapless spin sector -- so a fixed chi eventually")
    print("fails, but only logarithmically slowly.  (U = 0 is left out of the")
    print("ring table: its ground state is degenerate, and the double")
    print("occupancy is then not unique.)")


def _demo():
    for f in (demo_mpo, demo_pairing, demo_particle_hole, demo_hubbard):
        f()
        print()


if __name__ == "__main__":
    _demo()

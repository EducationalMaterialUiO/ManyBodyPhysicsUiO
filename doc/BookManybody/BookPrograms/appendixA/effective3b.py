"""
Effective interactions and induced three-body forces from full
configuration interaction.

Companion code to Appendix A, Project 2, of *Quantum mechanics for
Many-particle Systems* and to the FYS4480 lecture notes.

The question this program answers is the following.  Full configuration
interaction (chapter 5) is exact but limited by the exponential wall to a
few particles in a moderate single-particle basis.  Suppose we can solve
the two-, three- and four-particle problems exactly in a large basis of n
orbitals, but want to treat an N-particle system by FCI in a small *model
space* P spanned by the n_P lowest orbitals only.  Which interaction should
be diagonalised in P?  An effective two-body interaction that reproduces
exact two-particle energies in P is constructed with the Lee-Suzuki-Okubo
similarity transformation from the exact eigenpairs.  The same construction
for three particles yields an effective three-body Hamiltonian, and the
part of it that is *not* accounted for by the embedded one- and two-body
effective operators is an induced effective three-body interaction,

    V3_eff = H3_eff - sum_i t_i - sum_{i<j} V2_eff(ij).

The program implements this a-body cluster hierarchy on a toy model:
spinless fermions in a one-dimensional harmonic oscillator interacting
through a Gaussian two-body potential.  The ingredients are

  * ``HarmonicOscillator1D``  -- the single-particle basis and the one- and
    (antisymmetrised) two-body matrix elements on a grid;
  * ``DeterminantBasis``      -- Slater determinants as ordered tuples of
    occupied orbitals and as the bit strings of chapter 3 / chapter 5;
  * ``KBodyOperator``         -- a k-body operator stored through its matrix
    in the k-particle determinant basis, with a routine that embeds it in
    the N-particle space (the operator identity
    O = sum_{S,P} <S|O|P> a+_S a_P), built on the bit-string routines
    ``annihilate`` and ``create`` of chapter 5's ``fci.py``;
  * ``LeeSuzuki``             -- the similarity transformation that produces
    a Hermitian effective Hamiltonian in P from the exact eigenpairs;
  * ``ClusterHierarchy``      -- the a-body cluster construction that
    extracts V2_eff, V3_eff (and V4_eff) and tests them on N-particle systems.

Everything runs on numpy and scipy.linalg in a few seconds.  The program
imports ``fci.py`` from ``BookPrograms/chapter05``, so put the whole of
``BookPrograms/`` on the search path (the notebook shows how); a local copy
of the three bit-string routines keeps it runnable on its own.

Author: Morten Hjorth-Jensen
"""

import itertools
import numpy as np
from scipy.linalg import eigh, inv

try:
    # the bit-string representation of Slater determinants of chapter 5
    from fci import popcount, annihilate, create
except ImportError:                                  # stand-alone fallback
    def popcount(x):
        return bin(x).count("1")

    def annihilate(state, p):
        """a_p |state>, returning (sign, new state) or (0, 0) if it vanishes."""
        if not (state >> p) & 1:
            return 0, 0
        sign = -1 if popcount(state & ((1 << p) - 1)) & 1 else 1
        return sign, state ^ (1 << p)

    def create(state, p):
        """a+_p |state>, returning (sign, new state) or (0, 0) if it vanishes."""
        if (state >> p) & 1:
            return 0, 0
        sign = -1 if popcount(state & ((1 << p) - 1)) & 1 else 1
        return sign, state | (1 << p)


# ---------------------------------------------------------------------------
#  Single-particle basis: the one-dimensional harmonic oscillator
# ---------------------------------------------------------------------------
class HarmonicOscillator1D:
    """Harmonic-oscillator orbitals phi_n(x), n = 0, ..., n_orbitals-1.

    Units: hbar = m = omega = 1, so the oscillator length is one and the
    single-particle energies are n + 1/2.  The orbitals are tabulated on a
    uniform grid and all integrals are done by the trapezoidal rule, which
    is exponentially accurate for these rapidly decaying functions.
    """

    def __init__(self, n_orbitals, x_max=12.0, n_grid=601):
        self.n = n_orbitals
        self.x = np.linspace(-x_max, x_max, n_grid)
        self.w = np.full(n_grid, self.x[1] - self.x[0])
        self.w[0] = self.w[-1] = 0.5 * self.w[0]
        self.phi = self._orbitals()

    def _orbitals(self):
        """Normalised Hermite functions by the stable three-term recursion."""
        x = self.x
        phi = np.zeros((self.n, x.size))
        phi[0] = np.pi ** (-0.25) * np.exp(-0.5 * x ** 2)
        if self.n > 1:
            phi[1] = np.sqrt(2.0) * x * phi[0]
        for k in range(1, self.n - 1):
            phi[k + 1] = (np.sqrt(2.0 / (k + 1)) * x * phi[k]
                          - np.sqrt(k / (k + 1)) * phi[k - 1])
        return phi

    def energies(self):
        """Single-particle energies epsilon_n = n + 1/2."""
        return np.arange(self.n) + 0.5

    def one_body(self, gap=0.0, n_P=None):
        """The one-body Hamiltonian h_pq = epsilon_p delta_pq.

        With ``gap`` > 0 the orbitals n >= n_P (the excluded space) are
        shifted up by ``gap``: a schematic shell gap between the model space
        and the excluded space.  The pure oscillator has gap = 0, and there
        P-space and Q-space determinants with the same number of quanta are
        exactly degenerate -- the intruder-state problem of the project.
        """
        eps = self.energies().copy()
        if gap:
            eps[n_P:] += gap
        return np.diag(eps)

    def parity(self):
        """Parity of each orbital, (-1)^n."""
        return (-1) ** np.arange(self.n)

    def two_body(self, potential):
        """Antisymmetrised two-body matrix elements <pq|V|rs>_AS.

        <pq|V|rs> = int dx dy phi_p(x) phi_q(y) V(x-y) phi_r(x) phi_s(y),
        computed as a matrix product over the grid, and

        <pq|V|rs>_AS = <pq|V|rs> - <pq|V|sr>.
        """
        n = self.n
        Vxy = potential(self.x[:, None] - self.x[None, :])
        # Phi[(p,r), x] = phi_p(x) phi_r(x) w_x
        Phi = (self.phi[:, None, :] * self.phi[None, :, :]
               * self.w[None, None, :]).reshape(n * n, -1)
        v = (Phi @ Vxy @ Phi.T).reshape(n, n, n, n)     # v[p, r, q, s]
        v = v.transpose(0, 2, 1, 3)                     # v[p, q, r, s]
        return v - v.transpose(0, 1, 3, 2)


def gaussian(V0, sigma):
    """The Gaussian potential V(r) = V0 exp(-r^2 / (2 sigma^2))."""
    return lambda r: V0 * np.exp(-0.5 * (r / sigma) ** 2)


# ---------------------------------------------------------------------------
#  Slater determinants
# ---------------------------------------------------------------------------
class DeterminantBasis:
    """All Slater determinants of N particles in a given set of orbitals.

    A determinant is stored as an ordered tuple (p1 < p2 < ... < pN) and
    stands for the state

        |p1 p2 ... pN> = a+_{p1} a+_{p2} ... a+_{pN} |0>.

    The determinants built from the *lowest* n_P orbitals form the model
    space; ``DeterminantBasis(range(n_P), N)`` is a sub-basis of
    ``DeterminantBasis(range(n), N)`` and ``indices_in`` locates it there.
    """

    def __init__(self, orbitals, n_particles):
        self.orbitals = tuple(orbitals)
        self.N = n_particles
        self.dets = list(itertools.combinations(self.orbitals, n_particles))
        self.index = {d: i for i, d in enumerate(self.dets)}
        self.bits = np.array([sum(1 << p for p in d) for d in self.dets],
                             dtype=np.int64)

    def __len__(self):
        return len(self.dets)

    def indices_in(self, larger):
        """Positions of our determinants inside a larger basis."""
        return np.array([larger.index[d] for d in self.dets], dtype=int)

    def parity(self, orbital_parity):
        """Total parity of each determinant, prod_i (-1)^{n_i}."""
        return np.array([np.prod([orbital_parity[p] for p in d])
                         for d in self.dets], dtype=int)


# ---------------------------------------------------------------------------
#  k-body operators and their embedding in the N-particle space
# ---------------------------------------------------------------------------
class KBodyOperator:
    """A k-body operator O given by its matrix in the k-particle basis.

    The antisymmetrised matrix elements <S|O|P> between k-particle
    determinants define O uniquely as an operator on Fock space,

        O = sum_{S,P} <S|O|P>  a+_{s1} ... a+_{sk}  a_{pk} ... a_{p1},

    with ordered tuples S = (s1<...<sk), P = (p1<...<pk).  ``embed`` uses
    this identity to build the matrix of O in any N-particle determinant
    basis with N >= k.  A one-body operator is the case k = 1 (its matrix
    is just h_pq), the bare two-body interaction is k = 2 with
    <rs|V|pq> = <rs|V|pq>_AS, and the effective three-body interaction
    of the project is k = 3.  The fermionic phases come from ``annihilate``
    and ``create`` of chapter 5's ``fci.py``.
    """

    _structure_cache = {}

    def __init__(self, basis, matrix):
        self.basis = basis
        self.k = basis.N
        self.matrix = np.asarray(matrix, dtype=float)
        assert self.matrix.shape == (len(basis), len(basis))

    # -- constructors -------------------------------------------------------
    @classmethod
    def from_one_body(cls, h, orbitals=None):
        orbitals = range(h.shape[0]) if orbitals is None else orbitals
        basis = DeterminantBasis(orbitals, 1)
        idx = [d[0] for d in basis.dets]
        return cls(basis, h[np.ix_(idx, idx)])

    @classmethod
    def from_two_body(cls, v_as, orbitals=None):
        orbitals = range(v_as.shape[0]) if orbitals is None else orbitals
        basis = DeterminantBasis(orbitals, 2)
        m = np.zeros((len(basis), len(basis)))
        for i, (r, s) in enumerate(basis.dets):
            for j, (p, q) in enumerate(basis.dets):
                m[i, j] = v_as[r, s, p, q]
        return cls(basis, m)

    # -- algebra ------------------------------------------------------------
    def restrict(self, sub_basis):
        """The same operator restricted to a sub-basis (P O P)."""
        idx = sub_basis.indices_in(self.basis)
        return KBodyOperator(sub_basis, self.matrix[np.ix_(idx, idx)])

    def norm(self):
        """Frobenius norm of the k-particle matrix, a measure of the size of O."""
        return np.linalg.norm(self.matrix)

    def __add__(self, other):
        assert other.basis.dets == self.basis.dets
        return KBodyOperator(self.basis, self.matrix + other.matrix)

    def __sub__(self, other):
        assert other.basis.dets == self.basis.dets
        return KBodyOperator(self.basis, self.matrix - other.matrix)

    # -- embedding in the N-particle space ---------------------------------
    def _structure(self, target):
        """Sparse pattern of the embedding, cached: the matrix of O in the
        target basis is sum over entries of sign * matrix.flat[opidx] at
        (row, col).  Depends only on the two bases, not on the numbers."""
        key = (self.basis.orbitals, self.k, target.orbitals, target.N)
        if key in self._structure_cache:
            return self._structure_cache[key]
        rows, cols, signs, opidx = [], [], [], []
        nk = len(self.basis)
        kb = self.basis
        for col, ket in enumerate(target.dets):
            for P in itertools.combinations(ket, self.k):
                # a_{pk} ... a_{p1} |ket>: a_{p1} acts first
                sgn, bits = 1, int(target.bits[col])
                for p in P:
                    s, bits = annihilate(bits, p)
                    sgn *= s
                jP = kb.index[P]
                remaining = bits
                for iS, S in enumerate(kb.dets):
                    # a+_{s1} ... a+_{sk}: a+_{sk} acts first
                    sgn2, bits2 = sgn, remaining
                    for s_orb in reversed(S):
                        s, bits2 = create(bits2, s_orb)
                        if s == 0:
                            sgn2 = 0
                            break
                        sgn2 *= s
                    if sgn2 == 0:
                        continue
                    bra = tuple(p for p in target.orbitals if (bits2 >> p) & 1)
                    rows.append(target.index[bra])
                    cols.append(col)
                    signs.append(sgn2)
                    opidx.append(iS * nk + jP)
        structure = (np.array(rows, dtype=int), np.array(cols, dtype=int),
                     np.array(signs, dtype=float), np.array(opidx, dtype=int))
        self._structure_cache[key] = structure
        return structure

    def embed(self, target):
        """Matrix of this k-body operator in an N-particle determinant basis."""
        assert target.N >= self.k
        rows, cols, signs, opidx = self._structure(target)
        M = np.zeros((len(target), len(target)))
        np.add.at(M, (rows, cols), signs * self.matrix.flat[opidx])
        return M


# ---------------------------------------------------------------------------
#  The Lee-Suzuki-Okubo effective Hamiltonian
# ---------------------------------------------------------------------------
class LeeSuzuki:
    """Hermitian effective Hamiltonian in a model space P from exact eigenpairs.

    Given the full Hamiltonian H (a real symmetric matrix), the indices of
    the model-space basis states, and optionally a label of the symmetry
    sector of each basis state, the effective Hamiltonian is built sector by
    sector:

      1. diagonalise H in the sector, H|k> = E_k|k>;
      2. choose d_P eigenvectors (the lowest, or those with largest norm
         inside P) and collect them as columns of K;
      3. omega = (Q K)(P K)^{-1} solves Q|k> = omega P|k> for the chosen k;
      4. H_eff = (P + omega^+ omega)^{-1/2} (P + omega^+) H (P + omega)
                 (P + omega^+ omega)^{-1/2}.

    Since (P + omega) P|k> = |k>, one has (P + omega) = K (PK)^{-1} on P and
    H_eff = S^{-1/2} (PK)^{-+} diag(E) (PK)^{-1} S^{-1/2} with
    S = (PK)^{-+}(PK)^{-1}, which is what the code evaluates.  H_eff has
    exactly the chosen eigenvalues.  The non-Hermitian Lee-Suzuki form
    P H P + P H Q omega is also returned for comparison.
    """

    def __init__(self, H, p_indices, sectors=None, select="lowest"):
        self.H = np.asarray(H)
        self.p = np.asarray(p_indices, dtype=int)
        self.sectors = (np.zeros(self.H.shape[0], dtype=int)
                        if sectors is None else np.asarray(sectors))
        self.select = select
        self._solve()

    def _solve(self):
        n = self.H.shape[0]
        dP = len(self.p)
        pos_in_p = {idx: i for i, idx in enumerate(self.p)}
        in_p = np.zeros(n, dtype=bool)
        in_p[self.p] = True
        self.H_eff = np.zeros((dP, dP))
        self.H_eff_nonhermitian = np.zeros((dP, dP))
        self.chosen_energies = np.zeros(dP)
        self.condition_numbers = {}
        self.omega_norm = {}
        for label in np.unique(self.sectors):
            sec = np.where(self.sectors == label)[0]
            p_sec = [i for i in sec if in_p[i]]
            q_sec = [i for i in sec if not in_p[i]]
            d = len(p_sec)
            if d == 0:
                continue
            E, C = eigh(self.H[np.ix_(sec, sec)])
            rows_p = [list(sec).index(i) for i in p_sec]
            rows_q = [list(sec).index(i) for i in q_sec]
            if self.select == "lowest":
                chosen = np.arange(d)
            elif self.select == "overlap":
                pnorm = np.sum(C[rows_p, :] ** 2, axis=0)
                chosen = np.sort(np.argsort(-pnorm)[:d])
            else:
                raise ValueError("select must be 'lowest' or 'overlap'")
            K = C[:, chosen]
            PK = K[rows_p, :]
            QK = K[rows_q, :] if rows_q else np.zeros((0, d))
            self.condition_numbers[label] = np.linalg.cond(PK)
            PKinv = inv(PK)
            omega = QK @ PKinv
            self.omega_norm[label] = np.linalg.norm(omega)
            S = PKinv.T @ PKinv                    # = 1 + omega^T omega
            s, U = eigh(S)
            S_mhalf = U @ np.diag(s ** -0.5) @ U.T
            Heff = S_mhalf @ PKinv.T @ np.diag(E[chosen]) @ PKinv @ S_mhalf
            Heff = 0.5 * (Heff + Heff.T)
            # non-Hermitian version P H P + P H Q omega
            Hpp = self.H[np.ix_(p_sec, p_sec)]
            Hpq = self.H[np.ix_(p_sec, q_sec)] if q_sec else np.zeros((d, 0))
            Hnh = Hpp + Hpq @ omega
            loc = [pos_in_p[i] for i in p_sec]
            self.H_eff[np.ix_(loc, loc)] = Heff
            self.H_eff_nonhermitian[np.ix_(loc, loc)] = Hnh
            self.chosen_energies[loc] = E[chosen]
        self.chosen_energies = np.sort(self.chosen_energies)

    def check(self):
        """Largest deviation between eig(H_eff) and the chosen exact energies."""
        return np.max(np.abs(np.sort(eigh(self.H_eff, eigvals_only=True))
                             - self.chosen_energies))


# ---------------------------------------------------------------------------
#  The a-body cluster hierarchy
# ---------------------------------------------------------------------------
class ClusterHierarchy:
    """Effective one-, two- and three-body interactions in a model space.

    Parameters
    ----------
    h : (n, n) one-body matrix in the full single-particle basis
    v_as : (n, n, n, n) antisymmetrised two-body matrix elements
    n_P : number of model-space orbitals (the lowest n_P)
    orbital_parity : parity of each orbital, used to define symmetry sectors
    select : eigenvector selection rule handed to ``LeeSuzuki``

    After construction the object holds

      h_P      : the one-body operator restricted to P (a=1 cluster; exact
                 for a diagonal h)
      V2_eff   : effective two-body interaction, H2_eff - sum_i t_i
      V3_eff   : induced three-body interaction,
                 H3_eff - sum_i t_i - sum_{i<j} V2_eff(ij)
      V2_bare  : the bare interaction restricted to P, for comparison
    """

    def __init__(self, h, v_as, n_P, orbital_parity=None, select="lowest"):
        self.n = h.shape[0]
        self.n_P = n_P
        self.select = select
        self.parity = (np.ones(self.n, dtype=int) if orbital_parity is None
                       else np.asarray(orbital_parity))
        self.h1 = KBodyOperator.from_one_body(h)
        self.v2 = KBodyOperator.from_two_body(v_as)
        self.full = {}
        self.model = {}
        self.ls = {}
        # a = 1: h is diagonal in the oscillator basis, so P is exact
        self.h_P = self.h1.restrict(self.basis(1, model=True))
        # a = 2
        H2eff = self.effective_hamiltonian(2)
        B2P = self.basis(2, model=True)
        self.V2_bare = self.v2.restrict(B2P)
        self.V2_eff = KBodyOperator(B2P, H2eff - self.h_P.embed(B2P))
        # a = 3
        H3eff = self.effective_hamiltonian(3)
        B3P = self.basis(3, model=True)
        self.V3_eff = KBodyOperator(
            B3P, H3eff - self.h_P.embed(B3P) - self.V2_eff.embed(B3P))
        self.V4_eff = None

    # -- bases and Hamiltonians --------------------------------------------
    def basis(self, N, model=False):
        store, orbs = ((self.model, range(self.n_P)) if model
                       else (self.full, range(self.n)))
        if N not in store:
            store[N] = DeterminantBasis(orbs, N)
        return store[N]

    def bare_hamiltonian(self, N, model=False):
        """H = sum_i t_i + sum_{i<j} v_ij in the N-particle basis."""
        B = self.basis(N, model)
        return self.h1.embed(B) + self.v2.embed(B)

    def effective_hamiltonian(self, N):
        """Lee-Suzuki-Okubo H_eff for N particles in the model space."""
        B, BP = self.basis(N), self.basis(N, model=True)
        H = self.bare_hamiltonian(N)
        ls = LeeSuzuki(H, BP.indices_in(B), sectors=B.parity(self.parity),
                       select=self.select)
        self.ls[N] = ls
        return ls.H_eff

    def add_four_body(self):
        """Continue the hierarchy to a = 4 (the induced four-body force)."""
        H4eff = self.effective_hamiltonian(4)
        B4P = self.basis(4, model=True)
        self.V4_eff = KBodyOperator(
            B4P, H4eff - self.h_P.embed(B4P) - self.V2_eff.embed(B4P)
            - self.V3_eff.embed(B4P))
        return self.V4_eff

    # -- tests on N-particle systems ----------------------------------------
    def model_space_spectra(self, N, n_states=None):
        """Lowest eigenvalues of N particles: exact, and in the model space
        with the bare interaction, with V2_eff, and with V2_eff + V3_eff
        (and with V4_eff if available)."""
        B, BP = self.basis(N), self.basis(N, model=True)
        k = len(BP) if n_states is None else min(n_states, len(BP))
        out = {}
        out["exact"] = eigh(self.bare_hamiltonian(N), eigvals_only=True)[:k]
        T = self.h_P.embed(BP)
        out["bare (PHP)"] = eigh(T + self.V2_bare.embed(BP),
                                 eigvals_only=True)[:k]
        H2 = T + self.V2_eff.embed(BP)
        out["h + V2_eff"] = eigh(H2, eigvals_only=True)[:k]
        if N >= 3:
            H3 = H2 + self.V3_eff.embed(BP)
            out["h + V2_eff + V3_eff"] = eigh(H3, eigvals_only=True)[:k]
        if N >= 4 and self.V4_eff is not None:
            H4 = H3 + self.V4_eff.embed(BP)
            out["h + V2_eff + V3_eff + V4_eff"] = eigh(H4, eigvals_only=True)[:k]
        return out


# ---------------------------------------------------------------------------
#  Demonstration
# ---------------------------------------------------------------------------
def demo(n=14, n_P=6, V0=-2.0, sigma=1.0, gap=3.0):
    np.set_printoptions(precision=6, suppress=True, linewidth=110)
    ho = HarmonicOscillator1D(n)
    h, v = ho.one_body(gap, n_P), ho.two_body(gaussian(V0, sigma))
    print(f"1D oscillator, {n} orbitals, model space n_P = {n_P}, shell gap "
          f"{gap}, Gaussian V0 = {V0}, sigma = {sigma}")
    ch = ClusterHierarchy(h, v, n_P, ho.parity(), select="overlap")
    for N in (2, 3):
        print(f"\nN = {N}: Lee-Suzuki check max|eig(H_eff) - E_k| = "
              f"{ch.ls[N].check():.2e}, cond(PK) = "
              f"{max(ch.ls[N].condition_numbers.values()):.2f}")
    print(f"\n||V2_bare|| = {ch.V2_bare.norm():.4f}   "
          f"||V2_eff|| = {ch.V2_eff.norm():.4f}   "
          f"||V3_eff|| = {ch.V3_eff.norm():.4f}")
    for N in (3, 4):
        print(f"\nLowest energies, N = {N}:")
        for label, E in ch.model_space_spectra(N, 4).items():
            print(f"  {label:26s} {E}")
    return ch


if __name__ == "__main__":
    demo()

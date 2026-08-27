"""
The spectral decomposition of a Hermitian operator, and what it means.

Companion code to Section 1.11 of *Quantum mechanics for Many-particle
Systems*.  The spectral theorem writes a Hermitian matrix as

    A = sum_alpha a_alpha P_alpha ,

with real eigenvalues a_alpha and orthogonal projectors P_alpha onto the
eigenspaces.  In quantum mechanics every piece of this formula carries a
physical meaning, and this program makes each of them concrete:

    Observable      -- eigenvalues as measurement outcomes, projectors as the
                       states associated with each outcome, probabilities
                       <psi|P_alpha|psi>, functions f(A) = sum f(a) P, and the
                       variance as the variance of the outcome distribution
    TwoLevelSystem  -- the time-evolution operator exp(-iHt) from the spectral
                       decomposition of H, and the relative phase that drives
                       Ramsey oscillations
    DensityMatrix   -- eigenvalues as statistical weights, purity, the von
                       Neumann entropy, thermal states exp(-beta H)/Z, and the
                       reduced density matrix of a bipartite state
    oscillator      -- the truncated harmonic oscillator, whose projectors
                       count excitation quanta

Everything uses numpy.linalg.eigh, and everything is checked: the projectors
are idempotent and complete, the decomposition rebuilds the matrix, and the
degenerate-eigenspace projector is shown to be independent of the basis
chosen inside the eigenspace.

Author: Morten Hjorth-Jensen
"""

import numpy as np

# the Pauli matrices and the two-dimensional computational basis
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
KET0 = np.array([1.0, 0.0], dtype=complex)
KET1 = np.array([0.0, 1.0], dtype=complex)


def ket(*amplitudes):
    """A normalised state vector from its amplitudes."""
    v = np.array(amplitudes, dtype=complex)
    return v / np.linalg.norm(v)


def projector(*vectors):
    """P = sum_k |v_k><v_k| for orthonormal vectors v_k."""
    P = np.zeros((len(vectors[0]), len(vectors[0])), dtype=complex)
    for v in vectors:
        P += np.outer(v, v.conj())
    return P


# ---------------------------------------------------------------------------
#  Observables
# ---------------------------------------------------------------------------
class Observable:
    """A Hermitian matrix together with its spectral decomposition.

    Eigenvalues closer than ``degeneracy_tol`` are grouped into one
    eigenspace, so that ``values`` and ``projectors`` describe
    A = sum_alpha a_alpha P_alpha with one projector per *distinct*
    eigenvalue, the form that survives when the eigenvalues are degenerate.
    """

    def __init__(self, matrix, name="A", degeneracy_tol=1e-9):
        A = np.asarray(matrix, dtype=complex)
        if not np.allclose(A, A.conj().T):
            raise ValueError(f"{name} is not Hermitian")
        self.A = A
        self.name = name
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(A)
        # group into eigenspaces
        self.values, self.projectors, self.multiplicities = [], [], []
        k = 0
        n = len(self.eigenvalues)
        while k < n:
            j = k
            while j + 1 < n and abs(self.eigenvalues[j + 1]
                                    - self.eigenvalues[k]) < degeneracy_tol:
                j += 1
            vecs = [self.eigenvectors[:, m] for m in range(k, j + 1)]
            self.values.append(float(np.mean(self.eigenvalues[k:j + 1])))
            self.projectors.append(projector(*vecs))
            self.multiplicities.append(j - k + 1)
            k = j + 1
        self.values = np.array(self.values)

    # ------------------------------------------------------------------
    def reconstruct(self):
        """sum_alpha a_alpha P_alpha, which must equal A."""
        return sum(a * P for a, P in zip(self.values, self.projectors))

    def check(self):
        """Idempotence, orthogonality, completeness and reconstruction."""
        n = self.A.shape[0]
        idem = max(np.abs(P @ P - P).max() for P in self.projectors)
        orth = max((np.abs(P @ Q).max() for i, P in enumerate(self.projectors)
                    for j, Q in enumerate(self.projectors) if i != j),
                   default=0.0)
        comp = np.abs(sum(self.projectors) - np.eye(n)).max()
        rec = np.abs(self.reconstruct() - self.A).max()
        return dict(idempotent=idem, orthogonal=orth, complete=comp,
                    reconstruction=rec)

    # ------------------------------------------------------------------
    def probabilities(self, psi):
        """p_alpha = <psi|P_alpha|psi>: the probability of each outcome."""
        psi = np.asarray(psi, dtype=complex)
        return np.array([np.vdot(psi, P @ psi).real for P in self.projectors])

    def expectation(self, psi):
        return float(np.vdot(psi, self.A @ psi).real)

    def variance(self, psi):
        """<A^2> - <A>^2, which equals sum_alpha a_alpha^2 p_alpha - (sum a p)^2."""
        p = self.probabilities(psi)
        return float(np.sum(self.values**2 * p) - np.sum(self.values * p)**2)

    def function(self, f):
        """f(A) = sum_alpha f(a_alpha) P_alpha."""
        return sum(f(a) * P for a, P in zip(self.values, self.projectors))

    def collapse(self, psi, alpha):
        """The state after the outcome a_alpha: P_alpha|psi>/sqrt(p_alpha)."""
        v = self.projectors[alpha] @ np.asarray(psi, dtype=complex)
        return v / np.linalg.norm(v)


# ---------------------------------------------------------------------------
#  Time evolution
# ---------------------------------------------------------------------------
class TwoLevelSystem:
    """H = E0 |0><0| + E1 |1><1| in the basis of its own eigenstates.

    The evolution operator U(t) = exp(-iHt/hbar) is assembled from the
    spectral decomposition, U(t) = sum_n exp(-i E_n t/hbar) |E_n><E_n|,
    with hbar = 1.  The physics is in the relative phase
    exp(-i (E1 - E0) t), which is what a Ramsey experiment measures.
    """

    def __init__(self, E0=0.0, E1=1.0):
        self.E0, self.E1 = E0, E1
        self.H = Observable(np.diag([E0, E1]), name="H")

    def evolution(self, t):
        return self.H.function(lambda E: np.exp(-1j * E * t))

    def evolve(self, psi0, t):
        return self.evolution(t) @ np.asarray(psi0, dtype=complex)

    def sigma_x_expectation(self, psi0, times):
        """<sigma_x>(t) for the state psi0, from the spectral phases."""
        return np.array([np.vdot(self.evolve(psi0, t),
                                 SX @ self.evolve(psi0, t)).real
                         for t in times])

    def survival(self, psi0, times):
        """|<psi0|psi(t)>|^2."""
        return np.array([abs(np.vdot(psi0, self.evolve(psi0, t)))**2
                         for t in times])


# ---------------------------------------------------------------------------
#  Density matrices
# ---------------------------------------------------------------------------
class DensityMatrix:
    """A density operator: Hermitian, positive, unit trace.

    Its spectral decomposition rho = sum_n p_n |phi_n><phi_n| has a
    different reading from that of an observable: the eigenvalues are the
    statistical weights of orthogonal states, and every quantity that
    depends on rho alone -- purity, entropy -- depends only on them.
    """

    def __init__(self, matrix, name="rho"):
        self.rho = np.asarray(matrix, dtype=complex)
        self.name = name
        if not np.allclose(self.rho, self.rho.conj().T):
            raise ValueError("a density matrix must be Hermitian")
        if abs(np.trace(self.rho).real - 1.0) > 1e-10:
            raise ValueError("a density matrix must have unit trace")
        self.weights, self.states = np.linalg.eigh(self.rho)
        if self.weights.min() < -1e-10:
            raise ValueError("a density matrix must be positive")
        self.weights = np.clip(self.weights, 0.0, None)

    @classmethod
    def pure(cls, psi):
        psi = np.asarray(psi, dtype=complex)
        psi = psi / np.linalg.norm(psi)
        return cls(np.outer(psi, psi.conj()), name="|psi><psi|")

    @classmethod
    def thermal(cls, H, beta):
        """rho_beta = exp(-beta H)/Z from the spectral decomposition of H."""
        obs = H if isinstance(H, Observable) else Observable(H)
        weights = obs.function(lambda E: np.exp(-beta * E))
        Z = np.trace(weights).real
        rho = cls(weights / Z, name=f"exp(-{beta} H)/Z")
        rho.partition_function = Z
        return rho

    @classmethod
    def reduced(cls, psi, dims, keep=0):
        """Trace out one subsystem of a pure bipartite state.

        ``dims`` = (d_A, d_B) and ``keep`` selects which subsystem survives.
        rho_A = Tr_B |psi><psi| = C C^+ with psi reshaped to a d_A x d_B
        matrix C, so the eigenvalues of rho_A are the squared Schmidt
        coefficients of the Schmidt decomposition later in the chapter.
        """
        C = np.asarray(psi, dtype=complex).reshape(dims)
        C = C / np.linalg.norm(C)
        rho = C @ C.conj().T if keep == 0 else C.T @ C.conj()
        return cls(rho, name="rho_A" if keep == 0 else "rho_B")

    # ------------------------------------------------------------------
    def purity(self):
        return float(np.trace(self.rho @ self.rho).real)

    def entropy(self):
        """S = -Tr rho ln rho = -sum_n p_n ln p_n."""
        p = self.weights[self.weights > 1e-15]
        return max(float(-np.sum(p * np.log(p))), 0.0)

    def expectation(self, A):
        return float(np.trace(self.rho @ A).real)


# ---------------------------------------------------------------------------
#  The harmonic oscillator, truncated
# ---------------------------------------------------------------------------
def oscillator(nmax, omega=1.0):
    """H = omega (a^+ a + 1/2) in the number basis |0>, ..., |nmax>.

    Returns the Hamiltonian, the number operator and the ladder operator a,
    all as (nmax+1) x (nmax+1) matrices.  The projectors of H count quanta.
    """
    n = np.arange(nmax + 1)
    a = np.diag(np.sqrt(n[1:]), k=1)
    N = a.T @ a
    H = omega * (N + 0.5 * np.eye(nmax + 1))
    return H, N, a


# ---------------------------------------------------------------------------
#  Demonstrations
# ---------------------------------------------------------------------------
def demo_measurement():
    print("=" * 74)
    print("1. Eigenvalues are outcomes, projectors are the states of each outcome")
    print("=" * 74)
    sz = Observable(0.5 * SZ, name="S_z")
    print("S_z = (1/2) sigma_z has the spectral decomposition")
    for a, P in zip(sz.values, sz.projectors):
        print(f"   {a:+.1f} x P,   P =")
        print("       " + str(P.real).replace("\n", "\n       "))
    print("The checks (idempotent, orthogonal, complete, rebuilds S_z):")
    print("  ", {k: f"{v:.1e}" for k, v in sz.check().items()})
    print()
    psi = ket(np.cos(np.pi / 8), np.sin(np.pi / 8))
    print("For |psi> = cos(pi/8)|0> + sin(pi/8)|1> a measurement of S_z gives")
    for a, p in zip(sz.values, sz.probabilities(psi)):
        print(f"   S_z = {a:+.1f}  with probability  {p:.6f}")
    print(f"   <S_z>      = {sz.expectation(psi):+.6f}"
          f"   (sum_alpha a_alpha p_alpha = "
          f"{np.sum(sz.values * sz.probabilities(psi)):+.6f})")
    print(f"   Var(S_z)   = {sz.variance(psi):.6f}"
          f"   (<S_z^2> - <S_z>^2 = "
          f"{np.vdot(psi, sz.A @ sz.A @ psi).real - sz.expectation(psi)**2:.6f})")
    print("After the outcome +1/2 the state is P_+|psi>/sqrt(p_+) =",
          np.round(sz.collapse(psi, 1), 6))
    print()
    sx = Observable(0.5 * SX, name="S_x")
    print("The same state measured along x, S_x = (1/2) sigma_x:")
    for a, p in zip(sx.values, sx.probabilities(psi)):
        print(f"   S_x = {a:+.1f}  with probability  {p:.6f}")
    print("The two Stern-Gerlach magnets ask two different questions of one")
    print("state, and the spectral decompositions of S_z and S_x are the")
    print("questions.")


def demo_functions():
    print("=" * 74)
    print("2. Functions of an observable: f(A) = sum f(a_alpha) P_alpha")
    print("=" * 74)
    A = Observable(np.array([[2.0, 1.0, 0.0],
                             [1.0, 2.0, 0.0],
                             [0.0, 0.0, 3.0]]), name="A")
    print("A = [[2,1,0],[1,2,0],[0,0,3]] has eigenvalues",
          np.round(A.eigenvalues, 6),
          "-> distinct values", A.values, "with multiplicities",
          A.multiplicities)
    A2 = A.function(lambda a: a**2)
    print(f"  max |f(A) - A A|          for f(a) = a^2   : "
          f"{np.abs(A2 - A.A @ A.A).max():.1e}")
    expA = A.function(np.exp)
    from math import factorial
    series = sum(np.linalg.matrix_power(A.A, k) / factorial(k) for k in range(40))
    print(f"  max |exp(A) - Taylor series|                : "
          f"{np.abs(expA - series).max():.1e}")
    sqrtA = A.function(np.sqrt)
    print(f"  max |sqrt(A) sqrt(A) - A|                   : "
          f"{np.abs(sqrtA @ sqrtA - A.A).max():.1e}")
    print()
    print("The eigenvalue 3 is doubly degenerate.  Rotating the two")
    print("eigenvectors inside that eigenspace changes the eigenvectors but")
    print("not the projector:")
    v1, v2 = A.eigenvectors[:, 1], A.eigenvectors[:, 2]
    theta = 0.7
    w1 = np.cos(theta) * v1 + np.sin(theta) * v2
    w2 = -np.sin(theta) * v1 + np.cos(theta) * v2
    P_rot = projector(w1, w2)
    print(f"  max |P(rotated basis) - P(original basis)| = "
          f"{np.abs(P_rot - A.projectors[1]).max():.1e}")
    print("Inside a degenerate eigenspace the basis is a choice; the")
    print("projector is not, which is why P_alpha rather than the individual")
    print("eigenvectors carries the physics.")


def demo_evolution():
    print("=" * 74)
    print("3. Time evolution from the spectral decomposition of H")
    print("=" * 74)
    sys2 = TwoLevelSystem(E0=0.0, E1=1.0)
    psi0 = ket(1.0, 1.0)
    print("H = E_0|0><0| + E_1|1><1| with E_0 = 0, E_1 = 1 and hbar = 1.")
    print("U(t) = sum_n exp(-i E_n t)|E_n><E_n|; for |psi(0)> = (|0>+|1>)/sqrt2")
    print(f"{'t':>8s} {'|<0|psi>|^2':>12s} {'|<1|psi>|^2':>12s} "
          f"{'<sigma_x>':>10s} {'cos((E1-E0)t)':>15s}")
    for t in (0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi):
        psi = sys2.evolve(psi0, t)
        print(f"{t:8.4f} {abs(psi[0])**2:12.6f} {abs(psi[1])**2:12.6f} "
              f"{np.vdot(psi, SX @ psi).real:10.6f} "
              f"{np.cos((sys2.E1 - sys2.E0) * t):15.6f}")
    print()
    print("The populations never change -- each energy component only picks")
    print("up its own phase -- but the relative phase exp(-i(E_1 - E_0)t)")
    print("rotates the state, and <sigma_x> oscillates with it.  This is a")
    print("Ramsey fringe, and it is where interference in quantum mechanics")
    print("comes from.")
    print()
    U = sys2.evolution(0.37)
    print(f"U(t) is unitary: max |U^+ U - I| = "
          f"{np.abs(U.conj().T @ U - I2).max():.1e}")


def demo_oscillator():
    print("=" * 74)
    print("4. The harmonic oscillator: projectors count quanta")
    print("=" * 74)
    H, N, a = oscillator(nmax=6)
    obs = Observable(H, name="H_osc")
    print("H = omega(N + 1/2) in the number basis, truncated at n = 6:")
    print("  eigenvalues", np.round(obs.values, 4))
    print("  P_n |n> = |n> and P_n |m> = 0 for m /= n: each projector picks")
    print("  out the sector with n quanta.  The number operator has the same")
    print("  projectors with eigenvalues n:")
    Nobs = Observable(N, name="N")
    print("  max |P_n(H) - P_n(N)| =",
          f"{max(np.abs(P - Q).max() for P, Q in zip(obs.projectors, Nobs.projectors)):.1e}")
    beta = 1.0
    rho = DensityMatrix.thermal(obs, beta)
    print()
    print(f"Thermal state exp(-beta H)/Z at beta = {beta} (omega = 1):")
    print("  weights p_n =", np.round(rho.weights[::-1], 5))
    boltz = np.exp(-beta * obs.values)
    print("  Boltzmann   =", np.round(boltz / boltz.sum(), 5))
    print(f"  <N> = {rho.expectation(N):.5f}, entropy S = {rho.entropy():.5f}")
    print("Quantum statistical mechanics is the Boltzmann weighting of the")
    print("spectral decomposition, and nothing else.")


def demo_density_matrices():
    print("=" * 74)
    print("5. Density matrices: eigenvalues as statistical weights")
    print("=" * 74)
    psi = ket(1.0, 1.0)
    pure = DensityMatrix.pure(psi)
    mixed = DensityMatrix(0.5 * I2, name="I/2")
    print(f"{'state':>22s} {'spectrum':>16s} {'purity':>8s} {'entropy':>9s}")
    for rho in (pure, mixed):
        print(f"{rho.name:>22s} {str(np.round(rho.weights, 4)):>16s} "
              f"{rho.purity():8.4f} {rho.entropy():9.5f}")
    print(f"  (ln 2 = {np.log(2):.5f})")
    print()
    print("Reduced density matrices of two-qubit states |Psi> = "
          "cos(t)|00> + sin(t)|11>:")
    print(f"{'theta':>8s} {'spectrum of rho_A':>22s} {'S_A':>9s}")
    for theta in (0.0, np.pi / 12, np.pi / 8, np.pi / 6, np.pi / 4):
        psi = np.zeros(4, dtype=complex)
        psi[0], psi[3] = np.cos(theta), np.sin(theta)
        rho_A = DensityMatrix.reduced(psi, (2, 2))
        print(f"{theta:8.4f} {str(np.round(rho_A.weights, 5)):>22s} "
              f"{rho_A.entropy():9.5f}")
    print()
    print("At theta = 0 the state is a product and rho_A is pure; at")
    print("theta = pi/4 it is the Bell state (|00>+|11>)/sqrt2, rho_A = I/2 is")
    print("maximally mixed and S_A = ln 2 is maximal.  The eigenvalues of")
    print("rho_A are the squared Schmidt coefficients of the Schmidt section, so")
    print("the spectral decomposition of a reduced density matrix measures")
    print("entanglement.")


def _demo():
    for f in (demo_measurement, demo_functions, demo_evolution,
              demo_oscillator, demo_density_matrices):
        f()
        print()


if __name__ == "__main__":
    _demo()

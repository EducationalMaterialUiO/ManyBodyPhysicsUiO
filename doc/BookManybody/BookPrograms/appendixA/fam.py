"""
The finite amplitude method for the pairing plus particle-hole model.

Companion code to Appendix A, Project 1, of *Quantum mechanics for
Many-particle Systems*.  It builds on ``rpa.py`` of chapter 7 and follows
T. Nakatsukasa, T. Inakura and K. Yabana, Phys. Rev. C 76, 024318 (2007).

The random-phase approximation of chapter 7 was obtained by constructing the
matrices A and B explicitly and diagonalising the RPA matrix.  The finite
amplitude method (FAM) solves the *same* linear-response problem without ever
forming A and B.  For a fixed, in general complex, frequency omega and an
external one-body field F it solves

    (omega - eps_m + eps_i) X_mi - dh_mi[X, Y] = F_mi ,
   (-omega - eps_m + eps_i) Y_mi - dh_im[X, Y] = F_im ,

for the forward and backward amplitudes X_mi(omega), Y_mi(omega), where the
induced field dh is *not* expanded as (dh/drho) drho but evaluated as the
finite difference

    dh = ( h[<psi'|, |psi>] - h0 ) / eta ,
    |psi_i> = |phi_i> + eta |X_i>,   <psi'_i| = <phi_i| + eta <Y_i| ,

with the mean-field routine h[rho] of the static Hartree-Fock code applied to
the non-Hermitian density rho = sum_i |psi_i><psi'_i|.  The strength function

    S(F; omega) = sum_i ( <phi_i|F^+|X_i> + <Y_i|F^+|phi_i> ) ,
    dB/domega   = -(1/pi) Im S(F; omega + i Gamma/2) ,

follows directly from the amplitudes.  Everything is checked against the
matrix RPA of chapter 7 and against the exact response of the model, which
a schematic model -- unlike the deformed 20Ne of the paper -- makes possible.

Author: Morten Hjorth-Jensen
"""

import numpy as np
from scipy.linalg import eig, eigh, solve
from scipy.sparse.linalg import LinearOperator, gmres, bicgstab

import rpa
from rpa import FockSpace, PairingPH, orbital, popcount, UP, DN


# ---------------------------------------------------------------------------
#  External one-body fields, given in the model (spin-orbital) basis
# ---------------------------------------------------------------------------
def field_level(L):
    """Q_0 = sum_{p sigma} (p - 1) n_{p sigma}: the level operator, H_0 / xi.

    It is the discrete analogue of a monopole field: diagonal, and it does
    not break pairs.  Its response measures how the interaction moves
    particles between the equally spaced levels.
    """
    return np.diag([float(p - 1) for p in range(1, L + 1) for _ in (UP, DN)])


def field_hopping(L):
    """Q_1 = sum_{p sigma} ( a^+_{p+1 sigma} a_{p sigma} + h.c. ).

    The discrete analogue of a dipole field: it connects neighbouring levels
    and, acting on one spin at a time, it breaks pairs.
    """
    n = 2 * L
    F = np.zeros((n, n))
    for p in range(1, L):
        for s in (UP, DN):
            a, b = orbital(p + 1, s), orbital(p, s)
            F[a, b] = F[b, a] = 1.0
    return F


# ---------------------------------------------------------------------------
#  The "static Hartree-Fock code": h[rho] for an arbitrary density
# ---------------------------------------------------------------------------
class MeanField:
    """The single-particle Hamiltonian h[rho] of the model.

    This class plays the role of the static Hartree-Fock program in the paper.
    It knows how to build h[rho] = t + sum_{cd} vbar_{acbd} rho_{dc} from a
    one-body density, and nothing about RPA.  The only extension the finite
    amplitude method asks for is ``h_braket``, which accepts *independent* bra
    and ket orbitals and therefore a non-Hermitian density.  Everything is
    expressed in the model basis of 2L spin-orbitals; the Hartree-Fock
    orbitals are the columns of C.
    """

    def __init__(self, fock, N, g, f, xi=1.0):
        self.fock = fock
        self.N = N
        self.n = fock.norb
        self.g, self.f, self.xi = g, f, xi
        hf = rpa.hartree_fock(fock, N, g, f, xi)
        self.hf = hf
        self.t = np.diag(rpa.single_particle_energies(fock.L, xi))
        self.vbar = hf["vbar"]
        self.C = hf["C"]
        self.eps = hf["eps"]
        self.rho0 = self.C[:, :N] @ self.C[:, :N].T
        self.calls = 0
        self.h0 = self.h(self.rho0)

    # ------------------------------------------------------------------
    def h(self, rho):
        """h[rho]_ab = t_ab + sum_cd vbar_acbd rho_dc, rho arbitrary."""
        self.calls += 1
        return self.t + np.einsum("acbd,dc->ab", self.vbar, rho)

    def h_braket(self, bra, ket):
        """h[<psi'|, |psi>] with rho = sum_i |psi_i><psi'_i|.

        ``ket[:, i]`` holds the components of |psi_i> and ``bra[:, i]`` those
        of |psi'_i>, so that <psi'_i| carries the complex conjugate.
        """
        rho = ket @ bra.conj().T
        return self.h(rho)

    def energy(self):
        return self.hf["energy"]

    def to_hf_basis(self, F):
        """A one-body operator from the model basis to the HF basis."""
        return self.C.T @ F @ self.C


# ---------------------------------------------------------------------------
#  The particle-hole space and the transition strength
# ---------------------------------------------------------------------------
class ParticleHoleSpace:
    """Bookkeeping shared by all the linear-response solvers.

    Pairs (m, i) are ordered as in ``rpa.tda_rpa`` -- m runs over the
    unoccupied and i over the occupied Hartree-Fock orbitals -- so that the
    amplitudes line up with the A and B matrices of chapter 7.
    """

    def __init__(self, mf):
        self.mf = mf
        self.N, self.n = mf.N, mf.n
        self.pairs = [(m, i) for m in range(self.N, self.n)
                      for i in range(self.N)]
        self.m_index = np.array([m for m, i in self.pairs])
        self.i_index = np.array([i for m, i in self.pairs])
        self.delta = mf.eps[self.m_index] - mf.eps[self.i_index]
        self.size = len(self.pairs)

    def ph(self, F_hf):
        """f_mi = <m|F|i> and g_mi = <i|F|m> as vectors over the pairs."""
        f = F_hf[self.m_index, self.i_index]
        g = F_hf[self.i_index, self.m_index]
        return f, g

    def strength(self, F_hf, X, Y):
        """S(F; omega) = sum_i ( <phi_i|F^+|X_i> + <Y_i|F^+|phi_i> )."""
        f, g = self.ph(F_hf)
        return np.sum(f.conj() * X) + np.sum(g.conj() * Y)


def db_domega(S):
    """dB/domega = -(1/pi) Im S(F; omega + i Gamma/2)."""
    return -np.imag(S) / np.pi


def strength_curve(strength, F, omegas, gamma):
    """dB/domega on a grid of real frequencies with smoothing Gamma.

    ``strength`` is the ``strength`` method of a MatrixResponse, a
    FiniteAmplitudeMethod or an ExactResponse; ``F`` must be given in the
    basis that method expects (the HF basis for the first two, the model
    basis for the exact response).
    """
    return np.array([db_domega(strength(F, w + 0.5j * gamma))
                     for w in omegas])


# ---------------------------------------------------------------------------
#  1. The conventional route: A, B and the RPA matrix
# ---------------------------------------------------------------------------
class MatrixResponse:
    """Linear response from the explicit RPA matrix of chapter 7.

    The matrices A and B are taken from ``rpa.tda_rpa``, where they are
    evaluated as double commutators in the Fock space.  For a complex omega
    the amplitudes solve

        ( M - omega N ) (X, Y)^T = -(f, g)^T ,
        M = [[A, B], [B*, A*]],   N = diag(1, -1),

    which is Eq. (20) of Nakatsukasa et al.  This is the reference that the
    finite amplitude method is measured against.
    """

    def __init__(self, mf):
        self.mf = mf
        self.space = ParticleHoleSpace(mf)
        r = rpa.tda_rpa(mf.fock, mf.N, mf.g, mf.f, mf.xi)
        assert r["pairs"] == self.space.pairs
        self.A, self.B = r["A"], r["B"]
        n = self.space.size
        self.M = np.block([[self.A, self.B],
                           [self.B.conj(), self.A.conj()]])
        self.metric = np.diag(np.concatenate([np.ones(n), -np.ones(n)]))
        self.omega, self.X, self.Y = self._modes()

    def _modes(self):
        """Positive RPA roots and their amplitudes, normalised in the RPA metric.

        The roots of this model are degenerate, and inside a degenerate
        block the eigenvectors returned by ``eig`` are not orthogonal in the
        indefinite metric N = diag(1, -1).  The block is therefore
        orthonormalised with G = Z^+ N Z, Z -> Z G^{-1/2}, which leaves the
        eigenvectors of distinct roots untouched and makes every mode
        satisfy sum_mi (|X_mi|^2 - |Y_mi|^2) = 1, Eq. (11) of the paper.
        """
        n = self.space.size
        w, v = eig(self.metric @ self.M)
        keep = np.where((w.real > 1e-8) & (np.abs(w.imag) < 1e-8))[0]
        keep = keep[np.argsort(w.real[keep])]
        omega = w.real[keep]
        Z = v[:, keep]
        G = Z.conj().T @ self.metric @ Z
        G = 0.5 * (G + G.conj().T)
        lam, U = eigh(G)
        if lam.min() <= 0.0:
            raise RuntimeError("a positive root with non-positive norm")
        Z = Z @ U @ np.diag(1.0 / np.sqrt(lam)) @ U.conj().T
        return omega, Z[:n], Z[n:]

    def amplitudes(self, F_hf, omega):
        """X(omega), Y(omega) from a direct linear solve."""
        f, g = self.space.ph(F_hf)
        rhs = -np.concatenate([f, g]).astype(complex)
        z = solve(self.M - omega * self.metric, rhs)
        n = self.space.size
        return z[:n], z[n:]

    def strength(self, F_hf, omega):
        X, Y = self.amplitudes(F_hf, omega)
        return self.space.strength(F_hf, X, Y)

    def transition_amplitudes(self, F_hf):
        """<nu|F|0> = sum_mi ( X^nu*_mi f_mi + Y^nu*_mi g_mi )."""
        f, g = self.space.ph(F_hf)
        return self.X.conj().T @ f + self.Y.conj().T @ g

    def spectral_strength(self, F_hf, omega):
        """S from the eigenmodes, Eq. (35) of the paper."""
        t = self.transition_amplitudes(F_hf)
        tdag = self.transition_amplitudes(F_hf.conj().T)
        return (np.sum(np.abs(t)**2 / (omega - self.omega))
                - np.sum(np.abs(tdag)**2 / (omega + self.omega)))

    def moments(self, F_hf, k=(0, 1)):
        t = np.abs(self.transition_amplitudes(F_hf))**2
        return [float(np.sum(self.omega**p * t)) for p in k]


# ---------------------------------------------------------------------------
#  2. The finite amplitude method
# ---------------------------------------------------------------------------
class FiniteAmplitudeMethod:
    """Linear response without the residual interaction.

    Parameters
    ----------
    mf : MeanField
    eta : the finite-amplitude parameter.  With ``scale=True`` (the choice
        of the paper) the actual displacement is eta / max(|X|, |Y|), so
        that the orbitals are always perturbed by the same small amount.
    scheme : 'forward' evaluates (h[psi', psi] - h0)/eta as in Eq. (31) of
        the paper; 'central' uses (h[+eta] - h[-eta])/(2 eta), which removes
        the O(eta) error of the forward difference; 'linear' applies
        (dh/drho) drho exactly and serves as a check.
    """

    def __init__(self, mf, eta=1e-5, scale=True, scheme="forward"):
        self.mf = mf
        self.space = ParticleHoleSpace(mf)
        self.eta = eta
        self.scale = scale
        self.scheme = scheme
        self.n, self.N = mf.n, mf.N
        self.phi = mf.C[:, :self.N]          # the HF orbitals, as columns
        self.particles = mf.C[:, self.N:]
        self.h0_hf = mf.to_hf_basis(mf.h0)

    # ------------------------------------------------------------------
    def _orbitals(self, X, Y, eta):
        """|psi_i> = |phi_i> + eta |X_i>,  <psi'_i| = <phi_i| + eta <Y_i|."""
        nP = self.n - self.N
        Xm = X.reshape(nP, self.N)
        Ym = Y.reshape(nP, self.N)
        ket = self.phi + eta * (self.particles @ Xm)
        # |Y_i> = sum_m |phi_m> Y*_mi, so the components carry Y*, and the
        # bra <psi'_i| = <phi_i| + eta sum_m Y_mi <phi_m| carries Y.
        bra = self.phi + eta * (self.particles @ Ym.conj())
        return bra, ket

    def induced_field(self, X, Y):
        """dh(omega) in the HF basis, by finite differences."""
        eta = self.eta
        if self.scale:
            eta = eta / max(np.abs(X).max(), np.abs(Y).max(), 1e-300)
        if self.scheme == "forward":
            bra, ket = self._orbitals(X, Y, eta)
            dh = (self.mf.h_braket(bra, ket) - self.mf.h0) / eta
        elif self.scheme == "central":
            bra, ket = self._orbitals(X, Y, eta)
            brm, ktm = self._orbitals(X, Y, -eta)
            dh = (self.mf.h_braket(bra, ket)
                  - self.mf.h_braket(brm, ktm)) / (2.0 * eta)
        elif self.scheme == "linear":
            nP = self.n - self.N
            Xm = X.reshape(nP, self.N)
            Ym = Y.reshape(nP, self.N)
            drho = (self.particles @ Xm @ self.phi.T
                    + self.phi @ Ym.T @ self.particles.T)
            dh = self.mf.h(drho) - self.mf.t
        else:
            raise ValueError(self.scheme)
        return self.mf.to_hf_basis(dh)

    # ------------------------------------------------------------------
    def residual(self, F_hf, omega, X, Y):
        """The left-hand side minus the right-hand side of the FAM equations."""
        f, g = self.space.ph(F_hf)
        dh = self.induced_field(X, Y)
        dh_ph, dh_hp = self.space.ph(dh)
        rX = (omega - self.space.delta) * X - dh_ph - f
        rY = (-omega - self.space.delta) * Y - dh_hp - g
        return rX, rY

    def operator(self, omega):
        """The FAM map (X, Y) -> left-hand side, as a matrix-free operator."""
        n = self.space.size

        def matvec(z):
            X, Y = z[:n], z[n:]
            dh = self.induced_field(X, Y)
            dh_ph, dh_hp = self.space.ph(dh)
            return np.concatenate([(omega - self.space.delta) * X - dh_ph,
                                   (-omega - self.space.delta) * Y - dh_hp])

        return LinearOperator((2 * n, 2 * n), matvec=matvec, dtype=complex)

    def solve(self, F_hf, omega, method="gmres", tol=1e-6, maxiter=2000,
              mixing=0.5, x0=None):
        """X(omega), Y(omega) for the field F at the complex frequency omega.

        Returns the amplitudes, the number of iterations and the final
        relative residual.  ``method`` is 'gmres', 'bicgstab' (both
        matrix-free Krylov solvers) or 'fixed' for the plain iteration

            X <- (f + dh_ph)/(omega - Delta),  Y <- -(g + dh_hp)/(omega + Delta)

        with linear mixing, which is the simplest thing that can work.

        The tolerance cannot usefully be smaller than the accuracy of the
        finite difference itself: with the forward scheme the operator is
        linear only to a relative O(eta), and a Krylov solver asked for more
        than that stagnates.  The paper's choice is tol = eta = 1e-5.
        """
        n = self.space.size
        f, g = self.space.ph(F_hf)
        rhs = np.concatenate([f, g]).astype(complex)
        count = [0]

        if method == "fixed":
            X = np.zeros(n, complex) if x0 is None else x0[:n].copy()
            Y = np.zeros(n, complex) if x0 is None else x0[n:].copy()
            for it in range(1, maxiter + 1):
                dh = self.induced_field(X, Y)
                dh_ph, dh_hp = self.space.ph(dh)
                Xn = (f + dh_ph) / (omega - self.space.delta)
                Yn = -(g + dh_hp) / (omega + self.space.delta)
                X = (1 - mixing) * X + mixing * Xn
                Y = (1 - mixing) * Y + mixing * Yn
                rX, rY = self.residual(F_hf, omega, X, Y)
                res = np.linalg.norm(np.concatenate([rX, rY]))
                res /= np.linalg.norm(rhs)
                if res < tol or not np.isfinite(res) or res > 1e6:
                    break
            if not np.isfinite(res) or res > 1e6:
                res = np.inf
            return X, Y, it, res

        def callback(_):
            count[0] += 1

        op = self.operator(omega)
        if method == "gmres":
            restart = min(2 * n, 50)
            z, info = gmres(op, rhs, x0=x0, rtol=tol, atol=0.0,
                            restart=restart,
                            maxiter=max(1, maxiter // restart),
                            callback=callback, callback_type="pr_norm")
        elif method == "bicgstab":
            z, info = bicgstab(op, rhs, x0=x0, rtol=tol, atol=0.0,
                               maxiter=maxiter, callback=callback)
        else:
            raise ValueError(method)
        X, Y = z[:n], z[n:]
        rX, rY = self.residual(F_hf, omega, X, Y)
        res = np.linalg.norm(np.concatenate([rX, rY])) / np.linalg.norm(rhs)
        return X, Y, count[0], res

    def strength(self, F_hf, omega, **kw):
        X, Y, _, _ = self.solve(F_hf, omega, **kw)
        return self.space.strength(F_hf, X, Y)


# ---------------------------------------------------------------------------
#  3. The exact response, from the full diagonalisation of chapter 5
# ---------------------------------------------------------------------------
class ExactResponse:
    """The exact strength function of the N-particle system.

    The Hamiltonian and the field are built as sparse Fock-space operators
    and restricted to the N-particle sector; every eigenstate is then known
    and S(F; omega) is a sum over poles at the exact excitation energies.
    """

    def __init__(self, fock, N, g, f, xi=1.0):
        self.fock = fock
        self.N = N
        self.states = np.array([s for s in range(fock.dim)
                                if popcount(s) == N])
        H = fock.hamiltonian(g, f, xi).toarray()
        self.H = H[np.ix_(self.states, self.states)]
        self.E, self.V = eigh(self.H)
        self.omega = self.E[1:] - self.E[0]

    def operator(self, F):
        """sum_ab F_ab a^+_a a_b in the N-particle sector."""
        fock = self.fock
        op = None
        for a in range(fock.norb):
            for b in range(fock.norb):
                if abs(F[a, b]) > 1e-14:
                    term = F[a, b] * (fock.cdag[a] @ fock.c[b])
                    op = term if op is None else op + term
        return op.toarray()[np.ix_(self.states, self.states)]

    def transition_amplitudes(self, F):
        Fop = self.operator(F)
        return (self.V.T @ Fop @ self.V[:, 0])[1:]

    def strength(self, F, omega):
        t = self.transition_amplitudes(F)
        tdag = self.transition_amplitudes(F.conj().T)
        return (np.sum(np.abs(t)**2 / (omega - self.omega))
                - np.sum(np.abs(tdag)**2 / (omega + self.omega)))

    def moments(self, F, k=(0, 1)):
        t = np.abs(self.transition_amplitudes(F))**2
        return [float(np.sum(self.omega**p * t)) for p in k]


def energy_weighted_sum_rule(fock, reference, H, F):
    """m_1 = (1/2) <0|[F^+, [H, F]]|0>, evaluated as a double commutator.

    With |0> the Hartree-Fock determinant this is Thouless' theorem for the
    RPA; with the exact ground state it is the exact sum rule.
    """
    Fop = None
    for a in range(fock.norb):
        for b in range(fock.norb):
            if abs(F[a, b]) > 1e-14:
                term = F[a, b] * (fock.cdag[a] @ fock.c[b])
                Fop = term if Fop is None else Fop + term
    A, _ = rpa.build_AB(H, reference, [Fop.tocsr()])
    return 0.5 * float(A[0, 0].real)


# ---------------------------------------------------------------------------
#  4. Poles by contour integration
# ---------------------------------------------------------------------------
def contour_moments(strength, center, radius, npoints=32):
    """Contour integrals of the strength function around a circle.

    S(F; omega) is meromorphic with a simple pole at every RPA root, with
    residue |<nu|F|0>|^2.  For a circle C of the given centre and radius,

        (1/2 pi i) oint_C S(omega) d omega          = sum_{nu in C} |<nu|F|0>|^2
        (1/2 pi i) oint_C omega S(omega) d omega    = sum_{nu in C} omega_nu |<nu|F|0>|^2 ,

    so the ratio of the two returns the energy of a single enclosed pole.
    The trapezoidal rule on a circle converges exponentially.  ``strength``
    is any callable omega -> S(omega); with the finite amplitude method this
    needs one FAM solution per quadrature point, and no smoothing Gamma.
    """
    theta = 2.0 * np.pi * (np.arange(npoints) + 0.5) / npoints
    omegas = center + radius * np.exp(1j * theta)
    S = np.array([strength(w) for w in omegas])
    dw = 1j * radius * np.exp(1j * theta) * (2.0 * np.pi / npoints)
    m0 = np.sum(S * dw) / (2j * np.pi)
    m1 = np.sum(omegas * S * dw) / (2j * np.pi)
    return m0, m1


def correlation_energy_fam(solver, center, radius, npoints=48):
    """The RPA correlation energy from FAM solutions alone.

    Chapter 7 writes E_corr = (1/2)(sum_nu omega_nu - Tr A).  Both terms are
    available without A: for the unit field F_K = a^+_m a_i the strength
    function has residue |X^nu_K|^2 at +omega_nu and -|Y^nu_K|^2 at
    -omega_nu, so with a circle C+ around the positive roots and its mirror
    image C- around the negative ones

        sum_nu omega_nu = sum_K [ (1/2 pi i) oint_{C+} omega S(F_K) d omega
                                - (1/2 pi i) oint_{C-} omega S(F_K) d omega ] ,

    by the normalisation sum_K (|X^nu_K|^2 - |Y^nu_K|^2) = 1, while
    A_KK = Delta_K + dh_K[X = e_K, Y = 0] is one induced field per pair.
    """
    space = solver.space
    n, nn = space.size, solver.n
    sum_omega, trace_A = 0.0, 0.0
    for K, (m, i) in enumerate(space.pairs):
        F = np.zeros((nn, nn))
        F[m, i] = 1.0
        _, m1_plus = contour_moments(lambda z: solver.strength(F, z),
                                     center, radius, npoints)
        _, m1_minus = contour_moments(lambda z: solver.strength(F, z),
                                      -center, radius, npoints)
        sum_omega += (m1_plus - m1_minus).real
        X = np.zeros(n, complex)
        X[K] = 1.0
        dh_ph, _ = space.ph(solver.induced_field(X, np.zeros(n, complex)))
        trace_A += space.delta[K] + dh_ph[K].real
    return 0.5 * (sum_omega - trace_A), sum_omega, trace_A


# ---------------------------------------------------------------------------
#  Demonstrations
# ---------------------------------------------------------------------------
def setup(g=1.0, f=0.5, L=4, N=4):
    fock = FockSpace(L)
    mf = MeanField(fock, N, g, f)
    return fock, mf


def demo_setup():
    print("=" * 74)
    print("1. The mean field and the particle-hole space")
    print("=" * 74)
    fock, mf = setup()
    print(f"L = {fock.L} levels, N = {mf.N} particles, g = {mf.g}, f = {mf.f}")
    print(f"E_HF = {mf.energy():.6f}")
    print("HF single-particle energies:",
          "  ".join(f"{e:7.4f}" for e in mf.eps))
    ph = ParticleHoleSpace(mf)
    print(f"particle-hole pairs: {ph.size}, unperturbed energies "
          f"{np.unique(np.round(ph.delta, 4))}")
    print()
    print("h[rho] returns the HF energies when fed the HF density:")
    h0 = mf.to_hf_basis(mf.h0)
    print(f"  max |h0 - diag(eps)| = {np.abs(h0 - np.diag(mf.eps)).max():.1e}")
    print("and it is linear in rho, so the finite difference is exact up to")
    print("the quadratic term of rho = sum |psi><psi'| and to round-off.")


def demo_validation():
    print("=" * 74)
    print("2. FAM against the RPA matrix, one frequency at a time")
    print("=" * 74)
    fock, mf = setup()
    matrix = MatrixResponse(mf)
    F = mf.to_hf_basis(field_level(fock.L))
    print("The induced field from finite differences against (dh/drho) drho,")
    print("for a random amplitude vector:")
    rng = np.random.default_rng(1)
    n = matrix.space.size
    X = rng.normal(size=n) + 1j * rng.normal(size=n)
    Y = rng.normal(size=n) + 1j * rng.normal(size=n)
    dh_lin = FiniteAmplitudeMethod(mf, scheme="linear").induced_field(X, Y)
    dh_ph, dh_hp = matrix.space.ph(dh_lin)
    ref = np.concatenate([matrix.A @ X + matrix.B @ Y
                          - matrix.space.delta * X,
                          matrix.B.conj() @ X + matrix.A.conj() @ Y
                          - matrix.space.delta * Y])
    print(f"  max |dh(linear) - (A - Delta)X - BY| = "
          f"{np.abs(np.concatenate([dh_ph, dh_hp]) - ref).max():.1e}")
    for scheme in ("forward", "central"):
        for eta in (1e-2, 1e-4, 1e-6, 1e-8):
            fam = FiniteAmplitudeMethod(mf, eta=eta, scale=True,
                                        scheme=scheme)
            dh = fam.induced_field(X, Y)
            err = np.abs(dh - dh_lin).max()
            print(f"  {scheme:8s} eta = {eta:.0e}:  max |dh - dh(linear)| = "
                  f"{err:.1e}")
    print()
    print("The forward difference carries the O(eta) error of the term")
    print("eta^2 sum_i |X_i><Y_i| in the density; the central difference")
    print("cancels it, and both eventually lose to round-off, which grows as")
    print("1/eta.  This is the finite-difference trade-off of the paper.")
    print()
    for omega in (1.7 + 0.05j, 2.5 + 0.05j):
        print()
        print(f"Strength function at omega = {omega}:")
        S_mat = matrix.strength(F, omega)
        S_spec = matrix.spectral_strength(F, omega)
        print(f"  RPA matrix, linear solve      S = {S_mat:.10f}")
        print(f"  RPA matrix, spectral sum      S = {S_spec:.10f}")
        for scheme, method in (("forward", "gmres"), ("forward", "bicgstab"),
                               ("forward", "fixed"), ("central", "gmres")):
            fam = FiniteAmplitudeMethod(mf, scheme=scheme)
            X, Y, it, res = fam.solve(F, omega, method=method, tol=1e-6)
            S = fam.space.strength(F, X, Y)
            if np.isfinite(res):
                print(f"  FAM {scheme:7s} {method:8s} S = {S:.10f}   "
                      f"iterations {it:4d}   residual {res:.1e}   "
                      f"|S - S_RPA| = {abs(S - S_mat):.1e}")
            else:
                print(f"  FAM {scheme:7s} {method:8s} diverges after "
                      f"{it} iterations")
    print()
    print("With eta = 1e-5 the forward difference reproduces the matrix RPA")
    print("to five or six digits, which is the 'three to four digits' of the")
    print("paper with a smaller eta; the central difference is exact for a")
    print("Hamiltonian linear in the density.  The plain iteration diverges")
    print("close to a pole and converges away from it; the Krylov solvers do")
    print("not care.")


def demo_strength():
    print("=" * 74)
    print("3. Strength functions: exact, RPA matrix and FAM")
    print("=" * 74)
    fock, mf = setup()
    matrix = MatrixResponse(mf)
    exact = ExactResponse(fock, mf.N, mf.g, mf.f)
    fam = FiniteAmplitudeMethod(mf)
    gamma = 0.1
    print(f"smoothing Gamma = {gamma}, i.e. omega -> omega + i Gamma/2")
    for name, F0 in (("level operator Q_0", field_level(fock.L)),
                     ("hopping operator Q_1", field_hopping(fock.L))):
        F = mf.to_hf_basis(F0)
        print()
        print(f"  {name}")
        print(f"  {'omega':>6s} {'exact':>12s} {'RPA':>12s} {'FAM':>12s} "
              f"{'|FAM-RPA|':>10s} {'iter':>5s}")
        for w in (0.5, 1.0, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5):
            z = w + 0.5j * gamma
            X, Y, it, _ = fam.solve(F, z)
            s_fam = db_domega(fam.space.strength(F, X, Y))
            s_rpa = db_domega(matrix.strength(F, z))
            s_ex = db_domega(exact.strength(F0, z))
            print(f"  {w:6.2f} {s_ex:12.6f} {s_rpa:12.6f} {s_fam:12.6f} "
                  f"{abs(s_fam - s_rpa):10.1e} {it:5d}")
        print(f"  RPA roots       : {np.round(np.unique(np.round(matrix.omega, 6)), 4)}")
        print(f"  exact excitations with strength > 1e-3: "
              f"{np.round(exact.omega[np.abs(exact.transition_amplitudes(F0))**2 > 1e-3], 4)}")


def demo_sum_rules():
    print("=" * 74)
    print("4. Sum rules and the poles by contour integration")
    print("=" * 74)
    fock, mf = setup()
    matrix = MatrixResponse(mf)
    exact = ExactResponse(fock, mf.N, mf.g, mf.f)
    fam = FiniteAmplitudeMethod(mf)
    fam_c = FiniteAmplitudeMethod(mf, scheme="central")
    H = fock.hamiltonian(mf.g, mf.f, mf.xi)
    creators = fock.rotate(mf.C)
    hf_state = fock.determinant(creators, mf.N)
    gs = np.zeros(fock.dim)
    gs[exact.states] = exact.V[:, 0]
    for name, F0 in (("Q_0", field_level(fock.L)),
                     ("Q_1", field_hopping(fock.L))):
        F = mf.to_hf_basis(F0)
        m0_rpa, m1_rpa = matrix.moments(F)
        m0_ex, m1_ex = exact.moments(F0)
        m1_thouless = energy_weighted_sum_rule(fock, hf_state, H, F0)
        m1_exact_dc = energy_weighted_sum_rule(fock, gs, H, F0)
        print(f"  field {name}")
        print(f"    RPA   m0 = {m0_rpa:.8f}  m1 = {m1_rpa:.8f}   "
              f"(1/2)<HF|[F,[H,F]]|HF> = {m1_thouless:.8f}")
        print(f"    exact m0 = {m0_ex:.8f}  m1 = {m1_ex:.8f}   "
              f"(1/2)<0|[F,[H,F]]|0>   = {m1_exact_dc:.8f}")
        # contour around the lowest RPA root, and around all of them
        w1 = matrix.omega[0]
        for npts in (8, 16, 32):
            m0, m1 = contour_moments(lambda z: fam.strength(F, z),
                                     center=w1, radius=0.3, npoints=npts)
            print(f"    contour around omega_1 with {npts:2d} points: "
                  f"m0 = {m0.real:.8f}, m1/m0 = {(m1 / m0).real:.8f}"
                  f"   (omega_1 = {w1:.8f})")
        m0, m1 = contour_moments(lambda z: fam_c.strength(F, z),
                                 center=w1, radius=0.3, npoints=16)
        print(f"    the same with the central difference:    "
              f"m0 = {m0.real:.8f}, m1/m0 = {(m1 / m0).real:.8f}")
        m0, m1 = contour_moments(lambda z: fam.strength(F, z),
                                 center=3.0, radius=2.9, npoints=64)
        print(f"    contour enclosing every positive root:  m0 = {m0.real:.8f}"
              f"  m1 = {m1.real:.8f}")
    print()
    r = rpa.tda_rpa(fock, mf.N, mf.g, mf.f, mf.xi)
    print("The RPA correlation energy from contour integrals of the FAM")
    print("strength functions of the sixteen unit fields a^+_m a_i:")
    for scheme in ("forward", "central"):
        solver = FiniteAmplitudeMethod(mf, scheme=scheme)
        ecorr, sw, trA = correlation_energy_fam(solver, center=3.0,
                                                radius=2.9, npoints=48)
        print(f"  {scheme:8s} sum omega = {sw:.8f}  Tr A = {trA:.8f}  "
              f"E_corr = {ecorr:.8f}")
    print(f"  chapter 7 sum omega = {r['rpa'].sum():.8f}  "
          f"Tr A = {np.trace(r['A']):.8f}  E_corr = {r['ecorr']:.8f}")
    print()
    print("Thouless' theorem holds for the self-consistent RPA: the energy-")
    print("weighted sum of the RPA strengths equals the double commutator in")
    print("the Hartree-Fock state.  The contour integrals of the FAM strength")
    print("function return both moments without any smoothing, and the ratio")
    print("of the two locates the pole -- to the accuracy of the finite")
    print("difference, and to ten digits with the central one -- from a")
    print("solver that has never seen the matrices A and B.")


def demo_iterations():
    print("=" * 74)
    print("5. The cost: iterations against frequency and smoothing")
    print("=" * 74)
    fock, mf = setup()
    fam = FiniteAmplitudeMethod(mf)
    F = mf.to_hf_basis(field_hopping(fock.L))
    print(f"{'omega':>6s}", end="")
    gammas = (0.5, 0.2, 0.05)
    for gm in gammas:
        print(f"   gmres G={gm:<4}  fixed G={gm:<4}", end="")
    print()
    for w in (0.5, 1.0, 1.5, 1.65, 2.0, 2.7, 3.2, 4.0, 6.0):
        print(f"{w:6.2f}", end="")
        for gm in gammas:
            z = w + 0.5j * gm
            _, _, it_g, _ = fam.solve(F, z, method="gmres", tol=1e-6)
            _, _, it_f, res = fam.solve(F, z, method="fixed", tol=1e-6,
                                        mixing=0.5, maxiter=3000)
            flag = "" if res < 1e-6 else "*"
            print(f"   {it_g:12d}  {it_f:11d}{flag:1s}", end="")
        print()
    print("(* : not converged in 3000 iterations, or diverged)")
    print()
    print("The Krylov solver needs at most as many iterations as there are")
    print("unknowns, 2 x 16 here, and is insensitive to Gamma.  The plain")
    print("iteration is driven by the size of the residual interaction")
    print("relative to |omega - Delta|, so it slows down or fails near a pole")
    print("and for small Gamma.  The paper reports the same trend for the")
    print("coordinate-space case, where the unknowns number in the millions")
    print("and a Krylov solver is the only option.")


def _demo():
    for f in (demo_setup, demo_validation, demo_strength, demo_sum_rules,
              demo_iterations):
        f()
        print()


if __name__ == "__main__":
    _demo()

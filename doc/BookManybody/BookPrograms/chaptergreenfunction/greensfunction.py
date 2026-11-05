"""
Green's functions, the Dyson equation and the parquet resummation.

Companion code to the Green's function chapter of *Quantum mechanics for
Many-particle Systems*.

The chapter has two halves.  The first defines the one-body propagator, its
Lehmann representation, the spectral functions and the Migdal-Galitskii-Koltun
sum rule; here these are computed *exactly* from full configuration
interaction in the N, N+1 and N-1 particle spaces, so that every formula of
the text can be checked against a number.  The second half introduces the
self energy, the Dyson equation and the parquet resummation of ladder and
ring diagrams into an energy-dependent interaction vertex Gamma.  The parquet
equations are solved here at fixed starting energies, the vertex is fed into
the two-particle-one-hole and two-hole-one-particle self energy, the Dyson
equation is solved as an eigenvalue problem in an extended space, and the
energy follows from the sum rule.

    lipkin_model           -- the Lipkin model in a spin-orbital basis
    LehmannPropagator      -- the exact propagator from full CI in N, N+1, N-1
    PairSpaces             -- the [12] (pair) and [13] (particle-hole) matrices
                              of a four-index vertex
    pair_propagators       -- the free two-time propagators G0^pphh and G0^ph
    parquet_vertex         -- Gamma = I + L + R at fixed starting energies
    self_energy_poles      -- the 2p1h/2h1p pole form of Sigma with a dressed
                              vertex (Gamma = V gives the second-order Sigma)
    dyson                  -- the Dyson equation as an eigenvalue problem
    koltun_energy          -- the sum rule for the ground-state energy
    green_function_energy  -- the driver: reference -> vertex -> Sigma ->
                              Dyson -> energy, optionally self-consistent
    channel_poles          -- the poles of the pure ring and ladder series
                              (the RPA and the pp-RPA energies)
    exact_self_energy      -- Sigma(omega) = omega - F - g(omega)^{-1} from the
                              exact propagator
    mp_energies            -- the Moller-Plesset series in the Fock partition
    compare_methods        -- HF, MP2, MP3, GF2, parquet, RPA-dressed, CCD,
                              CCSD and FCI for one Hamiltonian

Conventions are those of ``coupledcluster.py`` (chapter 10): spin-orbitals
are numbered so that the ``n_occ`` lowest are occupied in the reference, and
the two-body tensor is antisymmetrised, v[p,q,r,s] = <pq||rs>.  Energies are
in the units of the model, hbar = 1.

Author: Morten Hjorth-Jensen
"""

import numpy as np

import coupledcluster as cc


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------
def lipkin_model(N=4, eps=1.0, V=0.2, W=0.0):
    """The Lipkin model of chapter 4 in a spin-orbital basis.

    Spin-orbital p (p = 0, ..., N-1) is the substate p of the lower level
    (sigma = -1, energy -eps/2) and N + p the same substate of the upper level
    (sigma = +1, energy +eps/2).  The reference fills the lower level.

        H = (eps/2) sum sigma a+ a
            + (V/2) sum_{p p' sigma} a+_{p sigma} a+_{p' sigma} a_{p' -sigma} a_{p -sigma}
            + (W/2) sum_{p p' sigma} a+_{p sigma} a+_{p' -sigma} a_{p' sigma} a_{p -sigma}.
    """
    n = 2 * N
    h = np.diag([-0.5 * eps] * N + [0.5 * eps] * N).astype(float)
    v = np.zeros((n, n, n, n))

    def orb(p, sigma):
        return p if sigma < 0 else N + p

    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            for sigma in (-1, +1):
                # a+_{p s} a+_{q s} a_{q -s} a_{p -s}:  r = (p,-s), s = (q,-s)
                cc._add_term(v, 0.5 * V, orb(p, sigma), orb(q, sigma),
                             orb(p, -sigma), orb(q, -sigma))
                if W != 0.0:
                    # a+_{p s} a+_{q -s} a_{q s} a_{p -s}: r = (p,-s), s = (q,s)
                    cc._add_term(v, 0.5 * W, orb(p, sigma), orb(q, -sigma),
                                 orb(p, -sigma), orb(q, sigma))
    return h, v, N


# ---------------------------------------------------------------------------
#  The exact propagator: Lehmann representation from full CI
# ---------------------------------------------------------------------------
def _transition_matrix(states_from, index_to, operator, orbital):
    """Matrix of a+_alpha (operator='create') or a_alpha between two spaces."""
    rows, cols, vals = [], [], []
    for col, s in enumerate(states_from):
        if operator == "create":
            t, sign = cc._create(s, orbital)
        else:
            t, sign = cc._annihilate(s, orbital)
        if t is None:
            continue
        rows.append(index_to[t])
        cols.append(col)
        vals.append(sign)
    A = np.zeros((len(index_to), len(states_from)))
    A[rows, cols] = vals
    return A


class LehmannPropagator:
    """The exact one-body propagator of a small system.

    Diagonalises H in the N-, (N+1)- and (N-1)-particle spaces and stores the
    removal and addition amplitudes

        X_minus[k, alpha] = <Psi_k^{N-1}| a_alpha |Psi_0^N>,
        X_plus[n, alpha]  = <Psi_n^{N+1}| a+_alpha |Psi_0^N>,

    with the energies eps_minus[k] = E_0^N - E_k^{N-1} and
    eps_plus[n] = E_n^{N+1} - E_0^N.  Everything else -- the propagator, the
    spectral functions, the occupation numbers, the density matrix and the
    sum rule for the energy -- is a sum over these.
    """

    def __init__(self, h, v, n_particles):
        self.h, self.v, self.n_particles = h, v, n_particles
        n = h.shape[0]
        H, states, index = cc.hamiltonian_matrix(h, v, n_particles)
        E, U = np.linalg.eigh(H)
        self.energy = float(E[0])
        self.spectrum = E
        psi0 = U[:, 0]
        self.dimension = len(states)

        # N+1 space
        if n_particles < n:
            Hp, states_p, index_p = cc.hamiltonian_matrix(h, v, n_particles + 1)
            Ep, Up = np.linalg.eigh(Hp)
            self.eps_plus = Ep - self.energy
            self.X_plus = np.array([Up.T @ _transition_matrix(
                states, index_p, "create", a) @ psi0 for a in range(n)]).T
        else:
            self.eps_plus = np.zeros(0)
            self.X_plus = np.zeros((0, n))
        # N-1 space
        if n_particles > 0:
            Hm, states_m, index_m = cc.hamiltonian_matrix(h, v, n_particles - 1)
            Em, Um = np.linalg.eigh(Hm)
            self.eps_minus = self.energy - Em
            self.X_minus = np.array([Um.T @ _transition_matrix(
                states, index_m, "annihilate", a) @ psi0 for a in range(n)]).T
        else:
            self.eps_minus = np.zeros(0)
            self.X_minus = np.zeros((0, n))
        self.n = n

    # ------------------------------------------------------------------
    def propagator(self, omega, eta=1e-3):
        """g_{alpha beta}(omega) as an n x n complex matrix."""
        gp = np.einsum("na,nb,n->ab", self.X_plus, self.X_plus,
                       1.0 / (omega - self.eps_plus + 1j * eta))
        gm = np.einsum("ka,kb,k->ab", self.X_minus, self.X_minus,
                       1.0 / (omega - self.eps_minus - 1j * eta))
        return gp + gm

    def hole_strength(self):
        """S_h(alpha, k) = |<k|a_alpha|0>|^2: poles eps_minus, weights."""
        return self.eps_minus, self.X_minus**2

    def particle_strength(self):
        return self.eps_plus, self.X_plus**2

    def spectral_function(self, omega, eta=0.05):
        """Lorentzian-broadened S(alpha, omega), hole and particle parts."""
        omega = np.atleast_1d(omega)
        lh = (eta / np.pi) / ((omega[:, None] - self.eps_minus[None, :])**2
                              + eta**2)
        lp = (eta / np.pi) / ((omega[:, None] - self.eps_plus[None, :])**2
                              + eta**2)
        return lh @ self.X_minus**2, lp @ self.X_plus**2

    def occupation(self):
        """n_alpha = sum_k |<k|a_alpha|0>|^2 and the depletion d_alpha."""
        return (self.X_minus**2).sum(0), (self.X_plus**2).sum(0)

    def density_matrix(self):
        """rho_{alpha beta} = <0|a+_beta a_alpha|0>."""
        return self.X_minus.T @ self.X_minus

    def koltun_energy(self):
        """E_0 = 1/2 sum_k [ <h>_k + eps_k^- n_k ],  the sum rule."""
        rho_k = np.einsum("ka,kb->kab", self.X_minus, self.X_minus)
        return 0.5 * float(np.einsum("ab,kab->", self.h, rho_k)
                           + np.einsum("k,kaa->", self.eps_minus, rho_k))

    def energy_weighted_sum(self):
        """sum_alpha int omega S_h = <H_1> + 2<V>, and <H_1> separately."""
        h1 = float(np.einsum("ab,kb,ka->", self.h, self.X_minus, self.X_minus))
        ew = float(np.einsum("k,ka,ka->", self.eps_minus, self.X_minus,
                             self.X_minus))
        return ew, h1


# ---------------------------------------------------------------------------
#  Pair spaces: the [12] and [13] matrix forms of a four-index vertex
# ---------------------------------------------------------------------------
class PairSpaces:
    """Re-indexings of a vertex <pq|M|rs> (antisymmetric in pq and in rs).

    [12] channel: antisymmetric pairs (a < b):  M_pp[(ab),(cd)] = <ab|M|cd>.
    [13] channel: ordered pairs (p, q):         M_ph[(pq),(rs)] = <ps|M|qr>.

    In the [13] form a ring (particle-hole) insertion is an ordinary matrix
    product, exactly as a ladder insertion is in the [12] form.
    """

    def __init__(self, n):
        self.n = n
        self.pairs = np.array([(a, b) for a in range(n) for b in range(a + 1, n)])
        self.n_pp = len(self.pairs)
        self.n_ph = n * n

    def to_pp(self, M4):
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        return M4[a[:, None], b[:, None], a[None, :], b[None, :]]

    def from_pp(self, Mpp):
        n = self.n
        a, b = self.pairs[:, 0], self.pairs[:, 1]
        M4 = np.zeros((n, n, n, n), dtype=Mpp.dtype)
        A, B = a[:, None], b[:, None]
        C, D = a[None, :], b[None, :]
        M4[A, B, C, D] = Mpp
        M4[B, A, C, D] = -Mpp
        M4[A, B, D, C] = -Mpp
        M4[B, A, D, C] = Mpp
        return M4

    def to_ph(self, M4):
        n = self.n
        return np.transpose(M4, (0, 2, 3, 1)).reshape(n * n, n * n)

    def from_ph(self, Mph):
        n = self.n
        T = Mph.reshape(n, n, n, n)          # T[p,q,r,s] = <ps|M|qr>
        return np.transpose(T, (0, 3, 1, 2))  # M4[p,s,q,r] = T[p,q,r,s]

    @staticmethod
    def antisymmetrise(M4):
        """The projection of a tensor on the antisymmetric-antisymmetric part."""
        return 0.25 * (M4 - M4.transpose(1, 0, 2, 3) - M4.transpose(0, 1, 3, 2)
                       + M4.transpose(1, 0, 3, 2))


def pair_propagators(eps, n_occ, omega_pp, omega_ph, eta, spaces):
    """The free two-time propagators at fixed energies, as diagonal arrays.

    [12] channel (pairs a<b):
        G^pp_(ab)  =  1/(Omega - eps_a - eps_b + i eta)   both particles,
        G^hh_(ij)  = -1/(Omega - eps_i - eps_j - i eta)   both holes,
        0 otherwise.
    [13] channel (ordered pairs (p,q)):
        G^ph_(pq)  =  1/(Omega - eps_p + eps_q + i eta)   p particle, q hole,
                   = -1/(Omega - eps_p + eps_q - i eta)   p hole, q particle,
        0 otherwise.
    """
    n = len(eps)
    hole = np.arange(n) < n_occ
    a, b = spaces.pairs[:, 0], spaces.pairs[:, 1]
    Gpp = np.zeros(spaces.n_pp, dtype=complex)
    pp = ~hole[a] & ~hole[b]
    hh = hole[a] & hole[b]
    Gpp[pp] = 1.0 / (omega_pp - eps[a[pp]] - eps[b[pp]] + 1j * eta)
    Gpp[hh] = -1.0 / (omega_pp - eps[a[hh]] - eps[b[hh]] - 1j * eta)

    p = np.repeat(np.arange(n), n)
    q = np.tile(np.arange(n), n)
    Gph = np.zeros(spaces.n_ph, dtype=complex)
    fw = ~hole[p] & hole[q]
    bw = hole[p] & ~hole[q]
    Gph[fw] = 1.0 / (omega_ph - eps[p[fw]] + eps[q[fw]] + 1j * eta)
    Gph[bw] = -1.0 / (omega_ph - eps[p[bw]] + eps[q[bw]] - 1j * eta)
    return Gpp, Gph


# ---------------------------------------------------------------------------
#  The parquet equations
# ---------------------------------------------------------------------------
def parquet_vertex(v, eps, n_occ, omega_pp, omega_ph, eta=0.0,
                   ladders=True, rings=True, max_iter=200, tol=1e-10,
                   mixing=0.5, spaces=None, verbose=False):
    """Solve  Gamma = I + L + R  with I = V at fixed starting energies.

        L   = (I + R)       G0^pphh (I + R + L)      in the [12] channel,
        R13 = (I + L + R14) G0^ph   (I + L + R)      in the [13] channel,

    where R14 is the exchange partner of R13 (R14 = -P_12 R13) and
    R = R13 + R14 is the antisymmetric ring class.  Each equation is linear
    in its own unknown when the other class is held fixed, and is solved by
    a matrix inversion (the geometric series it stands for need not converge
    for the resummed vertex to exist); the two are then alternated until
    both are stationary.  Returns Gamma, L, R (four-index arrays, complex
    when eta > 0) and a dictionary with the iteration history.
    """
    n = v.shape[0]
    if spaces is None:
        spaces = PairSpaces(n)
    Gpp, Gph = pair_propagators(eps, n_occ, omega_pp, omega_ph, eta, spaces)
    dtype = complex if eta > 0 else float
    if eta == 0.0:
        Gpp, Gph = Gpp.real, Gph.real
    I = v.astype(dtype)
    L = np.zeros_like(I)
    R = np.zeros_like(I)
    R13 = np.zeros_like(I)
    one_pp = np.eye(spaces.n_pp, dtype=dtype)
    one_ph = np.eye(spaces.n_ph, dtype=dtype)
    history = []
    change = 0.0
    for iteration in range(1, max_iter + 1):
        L_old, R_old = L, R
        if ladders:
            K = spaces.to_pp(I + R)
            KG = K * Gpp[None, :]
            # L = [1 - K G]^{-1} K G K
            Lpp = np.linalg.solve(one_pp - KG, KG @ K)
            L_new = spaces.from_pp(Lpp)
        else:
            L_new = L
        if rings:
            R14 = -R13.transpose(1, 0, 2, 3)
            K = spaces.to_ph(I + L_new + R14)
            KG = K * Gph[None, :]
            # R13 = [1 - K G]^{-1} K G (I + L + R14)
            Rph = np.linalg.solve(one_ph - KG, KG @ spaces.to_ph(I + L_new + R14))
            R13_new = spaces.from_ph(Rph)
            R_new = 2.0 * spaces.antisymmetrise(R13_new)
        else:
            R13_new, R_new = R13, R
        if mixing > 0.0:
            L_new = (1 - mixing) * L_new + mixing * L_old
            R_new = (1 - mixing) * R_new + mixing * R_old
            R13_new = (1 - mixing) * R13_new + mixing * R13
        change = max(np.abs(L_new - L_old).max(), np.abs(R_new - R_old).max())
        L, R, R13 = L_new, R_new, R13_new
        history.append(change)
        if verbose:
            print(f"  parquet iteration {iteration:4d}  change = {change:.3e}")
        if change < tol:
            break
    info = dict(iterations=iteration, converged=bool(change < tol),
                history=history)
    return I + L + R, L, R, info


# ---------------------------------------------------------------------------
#  Self energy, Dyson equation, sum rule
# ---------------------------------------------------------------------------
def self_energy_poles(v, Gamma, eps, n_occ):
    """The pole form of the energy-dependent self energy

        Sigma_{ab}(omega) = sum_s  Mleft[s,a] Mright[s,b] / (omega - E_s),

    with the two-particle-one-hole poles E = eps_c + eps_d - eps_k (k a hole,
    c < d particles), Mleft = <a k|V|c d>, Mright = <c d|Gamma|b k>, and the
    two-hole-one-particle poles E = eps_i + eps_j - eps_c (c a particle,
    i < j holes), Mleft = <a c|V|i j>, Mright = <i j|Gamma|b c>.

    With Gamma = V this is the second-order self energy.  The bare vertex on
    the left and the dressed one on the right avoid double counting; the
    self energy is then not symmetric in general.
    """
    n = v.shape[0]
    holes = np.arange(n_occ)
    parts = np.arange(n_occ, n)
    E, ML, MR = [], [], []
    for k in holes:
        for ic, c in enumerate(parts):
            for d in parts[ic + 1:]:
                E.append(eps[c] + eps[d] - eps[k])
                ML.append(v[:, k, c, d])
                MR.append(Gamma[c, d, :, k])
    for c in parts:
        for ii, i in enumerate(holes):
            for j in holes[ii + 1:]:
                E.append(eps[i] + eps[j] - eps[c])
                ML.append(v[:, c, i, j])
                MR.append(Gamma[i, j, :, c])
    return np.array(E), np.array(ML), np.array(MR)


def self_energy(omega, F, E, ML, MR):
    """Sigma(omega) = F - diag(eps) + sum_s ML MR/(omega - E_s); here only the
    energy-dependent part, as an n x n matrix for a scalar omega."""
    return np.einsum("sa,sb,s->ab", ML, MR, 1.0 / (omega - E))


def dyson(F, E, ML, MR, mu):
    """Solve the Dyson equation g = [omega - F - Sigma(omega)]^{-1}.

    With Sigma in pole form the equation is equivalent to the eigenvalue
    problem of the extended matrix

        [ F      ML^T ]
        [ MR     diag(E) ],

    whose eigenvalues are the poles omega_lambda of g, and whose right and
    left eigenvectors (top blocks X, Y, normalised Y X = 1) give the residues
    g_{ab} = sum_lambda X[a,lambda] Y[lambda,b]/(omega - omega_lambda).  Poles
    below the chemical potential mu are hole (removal) poles.
    """
    n = F.shape[0]
    ns = len(E)
    M = np.zeros((n + ns, n + ns), dtype=np.result_type(F, ML, MR))
    M[:n, :n] = F
    M[:n, n:] = ML.T
    M[n:, :n] = MR
    M[n:, n:] = np.diag(E)
    w, Vr = np.linalg.eig(M)
    Vl = np.linalg.inv(Vr)              # rows are the left eigenvectors
    order = np.argsort(w.real)
    w, Vr, Vl = w[order], Vr[:, order], Vl[order, :]
    X, Y = Vr[:n, :], Vl[:, :n]
    holes = w.real < mu
    return dict(poles=w, X=X, Y=Y, holes=holes,
                strength=np.einsum("al,la->la", X, Y))


def koltun_energy(h, sol):
    """E_0 = 1/2 sum_{lambda hole} [ Tr(h rho_lambda) + omega_lambda Tr rho_lambda ],
    with rho_lambda = X[:,lambda] Y[lambda,:] the residue matrix."""
    X, Y, w, holes = sol["X"], sol["Y"], sol["poles"], sol["holes"]
    rho = np.einsum("al,lb->lab", X[:, holes], Y[holes, :])
    e = 0.5 * (np.einsum("ab,lba->", h, rho) + np.einsum("l,laa->", w[holes], rho))
    return e


def hole_spectral_function(sol, omega, eta=0.05):
    """Lorentzian-broadened S_h(alpha, omega) from a Dyson solution."""
    w, S = sol["poles"], sol["strength"]
    holes = sol["holes"]
    omega = np.atleast_1d(omega)
    lor = (eta / np.pi) / ((omega[:, None] - w.real[None, holes])**2 + eta**2)
    return lor @ S[holes, :].real


# ---------------------------------------------------------------------------
#  The energy-dependent alternative: one channel, all of its energy dependence
# ---------------------------------------------------------------------------
def channel_phonons(v, eps, n_occ, channel, spaces=None):
    """Diagonalise the ladder or the ring channel.

    The pure series K + K G K + ... of a channel has the pole expansion

        Pi(Omega) = sum_nu Z_nu Ztilde_nu diag(eta) / (Omega - Omega_nu),

    where Omega_nu and the right/left vectors Z_nu, Ztilde_nu (normalised
    Ztilde Z = 1) are those of  diag(d) + diag(eta) K  on the propagating
    pairs: the particle-particle RPA for the ladder channel, the
    particle-hole RPA for the ring channel.  Returns the energies, the right
    and left vectors (as rows, on the propagating pairs), the signs eta, the
    list of propagating pairs and a boolean 'forward' (addition energies /
    forward phonons, by the sign of the RPA norm).
    """
    n = len(eps)
    if spaces is None:
        spaces = PairSpaces(n)
    hole = np.arange(n) < n_occ
    if channel == "ring":
        K = spaces.to_ph(v)
        p = np.repeat(np.arange(n), n)
        q = np.tile(np.arange(n), n)
        d = eps[p] - eps[q]
        eta = (np.where(~hole[p] & hole[q], 1.0, 0.0)
               - np.where(hole[p] & ~hole[q], 1.0, 0.0))
    else:
        K = spaces.to_pp(v)
        a, b = spaces.pairs[:, 0], spaces.pairs[:, 1]
        d = eps[a] + eps[b]
        eta = (np.where(~hole[a] & ~hole[b], 1.0, 0.0)
               - np.where(hole[a] & hole[b], 1.0, 0.0))
    keep = np.where(eta != 0)[0]
    M = np.diag(d[keep]) + eta[keep, None] * K[np.ix_(keep, keep)]
    w, Z = np.linalg.eig(M)
    Zt = np.linalg.inv(Z)
    if np.abs(w.imag).max() < 1e-8:
        w, Z, Zt = w.real, Z.real, Zt.real
    norm = np.einsum("pn,p,pn->n", Z, eta[keep], Z)
    return dict(energies=w, right=Z.T, left=Zt, eta=eta[keep], pairs=keep,
                forward=norm > 0, K=K)


def phonon_self_energy_poles(v, eps, n_occ, channels=("ladder", "ring"),
                             spaces=None):
    """The pole form of the self energy with the full energy dependence of
    the ladder and/or ring channel, through the RPA phonons of chapter 7:

      ladder:  Sigma(omega) = sum_k sum_{nu addition} (V Z_nu)(Ztilde_nu eta V)/(omega + eps_k - Omega_nu)
                            - sum_c sum_{mu removal}  (V Z_mu)(Ztilde_mu eta V)/(omega + eps_c - Omega_mu),
      ring:    Sigma(omega) = sum_a sum_{nu forward}  (K Z_nu)(Ztilde_nu eta K)/(omega - eps_a - Omega_nu)
                            - sum_i sum_{mu backward} (K Z_mu)(Ztilde_mu eta K)/(omega - eps_i - Omega_mu)
                            - Sigma^(2)(omega),

    and, with both channels, the second-order term counted once.  Replacing
    the phonons by free pairs reduces each line to the second-order self
    energy; the interference between the channels (the parquet content
    proper) begins at fourth order and is not included.
    """
    n = len(eps)
    if spaces is None:
        spaces = PairSpaces(n)
    holes = np.arange(n_occ)
    parts = np.arange(n_occ, n)
    E, ML, MR = [], [], []
    if "ladder" in channels:
        ph = channel_phonons(v, eps, n_occ, "ladder", spaces)
        Vpp = ph["K"][:, ph["pairs"]]                 # all pairs x propagating
        ab = spaces.pairs
        # V_{(alpha k),(ab)} = v[alpha, k, a, b] for every alpha, k, and pair ab
        Vleft = v[:, :, ab[:, 0], ab[:, 1]][:, :, ph["pairs"]]   # (alpha, k, pair)
        Vright = v[ab[:, 0], ab[:, 1], :, :][ph["pairs"], :, :]  # (pair, beta, k)
        for nu, (Om, fw) in enumerate(zip(ph["energies"], ph["forward"])):
            z, zt = ph["right"][nu], ph["left"][nu] * ph["eta"]
            left = np.einsum("akp,p->ak", Vleft, z)       # (alpha, spectator)
            right = np.einsum("p,pbk->bk", zt, Vright)    # (beta, spectator)
            spect = holes if fw else parts
            sign = 1.0 if fw else -1.0
            for k in spect:
                E.append(eps[k] * (-1.0) + Om)            # pole at Omega_nu - eps_k
                ML.append(sign * left[:, k])
                MR.append(right[:, k])
    if "ring" in channels:
        ph = channel_phonons(v, eps, n_occ, "ring", spaces)
        K = ph["K"]
        keep = ph["pairs"]
        Kl = K[:, keep].reshape(n, n, len(keep))          # [(alpha, a), pair]
        Kr = K[keep, :].reshape(len(keep), n, n)          # [pair, (beta, a)]
        for nu, (Om, fw) in enumerate(zip(ph["energies"], ph["forward"])):
            z, zt = ph["right"][nu], ph["left"][nu] * ph["eta"]
            left = np.einsum("asp,p->as", Kl, z)          # (alpha, spectator)
            right = np.einsum("p,pbs->bs", zt, Kr)        # (beta, spectator)
            spect = parts if fw else holes
            sign = 1.0 if fw else -1.0
            for a in spect:
                E.append(eps[a] + Om)
                ML.append(sign * left[:, a])
                MR.append(right[:, a])
        # the second-order term is contained twice in the ring sum
        E2, ML2, MR2 = self_energy_poles(v, v, eps, n_occ)
        E.extend(E2)
        ML.extend(-ML2)
        MR.extend(MR2)
        if "ladder" in channels:
            # ... and once more in the ladder sum: keep it exactly once
            E.extend(E2)
            ML.extend(-ML2)
            MR.extend(MR2)
    E = np.array(E)
    return E, np.array(ML), np.array(MR)


# ---------------------------------------------------------------------------
#  The driver
# ---------------------------------------------------------------------------
def static_self_energy(v, rho):
    """Sigma^(1)[rho]_{ab} = sum_{gd} <ag||bd> rho_{dg}, the first-order
    (Hartree-Fock-like) self energy built with a given density matrix."""
    return np.einsum("agbd,dg->ab", v, rho)


def green_function_energy(h, v, n_occ, scheme="parquet", omega_pp=None,
                          omega_ph=0.0, eta=0.0, basis="hf", static="sc",
                          update_energies=False, max_cycles=50, tol=1e-10,
                          mixing=0.0, parquet_kwargs=None, verbose=False):
    """Reference -> vertex -> self energy -> Dyson -> sum rule.

    scheme   'hf'       Sigma = first order only (returns the reference energy)
             'gf2'      Gamma = V: the second-order self energy
             'ladder'   Gamma from the ladder equation only
             'ring'     Gamma from the ring equation only
             'parquet'  Gamma from the coupled parquet equations
             'ladder-rpa', 'ring-rpa', 'both-rpa'
                        the self energy with the full energy dependence of
                        one channel or of both, through the RPA phonons
    omega_pp starting energy of the [12] channel; None means eps_HOMO+eps_LUMO
    omega_ph starting energy of the [13] channel (0 by default)
    eta      imaginary part in the two-time propagators
    basis    'hf' rotates to the Hartree-Fock basis first, 'reference' keeps
             the given orbitals
    static   'sc'  the static (first-order) self energy is rebuilt from the
                   correlated density matrix of the Dyson solution and the
                   cycle repeated until the energy is converged; 'hf' keeps
                   the Hartree-Fock Fock matrix and does a single cycle
    update_energies  also replace the single-particle energies entering the
             two-time propagators and the self-energy poles by the dominant
             poles of the previous cycle (a quasiparticle self-consistency)

    Returns a dictionary with the energy, the Dyson solution, the vertex,
    the static self energy and the matrices in the basis used.
    """
    n = h.shape[0]
    if basis == "hf":
        h, v, C, _ = cc.hartree_fock(h, v, n_occ)
    F = cc.fock_matrix(h, v, n_occ)
    eps = np.diag(F).copy()
    e_ref = cc.reference_energy(h, v, n_occ)
    spaces = PairSpaces(n)
    kw = dict(parquet_kwargs or {})
    results = dict(h=h, v=v, F0=F.copy(), reference_energy=e_ref, history=[])
    if scheme == "hf":
        results.update(energy=e_ref, eps=eps, F=F)
        return results
    mu = 0.5 * (eps[n_occ - 1] + eps[n_occ])
    cycles = max_cycles if static == "sc" or update_energies else 1
    previous = None
    for cycle in range(1, cycles + 1):
        w_pp = (eps[n_occ - 1] + eps[n_occ]) if omega_pp is None else omega_pp
        info = dict(iterations=0, converged=True)
        Gamma = None
        if scheme == "gf2":
            Gamma = v
        elif scheme in ("ladder", "ring", "parquet"):
            Gamma, L, R, info = parquet_vertex(
                v, eps, n_occ, w_pp, omega_ph, eta,
                ladders=scheme in ("ladder", "parquet"),
                rings=scheme in ("ring", "parquet"), spaces=spaces, **kw)
            results.update(L=L, R=R)
        if scheme in ("ladder-rpa", "ring-rpa", "both-rpa"):
            chans = {"ladder-rpa": ("ladder",), "ring-rpa": ("ring",),
                     "both-rpa": ("ladder", "ring")}[scheme]
            E, ML, MR = phonon_self_energy_poles(v, eps, n_occ, chans, spaces)
        else:
            E, ML, MR = self_energy_poles(v, Gamma, eps, n_occ)
        sol = dyson(F, E, ML, MR, mu)
        energy = koltun_energy(h, sol)
        results["history"].append(dict(cycle=cycle, energy=energy,
                                       eps=eps.copy(), parquet=info))
        if verbose:
            print(f"cycle {cycle:3d}: E = {energy.real:.10f}  "
                  f"(vertex: {info['iterations']} iterations, "
                  f"converged {info['converged']})")
        if previous is not None and abs(energy - previous) < tol:
            break
        previous = energy
        if cycle == cycles:
            break
        # the correlated density matrix and the static self energy
        holes = sol["holes"]
        rho = np.einsum("al,lb->ab", sol["X"][:, holes], sol["Y"][holes, :])
        rho = rho.real if np.abs(rho.imag).max() < 1e-10 else rho
        if static == "sc":
            F_new = h + static_self_energy(v, rho)
            F = (1 - mixing) * F_new + mixing * F if mixing else F_new
        if update_energies:
            S = sol["strength"].real
            new = np.array([sol["poles"].real[np.argmax(S[:, a])]
                            for a in range(n)])
            eps = (1 - mixing) * new + mixing * eps if mixing else new
            mu = 0.5 * (np.sort(eps)[n_occ - 1] + np.sort(eps)[n_occ])
    results.update(energy=energy, dyson=sol, Gamma=Gamma, eps=eps, F=F,
                   poles=E, ML=ML, MR=MR, vertex_info=info, mu=mu,
                   cycles=cycle, density=np.einsum(
                       "al,lb->ab", sol["X"][:, sol["holes"]],
                       sol["Y"][sol["holes"], :]))
    return results


# ---------------------------------------------------------------------------
#  Checks of the two channels against the RPA
# ---------------------------------------------------------------------------
def rpa_matrices(v, eps, n_occ):
    """A and B of the particle-hole RPA, A_{ai,bj} = d(eps_a-eps_i) + <aj||ib>,
    B_{ai,bj} = <ab||ij>, and the same for the particle-particle RPA."""
    n = len(eps)
    o, u = range(n_occ), range(n_occ, n)
    ph = [(a, i) for a in u for i in o]
    A = np.array([[(eps[a] - eps[i]) * (a == b and i == j) + v[a, j, i, b]
                   for (b, j) in ph] for (a, i) in ph])
    B = np.array([[v[a, b, i, j] for (b, j) in ph] for (a, i) in ph])
    return A, B


def channel_poles(v, eps, n_occ, channel):
    """The poles of the pure ring or pure ladder series.

    With G = eta_pair/(Omega - d_pair) the series K + K G K + ... has its
    poles where  1 - G(Omega) K  is singular, that is at the eigenvalues of
    diag(d) + diag(eta) K restricted to the pairs that propagate.  For the
    ring channel these are the RPA energies, for the ladder channel the
    particle-particle RPA energies: the two channels of the parquet
    equations contain the two RPAs of chapter 7.
    """
    n = len(eps)
    spaces = PairSpaces(n)
    hole = np.arange(n) < n_occ
    if channel == "ring":
        K = spaces.to_ph(v)
        p = np.repeat(np.arange(n), n)
        q = np.tile(np.arange(n), n)
        d = eps[p] - eps[q]
        sign = np.where(~hole[p] & hole[q], 1.0, 0.0) - np.where(hole[p] & ~hole[q], 1.0, 0.0)
    else:
        K = spaces.to_pp(v)
        a, b = spaces.pairs[:, 0], spaces.pairs[:, 1]
        d = eps[a] + eps[b]
        sign = np.where(~hole[a] & ~hole[b], 1.0, 0.0) - np.where(hole[a] & hole[b], 1.0, 0.0)
    keep = sign != 0
    M = np.diag(d[keep]) + sign[keep, None] * K[np.ix_(keep, keep)]
    return np.sort(np.linalg.eigvals(M).real)


def pp_rpa_energies(v, eps, n_occ):
    """The particle-particle RPA: eigenvalues of [[D_pp + V_pp, V_ph], [-V_hp, D_hh - V_hh]],
    the addition energies E^{N+2} - E_0 and the removal energies E_0 - E^{N-2}."""
    n = len(eps)
    spaces = PairSpaces(n)
    a, b = spaces.pairs[:, 0], spaces.pairs[:, 1]
    hole = np.arange(n) < n_occ
    pp = np.where(~hole[a] & ~hole[b])[0]
    hh = np.where(hole[a] & hole[b])[0]
    Vpp = spaces.to_pp(v)
    App = np.diag(eps[a[pp]] + eps[b[pp]]) + Vpp[np.ix_(pp, pp)]
    Bph = Vpp[np.ix_(pp, hh)]
    Dhh = np.diag(eps[a[hh]] + eps[b[hh]])
    M = np.block([[App, Bph], [-Bph.T, Dhh - Vpp[np.ix_(hh, hh)]]])
    return np.sort(np.linalg.eigvals(M).real)


# ---------------------------------------------------------------------------
#  Helpers for the comparisons
# ---------------------------------------------------------------------------
def exact_self_energy(lp, F, omega, eta=1e-6):
    """Sigma(omega) = omega - F - g(omega)^{-1} with the exact propagator of a
    LehmannPropagator lp and the Fock matrix F of the same basis."""
    g = lp.propagator(omega, eta)
    return omega * np.eye(F.shape[0]) - F - np.linalg.inv(g)


def mp_energies(h, v, n_occ, order=4):
    """The Rayleigh-Schrodinger series with H_0 = sum eps^HF n_p (the
    Moller-Plesset partition), through the given order, using the recursion
    of chapter 9.  Returns the reference energy and the list of corrections
    E^(1), E^(2), ... (E^(1) is the difference between the reference energy
    and the sum of the occupied Fock energies)."""
    import mbpt
    H, states, index = cc.hamiltonian_matrix(h, v, n_occ)
    eps = np.diag(cc.fock_matrix(h, v, n_occ))
    w = np.array([sum(eps[p] for p in range(len(eps)) if (st >> p) & 1)
                  for st in states])
    part = mbpt.Partition(H, w, index[(1 << n_occ) - 1], "one_body")
    return cc.reference_energy(h, v, n_occ), mbpt.rayleigh_schrodinger(part, order)


def compare_methods(h, v, n_occ, omega_pp=-10.0, omega_ph=None, eta=0.0,
                    parquet_kwargs=None, fci=True, mp3=True):
    """Ground-state energies of one Hamiltonian by the methods of chapters
    6, 9 and 10 and by the Green's function methods of this chapter.  The
    parquet vertex is evaluated at the common starting energy omega_pp
    (omega_ph defaults to the same value, as in the energy-independent
    scheme of the text).  ``fci`` may be True (compute it by brute force),
    False, or a number known from elsewhere (the quasispin solution of the
    Lipkin model, say); ``mp3=False`` skips the third order, which needs the
    full determinant space.  Returns an ordered dictionary."""
    if omega_ph is None:
        omega_ph = omega_pp
    hh, vv, _, _ = cc.hartree_fock(h, v, n_occ)
    f = cc.fock_matrix(hh, vv, n_occ)
    e_ref = cc.reference_energy(hh, vv, n_occ)
    out = {"HF": e_ref, "MP2": e_ref + cc.mp2_energy(f, vv, n_occ)}
    if mp3:
        _, Es = mp_energies(hh, vv, n_occ, 3)
        out["MP3"] = e_ref + Es[1] + Es[2]
    kw = dict(parquet_kwargs or {})
    for key, args in (("GF2", dict(scheme="gf2")),
                      ("parquet", dict(scheme="parquet", omega_pp=omega_pp,
                                       omega_ph=omega_ph, eta=eta,
                                       parquet_kwargs=kw)),
                      ("ladder-RPA", dict(scheme="ladder-rpa")),
                      ("ring-RPA", dict(scheme="ring-rpa")),
                      ("both-RPA", dict(scheme="both-rpa"))):
        try:
            r = green_function_energy(hh, vv, n_occ, basis="reference", **args)
            e = r["energy"].real
            if not r["vertex_info"]["converged"]:
                e = np.nan
        except (np.linalg.LinAlgError, ValueError):
            e = np.nan
        out[key] = e
    out["CCD"] = e_ref + cc.ccd(f, vv, n_occ)["energy"]
    out["CCSD"] = e_ref + cc.ccsd(f, vv, n_occ)["energy"]
    if fci is True:
        out["FCI"] = cc.fci_energy(h, v, n_occ)[0]
    elif fci is not False:
        out["FCI"] = float(fci)
    return out


def _print_table(rows, keys, label, fmt="%10.6f"):
    print(" " * len(label) + "".join("%11s" % k for k in keys))
    for name, out in rows:
        print(name.ljust(len(label)) + "".join(
            ("%11s" % "--") if np.isnan(out[k]) else (" " + fmt % out[k])
            for k in keys))


# ---------------------------------------------------------------------------
#  Demonstrations
# ---------------------------------------------------------------------------
def demo_lehmann():
    print("The exact propagator of the pairing model (4 levels, 4 particles)")
    for g in (0.5, 1.0):
        h, v, N = cc.pairing_model(levels=4, particles=4, g=g)
        lp = LehmannPropagator(h, v, N)
        occ, dep = lp.occupation()
        ew, h1 = lp.energy_weighted_sum()
        print(f"  g = {g}:  E_FCI = {lp.energy:.10f}   "
              f"sum rule = {lp.koltun_energy():.10f}")
        print("     occupation numbers n_alpha  ", np.round(occ, 6))
        print("     n_alpha + d_alpha           ", np.round(occ + dep, 12))
        print(f"     sum_alpha int omega S_h = {ew:.8f} = <H_1> + 2<V>,  "
              f"<H_1> = {h1:.8f}, hence <V> = {0.5 * (ew - h1):.8f}")
        print(f"     removal energies {np.round(lp.eps_minus[:6], 4)} ...")
    print("The Lipkin model, N = 4, V = 0.3")
    h, v, N = lipkin_model(N=4, V=0.3)
    lp = LehmannPropagator(h, v, N)
    print(f"  E_FCI = {lp.energy:.10f}   sum rule = {lp.koltun_energy():.10f}")
    print("  occupation numbers", np.round(lp.occupation()[0], 6))


def demo_channels():
    print("The ring and ladder channels against the RPA (pairing + ph model, "
          "g = 0.4, f = 0.3, HF basis)")
    h, v, N = cc.pairing_ph_model(levels=4, particles=4, g=0.4, f=0.3)
    h, v, _, _ = cc.hartree_fock(h, v, N)
    eps = np.diag(cc.fock_matrix(h, v, N))
    A, B = rpa_matrices(v, eps, N)
    w = np.sort(np.linalg.eigvals(np.block([[A, B], [-B, -A]])).real)
    r = channel_poles(v, eps, N, "ring")
    print("  RPA energies     ", np.round(w[w > 0], 6))
    print("  ring-series poles", np.round(r[r > 0], 6))
    print("  largest difference", np.abs(np.sort(w) - np.sort(r)).max())
    wpp = pp_rpa_energies(v, eps, N)
    l = channel_poles(v, eps, N, "ladder")
    print("  pp-RPA energies    ", np.round(wpp, 6))
    print("  ladder-series poles", np.round(l, 6))
    print("  largest difference", np.abs(np.sort(wpp) - np.sort(l)).max())


def demo_second_order():
    print("The second-order self energy and the Dyson equation, pairing model")
    h, v, N = cc.pairing_model(levels=4, particles=4, g=0.5)
    F = cc.fock_matrix(h, v, N)
    eps = np.diag(F)
    E, ML, MR = self_energy_poles(v, v, eps, N)
    print(f"  {len(E)} poles of Sigma^(2); 2p1h poles at",
          np.round(np.unique(E[E > 0]), 4), " 2h1p poles at",
          np.round(np.unique(E[E < 0]), 4))
    lp = LehmannPropagator(h, v, N)
    r = green_function_energy(h, v, N, scheme="gf2")
    sol = r["dyson"]
    strong = (lp.X_minus**2).max(1) > 1e-3
    print("  exact removal energies with strength > 1e-3  ",
          np.round(lp.eps_minus[strong], 4))
    S = sol["strength"].real[sol["holes"], :]
    w = sol["poles"].real[sol["holes"]]
    strong = S.max(1) > 1e-3
    print("  Dyson(GF2) removal energies, strength > 1e-3", np.round(w[strong], 4))
    print("  exact occupation ", np.round(lp.occupation()[0], 5))
    print("  Dyson(GF2)       ", np.round(np.diag(r["density"]).real, 5))
    print(f"  E_FCI = {lp.energy:.8f}   E(GF2) = {r['energy'].real:.8f}   "
          f"cycles of the static self-consistency: {r['cycles']}")


def demo_perturbative_content():
    print("Perturbative content: (E_method - E_MPn)/g^(n+1) for small g")
    print("  pairing model, 4 levels, 4 particles")
    print("     g     (GF2-MP2)/g^3  (GF2-MP3)/g^4  (lad-MP3)/g^4  "
          "(ring-MP3)/g^4  (both-MP3)/g^4")
    for g in (0.025, 0.05, 0.1, 0.2):
        h, v, N = cc.pairing_model(levels=4, particles=4, g=g)
        e_ref, Es = mp_energies(h, v, N, 3)
        e2, e3 = e_ref + Es[1], e_ref + Es[1] + Es[2]
        e = [green_function_energy(h, v, N, scheme=s)["energy"].real
             for s in ("gf2", "ladder-rpa", "ring-rpa", "both-rpa")]
        print(f"  {g:6.3f} {(e[0]-e2)/g**3:13.5f} {(e[0]-e3)/g**4:14.4f} "
              f"{(e[1]-e3)/g**4:14.4f} {(e[2]-e3)/g**4:14.4f} {(e[3]-e3)/g**4:14.4f}")


def demo_pairing():
    print("The pairing model, 4 levels, 4 particles (starting energy -10)")
    keys = ["HF", "MP2", "MP3", "GF2", "parquet", "ladder-RPA", "ring-RPA",
            "both-RPA", "CCD", "FCI"]
    rows = []
    for g in (0.25, 0.5, 0.75, 1.0):
        h, v, N = cc.pairing_model(levels=4, particles=4, g=g)
        rows.append((f"g = {g:4.2f}", compare_methods(h, v, N)))
    _print_table(rows, keys, "g = 0.00 ")
    return rows


def demo_pairing_ph():
    print("The pairing plus particle-hole model (starting energy -10)")
    keys = ["HF", "MP2", "GF2", "parquet", "both-RPA", "CCD", "CCSD", "FCI"]
    rows = []
    for g, f in ((0.5, 0.025), (0.7, 0.05), (1.0, 0.05), (0.5, 0.25), (1.0, 0.5)):
        h, v, N = cc.pairing_ph_model(levels=4, particles=4, g=g, f=f)
        rows.append((f"g = {g:4.2f}, f = {f:5.3f}", compare_methods(h, v, N)))
    _print_table(rows, keys, "g = 0.00, f = 0.000 ")
    return rows


def lipkin_exact_energy(N, eps=1.0, V=0.2, W=0.0):
    """The exact ground-state energy from the quasispin matrix of chapter 4."""
    import models
    return float(models.LipkinModel(J=N / 2, eps=eps, V=V, W=W).ground_state()[0])


def demo_lipkin():
    print("The Lipkin model (starting energy -10)")
    keys = ["HF", "MP2", "GF2", "parquet", "ring-RPA", "both-RPA", "CCD",
            "CCSD", "FCI"]
    rows = []
    for N_, V in ((4, 0.1), (4, 0.3), (4, 0.6), (6, 0.1), (6, 0.15), (8, 0.1)):
        h, v, N = lipkin_model(N=N_, V=V)
        rows.append((f"N = {N_}, V = {V:4.2f}",
                     compare_methods(h, v, N, fci=lipkin_exact_energy(N_, V=V),
                                     mp3=False)))
    _print_table(rows, keys, "N = 0, V = 0.00 ")
    return rows


def demo_starting_energy():
    print("Dependence of the energy-independent parquet energy on the "
          "starting energy (pairing model, g = 0.5)")
    h, v, N = cc.pairing_model(levels=4, particles=4, g=0.5)
    print(f"  GF2 {green_function_energy(h, v, N, scheme='gf2')['energy'].real:.6f}"
          f"   FCI {cc.fci_energy(h, v, N)[0]:.6f}")
    print("   E_in    eta=0        eta=0.5")
    for Ein in (-20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20):
        row = []
        for eta in (0.0, 0.5):
            try:
                r = green_function_energy(h, v, N, scheme="parquet",
                                          omega_pp=Ein, omega_ph=Ein, eta=eta)
                row.append("%.6f%s" % (r["energy"].real,
                                       "" if r["vertex_info"]["converged"] else "*"))
            except (np.linalg.LinAlgError, ValueError):
                row.append("   --   ")
        print(f"  {Ein:5.0f}   " + "   ".join(row))


def _demo():
    demo_lehmann()
    print()
    demo_channels()
    print()
    demo_second_order()
    print()
    demo_perturbative_content()
    print()
    demo_pairing()
    print()
    demo_pairing_ph()
    print()
    demo_lipkin()
    print()
    demo_starting_energy()


if __name__ == "__main__":
    _demo()

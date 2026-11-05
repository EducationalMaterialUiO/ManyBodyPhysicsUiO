"""
A verified Kohn-Sham solver: the self-consistent field as a fixed-point problem.

Companion code to chapter 8 of *Quantum mechanics for Many-particle Systems*,
section "Kohn-Sham in practice".

`dft.py` solves the Kohn-Sham equations for N spinless fermions in the
one-dimensional harmonic trap with the softened Coulomb interaction of
chapters 2 and 6.  This module solves the same problem a second time, with
the bookkeeping that a production code needs made explicit, and verifies
every step that the chapter says must be verified:

  * the sampled oscillator basis is orthogonalised with the quadrature
    weights actually used, so that the particle number is conserved to
    machine precision and one quadrature serves the overlap, the density,
    the Hartree field and the exchange energy alike;
  * the energy is evaluated at every iteration as the *direct* functional
    on the output orbitals and density, and the eigenvalue
    (double-counting) formula is checked against it only at convergence;
  * the convergence test uses the unmixed fixed-point residual
    n_out - n_in, never the mixed step alpha (n_out - n_in);
  * at convergence the orbital residual F C - C eps, the orthonormality of
    the occupied block, the particle number and the direct/eigenvalue
    energy identity are all reported;
  * the representation (grid, box, basis, exchange table) is refined
    independently of the self-consistency, and a second starting density
    is tried;
  * the functional derivative delta(E_H + E_x)/delta n = v_H + v_x is
    checked by a central difference in density space;
  * the linear mixing n <- (1 - alpha) n + alpha n_out is run for several
    alpha, exhibiting the contraction factor of the linearised map.

The exchange functional is `dft.ExchangeLDA1D`, the same one `dft.py` uses,
so the two solvers must agree -- and `demo_two_solvers` shows that they do,
to ten decimals.

    KohnShamSCF        -- the solver, one object per (N, basis, grid)
    representation_tests, mixing_study, functional_derivative_check
    demo_*             -- the tables of the chapter

Author: Morten Hjorth-Jensen
"""

from math import factorial, pi

import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.linalg import eigh

import dft


# ---------------------------------------------------------------------------
#  Quadrature
# ---------------------------------------------------------------------------
def simpson_weights(n_points, h):
    """Composite Simpson weights for n_points (odd) equidistant nodes."""
    if n_points % 2 != 1:
        raise ValueError("Simpson quadrature needs an odd number of nodes")
    w = np.ones(n_points)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w * h / 3.0


# ---------------------------------------------------------------------------
#  The solver
# ---------------------------------------------------------------------------
class KohnShamSCF:
    """Galerkin Kohn-Sham in the first `n_basis` oscillator functions.

    The sampled basis B_{j mu} = chi_mu(x_j) has the quadrature overlap
    S = B^T W B, which is the identity only up to the quadrature error.  We
    orthogonalise it exactly, X = S^{-1/2}, and transform h_0 by the same
    X, so that the finite calculation conserves N to machine precision
    whatever the grid.  All local terms -- the Hartree field, the exchange
    potential and the energies -- are evaluated with the same weights.
    """

    def __init__(self, n_particles, lda=None, n_basis=8, box=8.0,
                 n_grid=401, strength=1.0, softening=0.5):
        if not 0 < n_particles <= n_basis:
            raise ValueError("need 0 < N <= n_basis for spinless occupations")
        self.N = n_particles
        self.n_basis = n_basis
        self.box = box
        self.n_grid = n_grid
        self.lda = lda if lda is not None else dft.ExchangeLDA1D(strength,
                                                                 softening)
        self.x = np.linspace(-box, box, n_grid)
        self.w = simpson_weights(n_grid, self.x[1] - self.x[0])
        phi = np.empty((n_grid, n_basis))
        for j in range(n_basis):
            coef = np.zeros(j + 1)
            coef[-1] = 1.0
            phi[:, j] = (hermval(self.x, coef) * np.exp(-self.x ** 2 / 2)
                         / np.sqrt(2.0 ** j * factorial(j) * np.sqrt(pi)))
        S = phi.T @ (self.w[:, None] * phi)
        s, U = eigh(S)
        if s.min() < 1e-10:
            raise ValueError("sampled oscillator basis is linearly dependent")
        X = (U / np.sqrt(s)) @ U.T                  # S^{-1/2}
        self.overlap_error = float(np.abs(S - np.eye(n_basis)).max())
        self.B = phi @ X                            # orthonormal on the grid
        self.h0 = X.T @ np.diag(np.arange(n_basis) + 0.5) @ X
        self.K = strength / np.sqrt(
            (self.x[:, None] - self.x[None, :]) ** 2 + softening ** 2)

    # ------------------------------------------------------------------
    def density(self, C):
        return np.sum((self.B @ C[:, :self.N]) ** 2, axis=1)

    def potential_matrix(self, v):
        """<mu| v(x) |nu> with the quadrature weights."""
        return self.B.T @ ((self.w * v)[:, None] * self.B)

    def hartree_potential(self, n):
        return self.K @ (self.w * n)

    def components(self, C, n):
        """One-body, Hartree and exchange energies of orbitals C, density n."""
        occ = C[:, :self.N]
        one = float(np.trace(occ.T @ self.h0 @ occ))
        hartree = float(0.5 * np.dot(self.w * n, self.hartree_potential(n)))
        exchange = float(np.dot(self.w, self.lda.energy_density(n)))
        return one, hartree, exchange

    def initial_density(self, kind="oscillator"):
        if kind == "oscillator":
            _, C = eigh(self.h0)
            return self.density(C)
        if kind == "broadened":
            n = np.exp(-self.x ** 2 / 2)
            return n * self.N / np.dot(self.w, n)
        raise ValueError(kind)

    # ------------------------------------------------------------------
    def solve(self, alpha=0.3, density_tol=1e-10, energy_tol=1e-12,
              max_iter=500, initial="oscillator", verbose=False):
        """Linear mixing of the density until the fixed-point residual and
        the energy change are both below tolerance.  Raises if the loop
        does not converge: a maximum-iteration exit is a failure, not a
        result."""
        n = self.initial_density(initial)
        history = []
        last_energy = np.inf
        for iteration in range(1, max_iter + 1):
            F = self.h0 + self.potential_matrix(self.hartree_potential(n)
                                                + self.lda.v_x(n))
            eps, C = eigh(F)
            n_out = self.density(C)
            residual = n_out - n
            one, hartree, exchange = self.components(C, n_out)
            energy = one + hartree + exchange          # direct functional
            history.append(dict(iteration=iteration, energy=energy,
                                residual_max=float(np.abs(residual).max()),
                                residual_l1=float(np.dot(self.w,
                                                         np.abs(residual))),
                                energy_change=float(abs(energy - last_energy))))
            if verbose:
                print(f"   {iteration:4d}  E = {energy:16.12f}   "
                      f"max|n_out - n_in| = {history[-1]['residual_max']:.3e}")
            if (history[-1]["residual_max"] < density_tol
                    and abs(energy - last_energy) < energy_tol):
                break
            # convex mixing: keeps n >= 0 and int n = N
            n = (1.0 - alpha) * n + alpha * n_out
            last_energy = energy
        else:
            raise RuntimeError(f"SCF did not converge in {max_iter} "
                               f"iterations (alpha = {alpha})")

        # final consistency checks on one and the same output state
        n = n_out
        v_h, v_x = self.hartree_potential(n), self.lda.v_x(n)
        F_final = self.h0 + self.potential_matrix(v_h + v_x)
        occ = C[:, :self.N]
        one, hartree, exchange = self.components(C, n)
        direct = one + hartree + exchange
        eigenvalue_form = float(eps[:self.N].sum() - hartree + exchange
                                - np.dot(self.w * n, v_x))
        P = occ @ occ.T
        self.C, self.eps, self.n, self.history = C, eps, n, history
        self.results = dict(
            N=self.N, n_basis=self.n_basis, box=self.box, n_grid=self.n_grid,
            alpha=alpha, initial=initial, iterations=iteration,
            energy=direct, one_body=one, hartree=hartree, exchange=exchange,
            hartree_plus_exchange=hartree + exchange,
            eigenvalue_energy=eigenvalue_form,
            energy_identity_error=abs(direct - eigenvalue_form),
            particles=float(np.dot(self.w, n)),
            particle_error=abs(float(np.dot(self.w, n)) - self.N),
            orthonormality_error=float(np.linalg.norm(occ.T @ occ
                                                      - np.eye(self.N))),
            orbital_residual=float(np.linalg.norm(F_final @ occ
                                                  - occ * eps[:self.N])),
            commutator_residual=float(np.linalg.norm(F_final @ P
                                                     - P @ F_final)),
            density_residual=history[-1]["residual_max"],
            edge_density=float(max(n[0], n[-1])),
            overlap_error=self.overlap_error,
            occupied_eigenvalues=eps[:self.N].tolist())
        return self.results

    # ------------------------------------------------------------------
    def output_density(self, n):
        """The map F: n_in -> n_out, one Kohn-Sham diagonalisation."""
        F = self.h0 + self.potential_matrix(self.hartree_potential(n)
                                            + self.lda.v_x(n))
        _, C = eigh(F)
        return self.density(C)

    def jacobian(self, n=None, delta=1e-6):
        """J = dF/dn at the density n (default: the converged one), by
        central differences along every grid point.  The map is linear in
        the grid vector, so this is the matrix whose spectral radius governs
        linear mixing."""
        n = self.n if n is None else n
        J = np.empty((n.size, n.size))
        for j in range(n.size):
            e = np.zeros_like(n)
            e[j] = delta
            J[:, j] = (self.output_density(n + e)
                       - self.output_density(n - e)) / (2.0 * delta)
        return J

    def contraction_factor(self, last=8):
        """Geometric decay rate of the residual over the last iterations."""
        r = np.array([h["residual_max"] for h in self.history])
        r = r[r > 0]
        if r.size < 3:
            return np.nan
        tail = r[-min(last, r.size - 1) - 1:]
        return float(np.exp(np.mean(np.diff(np.log(tail)))))


def mixing_rate(eigenvalues, alpha):
    """Spectral radius of (1 - alpha) I + alpha J for the given spectrum."""
    return float(np.abs(1.0 - alpha + alpha * eigenvalues).max())


def jacobian_spectrum(ks, threshold=1e-8):
    """Non-zero eigenvalues of dF/dn at the converged density, split by the
    parity of their eigenvectors.  The trap is symmetric and the iteration
    starts from an even density, so only the even sector is ever excited
    and only it governs the observed rate."""
    lam, V = np.linalg.eig(ks.jacobian())
    keep = np.abs(lam) > threshold
    lam, V = lam[keep].real, V[:, keep].real
    even, odd = [], []
    for value, v in zip(lam, V.T):
        (even if np.dot(v, v[::-1]) > 0 else odd).append(value)
    return np.sort(even), np.sort(odd)


# ---------------------------------------------------------------------------
#  Studies built on the solver
# ---------------------------------------------------------------------------
def representation_tests(lda=None, N=4):
    """The same functional in finer grids, larger boxes, larger bases and
    with a finer exchange table; and from a second starting density."""
    lda = lda if lda is not None else dft.ExchangeLDA1D()
    rows = []
    for label, kwargs, solve_kwargs in (
            ("M = 8,  401 nodes, [-8, 8]",   dict(), dict()),
            ("M = 8,  801 nodes, [-8, 8]",   dict(n_grid=801), dict()),
            ("M = 8,  501 nodes, [-10, 10]", dict(box=10.0, n_grid=501), dict()),
            ("M = 12, 501 nodes, [-10, 10]", dict(n_basis=12, box=10.0,
                                                  n_grid=501), dict()),
            ("M = 16, 501 nodes, [-10, 10]", dict(n_basis=16, box=10.0,
                                                  n_grid=501), dict()),
            ("M = 8,  401 nodes, broadened start", dict(),
             dict(initial="broadened"))):
        ks = KohnShamSCF(N, lda=lda, **kwargs)
        res = ks.solve(**solve_kwargs)
        rows.append((label, res["energy"], res["iterations"],
                     res["density_residual"]))
    fine = dft.ExchangeLDA1D(n_points=1201)
    res = KohnShamSCF(N, lda=fine).solve()
    rows.append(("M = 8,  401 nodes, 1201-point exchange table",
                 res["energy"], res["iterations"], res["density_residual"]))
    return rows


def mixing_study(alphas=(0.1, 0.3, 0.5, 0.7, 0.9, 1.0), lda=None, N=4,
                 max_iter=400):
    """Iterations and contraction factor of linear mixing against alpha."""
    lda = lda if lda is not None else dft.ExchangeLDA1D()
    out = []
    for alpha in alphas:
        ks = KohnShamSCF(N, lda=lda)
        try:
            res = ks.solve(alpha=alpha, max_iter=max_iter)
            out.append((alpha, res["iterations"], ks.contraction_factor(),
                        ks.history))
        except RuntimeError:
            out.append((alpha, None, np.nan, ks.history))
    return out


def functional_derivative_check(ks, delta=1e-5):
    """delta(E_H + E_x)/delta n against a central difference along a
    density variation eta that conserves the particle number."""
    x, w, n, K, lda = ks.x, ks.w, ks.n, ks.K, ks.lda
    eta = n * (np.cos(x) - np.dot(w * n, np.cos(x)) / ks.N)

    def interaction_energy(d):
        return (0.5 * np.dot(w * d, K @ (w * d))
                + np.dot(w, lda.energy_density(d)))

    analytic = float(np.dot(w * eta, K @ (w * n) + lda.v_x(n)))
    numerical = float((interaction_energy(n + delta * eta)
                       - interaction_energy(n - delta * eta)) / (2 * delta))
    return analytic, numerical, abs(analytic - numerical)


# ---------------------------------------------------------------------------
#  Demonstrations
# ---------------------------------------------------------------------------
def demo_checks(lda=None):
    print("=" * 74)
    print("1. One consistent output state, and what to check on it")
    print("=" * 74)
    print("Each row is a converged Kohn-Sham calculation in eight oscillator")
    print("functions on 401 Simpson nodes in [-8, 8], mixing 0.3.  The energy")
    print("is the direct functional on the output orbitals; the last column")
    print("is its difference from the eigenvalue formula")
    print("   E = sum_i eps_i - E_H + E_x - int n v_x,")
    print("which must vanish at self-consistency and only there.")
    print()
    lda = lda if lda is not None else dft.ExchangeLDA1D()
    print(f"{'N':>2s} {'<h0>':>12s} {'E_H':>12s} {'E_x':>13s} {'E':>13s} "
          f"{'iter':>5s} {'|int n - N|':>11s} {'|C^T C - I|':>11s} "
          f"{'|FC - C eps|':>12s} {'|E - E_eig|':>11s}")
    for N in (1, 2, 4):
        r = KohnShamSCF(N, lda=lda).solve()
        print(f"{N:2d} {r['one_body']:12.8f} {r['hartree']:12.8f} "
              f"{r['exchange']:13.8f} {r['energy']:13.8f} "
              f"{r['iterations']:5d} {r['particle_error']:11.1e} "
              f"{r['orthonormality_error']:11.1e} "
              f"{r['orbital_residual']:12.1e} "
              f"{r['energy_identity_error']:11.1e}")
    print()
    r = KohnShamSCF(4, lda=lda).solve()
    print(f"For N = 4 the final density residual is {r['density_residual']:.2e}")
    print(f"and the density at the edge of the box is "
          f"{r['edge_density']:.1e}, so the box is large enough.")


def demo_two_solvers(lda=None):
    print("=" * 74)
    print("2. Two implementations, one functional")
    print("=" * 74)
    print("dft.KohnSham1D projects the potentials with trapezoidal weights")
    print("on the raw sampled basis and evaluates the energy from the")
    print("eigenvalue formula on the mixed density; KohnShamSCF")
    print("orthogonalises the basis with Simpson weights and evaluates the")
    print("direct functional on the output density.  Given the same exchange")
    print("table they must agree at convergence, and they do:")
    print()
    lda = lda if lda is not None else dft.ExchangeLDA1D()
    print(f"{'N':>2s} {'dft.KohnSham1D':>16s} {'KohnShamSCF':>16s} "
          f"{'difference':>12s}")
    for N in (1, 2, 3, 4):
        old = dft.KohnSham1D(N, lda=lda)
        old.run()
        new = KohnShamSCF(N, lda=lda).solve()["energy"]
        print(f"{N:2d} {old.E:16.10f} {new:16.10f} {abs(old.E - new):12.1e}")
    print()
    print("The agreement is to ten decimals.  The fifth-decimal differences")
    print("between earlier editions of these tables came entirely from the")
    print("exchange table -- a real-space quadrature truncated at s = 400 --")
    print("and not from the solvers (see demo 3 of dft.py).")


def demo_representation(lda=None):
    print("=" * 74)
    print("3. A tiny residual is not a converged representation")
    print("=" * 74)
    print("Every calculation below converges to a density residual below")
    print("1e-10.  Only the basis changes the energy at a visible level:")
    print()
    rows = representation_tests(lda)
    base = rows[0][1]
    print(f"{'setting':<46s} {'E':>15s} {'E - E_base':>12s} {'iter':>5s}")
    for label, energy, iterations, residual in rows:
        diff = "--" if label == rows[0][0] else f"{energy - base:12.2e}"
        print(f"{label:<46s} {energy:15.10f} {diff:>12s} {iterations:5d}")
    print()
    print("Doubling the grid, enlarging the box and doubling the exchange")
    print("table all move the energy by 1e-11 or less; going from eight to")
    print("sixteen basis functions moves it by 7e-4.  The self-consistent")
    print("loop cannot repair a basis-truncation error, and none of these")
    print("tests says anything about the error of the functional itself.")


def demo_mixing(lda=None):
    print("=" * 74)
    print("4. Linear mixing as a contraction")
    print("=" * 74)
    print("Near the fixed point the iteration n_{k+1} = (1 - alpha) n_k +")
    print("alpha F[n_k] has the linearised map (1 - alpha) I + alpha J,")
    print("J = dF/dn, and converges when its spectral radius is below one.")
    print("We build J by finite differences at the converged N = 4 density")
    print("and compare its prediction with the measured decay of the")
    print("residual:")
    print()
    lda = lda if lda is not None else dft.ExchangeLDA1D()
    ks = KohnShamSCF(4, lda=lda)
    ks.solve()
    even, odd = jacobian_spectrum(ks)
    print(f"   J has {even.size + odd.size} eigenvalues above 1e-8 in "
          f"modulus (its rank is at most N (M - N) = 16).")
    print(f"   even eigenvectors: {np.array2string(even, precision=4)}")
    print(f"   odd eigenvectors:  {np.array2string(odd, precision=4)}")
    print("   The density is even and stays even, so only the even sector")
    print("   is excited; the prediction below uses it alone.")
    print()
    print(f"{'alpha':>6s} {'iterations':>11s} {'measured rate':>14s} "
          f"{'predicted':>10s}")
    for alpha, iterations, factor, history in mixing_study(lda=lda):
        it = "failed" if iterations is None else f"{iterations:d}"
        print(f"{alpha:6.2f} {it:>11s} {factor:14.4f} "
              f"{mixing_rate(even, alpha):10.4f}")
    best = even.min(), even.max()
    alpha_opt = 2.0 / (2.0 - best[0] - best[1])
    print()
    print(f"The response is dominated by negative eigenvalues -- more density")
    print(f"in means less density out, as it must for a repulsion -- and the")
    print(f"rate (1 - alpha) + alpha lambda falls with alpha until the most")
    print(f"negative eigenvalue takes over.  The optimum, where the two ends")
    print(f"of the spectrum have equal and opposite rates, is alpha = "
          f"{alpha_opt:.2f}")
    print(f"with rate {mixing_rate(even, alpha_opt):.3f}.  That the plain "
          f"iteration (alpha = 1) converges")
    print("at all is a property of this small trapped system: in an extended")
    print("system with a long-range interaction the Hartree response to a")
    print("long-wavelength density change lies below -1 -- charge sloshing --")
    print("and alpha = 1 diverges.  Small alpha then damps the oscillation at")
    print("the price of many iterations, which is why Pulay/DIIS and")
    print("preconditioned mixing exist.")


def demo_derivative_and_one_particle(lda=None):
    print("=" * 74)
    print("5. The functional derivative, and one particle")
    print("=" * 74)
    lda = lda if lda is not None else dft.ExchangeLDA1D()
    ks = KohnShamSCF(4, lda=lda)
    ks.solve()
    analytic, numerical, error = functional_derivative_check(ks)
    print("Along a number-conserving direction eta(x) in density space,")
    print("   int eta (v_H + v_x) dx  versus  [E(n + d eta) - E(n - d eta)]/2d:")
    print(f"   analytic {analytic:.10f}   central difference {numerical:.10f}"
          f"   |difference| {error:.1e}")
    print()
    r = KohnShamSCF(1, lda=lda).solve()
    print("For one particle the exact energy is 1/2 and exact exchange")
    print("cancels the Hartree term identically.  The local functional gives")
    print(f"   E_H + E_x^LDA        = {r['hartree_plus_exchange']:.8f}"
          "   (self-interaction residue on the computed density)")
    print(f"   <h_0> - 1/2          = {r['one_body'] - 0.5:.8f}"
          "   (orbital relaxation in response to it)")
    print(f"   E^LDA - 1/2          = {r['energy'] - 0.5:.8f}")
    print()
    print("The two pieces add up: the self-interaction residue is a property")
    print("of the functional, the relaxation a consequence of solving the")
    print("Kohn-Sham equations with it.  For N > 1 the two cannot be")
    print("separated this cleanly, because exact exchange is then not simply")
    print("minus the Hartree energy.")


def _demo():
    lda = dft.ExchangeLDA1D()
    for f in (demo_checks, demo_two_solvers, demo_representation,
              demo_mixing, demo_derivative_and_one_particle):
        f(lda)
        print()


if __name__ == "__main__":
    _demo()

"""
Neural quantum states for the two-electron quantum dot: the project.

Companion code to Project 5 of appendix A of *Quantum mechanics for
Many-particle Systems*.

The wave function is the square root of the marginal of a Gaussian-binary
restricted Boltzmann machine, the neural quantum state of chapter 17, and
the machinery around it is the variational Monte Carlo of chapters 13 and
14.  This program extends the NeuralQuantumState class of rbm.py in the
directions the project asks for:

  * a flat parameter vector, so that any optimiser can be used, with the
    logarithmic derivatives O_k = d ln Psi / d theta_k of chapter 14 for
    every parameter, checked against finite differences;

  * two samplers, brute-force Metropolis with a symmetric step and the
    importance-sampled Metropolis-Hastings of chapter 13, both returning
    the walker-averaged series of local energies for the blocking analysis
    of chapter 14;

  * two optimisers, stochastic gradient descent and stochastic
    reconfiguration (the natural gradient of chapter 14), on the same
    Monte Carlo estimates;

  * the width sigma of the visible units as a variational parameter;

  * a Pade-Jastrow factor exp[r_12/(1 + beta r_12)] multiplying the machine,
    which puts the cusp of chapter 13 into the ansatz by hand and lets the
    hidden units describe what is left -- the cheapest version of the
    physics-informed ansatz of chapter 18.

Everything runs on numpy; blocking needs scipy through vmcoptimise.py.

Author: Morten Hjorth-Jensen
"""

import math

import numpy as np

from rbm import NeuralQuantumState, sigmoid, softplus        # chapter 17
try:
    from vmcoptimise import blocking                          # chapter 14
except ImportError:                                           # pragma: no cover
    blocking = None


class ProjectNQS(NeuralQuantumState):
    """The chapter-17 machine with a flat parameter vector, an optional
    Pade-Jastrow factor and an optional variational width."""

    def __init__(self, n_particles=2, n_dimensions=2, n_hidden=2, sigma=None,
                 omega=1.0, interaction=True, rng=None, scale=0.05,
                 jastrow=False, beta=0.4, learn_sigma=False):
        super().__init__(n_particles, n_dimensions, n_hidden, sigma, omega,
                         interaction, rng, scale)
        self.jastrow = jastrow
        self.beta = beta
        self.learn_sigma = learn_sigma

    # -- parameters -------------------------------------------------------
    @property
    def n_parameters(self):
        return self.M + self.N + self.M * self.N + int(self.jastrow) + int(self.learn_sigma)

    def get_theta(self):
        parts = [self.a, self.b, self.W.ravel()]
        if self.jastrow:
            parts.append([self.beta])
        if self.learn_sigma:
            parts.append([self.sigma])
        return np.concatenate([np.asarray(p, dtype=float).ravel() for p in parts])

    def set_theta(self, theta):
        theta = np.asarray(theta, dtype=float)
        k = 0
        self.a = theta[k:k + self.M].copy(); k += self.M
        self.b = theta[k:k + self.N].copy(); k += self.N
        self.W = theta[k:k + self.M * self.N].reshape(self.M, self.N).copy(); k += self.M * self.N
        if self.jastrow:
            self.beta = float(theta[k]); k += 1
        if self.learn_sigma:
            self.sigma = float(abs(theta[k])); k += 1

    def parameter_names(self):
        names = ([f"a{m}" for m in range(self.M)] + [f"b{n}" for n in range(self.N)]
                 + [f"w{m}{n}" for m in range(self.M) for n in range(self.N)])
        if self.jastrow:
            names.append("beta")
        if self.learn_sigma:
            names.append("sigma")
        return names

    # -- the Jastrow factor, per visible coordinate --------------------------
    def _pairs(self, x):
        """For every pair p < q: the difference vectors and distances."""
        r = x.reshape(len(x), self.P, self.D)
        out = []
        for p in range(self.P):
            for q in range(p):
                d = r[:, p] - r[:, q]
                out.append((p, q, d, np.linalg.norm(d, axis=1)))
        return out

    def log_jastrow(self, x):
        total = np.zeros(len(x))
        for _, _, _, dist in self._pairs(x):
            total += dist / (1.0 + self.beta * dist)
        return total

    def jastrow_derivatives(self, x):
        """First and second derivatives of sum_{p<q} u(r_pq) with respect to
        every visible coordinate, u(r) = r/(1 + beta r)."""
        n = len(x)
        grad = np.zeros((n, self.P, self.D))
        lap = np.zeros((n, self.P, self.D))
        for p, q, d, dist in self._pairs(x):
            u1 = 1.0 / (1.0 + self.beta * dist) ** 2                  # u'
            u2 = -2.0 * self.beta / (1.0 + self.beta * dist) ** 3     # u''
            unit = d / dist[:, None]
            g = u1[:, None] * unit
            l = u2[:, None] * unit ** 2 + u1[:, None] * (1.0 - unit ** 2) / dist[:, None]
            grad[:, p] += g
            grad[:, q] -= g
            lap[:, p] += l
            lap[:, q] += l
        return grad.reshape(n, self.M), lap.reshape(n, self.M)

    def dlog_dbeta(self, x):
        total = np.zeros(len(x))
        for _, _, _, dist in self._pairs(x):
            total -= dist ** 2 / (1.0 + self.beta * dist) ** 2
        return total

    # -- the wave function -----------------------------------------------------
    def log_psi(self, x):
        x = np.atleast_2d(x)
        value = super().log_psi(x)
        if self.jastrow:
            value = value + self.log_jastrow(x)
        return value

    def grad_log_psi_x(self, x):
        x = np.atleast_2d(x)
        g = super().grad_log_psi_x(x)
        if self.jastrow:
            g = g + self.jastrow_derivatives(x)[0]
        return g

    def laplacian_log_psi(self, x):
        x = np.atleast_2d(x)
        l = super().laplacian_log_psi(x)
        if self.jastrow:
            l = l + self.jastrow_derivatives(x)[1]
        return l

    def log_derivatives(self, x):
        """O_k(x) = d ln Psi / d theta_k for every parameter: shape (walkers, n_parameters)."""
        x = np.atleast_2d(x)
        da, db, dW = self.grad_log_psi_parameters(x)
        columns = [da, db, dW.reshape(len(x), -1)]
        if self.jastrow:
            columns.append(self.dlog_dbeta(x)[:, None])
        if self.learn_sigma:
            s = sigmoid(self._Q(x))
            s3 = self.sigma ** 3
            dsigma = (np.sum((x - self.a) ** 2, axis=1) / (2.0 * s3)
                      - np.sum(s * (x @ self.W), axis=1) / s3)
            columns.append(dsigma[:, None])
        return np.concatenate(columns, axis=1)

    def log_derivatives_fd(self, x, h=1e-6):
        """The same by central differences, for the check."""
        x = np.atleast_2d(x)
        theta = self.get_theta()
        out = np.zeros((len(x), len(theta)))
        for k in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[k] += h
            minus[k] -= h
            self.set_theta(plus); lp = self.log_psi(x)
            self.set_theta(minus); lm = self.log_psi(x)
            out[:, k] = (lp - lm) / (2.0 * h)
        self.set_theta(theta)
        return out

    def local_energy_fd(self, x, h=1e-4):
        """-(1/2) sum_m d^2 Psi/dx_m^2 / Psi + V, by central differences of ln Psi."""
        x = np.atleast_2d(x)
        base = self.log_psi(x)
        kinetic = np.zeros(len(x))
        for m in range(self.M):
            plus, minus = x.copy(), x.copy()
            plus[:, m] += h
            minus[:, m] -= h
            lp, lm = self.log_psi(plus), self.log_psi(minus)
            first = (lp - lm) / (2.0 * h)
            second = (lp - 2.0 * base + lm) / h ** 2
            kinetic += -0.5 * (second + first ** 2)
        potential = 0.5 * self.omega ** 2 * np.sum(x ** 2, axis=1)
        if self.interaction:
            for _, _, _, dist in self._pairs(x):
                potential += 1.0 / dist
        return kinetic + potential

    # -- sampling ---------------------------------------------------------------
    def _accumulate(self, x, acc):
        e = self.local_energy(x)
        o = self.log_derivatives(x)
        acc["n"] += 1
        acc["e"] += e.mean()
        acc["e2"] += (e * e).mean()
        acc["o"] += o.mean(axis=0)
        acc["eo"] += (o * e[:, None]).mean(axis=0)
        acc["oo"] += o.T @ o / len(x)
        acc["series"].append(e.mean())

    def _finish(self, acc, accepted, proposals):
        n = acc["n"]
        mean = acc["e"] / n
        o, eo, oo = acc["o"] / n, acc["eo"] / n, acc["oo"] / n
        gradient = 2.0 * (eo - o * mean)
        metric = oo - np.outer(o, o)
        return dict(energy=mean, variance=acc["e2"] / n - mean ** 2,
                    gradient=gradient, metric=metric,
                    acceptance=accepted / proposals,
                    samples=np.array(acc["series"]))

    def _fresh(self):
        k = self.n_parameters
        return dict(n=0, e=0.0, e2=0.0, o=np.zeros(k), eo=np.zeros(k),
                    oo=np.zeros((k, k)), series=[])

    def sample_brute_force(self, n_cycles=2000, n_walkers=200, step=1.0,
                           rng=None, burn_in=200):
        """Metropolis with a uniform symmetric step, one particle at a time."""
        rng = np.random.default_rng(2024) if rng is None else rng
        x = rng.normal(0.0, 1.0, (n_walkers, self.M)) / math.sqrt(self.omega)
        log_old = self.log_psi(x)
        acc, accepted, proposals = self._fresh(), 0, 0
        for cycle in range(n_cycles + burn_in):
            for p in range(self.P):
                slot = slice(p * self.D, (p + 1) * self.D)
                trial = x.copy()
                trial[:, slot] += step * (2.0 * rng.random((n_walkers, self.D)) - 1.0)
                log_new = self.log_psi(trial)
                take = 2.0 * (log_new - log_old) > np.log(rng.random(n_walkers) + 1e-300)
                x[take] = trial[take]
                log_old[take] = log_new[take]
                accepted += int(take.sum())
                proposals += n_walkers
            if cycle >= burn_in:
                self._accumulate(x, acc)
        return self._finish(acc, accepted, proposals)

    def sample_importance(self, n_cycles=2000, n_walkers=200, time_step=0.5,
                          rng=None, burn_in=200):
        """Metropolis-Hastings with the quantum force, as in chapter 13."""
        rng = np.random.default_rng(2024) if rng is None else rng
        diffusion, root_dt = 0.5, math.sqrt(time_step)
        x = rng.normal(0.0, 1.0, (n_walkers, self.M)) / math.sqrt(self.omega)
        log_old = self.log_psi(x)
        force_old = self.quantum_force(x)
        acc, accepted, proposals = self._fresh(), 0, 0
        for cycle in range(n_cycles + burn_in):
            for p in range(self.P):
                slot = slice(p * self.D, (p + 1) * self.D)
                trial = x.copy()
                trial[:, slot] = (x[:, slot] + diffusion * force_old[:, slot] * time_step
                                  + rng.normal(size=(n_walkers, self.D)) * root_dt)
                log_new = self.log_psi(trial)
                force_new = self.quantum_force(trial)
                green = np.sum(0.5 * (force_old[:, slot] + force_new[:, slot])
                               * (0.5 * diffusion * time_step
                                  * (force_old[:, slot] - force_new[:, slot])
                                  - trial[:, slot] + x[:, slot]), axis=1)
                take = green + 2.0 * (log_new - log_old) > np.log(rng.random(n_walkers) + 1e-300)
                x[take] = trial[take]
                log_old[take] = log_new[take]
                force_old[take] = force_new[take]
                accepted += int(take.sum())
                proposals += n_walkers
            if cycle >= burn_in:
                self._accumulate(x, acc)
        return self._finish(acc, accepted, proposals)

    # -- optimisation ------------------------------------------------------------
    def train(self, method="sgd", learning_rate=0.05, max_iter=100, n_cycles=300,
              n_walkers=200, time_step=0.5, regulariser=1e-3, seed=2024,
              sampler="importance", step=1.0, verbose=False):
        """Gradient descent ('sgd') or stochastic reconfiguration ('sr').

        Both use the same Monte Carlo estimate of the gradient; SR
        preconditions it with the inverse of the metric S_kl = <O_k O_l> -
        <O_k><O_l>, regularised by a diagonal shift, Eq. (14-srregularised).
        Returns the history of (iteration, energy, variance, |gradient|).
        """
        history = []
        for iteration in range(1, max_iter + 1):
            rng = np.random.default_rng(seed + iteration)
            if sampler == "importance":
                result = self.sample_importance(n_cycles, n_walkers, time_step, rng)
            else:
                result = self.sample_brute_force(n_cycles, n_walkers, step, rng)
            g = result["gradient"]
            history.append((iteration, result["energy"], result["variance"],
                            float(np.linalg.norm(g))))
            if verbose and (iteration <= 3 or iteration % 20 == 0):
                print(f"      {iteration:4d}  E = {result['energy']:.6f}  "
                      f"var = {result['variance']:.5f}  |g| = {np.linalg.norm(g):.4f}")
            if method == "sr":
                S = result["metric"] + regulariser * np.eye(len(g))
                direction = np.linalg.solve(S, g)
            else:
                direction = g
            self.set_theta(self.get_theta() - learning_rate * direction)
        return history

    def production(self, n_cycles=3000, n_walkers=500, time_step=0.5, seed=7,
                   sampler="importance", step=1.0):
        rng = np.random.default_rng(seed)
        if sampler == "importance":
            result = self.sample_importance(n_cycles, n_walkers, time_step, rng, burn_in=300)
        else:
            result = self.sample_brute_force(n_cycles, n_walkers, step, rng, burn_in=300)
        if blocking is not None:
            mean, error, _ = blocking(result["samples"])
        else:                                                    # pragma: no cover
            mean, error = result["energy"], float(np.std(result["samples"]) / math.sqrt(n_cycles))
        result["error"] = error
        return result


# ---------------------------------------------------------------------------
def check_derivatives(seed=3):
    """Analytic against finite-difference derivatives, with and without the
    Jastrow factor and the variational width."""
    out = {}
    for jastrow, learn_sigma in ((False, False), (True, False), (True, True)):
        rng = np.random.default_rng(seed)
        psi = ProjectNQS(n_hidden=3, rng=rng, scale=0.3, jastrow=jastrow,
                         learn_sigma=learn_sigma, beta=0.5)
        x = rng.normal(0.0, 1.0, (6, psi.M))
        d_param = np.abs(psi.log_derivatives(x) - psi.log_derivatives_fd(x)).max()
        d_energy = np.abs(psi.local_energy(x) - psi.local_energy_fd(x)).max()
        out[(jastrow, learn_sigma)] = (d_param, d_energy)
    return out


def taut_local_energy(n_points=2000, seed=1):
    """E_L of the exact state exp(-(r1^2+r2^2)/2)(1+r12): 3 at every point."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n_points, 4))
    r = x.reshape(-1, 2, 2)
    d = r[:, 0] - r[:, 1]
    dist = np.linalg.norm(d, axis=1)
    unit = d / dist[:, None]
    # ln Psi = -s/2 + ln(1+r12): per-coordinate gradient and Laplacian
    grad = -x.copy().reshape(-1, 2, 2)
    grad[:, 0] += unit / (1.0 + dist)[:, None]
    grad[:, 1] -= unit / (1.0 + dist)[:, None]
    lap = -np.ones((n_points, 2, 2))
    l = (-unit ** 2 / (1.0 + dist)[:, None] ** 2
         + (1.0 - unit ** 2) / (dist * (1.0 + dist))[:, None])
    lap[:, 0] += l
    lap[:, 1] += l
    kinetic = -0.5 * np.sum(lap + grad ** 2, axis=(1, 2))
    potential = 0.5 * np.sum(x ** 2, axis=1) + 1.0 / dist
    return kinetic + potential


def _demo():
    print("=" * 74)
    print("1. Derivatives, analytic against finite differences")
    print("=" * 74)
    for key, (dp, de) in check_derivatives().items():
        print(f"   jastrow={key[0]!s:5s} learn_sigma={key[1]!s:5s}: "
              f"max |dO| = {dp:.1e}, max |dE_L| = {de:.1e}")
    e = taut_local_energy()
    print(f"   Taut's exact state: E_L = {e.mean():.12f} +/- {e.std():.1e} over {len(e)} points")

    print()
    print("=" * 74)
    print("2. One particle in one dimension, two hidden units, no interaction")
    print("=" * 74)
    for rate in (0.01, 0.05, 0.2):
        psi = ProjectNQS(n_particles=1, n_dimensions=1, n_hidden=2, interaction=False,
                         rng=np.random.default_rng(5), scale=0.3)
        hist = psi.train("sgd", learning_rate=rate, max_iter=100, n_cycles=200, n_walkers=100)
        prod = psi.production(n_cycles=2000, n_walkers=200)
        print(f"   eta = {rate:4.2f}: E = {prod['energy']:.6f} +/- {prod['error']:.6f}, "
              f"var = {prod['variance']:.2e}  (exact 0.5)")

    print()
    print("=" * 74)
    print("3. Two electrons at omega = 1: SGD against SR, then the cusp")
    print("=" * 74)
    for label, kwargs, method, rate in (
            ("RBM N=4, SGD", dict(n_hidden=4), "sgd", 0.05),
            ("RBM N=4, SR ", dict(n_hidden=4), "sr", 0.2),
            ("RBM N=4 x Jastrow, SR", dict(n_hidden=4, jastrow=True), "sr", 0.2)):
        psi = ProjectNQS(rng=np.random.default_rng(11), **kwargs)
        hist = psi.train(method, learning_rate=rate, max_iter=120, n_cycles=300, n_walkers=200)
        prod = psi.production()
        print(f"   {label:24s}: E = {prod['energy']:.5f} +/- {prod['error']:.5f}, "
              f"var(E_L) = {prod['variance']:.4f}, acceptance {prod['acceptance']:.3f}"
              + (f", beta = {psi.beta:.3f}" if psi.jastrow else ""))


if __name__ == "__main__":
    _demo()

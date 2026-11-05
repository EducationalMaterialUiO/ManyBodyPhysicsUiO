"""
Boltzmann machines: energy models, Gibbs sampling and neural quantum states.

Companion code to chapter 17 of *Quantum Mechanics for Many-particle Systems*.

The chapter has three strands and so does this program.

  1. **Gibbs sampling.**  The workhorse of energy-based models.  It is
     validated here on a bivariate Gaussian, where the exact answer is known,
     and on an Ising chain, where it is compared with Metropolis.

  2. **Restricted Boltzmann machines.**  The binary-binary and the
     Gaussian-binary RBM, with every marginal and conditional derived in the
     chapter checked here against brute-force enumeration of the partition
     function.  A small binary-binary machine is then trained by contrastive
     divergence on the bars-and-stripes data set, which is small enough that
     the exact log-likelihood and the exact Kullback-Leibler divergence can be
     computed at every step -- so we can watch the thing actually learn.

  3. **Neural quantum states.**  The Gaussian-binary marginal is used as a
     variational wave function for the two-dimensional quantum dot of
     chapters 11 and 13, optimised with the energy gradient of chapter 14 and
     error-analysed with the blocking of the same chapter.

Everything runs on numpy alone; the demonstration takes about three minutes.

Author: Morten Hjorth-Jensen
"""

import itertools
import math

import numpy as np

try:                                    # reuse chapter 14 where available
    from vmcoptimise import blocking
except ImportError:                     # pragma: no cover
    blocking = None


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def sigmoid(x):
    """The logistic function 1/(1+e^-x), written to avoid overflow.

    This function is not an arbitrary choice of nonlinearity.  It is what the
    conditional probability of a binary unit *is*, for any energy that is
    linear in that unit -- see Eqs. (17-bbhidden) and (17-gbhidden).
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    z = np.exp(x[~positive])
    out[~positive] = z / (1.0 + z)
    return out


def softplus(x):
    """log(1 + e^x), computed stably.  The free energy of one hidden unit."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def all_binary_states(n):
    """Every one of the 2^n binary vectors of length n, as an array."""
    return np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)


# ---------------------------------------------------------------------------
#  Gibbs sampling, in general
# ---------------------------------------------------------------------------
def gibbs_bivariate_gaussian(mean, cov, n_samples, rng=None, start=None):
    """Gibbs sampling from a bivariate Gaussian, one coordinate at a time.

    The point of the exercise is that we never touch the joint distribution.
    For a Gaussian the conditionals are Gaussian too,

        x | y ~ N( mu_x + rho (sigma_x/sigma_y)(y - mu_y), sigma_x^2(1-rho^2) ),

    and alternating exact draws from the two conditionals reconstructs the
    joint -- which is the whole claim of the method, here in a case where it
    can be checked against a direct Cholesky draw.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    sx, sy = math.sqrt(cov[0, 0]), math.sqrt(cov[1, 1])
    rho = cov[0, 1] / (sx * sy)

    samples = np.empty((n_samples, 2))
    state = np.array([3.0, -3.0]) if start is None else np.array(start, float)
    for t in range(n_samples):
        # x given y
        mu = mean[0] + rho * (sx / sy) * (state[1] - mean[1])
        state[0] = mu + sx * math.sqrt(1.0 - rho**2) * rng.normal()
        # y given x
        mu = mean[1] + rho * (sy / sx) * (state[0] - mean[0])
        state[1] = mu + sy * math.sqrt(1.0 - rho**2) * rng.normal()
        samples[t] = state
    return samples


def gibbs_ising_chain(n_spins, beta, coupling=1.0, field=0.0, n_sweeps=20000,
                      rng=None, burn_in=2000):
    """Gibbs sampling of a periodic Ising chain, s_i = +-1.

    The conditional of one spin in a frozen environment is
    P(s_i = +1 | rest) = sigmoid(2 beta H_i) with H_i the local field, so the
    update is a single logistic draw and is never rejected.  Returns the mean
    energy per spin, the mean magnetisation and the acceptance-free sweep
    count, for comparison with `metropolis_ising_chain`.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    s = rng.choice([-1.0, 1.0], size=n_spins)
    energies, magnetisations = [], []
    for sweep in range(n_sweeps + burn_in):
        for i in range(n_spins):
            local = coupling * (s[(i - 1) % n_spins] + s[(i + 1) % n_spins]) \
                + field
            p_up = 1.0 / (1.0 + math.exp(-2.0 * beta * local))
            s[i] = 1.0 if rng.random() < p_up else -1.0
        if sweep >= burn_in:
            energy = -coupling * float(np.sum(s * np.roll(s, -1))) \
                - field * float(np.sum(s))
            energies.append(energy / n_spins)
            magnetisations.append(float(np.mean(s)))
    return dict(energy=float(np.mean(energies)),
                magnetisation=float(np.mean(np.abs(magnetisations))),
                series=np.array(energies))


def metropolis_ising_chain(n_spins, beta, coupling=1.0, field=0.0,
                           n_sweeps=20000, rng=None, burn_in=2000,
                           random_order=True):
    """The same chain by single-spin-flip Metropolis, for contrast.

    `random_order` selects the site at random rather than sweeping through
    them in order, and it is not a cosmetic choice.  The Metropolis rule
    accepts a move with Delta E = 0 with probability *one*.  On the Ising
    chain such a move is exactly a domain wall hopping one site, so under a
    deterministic left-to-right sweep every wall moves in lockstep, walls
    never meet, and the number of walls is conserved: the chain is not
    ergodic and the measured energy stays pinned at its initial value.  Set
    `random_order=False` to watch this happen -- it is Exercise 3 of the
    chapter.

    Gibbs sampling has no such pathology, because at zero local field it
    draws the spin from sigmoid(0) = 1/2 rather than flipping it with
    certainty.  This is a small but instructive difference between drawing
    from the exact conditional and accepting a proposal.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    s = rng.choice([-1.0, 1.0], size=n_spins)
    energies, magnetisations = [], []
    accepted = proposed = 0
    for sweep in range(n_sweeps + burn_in):
        sites = rng.integers(0, n_spins, n_spins) if random_order \
            else range(n_spins)
        for i in sites:
            local = coupling * (s[(i - 1) % n_spins] + s[(i + 1) % n_spins]) \
                + field
            delta = 2.0 * s[i] * local           # energy cost of the flip
            proposed += 1
            if delta <= 0.0 or rng.random() < math.exp(-beta * delta):
                s[i] = -s[i]
                accepted += 1
        if sweep >= burn_in:
            energy = -coupling * float(np.sum(s * np.roll(s, -1))) \
                - field * float(np.sum(s))
            energies.append(energy / n_spins)
            magnetisations.append(float(np.mean(s)))
    return dict(energy=float(np.mean(energies)),
                magnetisation=float(np.mean(np.abs(magnetisations))),
                acceptance=accepted / proposed, series=np.array(energies))


def ising_chain_exact(n_spins, beta, coupling=1.0):
    """Exact energy per spin of the periodic zero-field Ising chain.

    From the transfer matrix, e = -J tanh(beta J) in the thermodynamic limit;
    for finite N the exact result is

        e = -J [ tanh(bJ) + (tanh(bJ))^{N-1} ] / [ 1 + (tanh(bJ))^N ] .
    """
    t = math.tanh(beta * coupling)
    return -coupling * (t + t**(n_spins - 1)) / (1.0 + t**n_spins)


# ---------------------------------------------------------------------------
#  The binary-binary restricted Boltzmann machine
# ---------------------------------------------------------------------------
class BinaryBinaryRBM:
    """An RBM with binary visible and binary hidden units, x_i, h_j in {0,1}.

    The energy is

        E(x, h) = - a.x - b.h - x^T W h ,

    and every quantity the chapter derives is implemented twice: once from the
    closed-form result, and once by brute-force enumeration of all 2^M x 2^N
    configurations.  `check_identities` compares them.
    """

    def __init__(self, n_visible, n_hidden, rng=None, scale=0.1):
        self.M, self.N = n_visible, n_hidden
        rng = np.random.default_rng(2024) if rng is None else rng
        self.a = rng.normal(0.0, scale, n_visible)
        self.b = rng.normal(0.0, scale, n_hidden)
        self.W = rng.normal(0.0, scale, (n_visible, n_hidden))

    # -- energy and free energy ------------------------------------------
    def energy(self, x, h):
        """E(x, h) = -a.x - b.h - x^T W h, for one pair or a batch."""
        x, h = np.atleast_2d(x), np.atleast_2d(h)
        return -(x @ self.a) - (h @ self.b) - np.sum((x @ self.W) * h, axis=1)

    def free_energy(self, x):
        """F(x) = -a.x - sum_j softplus(b_j + x.w_{*j}).

        Summing the joint over the hidden units gives
        p(x) = e^{-F(x)}/Z, so F is the *effective* energy of a visible
        configuration after the hidden units have been integrated out.  This is
        Eq. (17-bbmarginalx) written as an energy.
        """
        x = np.atleast_2d(x)
        return -(x @ self.a) - np.sum(softplus(self.b + x @ self.W), axis=1)

    # -- exact quantities, by enumeration --------------------------------
    def partition_function(self):
        """Z, summed over all 2^M visible states via the free energy."""
        return float(np.sum(np.exp(-self.free_energy(all_binary_states(self.M)))))

    def partition_function_bruteforce(self):
        """Z summed over all 2^M x 2^N pairs, the definition."""
        total = 0.0
        for x in all_binary_states(self.M):
            for h in all_binary_states(self.N):
                total += math.exp(-float(self.energy(x, h)[0]))
        return total

    def marginal_x(self, x=None):
        """p(x) for every visible configuration, exactly."""
        states = all_binary_states(self.M) if x is None else np.atleast_2d(x)
        return np.exp(-self.free_energy(states)) / self.partition_function()

    def marginal_x_formula(self, x=None):
        """The closed form  e^{a.x} prod_j (1 + e^{b_j + x.w_{*j}}) / Z."""
        states = all_binary_states(self.M) if x is None else np.atleast_2d(x)
        value = np.exp(states @ self.a) * np.prod(
            1.0 + np.exp(self.b + states @ self.W), axis=1)
        return value / self.partition_function()

    def marginal_h_formula(self, h=None):
        """The closed form  e^{b.h} prod_i (1 + e^{a_i + w_{i*}.h}) / Z."""
        states = all_binary_states(self.N) if h is None else np.atleast_2d(h)
        value = np.exp(states @ self.b) * np.prod(
            1.0 + np.exp(self.a + states @ self.W.T), axis=1)
        return value / self.partition_function()

    # -- conditionals ------------------------------------------------------
    def p_h_given_x(self, x):
        """p(h_j = 1 | x) = sigmoid(b_j + x.w_{*j}), independently for each j."""
        return sigmoid(self.b + np.atleast_2d(x) @ self.W)

    def p_x_given_h(self, h):
        """p(x_i = 1 | h) = sigmoid(a_i + w_{i*}.h), independently for each i."""
        return sigmoid(self.a + np.atleast_2d(h) @ self.W.T)

    def sample_h(self, x, rng):
        p = self.p_h_given_x(x)
        return (rng.random(p.shape) < p).astype(float)

    def sample_x(self, h, rng):
        p = self.p_x_given_h(h)
        return (rng.random(p.shape) < p).astype(float)

    # -- block Gibbs sampling ---------------------------------------------
    def gibbs(self, n_samples, rng=None, burn_in=1000, start=None):
        """Block Gibbs: resample the whole hidden layer, then the whole visible.

        The bipartite structure is what makes this legitimate.  Given x, the
        hidden units are conditionally independent, so all N of them can be
        drawn at once; likewise for the visible layer.  Two draws therefore
        constitute a complete sweep of all M + N variables.
        """
        if rng is None:
            rng = np.random.default_rng(2024)
        x = rng.integers(0, 2, (1, self.M)).astype(float) if start is None \
            else np.atleast_2d(start).astype(float)
        out = np.empty((n_samples, self.M))
        for t in range(n_samples + burn_in):
            h = self.sample_h(x, rng)
            x = self.sample_x(h, rng)
            if t >= burn_in:
                out[t - burn_in] = x[0]
        return out

    # -- training ----------------------------------------------------------
    def contrastive_divergence(self, data, k=1, rng=None):
        """One CD-k gradient estimate, returning (grad_a, grad_b, grad_W).

        The exact gradient of the negative log-likelihood is a difference of
        two moments, Eq. (17-gradmoments); the first is an average over the
        data and is cheap, the second is an average over the model and is not.
        Contrastive divergence replaces the second by a short Gibbs chain
        started *at the data* rather than at equilibrium.  It is a biased
        estimator of the gradient -- and it works.
        """
        if rng is None:
            rng = np.random.default_rng(2024)
        data = np.atleast_2d(data)
        n = len(data)

        p_h_data = self.p_h_given_x(data)              # positive phase
        x = data
        for _ in range(k):                             # k Gibbs steps
            h = (rng.random(p_h_data.shape) < self.p_h_given_x(x)) \
                .astype(float)
            x = self.sample_x(h, rng)
        p_h_model = self.p_h_given_x(x)                # negative phase

        grad_a = (data.mean(axis=0) - x.mean(axis=0))
        grad_b = (p_h_data.mean(axis=0) - p_h_model.mean(axis=0))
        grad_W = (data.T @ p_h_data - x.T @ p_h_model) / n
        return grad_a, grad_b, grad_W

    def log_likelihood(self, data):
        """The exact average log-likelihood, only usable for small M."""
        data = np.atleast_2d(data)
        return float(np.mean(-self.free_energy(data))
                     - math.log(self.partition_function()))

    def kullback_leibler(self, data):
        """KL(f || p) with f the empirical distribution of `data`.

        Exact, by enumeration.  Chapter 17 shows that minimising this is the
        same as maximising the log-likelihood; here the two curves can be
        plotted against each other and the statement checked.
        """
        data = np.atleast_2d(data)
        states = all_binary_states(self.M)
        index = {tuple(s): i for i, s in enumerate(states)}
        empirical = np.zeros(len(states))
        for row in data:
            empirical[index[tuple(row)]] += 1.0
        empirical /= len(data)
        model = self.marginal_x()
        mask = empirical > 0
        return float(np.sum(empirical[mask]
                            * np.log(empirical[mask] / model[mask])))

    def check_identities(self, rng=None):
        """Verify every closed form of the chapter against brute force."""
        rng = np.random.default_rng(7) if rng is None else rng
        z_free = self.partition_function()
        z_brute = self.partition_function_bruteforce()
        px_a, px_b = self.marginal_x(), self.marginal_x_formula()
        ph = self.marginal_h_formula()

        # p(x) and p(h) must both be normalised, and both must reproduce Z
        # the joint p(x,h) = p(h|x)p(x) must sum correctly
        joint_error = 0.0
        for x in all_binary_states(self.M):
            p_x = float(self.marginal_x_formula(x)[0])
            p_h_x = self.p_h_given_x(x)[0]
            for h in all_binary_states(self.N):
                factorised = np.prod(np.where(h > 0.5, p_h_x, 1.0 - p_h_x))
                exact = math.exp(-float(self.energy(x, h)[0])) / z_brute
                joint_error = max(joint_error, abs(p_x * factorised - exact))

        return dict(partition=abs(z_free - z_brute) / z_brute,
                    marginal_x=float(np.abs(px_a - px_b).max()),
                    normalisation_x=abs(float(px_a.sum()) - 1.0),
                    normalisation_h=abs(float(ph.sum()) - 1.0),
                    conditional_factorisation=joint_error)


# ---------------------------------------------------------------------------
#  The Gaussian-binary restricted Boltzmann machine
# ---------------------------------------------------------------------------
class GaussianBinaryRBM:
    """Continuous visible units, binary hidden units.

        E(x, h) = sum_i (x_i-a_i)^2/(2 sigma^2) - b.h - (x/sigma^2)^T W h .

    The marginal over the hidden units is the object chapter 17 turns into a
    wave function, and the conditional p(x_i | h) is a Gaussian with mean
    a_i + w_{i*}.h -- verified numerically below.
    """

    def __init__(self, n_visible, n_hidden, sigma=1.0, rng=None, scale=0.1):
        self.M, self.N = n_visible, n_hidden
        self.sigma = sigma
        rng = np.random.default_rng(2024) if rng is None else rng
        self.a = rng.normal(0.0, scale, n_visible)
        self.b = rng.normal(0.0, scale, n_hidden)
        self.W = rng.normal(0.0, scale, (n_visible, n_hidden))

    def energy(self, x, h):
        x, h = np.atleast_2d(x), np.atleast_2d(h)
        s2 = self.sigma**2
        return (np.sum((x - self.a)**2, axis=1) / (2.0 * s2)
                - h @ self.b
                - np.sum(((x / s2) @ self.W) * h, axis=1))

    def unnormalised_marginal_x(self, x):
        """e^{-||x-a||^2/2sigma^2} prod_j (1 + e^{b_j + (x/sigma^2).w_{*j}}).

        This is Z p_GB(x), Eq. (17-gbmarginalx), and it is what the neural
        quantum state below uses -- the normalisation cancels in every ratio.
        """
        x = np.atleast_2d(x)
        s2 = self.sigma**2
        gaussian = np.exp(-np.sum((x - self.a)**2, axis=1) / (2.0 * s2))
        return gaussian * np.prod(1.0 + np.exp(self.b + (x / s2) @ self.W),
                                  axis=1)

    def unnormalised_marginal_h(self, h):
        """The closed form for p_GB(h), with the Gaussian integral done.

            prod_i sqrt(2 pi sigma_i^2) e^{(2 a_i w_{i*}.h + (w_{i*}.h)^2)/2sigma^2}
        """
        h = np.atleast_2d(h)
        s2 = self.sigma**2
        wh = h @ self.W.T                                   # shape (n, M)
        prefactor = (2.0 * math.pi * s2)**(self.M / 2.0)
        return np.exp(h @ self.b) * prefactor * np.exp(
            np.sum(2.0 * self.a * wh + wh**2, axis=1) / (2.0 * s2))

    def p_h_given_x(self, x):
        """sigmoid(b_j + (x/sigma^2).w_{*j}) -- the same logistic form."""
        return sigmoid(self.b + np.atleast_2d(x) @ self.W / self.sigma**2)

    def mean_x_given_h(self, h):
        """p(x_i | h) = N(x_i ; a_i + w_{i*}.h, sigma^2): this is the mean."""
        return self.a + np.atleast_2d(h) @ self.W.T

    def sample_x_given_h(self, h, rng):
        mean = self.mean_x_given_h(h)
        return mean + self.sigma * rng.normal(size=mean.shape)

    def gibbs(self, n_samples, rng=None, burn_in=1000):
        """Block Gibbs, alternating a binary layer and a Gaussian layer."""
        if rng is None:
            rng = np.random.default_rng(2024)
        x = self.a + self.sigma * rng.normal(size=(1, self.M))
        out = np.empty((n_samples, self.M))
        for t in range(n_samples + burn_in):
            p = self.p_h_given_x(x)
            h = (rng.random(p.shape) < p).astype(float)
            x = self.sample_x_given_h(h, rng)
            if t >= burn_in:
                out[t - burn_in] = x[0]
        return out

    def check_identities(self, n_grid=60, rng=None):
        """Check the marginals and the conditional against numerics.

        The marginal over h is checked by summing the exact joint over the
        hidden states and integrating over x on a grid; the conditional
        p(x|h) is checked by comparing its numerically computed mean and
        variance with a_i + w.h and sigma^2.
        """
        rng = np.random.default_rng(11) if rng is None else rng
        s2 = self.sigma**2

        # p(x) by explicit summation over h, against the closed form
        grid = np.linspace(-6.0, 6.0, n_grid)
        points = rng.normal(self.a, 3.0 * self.sigma, size=(200, self.M))
        summed = np.zeros(len(points))
        for h in all_binary_states(self.N):
            summed += np.exp(-self.energy(points, h))
        marginal_error = float(np.abs(
            summed - self.unnormalised_marginal_x(points)).max()
            / np.abs(summed).max())

        # p(h) by integrating the joint over x on a product grid (M = 1 only)
        integral_error = None
        if self.M == 1:
            x_grid = np.linspace(-30.0, 30.0, 20001).reshape(-1, 1)
            dx = x_grid[1, 0] - x_grid[0, 0]
            states = all_binary_states(self.N)
            numeric = np.array([np.sum(np.exp(-self.energy(x_grid, h))) * dx
                                for h in states])
            closed = self.unnormalised_marginal_h(states)
            integral_error = float(np.abs(numeric - closed).max()
                                   / np.abs(closed).max())

        # the conditional p(x|h) must equal N(x ; a + W h, sigma^2) exactly.
        # Comparing densities rather than sampling gives machine precision.
        density_error = None
        if self.M == 1:
            h = (rng.random(self.N) < 0.5).astype(float)
            x_grid = np.linspace(-30.0, 30.0, 20001).reshape(-1, 1)
            joint = np.exp(-self.energy(x_grid, h))
            conditional = joint / float(self.unnormalised_marginal_h(h)[0])
            mu = float(self.mean_x_given_h(h)[0, 0])
            gaussian = (np.exp(-(x_grid[:, 0] - mu)**2 / (2.0 * s2))
                        / math.sqrt(2.0 * math.pi * s2))
            density_error = float(np.abs(conditional - gaussian).max()
                                  / gaussian.max())

        # and a direct check that sampling it reproduces that mean and variance
        h = (rng.random(self.N) < 0.5).astype(float)
        mean = self.mean_x_given_h(h)
        draws = mean + self.sigma * rng.normal(size=(400000, self.M))
        mean_error = float(np.abs(draws.mean(axis=0) - mean[0]).max())
        var_error = float(np.abs(draws.var(axis=0) - s2).max())
        return dict(marginal_x=marginal_error, marginal_h=integral_error,
                    conditional_density=density_error,
                    conditional_mean=mean_error,
                    conditional_variance=var_error)


# ---------------------------------------------------------------------------
#  Bars and stripes: a data set small enough to check everything on
# ---------------------------------------------------------------------------
def bars_and_stripes(side=3):
    """All patterns whose rows are constant, or whose columns are constant.

    For side = 3 this is 2^3 + 2^3 - 2 = 14 distinct patterns out of the 512
    possible 3x3 binary images.  Small enough that Z, the exact
    log-likelihood and the exact KL divergence can all be enumerated, and
    structured enough that an RBM has something to learn.
    """
    patterns = []
    for bits in itertools.product([0, 1], repeat=side):
        rows = np.array([[b] * side for b in bits], dtype=float)
        patterns.append(rows.ravel())
        patterns.append(rows.T.ravel())
    unique = {tuple(p) for p in patterns}
    return np.array(sorted(unique), dtype=float)


def train_bars_and_stripes(n_hidden=8, epochs=3000, learning_rate=0.1, k=1,
                           seed=3, report_every=500):
    """Train a binary-binary RBM by contrastive divergence, exactly monitored."""
    rng = np.random.default_rng(seed)
    data = bars_and_stripes(3)
    machine = BinaryBinaryRBM(9, n_hidden, rng=rng, scale=0.01)
    history = []
    for epoch in range(epochs + 1):
        if epoch % report_every == 0:
            history.append((epoch, machine.log_likelihood(data),
                            machine.kullback_leibler(data)))
        ga, gb, gW = machine.contrastive_divergence(data, k=k, rng=rng)
        machine.a += learning_rate * ga
        machine.b += learning_rate * gb
        machine.W += learning_rate * gW
    return machine, data, history


# ---------------------------------------------------------------------------
#  Neural quantum states: the Gaussian-binary RBM as a wave function
# ---------------------------------------------------------------------------
class NeuralQuantumState:
    """Psi(x) = sqrt(F_rbm(x)) for the two-dimensional quantum dot.

    The visible units are the particle coordinates, M = P x D of them, and the
    hidden units supply the correlations.  Taking the square root of the RBM
    marginal rather than the marginal itself is what keeps |Psi|^2 equal to a
    genuine RBM distribution, so that the machine could also be sampled by
    Gibbs; it costs only a factor of one half in every logarithmic derivative.

    With Q_n = b_n + sum_i x_i w_{in}/sigma^2,

        ln Psi = const - sum_m (x_m - a_m)^2/(4 sigma^2)
                 + (1/2) sum_n softplus(Q_n) .

    Note the default width.  Setting all weights to zero leaves the Gaussian
    factor exp(-sum_m x_m^2 / 4 sigma^2), which is the harmonic-oscillator
    ground state exactly when 1/(4 sigma^2) = omega/2, that is

        sigma^2 = 1/(2 omega) .

    With that choice the non-interacting dot is *exactly* representable and
    the hidden units have only the correlations left to describe -- which is
    the natural division of labour, and makes the variance of the local energy
    a clean measure of what the hidden units have achieved.
    """

    def __init__(self, n_particles=2, n_dimensions=2, n_hidden=2, sigma=None,
                 omega=1.0, interaction=True, rng=None, scale=0.05):
        rng = np.random.default_rng(2024) if rng is None else rng
        self.P, self.D, self.N = n_particles, n_dimensions, n_hidden
        self.M = n_particles * n_dimensions
        self.omega, self.interaction = omega, interaction
        self.sigma = 1.0 / math.sqrt(2.0 * omega) if sigma is None else sigma
        self.a = rng.normal(0.0, scale, self.M)
        self.b = rng.normal(0.0, scale, self.N)
        self.W = rng.normal(0.0, scale, (self.M, self.N))

    # -- the wave function and its derivatives ----------------------------
    #  Every routine takes an array of shape (n_walkers, M) and returns the
    #  corresponding array, so that a whole ensemble is advanced at once.
    def _Q(self, x):
        return self.b + x @ self.W / self.sigma**2

    def log_psi(self, x):
        x = np.atleast_2d(x)
        s2 = self.sigma**2
        return (-np.sum((x - self.a)**2, axis=1) / (4.0 * s2)
                + 0.5 * np.sum(softplus(self._Q(x)), axis=1))

    def grad_log_psi_x(self, x):
        """d ln Psi / d x_m -- half the quantum force."""
        x = np.atleast_2d(x)
        s2 = self.sigma**2
        return (-(x - self.a) + sigmoid(self._Q(x)) @ self.W.T) / (2.0 * s2)

    def laplacian_log_psi(self, x):
        """d^2 ln Psi / d x_m^2, one term per visible unit."""
        x = np.atleast_2d(x)
        s2 = self.sigma**2
        s = sigmoid(self._Q(x))
        return (-1.0 / (2.0 * s2)
                + (s * (1.0 - s)) @ (self.W**2).T / (2.0 * s2**2))

    def quantum_force(self, x):
        return 2.0 * self.grad_log_psi_x(x)

    def local_energy(self, x):
        """E_L = (1/2) sum_m [ -(d_m ln Psi)^2 - d^2_m ln Psi + w^2 x_m^2 ]
        plus the Coulomb repulsion."""
        x = np.atleast_2d(x)
        first = self.grad_log_psi_x(x)
        second = self.laplacian_log_psi(x)
        energy = 0.5 * np.sum(-first**2 - second + self.omega**2 * x**2,
                              axis=1)
        if self.interaction:
            r = x.reshape(len(x), self.P, self.D)
            for p in range(self.P):
                for q in range(p):
                    energy += 1.0 / np.linalg.norm(r[:, p] - r[:, q], axis=1)
        return energy

    def grad_log_psi_parameters(self, x):
        """d ln Psi / d(a, b, W) averaged over walkers, the O_theta of ch. 14.

        Returned as three arrays with a leading walker axis, so that the
        covariance with the local energy can be formed directly.
        """
        x = np.atleast_2d(x)
        s2 = self.sigma**2
        s = sigmoid(self._Q(x))
        return ((x - self.a) / (2.0 * s2), 0.5 * s,
                x[:, :, None] * s[:, None, :] / (2.0 * s2))

    # -- sampling -----------------------------------------------------------
    def sample(self, n_cycles=2000, n_walkers=200, time_step=0.5, rng=None,
               burn_in=200, keep_samples=False):
        """Importance-sampled Metropolis-Hastings, vectorised over walkers.

        One particle is moved at a time, as in chapter 13, but the move is
        made for every walker in the ensemble simultaneously.  Returned are
        the mean energy, its variance, the parameter gradient of chapter 14
        and, optionally, the walker-averaged series for blocking.
        """
        if rng is None:
            rng = np.random.default_rng(2024)
        diffusion, root_dt = 0.5, math.sqrt(time_step)
        x = rng.normal(0.0, 1.0, (n_walkers, self.M)) * root_dt
        log_old = self.log_psi(x)
        force_old = self.quantum_force(x)

        energy_sum = energy_sq = 0.0
        grads = [np.zeros_like(self.a), np.zeros_like(self.b),
                 np.zeros_like(self.W)]
        grads_e = [np.zeros_like(self.a), np.zeros_like(self.b),
                   np.zeros_like(self.W)]
        accepted = proposals = 0
        series = np.empty(n_cycles) if keep_samples else None

        for cycle in range(n_cycles + burn_in):
            for p in range(self.P):
                slot = slice(p * self.D, (p + 1) * self.D)
                trial = x.copy()
                trial[:, slot] = (x[:, slot]
                                  + diffusion * force_old[:, slot] * time_step
                                  + rng.normal(size=(n_walkers, self.D))
                                  * root_dt)
                log_new = self.log_psi(trial)
                force_new = self.quantum_force(trial)
                green = np.sum(
                    0.5 * (force_old[:, slot] + force_new[:, slot])
                    * (0.5 * diffusion * time_step
                       * (force_old[:, slot] - force_new[:, slot])
                       - trial[:, slot] + x[:, slot]), axis=1)
                take = (green + 2.0 * (log_new - log_old)
                        > np.log(rng.random(n_walkers) + 1e-300))
                x[take] = trial[take]
                log_old[take] = log_new[take]
                force_old[take] = force_new[take]
                accepted += int(take.sum())
                proposals += n_walkers
            if cycle >= burn_in:
                e = self.local_energy(x)
                o = self.grad_log_psi_parameters(x)
                energy_sum += float(e.mean())
                energy_sq += float((e * e).mean())
                for k in range(3):
                    grads[k] += o[k].mean(axis=0)
                    grads_e[k] += (o[k] * e.reshape((-1,) + (1,) * (o[k].ndim
                                                                    - 1))
                                   ).mean(axis=0)
                if keep_samples:
                    series[cycle - burn_in] = float(e.mean())

        mean = energy_sum / n_cycles
        gradient = [2.0 * (grads_e[k] / n_cycles - grads[k] / n_cycles * mean)
                    for k in range(3)]
        return dict(energy=mean, gradient=gradient,
                    variance=energy_sq / n_cycles - mean**2,
                    acceptance=accepted / proposals, samples=series)

    def optimise(self, learning_rate=0.05, max_iter=60, n_cycles=400,
                 n_walkers=200, time_step=0.5, seed=2024, tol=1e-4,
                 verbose=False):
        """Plain gradient descent on the RBM parameters, chapter 14 style."""
        history = []
        for iteration in range(1, max_iter + 1):
            result = self.sample(n_cycles=n_cycles, n_walkers=n_walkers,
                                 time_step=time_step,
                                 rng=np.random.default_rng(seed + iteration))
            norm = math.sqrt(sum(float(np.sum(g**2))
                                 for g in result["gradient"]))
            history.append((iteration, result["energy"], result["variance"],
                            norm))
            if verbose and (iteration <= 3 or iteration % 10 == 0):
                print(f"      {iteration:3d}  E = {result['energy']:.6f}  "
                      f"var = {result['variance']:.6f}  |grad| = {norm:.5f}")
            if norm < tol:
                break
            self.a -= learning_rate * result["gradient"][0]
            self.b -= learning_rate * result["gradient"][1]
            self.W -= learning_rate * result["gradient"][2]
        return history

    def optimise_annealed(self, rates=(0.05, 0.02, 0.01, 0.005), stage=60,
                          n_cycles=300, n_walkers=200, verbose=False):
        """Gradient descent with a decreasing learning rate.

        A fixed rate large enough to escape the initial configuration is too
        large to settle at the end; a schedule is the cheapest fix, and it
        matters here far more than the number of hidden units does.
        """
        history = []
        for rate in rates:
            history += self.optimise(learning_rate=rate, max_iter=stage,
                                     n_cycles=n_cycles, n_walkers=n_walkers,
                                     seed=int(round(rate * 10000)),
                                     verbose=verbose)
        return history


def check_nqs_derivatives(seed=5):
    """Every analytic derivative of ln Psi against a central difference."""
    rng = np.random.default_rng(seed)
    nqs = NeuralQuantumState(n_particles=2, n_dimensions=2, n_hidden=3,
                             rng=rng, scale=0.3)
    x = rng.normal(0.0, 1.0, (1, nqs.M))
    h = 1e-5
    h2 = 1e-3                       # a second difference loses two more digits
    value = lambda y: float(nqs.log_psi(y)[0])

    grad = np.empty(nqs.M)
    lap = np.empty(nqs.M)
    for m in range(nqs.M):
        step = np.zeros((1, nqs.M))
        step[0, m] = h
        grad[m] = (value(x + step) - value(x - step)) / (2 * h)
        step[0, m] = h2
        lap[m] = (value(x + step) - 2 * value(x) + value(x - step)) / h2**2

    da, db, dW = nqs.grad_log_psi_parameters(x)
    da, db, dW = da[0], db[0], dW[0]

    def central(array, index):
        array[index] += h
        plus = value(x)
        array[index] -= 2 * h
        minus = value(x)
        array[index] += h
        return (plus - minus) / (2 * h)

    fd_a = np.array([central(nqs.a, m) for m in range(nqs.M)])
    fd_b = np.array([central(nqs.b, n) for n in range(nqs.N)])
    fd_W = np.array([[central(nqs.W, (m, n)) for n in range(nqs.N)]
                     for m in range(nqs.M)])

    # and the local energy against a finite-difference Laplacian of Psi itself
    psi = lambda y: math.exp(value(y))
    kinetic = 0.0
    for m in range(nqs.M):
        step = np.zeros((1, nqs.M))
        step[0, m] = h2
        kinetic += (psi(x + step) - 2 * psi(x) + psi(x - step)) / h2**2
    numeric_energy = (-0.5 * kinetic / psi(x)
                      + 0.5 * nqs.omega**2 * float(np.sum(x**2)))
    if nqs.interaction:
        r = x.reshape(nqs.P, nqs.D)
        numeric_energy += 1.0 / np.linalg.norm(r[0] - r[1])

    return dict(
        grad_x=float(np.abs(nqs.grad_log_psi_x(x)[0] - grad).max()),
        laplacian_x=float(np.abs(nqs.laplacian_log_psi(x)[0] - lap).max()),
        grad_a=float(np.abs(da - fd_a).max()),
        grad_b=float(np.abs(db - fd_b).max()),
        grad_W=float(np.abs(dW - fd_W).max()),
        local_energy=abs(float(nqs.local_energy(x)[0]) - numeric_energy))


# ---------------------------------------------------------------------------
def _demo():
    print("=" * 74)
    print("1. Gibbs sampling on a bivariate Gaussian")
    print("=" * 74)
    print("Gibbs never touches the joint distribution: it alternates exact")
    print("draws from the two univariate conditionals.  If the claim of the")
    print("method is right, that must reconstruct the joint.  Here the joint")
    print("is a Gaussian with correlation 0.8, so we can check.")
    print()
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, 0.8], [0.8, 1.0]])
    rng = np.random.default_rng(2024)
    direct = (np.linalg.cholesky(cov) @ rng.normal(size=(2, 200000))).T + mean
    gibbs = gibbs_bivariate_gaussian(mean, cov, 200000,
                                     rng=np.random.default_rng(7))
    print(f"{'quantity':>18s} {'exact':>10s} {'direct draw':>13s} "
          f"{'Gibbs':>10s}")
    print(f"{'mean x':>18s} {0.0:10.4f} {direct[:,0].mean():13.4f} "
          f"{gibbs[:,0].mean():10.4f}")
    print(f"{'mean y':>18s} {0.0:10.4f} {direct[:,1].mean():13.4f} "
          f"{gibbs[:,1].mean():10.4f}")
    print(f"{'var x':>18s} {1.0:10.4f} {direct[:,0].var():13.4f} "
          f"{gibbs[:,0].var():10.4f}")
    print(f"{'var y':>18s} {1.0:10.4f} {direct[:,1].var():13.4f} "
          f"{gibbs[:,1].var():10.4f}")
    print(f"{'correlation':>18s} {0.8:10.4f} "
          f"{np.corrcoef(direct.T)[0,1]:13.4f} "
          f"{np.corrcoef(gibbs.T)[0,1]:10.4f}")
    print()
    print("   The price is correlation between successive samples: Gibbs")
    print("   moves one coordinate at a time, so the chain creeps along the")
    print("   ridge of a correlated distribution.  Correlation times of the")
    print("   x coordinate at several values of rho:")
    print()
    print(f"{'rho':>8s} {'tau':>10s}")
    for rho in (0.0, 0.5, 0.8, 0.95, 0.99):
        c = np.array([[1.0, rho], [rho, 1.0]])
        s = gibbs_bivariate_gaussian(mean, c, 40000,
                                     rng=np.random.default_rng(3))[:, 0]
        d = s - s.mean()
        var = float(np.dot(d, d) / len(d))
        tau, k = 1.0, 1
        while k < 400:
            rho_k = float(np.dot(d[:-k], d[k:]) / len(d) / var)
            if rho_k < 0.0:
                break
            tau += 2.0 * rho_k
            k += 1
        print(f"{rho:8.2f} {tau:10.2f}")
    print()
    print("   At rho = 0.99 successive samples are worth a small fraction of")
    print("   an independent one.  This is the generic weakness of Gibbs, and")
    print("   the reason blocked updates matter.")

    print()
    print("=" * 74)
    print("2. Gibbs against Metropolis on the Ising chain")
    print("=" * 74)
    print("Both are correct; they differ in that Gibbs never rejects.  The")
    print("exact energy per spin of the periodic chain is known from the")
    print("transfer matrix, so both can be checked.")
    print()
    print(f"{'beta':>7s} {'exact e':>11s} {'Gibbs':>11s} {'Metropolis':>12s} "
          f"{'M accept':>10s}")
    for beta in (0.25, 0.5, 1.0, 2.0):
        exact = ising_chain_exact(20, beta)
        g = gibbs_ising_chain(20, beta, n_sweeps=20000,
                              rng=np.random.default_rng(1))
        m = metropolis_ising_chain(20, beta, n_sweeps=20000,
                                   rng=np.random.default_rng(1))
        print(f"{beta:7.2f} {exact:11.5f} {g['energy']:11.5f} "
              f"{m['energy']:12.5f} {m['acceptance']:10.1%}")
    print()
    print("   Both reproduce the exact energy.  But look at the last column:")
    print("   at beta = 2 the Metropolis acceptance has fallen to one per")
    print("   cent, so ninety-nine of every hundred proposals are thrown")
    print("   away, while Gibbs draws the spin from its exact conditional and")
    print("   keeps every result.  That is the practical argument for Gibbs")
    print("   whenever the conditionals are available in closed form -- which,")
    print("   for a restricted Boltzmann machine, they always are.")
    print()
    print("   There is a subtler difference too.  Metropolis accepts a move")
    print("   with Delta E = 0 with probability one, and on this chain such a")
    print("   move is a domain wall hopping one site.  Under a deterministic")
    print("   sweep every wall then moves in lockstep, walls never meet and")
    print("   never annihilate, and the number of walls is conserved:")
    print()
    print(f"{'beta':>7s} {'exact e':>11s} {'random order':>14s} "
          f"{'in-order sweep':>16s}")
    for beta in (0.5, 2.0):
        exact = ising_chain_exact(20, beta)
        good = metropolis_ising_chain(20, beta, n_sweeps=5000,
                                      rng=np.random.default_rng(1))
        bad = metropolis_ising_chain(20, beta, n_sweeps=5000,
                                     rng=np.random.default_rng(1),
                                     random_order=False)
        print(f"{beta:7.2f} {exact:11.5f} {good['energy']:14.5f} "
              f"{bad['energy']:16.5f}")
    print()
    print("   The in-order column is stuck at the energy of its random")
    print("   starting configuration: the chain is not ergodic.  Gibbs")
    print("   sampling has no such pathology, because at zero local field it")
    print("   draws from sigmoid(0) = 1/2 rather than flipping with")
    print("   certainty.  Drawing from the conditional and accepting a")
    print("   proposal are not always the same thing.")

    print()
    print("=" * 74)
    print("3. The binary-binary RBM: every closed form checked")
    print("=" * 74)
    print("Marginals and conditionals derived in the chapter, against a")
    print("brute-force sum over all 2^M x 2^N configurations of a machine")
    print("with M = 6 visible and N = 4 hidden units.")
    print()
    machine = BinaryBinaryRBM(6, 4, rng=np.random.default_rng(1), scale=0.8)
    checks = machine.check_identities()
    labels = (("partition", "Z from free energy vs. brute force"),
              ("marginal_x", "p(x) enumerated vs. closed form"),
              ("normalisation_x", "sum_x p(x) - 1"),
              ("normalisation_h", "sum_h p(h) - 1"),
              ("conditional_factorisation",
               "p(x,h) - p(x) prod_j p(h_j|x)"))
    for key, label in labels:
        print(f"      {label:<40s} {checks[key]:.2e}")
    print()
    print("   The last line is the one that matters most.  It says that the")
    print("   hidden units really are conditionally independent given the")
    print("   visible ones -- which is what the missing intra-layer couplings")
    print("   buy, and what makes block Gibbs sampling possible.")
    print()
    print("   And Gibbs sampling of the machine, against its exact marginal:")
    print()
    small = BinaryBinaryRBM(4, 3, rng=np.random.default_rng(2), scale=1.2)
    exact = small.marginal_x()
    draws = small.gibbs(400000, rng=np.random.default_rng(9))
    states = all_binary_states(4)
    index = {tuple(s): i for i, s in enumerate(states)}
    empirical = np.zeros(len(states))
    for row in draws:
        empirical[index[tuple(row)]] += 1.0
    empirical /= len(draws)
    order = np.argsort(-exact)
    print(f"{'state':>10s} {'exact p(x)':>12s} {'Gibbs':>10s}")
    for i in order[:6]:
        label = "".join(str(int(v)) for v in states[i])
        print(f"{label:>10s} {exact[i]:12.5f} {empirical[i]:10.5f}")
    print(f"   largest absolute deviation over all 16 states "
          f"{np.abs(exact-empirical).max():.5f}")

    print()
    print("=" * 74)
    print("4. Training by contrastive divergence")
    print("=" * 74)
    print("Bars and stripes on a 3x3 grid: 14 of the 512 possible images have")
    print("either all rows constant or all columns constant.  With only nine")
    print("visible units the partition function can be summed exactly, so the")
    print("log-likelihood and the Kullback-Leibler divergence are known at")
    print("every step -- and the chapter's claim that minimising one is")
    print("maximising the other can be watched directly.")
    print()
    machine, data, history = train_bars_and_stripes(
        n_hidden=8, epochs=20000, learning_rate=0.1, k=5, report_every=2500)
    print(f"   {len(data)} distinct patterns, uniform target probability "
          f"{1/len(data):.5f}")
    print(f"   a perfect model would give log-likelihood "
          f"{math.log(1/len(data)):.4f} and KL exactly 0")
    print()
    print(f"{'epoch':>8s} {'log-likelihood':>16s} {'KL(f||p)':>12s}")
    for epoch, ll, kl in history:
        print(f"{epoch:8d} {ll:16.4f} {kl:12.4f}")
    print()
    total = float(np.sum(machine.marginal_x_formula(data)))
    print(f"   The two columns move together, exactly as the chapter's")
    print(f"   identity KL = <log f> + C_LL requires: they differ by a")
    print(f"   constant, the entropy of the data, which here is")
    print(f"   -log(1/14) = {-math.log(1/len(data)):.4f}.")
    print()
    print(f"   After training the model puts {total:.1%} of its probability on")
    print(f"   the 14 valid patterns, against {14/512:.1%} at random.  It has")
    print(f"   learned the rule 'rows constant or columns constant' without")
    print(f"   ever being told it, from 14 examples and 8 hidden units.")
    print()
    print("   The number of Gibbs steps k in CD-k matters.  Same budget of")
    print("   20000 updates, same learning rate:")
    print()
    print(f"{'k':>6s} {'log-likelihood':>16s} {'KL(f||p)':>12s}")
    for k in (1, 2, 5, 10):
        _, _, h = train_bars_and_stripes(n_hidden=8, epochs=20000,
                                         learning_rate=0.1, k=k,
                                         report_every=20000)
        print(f"{k:6d} {h[-1][1]:16.4f} {h[-1][2]:12.4f}")
    print()
    print("   CD-1 is the cheapest and the most biased: one Gibbs step from")
    print("   the data is nowhere near the model's equilibrium, so the")
    print("   negative phase is badly estimated.  It still works, which is")
    print("   the surprising and much-discussed fact about the algorithm.")

    print()
    print("=" * 74)
    print("5. The Gaussian-binary RBM")
    print("=" * 74)
    print("The continuous case.  The marginal over the hidden units, the")
    print("marginal over the visible ones (which needs a Gaussian integral),")
    print("and the conditional p(x|h) = N(a + Wh, sigma^2):")
    print()
    gb = GaussianBinaryRBM(1, 3, sigma=1.0, rng=np.random.default_rng(4),
                           scale=0.7)
    checks = gb.check_identities()
    print(f"      p(x): summed over h vs. closed form     "
          f"{checks['marginal_x']:.2e}")
    print(f"      p(h): integrated over x vs. closed form "
          f"{checks['marginal_h']:.2e}")
    print(f"      p(x|h) vs. N(a + Wh, sigma^2)           "
          f"{checks['conditional_density']:.2e}")
    print(f"      sampled mean of p(x|h) vs. a + W h      "
          f"{checks['conditional_mean']:.2e}")
    print(f"      sampled variance vs. sigma^2            "
          f"{checks['conditional_variance']:.2e}")
    print()
    print("   The second line is the one that required work in the chapter:")
    print("   completing the square in the exponent and doing the integral")
    print("   over each continuous visible unit in turn.")

    print()
    print("=" * 74)
    print("6. Neural quantum states: derivatives")
    print("=" * 74)
    print("Psi = sqrt(F_rbm) as a trial wave function.  Everything the local")
    print("energy and the optimiser need, against central differences:")
    print()
    d = check_nqs_derivatives()
    for key, label in (("grad_x", "d ln Psi / d x_m"),
                       ("laplacian_x", "d^2 ln Psi / d x_m^2"),
                       ("grad_a", "d ln Psi / d a_m"),
                       ("grad_b", "d ln Psi / d b_n"),
                       ("grad_W", "d ln Psi / d w_mn"),
                       ("local_energy", "E_L vs. numerical H Psi / Psi")):
        print(f"      {label:<32s} {d[key]:.2e}")

    print()
    print("=" * 74)
    print("7. Two electrons in a quantum dot")
    print("=" * 74)
    print("The system of chapters 11, 13 and 16: two electrons in a")
    print("two-dimensional harmonic trap at omega = 1, where Taut's exact")
    print("ground-state energy is 3 a.u. with the Coulomb repulsion and 2")
    print("without it.  The trial function is an RBM with no physics in it at")
    print("all -- no cusp condition, no Pade-Jastrow, no knowledge that the")
    print("particles repel.  The width is fixed by sigma^2 = 1/(2 omega),")
    print("which makes the Gaussian factor the exact non-interacting state.")
    print()
    print("   Non-interacting, exact 2.000000:")
    for n_hidden in (2, 4):
        nqs = NeuralQuantumState(2, 2, n_hidden, interaction=False,
                                 rng=np.random.default_rng(17), scale=0.05)
        nqs.optimise(learning_rate=0.05, max_iter=60, n_cycles=400,
                     n_walkers=200)
        final = nqs.sample(n_cycles=3000, n_walkers=500,
                           rng=np.random.default_rng(99), keep_samples=True)
        series = final["samples"]
        error = blocking(series)[1] if blocking is not None else \
            series.std(ddof=1) / math.sqrt(len(series))
        print(f"      N = {n_hidden} hidden:  E = {final['energy']:.6f} "
              f"+/- {error:.6f}   variance {final['variance']:.2e}")
    print()
    print("   The variance collapses towards zero, which is the zero-variance")
    print("   principle of chapter 14 in action: with all weights driven to")
    print("   zero the RBM *is* the exact ground state, and the optimiser")
    print("   finds it.  This is the calibration run -- if it does not come")
    print("   out at 2 with a vanishing variance, something is wrong.")
    print()
    print("   Interacting, exact 3.000000:")
    print(f"{'':>6s}{'N':>4s} {'E':>11s} {'error':>10s} {'variance':>10s} "
          f"{'E - exact':>11s}")
    for n_hidden in (2, 4, 8):
        nqs = NeuralQuantumState(2, 2, n_hidden, interaction=True,
                                 rng=np.random.default_rng(17), scale=0.4)
        nqs.optimise_annealed()
        final = nqs.sample(n_cycles=3000, n_walkers=500,
                           rng=np.random.default_rng(99), keep_samples=True)
        series = final["samples"]
        error = blocking(series)[1] if blocking is not None else \
            series.std(ddof=1) / math.sqrt(len(series))
        print(f"{'':>6s}{n_hidden:4d} {final['energy']:11.5f} {error:10.5f} "
              f"{final['variance']:10.4f} {final['energy']-3.0:+11.5f}")
    print()
    print("   Set against the same system solved by other means:")
    print()
    print(f"{'method':>34s} {'energy':>12s} {'error':>12s}")
    for label, value in (("Hartree-Fock, 42 orbitals", 3.161921),
                         ("MP2, 42 orbitals", 3.027038),
                         ("CCSD, 42 orbitals", 3.013626),
                         ("VMC, Pade-Jastrow (chapter 14)", 3.000549),
                         ("exact (Taut)", 3.000000)):
        print(f"{label:>34s} {value:12.6f} {value-3.0:12.6f}")
    print()
    print("   Two honest observations.  The RBM is competitive with")
    print("   Hartree-Fock without being given a single-particle basis, an")
    print("   orbital, or any notion of antisymmetry -- it learns the")
    print("   correlations from the energy alone.  And it is beaten")
    print("   comfortably by the two-parameter Pade-Jastrow function of")
    print("   chapter 13, because that function was built to satisfy the cusp")
    print("   condition and the RBM was not.  The large variance is the same")
    print("   fact seen from the other side: the local energy diverges as")
    print("   1/r_12 wherever the electrons meet, and no amount of training")
    print("   removes a divergence the ansatz cannot represent.  That is the")
    print("   motivation for the neural quantum states of chapter 18.")


if __name__ == "__main__":
    _demo()

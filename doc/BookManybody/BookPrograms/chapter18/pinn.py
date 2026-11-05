"""
Neural quantum states and physics-informed neural networks.

Companion code to chapter 18 of *Quantum Mechanics for Many-particle Systems*.

Three things are demonstrated, in increasing order of physical interest.

  1. **Automatic differentiation, done by hand.**  A PINN needs second
     derivatives of a network with respect to its inputs.  `Correlator`
     propagates value, gradient and Hessian forward through a tanh network in
     one pass -- which is what an AD framework does for you, written out so
     that it can be seen.  It is checked against central differences.

  2. **The variational principle, and how a PINN can lose it.**  On the
     quartic oscillator, where the exact ground-state energy is known to
     machine precision from a grid diagonalisation, the *same* one-parameter
     trial family is optimised two ways: by minimising the Rayleigh quotient,
     and by minimising the residual of the Schroedinger equation at uniformly
     sampled collocation points.  The first returns an upper bound.  The
     second returns energies far *below* the exact ground state, and
     unboundedly so as the collocation domain grows.

  3. **A Slater-Jastrow-neural ansatz for the two-electron quantum dot.**  The
     Pade-Jastrow function of chapter 13 multiplied by a permutation-invariant
     neural correlator, optimised with the energy gradient of chapter 14.
     This is the hybrid recommended at the end of chapter 17: the cusp and the
     antisymmetry are exact by construction, and the network supplies the
     remainder.

  A fourth section checks numerically that a permutation-equivariant backflow
  map preserves the antisymmetry of a Slater determinant and that a
  non-equivariant one destroys it.

Everything runs on numpy; `scipy.linalg` is used once, for the reference
diagonalisation.  The demonstration takes about five minutes.

Author: Morten Hjorth-Jensen
"""

import math

import numpy as np

try:                                    # reuse chapter 14 where available
    from vmcoptimise import blocking
except ImportError:                     # pragma: no cover
    blocking = None


# ---------------------------------------------------------------------------
#  1. A small network that knows its own first and second derivatives
# ---------------------------------------------------------------------------
class Correlator:
    """A tanh multilayer perceptron with forward-mode derivative propagation.

    The network maps a vector of `n_in` features to a scalar.  What makes it
    useful for a PINN is `value_grad_hess`, which returns the value, the
    gradient and the full Hessian with respect to the *inputs* in a single
    forward pass, exactly and at floating-point precision.  The recursion is

        z = W a + b,            a' = tanh(z),
        dz/du_k = W (da/du_k),
        d2z/du_k du_l = W (d2a/du_k du_l),
        da'/du_k = (1 - t^2) dz/du_k,
        d2a'/du_k du_l = (1 - t^2) d2z/du_k du_l
                         - 2 t (1 - t^2) (dz/du_k)(dz/du_l),

    with t = tanh(z).  This is forward-mode automatic differentiation carried
    to second order, written out.  A framework such as JAX or PyTorch does
    exactly this, and the point of writing it by hand here is that the cost is
    then visible: each extra derivative order multiplies the work by the
    number of inputs.

    tanh is not an arbitrary choice.  A PINN differentiates its network twice,
    and ReLU has a second derivative that is zero almost everywhere -- it is
    unusable for any second-order operator, the Laplacian included.
    """

    def __init__(self, n_in, widths=(16, 16), rng=None, scale=0.1):
        rng = np.random.default_rng(2024) if rng is None else rng
        sizes = (n_in,) + tuple(widths) + (1,)
        self.n_in = n_in
        self.W = [rng.normal(0.0, scale / math.sqrt(sizes[i]),
                             (sizes[i + 1], sizes[i]))
                  for i in range(len(sizes) - 1)]
        self.b = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    # -- parameters as a flat vector ---------------------------------------
    @property
    def n_parameters(self):
        return sum(w.size for w in self.W) + sum(v.size for v in self.b)

    def get_parameters(self):
        return np.concatenate([w.ravel() for w in self.W]
                              + [v.ravel() for v in self.b])

    def set_parameters(self, flat):
        i = 0
        for w in self.W:
            w[...] = flat[i:i + w.size].reshape(w.shape)
            i += w.size
        for v in self.b:
            v[...] = flat[i:i + v.size]
            i += v.size

    # -- forward pass -------------------------------------------------------
    def __call__(self, u):
        """The scalar output, for a batch of inputs of shape (n, n_in)."""
        a = np.atleast_2d(u)
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w.T + b
            a = np.tanh(z) if k < len(self.W) - 1 else z
        return a[:, 0]

    def value_grad_hess(self, u):
        """Return (f, df/du, d2f/du2) for a batch of inputs (n, n_in).

        Shapes are (n,), (n, n_in) and (n, n_in, n_in).
        """
        u = np.atleast_2d(u)
        n, d = u.shape
        a = u
        da = np.repeat(np.eye(d)[None, :, :], n, axis=0)      # (n, d, d)
        d2a = np.zeros((n, d, d, d))                          # (n, unit, k, l)

        for k, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w.T + b                                   # (n, m)
            dz = np.einsum("mj,njk->nmk", w, da)              # (n, m, d)
            d2z = np.einsum("mj,njkl->nmkl", w, d2a)          # (n, m, d, d)
            if k < len(self.W) - 1:
                t = np.tanh(z)
                s = 1.0 - t**2
                a = t
                da = s[:, :, None] * dz
                d2a = (s[:, :, None, None] * d2z
                       - 2.0 * (t * s)[:, :, None, None]
                       * dz[:, :, :, None] * dz[:, :, None, :])
            else:
                a, da, d2a = z, dz, d2z
        return a[:, 0], da[:, 0, :], d2a[:, 0, :, :]

    def value_grad(self, u):
        """(f, df/du) only -- first order, and about half the cost.

        The Metropolis loop needs the drift but not the Laplacian, and the
        Hessian propagation is the expensive part, so it is worth having a
        first-order path.
        """
        u = np.atleast_2d(u)
        n, d = u.shape
        a = u
        da = np.repeat(np.eye(d)[None, :, :], n, axis=0)
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w.T + b
            dz = np.einsum("mj,njk->nmk", w, da)
            if k < len(self.W) - 1:
                t = np.tanh(z)
                a = t
                da = (1.0 - t**2)[:, :, None] * dz
            else:
                a, da = z, dz
        return a[:, 0], da[:, 0, :]

    def parameter_gradient(self, u):
        """d f / d(theta) for a batch, returned as (n, n_parameters).

        Ordinary backpropagation: the forward activations are stored, the
        adjoint is propagated backwards, and the outer product of adjoint and
        activation gives the weight gradient.
        """
        u = np.atleast_2d(u)
        n = len(u)
        activations = [u]
        a = u
        for k, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w.T + b
            a = np.tanh(z) if k < len(self.W) - 1 else z
            activations.append(a)

        grads_w = [None] * len(self.W)
        grads_b = [None] * len(self.b)
        delta = np.ones((n, 1))                     # d(output)/d(output)
        for k in range(len(self.W) - 1, -1, -1):
            grads_w[k] = delta[:, :, None] * activations[k][:, None, :]
            grads_b[k] = delta
            if k > 0:
                delta = (delta @ self.W[k]) * (1.0 - activations[k]**2)
        return np.concatenate([g.reshape(n, -1) for g in grads_w]
                              + [g.reshape(n, -1) for g in grads_b], axis=1)


def check_autodiff(seed=5):
    """The hand-propagated derivatives against central differences."""
    rng = np.random.default_rng(seed)
    net = Correlator(2, widths=(12, 12), rng=rng, scale=0.8)
    u = rng.normal(0.0, 1.0, (1, 2))
    h = 1e-5

    value, grad, hess = net.value_grad_hess(u)
    fd_grad = np.empty(2)
    fd_hess = np.empty((2, 2))
    h2 = 1e-3                       # a second difference loses two more digits
    for k in range(2):
        step = np.zeros((1, 2))
        step[0, k] = h
        fd_grad[k] = (net(u + step)[0] - net(u - step)[0]) / (2 * h)
    for k in range(2):
        for l in range(2):
            sk = np.zeros((1, 2)); sk[0, k] = h2
            sl = np.zeros((1, 2)); sl[0, l] = h2
            fd_hess[k, l] = (net(u + sk + sl)[0] - net(u + sk - sl)[0]
                             - net(u - sk + sl)[0] + net(u - sk - sl)[0]) \
                / (4 * h2 * h2)

    flat = net.get_parameters()
    analytic = net.parameter_gradient(u)[0]
    fd_param = np.empty_like(flat)
    for i in range(len(flat)):
        shifted = flat.copy(); shifted[i] += h
        net.set_parameters(shifted); plus = net(u)[0]
        shifted[i] -= 2 * h
        net.set_parameters(shifted); minus = net(u)[0]
        fd_param[i] = (plus - minus) / (2 * h)
    net.set_parameters(flat)

    return dict(gradient=float(np.abs(grad - fd_grad).max()),
                hessian=float(np.abs(hess - fd_hess).max()),
                parameters=float(np.abs(analytic - fd_param).max()),
                n_parameters=net.n_parameters)


# ---------------------------------------------------------------------------
#  2. Does a PINN respect the variational principle?
# ---------------------------------------------------------------------------
#  The quartic oscillator H = -1/2 d^2/dx^2 + x^4/2, whose ground-state energy
#  we obtain to machine precision by diagonalising on a grid.  The trial family
#  is a single Gaussian, exp(-a x^2 / 2), which cannot represent the exact
#  state -- so both methods must be wrong, and the question is *how*.
# ---------------------------------------------------------------------------
def quartic_exact(n_states=3, half_width=8.0, n_grid=6001):
    """Ground and low excited states of -1/2 psi'' + x^4 psi / 2 on a grid."""
    from scipy.linalg import eigh_tridiagonal
    x = np.linspace(-half_width, half_width, n_grid)
    h = x[1] - x[0]
    diagonal = 1.0 / h**2 + 0.5 * x**4
    off = -0.5 / h**2 * np.ones(n_grid - 1)
    values, _ = eigh_tridiagonal(diagonal, off, select="i",
                                 select_range=(0, n_states - 1))
    return values


def quartic_variational(a):
    """The Rayleigh quotient for psi = exp(-a x^2/2).

    With <T> = a/4, <x^2> = 1/(2a) and <x^4> = 3/(4a^2),

        E(a) = a/4 + 3/(8 a^2),

    minimised at a = 3^(1/3).  Being a Rayleigh quotient, this is a rigorous
    upper bound on E_0 for every a.
    """
    return 0.25 * a + 3.0 / (8.0 * a**2)


def quartic_local_energy(x, a):
    """E_L(x) = (H psi)/psi for the Gaussian trial function."""
    return 0.5 * a - 0.5 * a**2 * x**2 + 0.5 * x**4


def quartic_residual(a, half_width, n_collocation=2001):
    """Minimise the collocation residual over E at fixed a.

    A PINN in its residual formulation minimises

        L(theta, E) = sum_j | (H - E) psi_theta (xi_j) |^2

    at collocation points xi_j drawn from some distribution over the domain --
    uniformly, in the simplest and commonest case.  Dividing through by psi,
    which is what one does to avoid the amplitude collapsing, this is

        L(theta, E) = sum_j ( E_L(xi_j) - E )^2 ,

    and for fixed theta the optimal E is the *unweighted* mean of the local
    energy over the collocation points.  That is not a Rayleigh quotient: the
    measure is the collocation measure, not |psi|^2.  Nothing bounds it below.
    """
    x = np.linspace(-half_width, half_width, n_collocation)
    local = quartic_local_energy(x, a)
    energy = float(local.mean())
    return float(np.mean((local - energy)**2)), energy


def quartic_residual_optimum(half_width, n_scan=4000):
    """Scan a to minimise the collocation residual, and report the energy."""
    best = None
    for a in np.linspace(0.05, 8.0, n_scan):
        loss, energy = quartic_residual(a, half_width)
        if best is None or loss < best[0]:
            best = (loss, a, energy)
    return best


def amplitude_collapse(scale, a=1.4422, half_width=4.0, n_collocation=2001):
    """The residual of an *unnormalised* network, as the amplitude shrinks.

    Minimising || (H - E) psi ||^2 without normalising psi has a trivial
    global minimum: psi = 0, for any E whatever.  Here the trial function is
    multiplied by a constant and the raw residual recomputed; it falls as the
    square of the constant while the energy is entirely undetermined.
    """
    x = np.linspace(-half_width, half_width, n_collocation)
    psi = scale * np.exp(-a * x**2 / 2.0)
    h_psi = quartic_local_energy(x, a) * psi
    energy = float(np.dot(h_psi, psi) / np.dot(psi, psi))
    return float(np.mean((h_psi - energy * psi)**2)), energy


# ---------------------------------------------------------------------------
#  3. Slater-Jastrow-neural ansatz for the two-electron quantum dot
# ---------------------------------------------------------------------------
#  ln Psi = -alpha omega s / 2 + r12/(1 + beta r12) + W_theta(s, r12)
#
#  with s = r_1^2 + r_2^2.  Every term is a function of the two
#  permutation-invariant features u = (s, r12), which makes the whole trial
#  function manifestly symmetric under exchange -- correct for the spin
#  singlet ground state, where the spatial part is symmetric and the Slater
#  determinant reduces to a constant.  The first two terms are the
#  Pade-Jastrow function of chapter 13; the third is the neural correlator.
# ---------------------------------------------------------------------------
class SlaterJastrowNeural:
    """The chapter-13 trial function multiplied by a neural correlator."""

    def __init__(self, alpha=1.0, beta=0.4, omega=1.0, widths=(16, 16),
                 rng=None, scale=0.05, use_network=True):
        self.alpha, self.beta, self.omega = alpha, beta, omega
        self.use_network = use_network
        self.net = Correlator(2, widths=widths, rng=rng, scale=scale)
        self.M = 4                      # two particles in two dimensions

    # -- the permutation-invariant features and their derivatives ----------
    def features(self, x):
        """u = (s, r12) with s = sum_m x_m^2, plus du/dx and d2u/dx2.

        Shapes: u is (n, 2), du is (n, 4, 2), d2u is (n, 4, 2).  Only the
        diagonal of the second derivative is needed, because the Laplacian
        wants d^2/dx_m^2 and nothing else.
        """
        x = np.atleast_2d(x)
        n = len(x)
        s = np.sum(x * x, axis=1)
        delta = x[:, :2] - x[:, 2:]
        r12 = np.linalg.norm(delta, axis=1)
        u = np.stack([s, r12], axis=1)

        du = np.zeros((n, 4, 2))
        d2u = np.zeros((n, 4, 2))
        du[:, :, 0] = 2.0 * x
        d2u[:, :, 0] = 2.0
        signed = np.concatenate([delta, -delta], axis=1)     # d r12 / d x_m
        du[:, :, 1] = signed / r12[:, None]
        d2u[:, :, 1] = (1.0 - (signed / r12[:, None])**2) / r12[:, None]
        return u, du, d2u, r12

    # -- ln Psi as a function of the features ------------------------------
    def _G(self, u, need_hessian=True):
        """G(u) = ln Psi, together with dG/du and (optionally) d2G/du du."""
        s, r12 = u[:, 0], u[:, 1]
        d = 1.0 / (1.0 + self.beta * r12)

        value = -0.5 * self.alpha * self.omega * s + r12 * d
        grad = np.zeros_like(u)
        grad[:, 0] = -0.5 * self.alpha * self.omega
        grad[:, 1] = d * d
        hess = None
        if need_hessian:
            hess = np.zeros((len(u), 2, 2))
            hess[:, 1, 1] = -2.0 * self.beta * d**3

        if self.use_network:
            if need_hessian:
                w, dw, d2w = self.net.value_grad_hess(u)
                hess = hess + d2w
            else:
                w, dw = self.net.value_grad(u)
            value = value + w
            grad = grad + dw
        return value, grad, hess

    # -- the wave function and the local energy ----------------------------
    def log_psi(self, x):
        u, _, _, _ = self.features(x)
        return self._G(u, need_hessian=False)[0]

    def derivatives(self, x):
        """d ln Psi / d x_m and d^2 ln Psi / d x_m^2, by the chain rule.

            d G / d x_m       = sum_k G_k  du_k/dx_m
            d^2 G / d x_m^2   = sum_kl G_kl (du_k/dx_m)(du_l/dx_m)
                                + sum_k G_k d^2u_k/dx_m^2
        """
        u, du, d2u, r12 = self.features(x)
        _, grad, hess = self._G(u)
        first = np.einsum("nk,nmk->nm", grad, du)
        second = (np.einsum("nkl,nmk,nml->nm", hess, du, du)
                  + np.einsum("nk,nmk->nm", grad, d2u))
        return first, second, r12

    def local_energy(self, x, interaction=True):
        x = np.atleast_2d(x)
        first, second, r12 = self.derivatives(x)
        energy = 0.5 * np.sum(-first**2 - second + self.omega**2 * x**2,
                              axis=1)
        if interaction:
            energy = energy + 1.0 / r12
        return energy

    def quantum_force(self, x):
        return 2.0 * self.derivatives(x)[0]

    def log_psi_and_force(self, x):
        """Both at once.

        The Metropolis-Hastings loop needs the value and the drift at every
        proposed configuration, and both come from the same forward pass
        through the network.  Computing them separately doubles the cost of
        the inner loop, which is where all the time goes.
        """
        u, du, _, _ = self.features(x)
        value, grad, _ = self._G(u, need_hessian=False)
        return value, 2.0 * np.einsum("nk,nmk->nm", grad, du)

    def parameter_gradient(self, x):
        """d ln Psi / d(alpha, beta, theta) for a batch, shape (n, P)."""
        u, _, _, _ = self.features(x)
        s, r12 = u[:, 0], u[:, 1]
        d = 1.0 / (1.0 + self.beta * r12)
        columns = [(-0.5 * self.omega * s)[:, None],
                   (-(r12 * d)**2)[:, None]]
        if self.use_network:
            columns.append(self.net.parameter_gradient(u))
        return np.concatenate(columns, axis=1)

    def get_parameters(self):
        head = np.array([self.alpha, self.beta])
        return np.concatenate([head, self.net.get_parameters()]) \
            if self.use_network else head

    def set_parameters(self, flat):
        self.alpha, self.beta = float(flat[0]), float(flat[1])
        if self.use_network:
            self.net.set_parameters(flat[2:])

    # -- sampling and optimisation -----------------------------------------
    def sample(self, n_cycles=400, n_walkers=400, time_step=0.4, rng=None,
               burn_in=100, interaction=True, keep_samples=False):
        """Importance-sampled Metropolis-Hastings, vectorised over walkers."""
        if rng is None:
            rng = np.random.default_rng(2024)
        diffusion, root_dt = 0.5, math.sqrt(time_step)
        x = rng.normal(0.0, 1.0, (n_walkers, self.M)) * root_dt
        log_old, force_old = self.log_psi_and_force(x)

        energy_sum = energy_sq = 0.0
        n_par = len(self.get_parameters())
        grad_sum = np.zeros(n_par)
        grad_energy_sum = np.zeros(n_par)
        accepted = proposals = 0
        series = np.empty(n_cycles) if keep_samples else None

        for cycle in range(n_cycles + burn_in):
            for p in range(2):
                slot = slice(2 * p, 2 * p + 2)
                trial = x.copy()
                trial[:, slot] = (x[:, slot]
                                  + diffusion * force_old[:, slot] * time_step
                                  + rng.normal(size=(n_walkers, 2)) * root_dt)
                log_new, force_new = self.log_psi_and_force(trial)
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
                e = self.local_energy(x, interaction=interaction)
                o = self.parameter_gradient(x)
                energy_sum += float(e.mean())
                energy_sq += float((e * e).mean())
                grad_sum += o.mean(axis=0)
                grad_energy_sum += (o * e[:, None]).mean(axis=0)
                if keep_samples:
                    series[cycle - burn_in] = float(e.mean())

        mean = energy_sum / n_cycles
        gradient = 2.0 * (grad_energy_sum / n_cycles
                          - grad_sum / n_cycles * mean)
        return dict(energy=mean, gradient=gradient,
                    variance=energy_sq / n_cycles - mean**2,
                    acceptance=accepted / proposals, samples=series)

    def optimise(self, rates=(0.02, 0.01, 0.005), stage=40, n_cycles=300,
                 n_walkers=400, interaction=True, verbose=False, seed=11):
        """Gradient descent on the Rayleigh quotient, chapter-14 style."""
        history = []
        step = 0
        for rate in rates:
            for _ in range(stage):
                step += 1
                result = self.sample(n_cycles=n_cycles, n_walkers=n_walkers,
                                     interaction=interaction,
                                     rng=np.random.default_rng(seed + step))
                history.append((step, result["energy"], result["variance"]))
                if verbose and step % 20 == 0:
                    print(f"      {step:3d}  E = {result['energy']:.6f}  "
                          f"var = {result['variance']:.6f}")
                self.set_parameters(self.get_parameters()
                                    - rate * result["gradient"])
        return history


class TautExact(SlaterJastrowNeural):
    """The closed-form exact ground state of the two-electron dot at omega = 1.

    Separating into centre-of-mass and relative coordinates, R = (r1+r2)/2 and
    r = r1 - r2, the Hamiltonian of chapter 11 becomes

        H = [ -(1/4) grad_R^2 + omega^2 R^2 ]
          + [ -grad_r^2 + (omega^2/4) r^2 + 1/r ] .

    The centre-of-mass part is a two-dimensional oscillator with ground state
    e^{-omega R^2} and energy omega.  For the relative part, inserting the
    ansatz e^{-omega r^2/4}(1 + a r) and demanding that the 1/r term cancel
    forces a = 1, and matching the remaining constant and linear terms gives
    E_rel = 2 omega together with the condition omega = 1.  Adding the two,

        E = 3,   Psi = e^{-omega(|r1+r2|^2 + |r1-r2|^2)/4} (1 + r12)
                     = e^{-omega (r_1^2 + r_2^2)/2} (1 + r12) .

    This is Taut's solution, and it is the reference energy used throughout
    chapters 11 to 17.  Two things make it worth having in code.  It is an
    exact zero-variance test of the whole local-energy machinery -- E_L must
    come out at 3 for *every* configuration.  And, written as

        ln Psi = -s/2 + ln(1 + r12),

    it is a function of the same two invariant features (s, r12) that the
    Slater-Jastrow-neural ansatz uses.  The ansatz therefore *contains* the
    exact state, at alpha = 1 and W(s, r12) = ln(1+r12) - r12/(1 + beta r12).
    A perfect optimiser would find it and the variance would vanish; how far
    the variance actually falls is a measure of the optimisation, not of the
    ansatz.
    """

    def __init__(self, omega=1.0):
        super().__init__(alpha=1.0, beta=0.0, omega=omega, use_network=False)

    def _G(self, u, need_hessian=True):
        s, r12 = u[:, 0], u[:, 1]
        value = -0.5 * self.omega * s + np.log1p(r12)
        grad = np.zeros_like(u)
        grad[:, 0] = -0.5 * self.omega
        grad[:, 1] = 1.0 / (1.0 + r12)
        hess = None
        if need_hessian:
            hess = np.zeros((len(u), 2, 2))
            hess[:, 1, 1] = -1.0 / (1.0 + r12)**2
        return value, grad, hess


def check_taut(n=2000, seed=0):
    """E_L of the exact state, over random configurations.  Must be 3."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.5, (n, 4))
    energies = TautExact().local_energy(x)
    return dict(mean=float(energies.mean()),
                spread=float(energies.max() - energies.min()))


# ---------------------------------------------------------------------------
#  4. Backflow and antisymmetry
# ---------------------------------------------------------------------------
def slater_matrix(r, orbitals):
    """M_ia = phi_a(r_i) for a list of single-particle orbitals."""
    return np.array([[phi(ri) for phi in orbitals] for ri in r])


def equivariant_backflow(r, strength=0.3, width=1.0):
    """A permutation-equivariant displacement, the standard backflow form.

        xi_i = sum_{j != i} eta(r_ij) (r_i - r_j),
        eta(r) = strength * exp(-r^2 / width^2).

    Equivariant means F_{P(i)}(P R) = F_i(R): relabelling the particles
    relabels the displacements in the same way.  The sum over j runs over all
    other particles with a function of the distance only, so this holds by
    inspection.
    """
    n = len(r)
    out = np.zeros_like(r)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = r[i] - r[j]
            out[i] += strength * math.exp(-float(d @ d) / width**2) * d
    return out


def broken_backflow(r, strength=0.3):
    """A displacement that is *not* equivariant: it privileges particle 0.

    Physically this is what happens if one feeds a network the particle index,
    or sorts the particles, or otherwise breaks the exchange symmetry.  The
    determinant then loses its antisymmetry, and the resulting object is not a
    fermionic wave function at all.
    """
    out = np.zeros_like(r)
    for i in range(len(r)):
        out[i] = strength * (i + 1) * (r[i] - r[0])
    return out


def check_backflow_antisymmetry(n_particles=4, seed=3):
    """Exchange two particles and see what the determinant does.

    Returns the ratio D(swapped)/D(original) for the plain determinant and for
    both backflow maps.  It must be exactly -1 whenever the construction is a
    legitimate fermionic ansatz.
    """
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 1.0, (n_particles, 2))
    orbitals = [lambda v: math.exp(-0.5 * float(v @ v)),
                lambda v: v[0] * math.exp(-0.5 * float(v @ v)),
                lambda v: v[1] * math.exp(-0.5 * float(v @ v)),
                lambda v: v[0] * v[1] * math.exp(-0.5 * float(v @ v))][
                    :n_particles]

    swapped = r.copy()
    swapped[[0, 1]] = swapped[[1, 0]]

    out = {}
    for name, displacement in (("no backflow", lambda q: np.zeros_like(q)),
                               ("equivariant backflow", equivariant_backflow),
                               ("non-equivariant map", broken_backflow)):
        d_plain = np.linalg.det(slater_matrix(r + displacement(r), orbitals))
        d_swap = np.linalg.det(slater_matrix(swapped + displacement(swapped),
                                             orbitals))
        out[name] = float(d_swap / d_plain)
    return out


def backflow_moves_the_nodes(n_grid=2001, strength=0.6, width=2.0, seed=0,
                             degenerate=False):
    """How far the nodal surface moves under backflow, and under a Jastrow.

    A Jastrow factor is positive and symmetric, so it multiplies the
    determinant without changing where the determinant vanishes: the nodes are
    exactly where they were.  Backflow changes the arguments of the orbitals
    and can move them.  Measured here as the position of the node along a
    line, as one particle is dragged past the others.

    Two sets of orbitals are available, and the difference between them is
    instructive.

    * `degenerate=True` uses three orbitals sharing one Gaussian factor,
      {1, x, y} e^{-r^2/2}.  The common factor comes out of the determinant,
      the node is exactly the condition that the three effective positions be
      *collinear* -- and pairwise backflow cannot move it.  At a collinear
      configuration every difference vector r_i - r_j already lies along the
      line, so a displacement built from those vectors keeps the points on it.
      This is not a numerical accident: it is a structural limitation of the
      analytic pairwise form, and precisely the sort of thing a learned
      many-body map is introduced to escape.

    * `degenerate=False` uses orbitals with different widths, so no common
      factor separates, the node is no longer pure collinearity, and the same
      backflow moves it.
    """
    rng = np.random.default_rng(seed)
    r0 = rng.normal(0.0, 1.0, (3, 2))
    if degenerate:
        orbitals = [lambda v: math.exp(-0.5 * float(v @ v)),
                    lambda v: v[0] * math.exp(-0.5 * float(v @ v)),
                    lambda v: v[1] * math.exp(-0.5 * float(v @ v))]
    else:
        orbitals = [lambda v: math.exp(-0.5 * float(v @ v)),
                    lambda v: v[0] * math.exp(-0.5 * float(v @ v)),
                    lambda v: math.exp(-0.25 * float(v @ v))]
    line = np.linspace(-4.0, 4.0, n_grid)

    def value(t, displacement, jastrow):
        r = r0.copy()
        r[0, 0] = t
        d = np.linalg.det(slater_matrix(r + displacement(r), orbitals))
        if jastrow:                       # any positive symmetric factor
            pair = sum(1.0 / (1.0 + np.linalg.norm(r[i] - r[j]))
                       for i in range(3) for j in range(i))
            d *= math.exp(pair)
        return d

    def first_node(displacement, jastrow):
        """Bracket the first sign change on the grid, then bisect it."""
        values = np.array([value(t, displacement, jastrow) for t in line])
        crossings = np.nonzero(np.diff(np.sign(values)))[0]
        if not len(crossings):
            return float("nan")
        lo, hi = line[crossings[0]], line[crossings[0] + 1]
        f_lo = value(lo, displacement, jastrow)
        for _ in range(80):               # bisection, well past grid accuracy
            mid = 0.5 * (lo + hi)
            f_mid = value(mid, displacement, jastrow)
            if f_lo * f_mid <= 0.0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return float(0.5 * (lo + hi))

    nothing = lambda q: np.zeros_like(q)
    backflow = lambda q: equivariant_backflow(q, strength=strength,
                                              width=width)
    return dict(plain=first_node(nothing, False),
                jastrow=first_node(nothing, True),
                backflow=first_node(backflow, False))


# ---------------------------------------------------------------------------
def _demo():
    print("=" * 74)
    print("1. Automatic differentiation, done by hand")
    print("=" * 74)
    print("A PINN needs second derivatives of a network with respect to its")
    print("inputs.  `Correlator.value_grad_hess` propagates value, gradient")
    print("and Hessian forward through a tanh network in one pass -- which is")
    print("what an AD framework does for you.  Against central differences:")
    print()
    checks = check_autodiff()
    print(f"      network with {checks['n_parameters']} parameters")
    print(f"      gradient  d f / d u_k        {checks['gradient']:.2e}")
    print(f"      Hessian   d2 f / d u_k d u_l {checks['hessian']:.2e}")
    print(f"      parameters d f / d theta     {checks['parameters']:.2e}")
    print()
    print("   Note which activation is used.  A PINN differentiates its")
    print("   network twice, and ReLU has a second derivative that vanishes")
    print("   almost everywhere: it is unusable for the Laplacian.  Smooth")
    print("   activations -- tanh, GELU, SiLU, softplus -- are not a matter")
    print("   of taste here.")

    print()
    print("=" * 74)
    print("2. Does a PINN respect the variational principle?")
    print("=" * 74)
    print("The quartic oscillator H = -d^2/dx^2/2 + x^4/2, whose spectrum we")
    print("get to machine precision from a grid diagonalisation.  The trial")
    print("family is one Gaussian, exp(-a x^2/2), which cannot represent the")
    print("exact state -- so both methods must be wrong.  The question is how.")
    print()
    spectrum = quartic_exact(3)
    print(f"   exact:  E_0 = {spectrum[0]:.6f}   E_1 = {spectrum[1]:.6f}"
          f"   E_2 = {spectrum[2]:.6f}")
    print()
    a_star = 3.0**(1.0 / 3.0)
    e_var = quartic_variational(a_star)
    print("   (a) Variational: minimise the Rayleigh quotient E(a) =")
    print("       a/4 + 3/(8 a^2), stationary at a = 3^(1/3).")
    print(f"       a* = {a_star:.4f}   E = {e_var:.6f}"
          f"   E - E_0 = {e_var - spectrum[0]:+.6f}")
    print()
    print("   (b) Residual: minimise the Schroedinger residual at uniformly")
    print("       sampled collocation points on [-L, L].  Dividing by psi to")
    print("       stop the amplitude collapsing, this is the mean squared")
    print("       deviation of the local energy from a free parameter E.")
    print()
    print(f"{'L':>8s} {'a*':>9s} {'E*':>13s} {'E* - E_0':>13s} "
          f"{'residual':>12s}")
    for half_width in (2.0, 3.0, 4.0, 6.0):
        loss, a, energy = quartic_residual_optimum(half_width)
        print(f"{half_width:8.1f} {a:9.4f} {energy:13.6f} "
              f"{energy - spectrum[0]:+13.6f} {loss:12.3e}")
    print()
    print("   Every one of those energies lies BELOW the exact ground state,")
    print("   and the deficit grows without bound as the collocation domain")
    print("   is enlarged.  The reason is visible in the formula: the optimal")
    print("   E is the mean of the local energy over the *collocation*")
    print("   measure, and only the mean over |psi|^2 is a Rayleigh quotient.")
    print("   Residual minimisation is not energy minimisation.")
    print()
    print("   And without a normalisation constraint it is worse still.  The")
    print("   raw residual ||(H - E)psi||^2 has a trivial global minimum at")
    print("   psi = 0, for any E whatever:")
    print()
    print(f"{'amplitude':>12s} {'residual':>14s} {'apparent E':>13s}")
    for scale in (1.0, 0.1, 0.01, 0.001):
        loss, energy = amplitude_collapse(scale)
        print(f"{scale:12.3f} {loss:14.3e} {energy:13.6f}")
    print()
    print("   The residual falls as the square of the amplitude while the")
    print("   energy is untouched -- the optimiser is rewarded for making the")
    print("   wave function small, which is not physics.  This is why the")
    print("   energy quotient, in which the normalisation cancels, should be")
    print("   the primary objective, and the residual at most an auxiliary")
    print("   term used for pretraining.")

    print()
    print("=" * 74)
    print("3. A Slater-Jastrow-neural ansatz for the two-electron dot")
    print("=" * 74)
    print("The hybrid recommended at the end of chapter 17: keep the physics")
    print("that is known -- the Gaussian envelope and the Pade-Jastrow cusp")
    print("factor of chapter 13 -- and let a small permutation-invariant")
    print("network supply the rest.  Both act on the invariant features")
    print("(s, r12) with s = r_1^2 + r_2^2, so exchange symmetry is exact.")
    print()
    print("   First, a reference.  At omega = 1 the two-electron dot has a")
    print("   closed-form ground state, Taut's solution")
    print()
    print("      Psi = exp(-(r_1^2 + r_2^2)/2) (1 + r_12),   E = 3 exactly,")
    print()
    print("   whose local energy must therefore be constant.  Over 2000")
    print("   random configurations:")
    taut = check_taut()
    print(f"      mean E_L = {taut['mean']:.12f}   "
          f"spread = {taut['spread']:.2e}")
    print()
    print("   That is a zero-variance check on every derivative in the code.")
    print("   It also tells us something about the ansatz: written as")
    print("   ln Psi = -s/2 + ln(1 + r12), the exact state is a function of")
    print("   the same two invariant features, so the Slater-Jastrow-neural")
    print("   form *contains* it -- at alpha = 1 and")
    print("   W = ln(1+r12) - r12/(1 + beta r12).  The Pade factor is a")
    print("   rational approximation to that logarithm: exact at the cusp,")
    print("   where both behave as r12, and wrong further out.  So the")
    print("   network has something well defined to learn.")
    print()
    results = {}
    for label, use_network in (("Pade-Jastrow only (chapter 13)", False),
                               ("Pade-Jastrow x neural correlator", True)):
        model = SlaterJastrowNeural(alpha=0.95, beta=0.3,
                                    rng=np.random.default_rng(4),
                                    scale=1e-3, widths=(8, 8),
                                    use_network=False)
        # phase 1: the two analytic parameters alone
        model.optimise(rates=(0.02, 0.01), stage=15, n_cycles=200,
                       n_walkers=300)
        if use_network:
            # phase 2: switch the correlator on, starting from that solution
            model.use_network = True
            model.optimise(rates=(0.05, 0.02, 0.01), stage=25, n_cycles=200,
                           n_walkers=300, seed=77)
        final = model.sample(n_cycles=3000, n_walkers=600,
                             rng=np.random.default_rng(99), keep_samples=True)
        series = final["samples"]
        error = blocking(series)[1] if blocking is not None else \
            series.std(ddof=1) / math.sqrt(len(series))
        results[label] = (final, error, len(model.get_parameters()) - 2)
        print(f"   {label}")
        print(f"      parameters: 2 variational + "
              f"{len(model.get_parameters()) - 2} network")
        print(f"      E = {final['energy']:.6f} +/- {error:.6f}   "
              f"variance {final['variance']:.6f}   "
              f"E - exact = {final['energy'] - 3.0:+.6f}")
        print(f"      alpha = {model.alpha:.5f}, beta = {model.beta:.5f}")
    print()
    plain = results["Pade-Jastrow only (chapter 13)"][0]
    hybrid = results["Pade-Jastrow x neural correlator"][0]
    print(f"   The correlator lowers the energy from "
          f"{plain['energy']:.6f} to {hybrid['energy']:.6f}")
    print(f"   and the variance from {plain['variance']:.5f} to "
          f"{hybrid['variance']:.5f}, a reduction of")
    print(f"   {100*(1 - hybrid['variance']/plain['variance']):.0f} per cent."
          f"  The variance is the honest measure here: it")
    print("   would vanish for the exact state, and the ansatz contains the")
    print("   exact state, so what is left is the optimisation, not the form.")
    print()
    print("   Set against the restricted Boltzmann machine of chapter 17,")
    print("   which reached 3.082 with a variance of three and no physics")
    print("   built in at all.  Putting the cusp back is worth more than any")
    print("   amount of extra flexibility -- and once it is back, the network")
    print("   improves on what the two analytic parameters can do alone.")

    print()
    print("=" * 74)
    print("4. Backflow: antisymmetry and the nodal surface")
    print("=" * 74)
    print("Backflow evaluates the orbitals at effective coordinates,")
    print("phi_a(r_i) -> phi_a(r_i + F_i(R)), so every row of the Slater")
    print("matrix becomes a many-body object.  The determinant stays")
    print("antisymmetric if and only if the map is permutation-equivariant.")
    print("Exchanging two particles and taking the ratio of determinants:")
    print()
    for name, ratio in check_backflow_antisymmetry().items():
        verdict = "antisymmetric" if abs(ratio + 1.0) < 1e-10 \
            else "NOT antisymmetric"
        print(f"      {name:<24s} D(swap)/D = {ratio:+.6f}   {verdict}")
    print()
    print("   The third line is the warning.  Feed a network the particle")
    print("   index, or sort the particles before passing them in, and the")
    print("   equivariance is broken; the object that comes out is not a")
    print("   fermionic wave function, whatever its energy may look like.")
    print()
    print("   And what backflow is *for*.  A Jastrow factor is positive and")
    print("   symmetric, so it cannot move the zeros of the determinant; the")
    print("   nodal surface is exactly where it was.  Backflow changes the")
    print("   arguments of the orbitals and can move it.  The first node")
    print("   along a line, as one particle is dragged past the others:")
    print()
    print(f"{'orbitals':>34s} {'plain':>10s} {'x Jastrow':>11s} "
          f"{'+ backflow':>12s}")
    for degenerate, label in ((False, "different widths"),
                              (True, "common Gaussian factor")):
        nodes = backflow_moves_the_nodes(degenerate=degenerate)
        print(f"{label:>34s} {nodes['plain']:10.5f} {nodes['jastrow']:11.5f} "
              f"{nodes['backflow']:12.5f}")
    print()
    print("   The Jastrow column never moves: that is the whole reason")
    print("   backflow exists.  But look at the second row.  When the three")
    print("   orbitals share one Gaussian factor it comes out of the")
    print("   determinant, the node becomes exactly the condition that the")
    print("   effective positions be collinear -- and pairwise backflow")
    print("   cannot move it either, because at a collinear configuration")
    print("   every difference vector r_i - r_j already lies along the line,")
    print("   so a displacement built from those vectors keeps the points on")
    print("   it.  That is not a numerical accident but a structural limit of")
    print("   the analytic pairwise form, and it is exactly the sort of thing")
    print("   a learned many-body map -- the CTNN backflow of section 18.6 --")
    print("   is introduced to escape.")


if __name__ == "__main__":
    _demo()

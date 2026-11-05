"""
Optimising the variational parameters, and resampling the error.

Companion code to chapter 14 of *Quantum Mechanics for Many-particle Systems*.

Chapter 13 located the variational minimum by scanning a grid.  That works for
two parameters and for nothing else: a grid of $m$ points per parameter costs
$m^p$ Monte Carlo runs, and modern trial functions have hundreds or thousands
of parameters.  This program does it properly, in three stages.

  1. The gradient of the energy with respect to the parameters is available
     as a Monte Carlo average,

         dE/dtheta = 2 ( <(d ln Psi/dtheta) E_L> - <d ln Psi/dtheta><E_L> ),

     which costs no more than the energy itself.  It is derived in the text
     and checked here against finite differences.

  2. That gradient drives an optimiser -- plain gradient descent, or a
     quasi-Newton method, or stochastic reconfiguration, which preconditions
     the gradient with the metric of the variational manifold.  Short, noisy
     Monte Carlo runs are enough at this stage: we are locating a minimum, not
     measuring an energy.

  3. Once the parameters are fixed, one long production run measures the
     energy, and the error on it is obtained by resampling -- blocking,
     bootstrap or jackknife -- because the samples are correlated and the
     naive sigma/sqrt(N) is wrong, as chapter 12 showed.

The trial function, local energy and quantum force are imported unchanged from
`vmc.py`.

Author: Morten Hjorth-Jensen
"""

import math

import numpy as np

from vmc import (TAUT_ENERGY, correlation_time, local_energy, log_psi,
                 quantum_force, statistics)


# ---------------------------------------------------------------------------
#  Derivatives of the trial function with respect to the parameters
# ---------------------------------------------------------------------------
def log_psi_gradient(r, alpha, beta, omega=1.0):
    """d ln Psi_T / d(alpha, beta) for one configuration.

        d/dalpha = -omega (r_1^2 + r_2^2) / 2 ,
        d/dbeta  = -r_12^2 / (1 + beta r_12)^2 .

    These are the only new derivatives the optimisation needs, and both are
    cheap: no second derivatives and no extra sampling.
    """
    r_squared = r[0, 0]**2 + r[0, 1]**2 + r[1, 0]**2 + r[1, 1]**2
    r12 = math.hypot(r[0, 0] - r[1, 0], r[0, 1] - r[1, 1])
    d = 1.0 / (1.0 + beta * r12)
    return np.array([-0.5 * omega * r_squared, -(r12 * d)**2])


def log_psi_gradient_fd(r, alpha, beta, omega=1.0, h=1e-5):
    """The same by finite differences, for verification only."""
    out = np.empty(2)
    for k, (da, db) in enumerate(((h, 0.0), (0.0, h))):
        plus = log_psi(r, alpha + da, beta + db, omega)
        minus = log_psi(r, alpha - da, beta - db, omega)
        out[k] = (plus - minus) / (2.0 * h)
    return out


# ---------------------------------------------------------------------------
#  A sampler that also accumulates the gradient
# ---------------------------------------------------------------------------
def sample(alpha, beta, n_cycles=20000, time_step=0.5, omega=1.0, rng=None,
           burn_in=2000, keep_samples=False):
    """Importance-sampled VMC returning the energy and its parameter gradient.

    The gradient is the Monte Carlo estimate

        dE/dtheta = 2 ( <O_theta E_L> - <O_theta><E_L> ),
        O_theta = d ln Psi_T / dtheta ,

    accumulated in the same loop as the energy, so it is essentially free.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    diffusion = 0.5
    root_dt = math.sqrt(time_step)

    position = rng.normal(0.0, 1.0, (2, 2)) * root_dt
    log_old = log_psi(position, alpha, beta, omega)
    force_old = quantum_force(position, alpha, beta, omega)

    energy_sum = 0.0
    energy_squared = 0.0
    gradient_sum = np.zeros(2)
    gradient_energy_sum = np.zeros(2)
    accepted = proposals = 0
    samples = np.empty(n_cycles) if keep_samples else None

    for cycle in range(n_cycles + burn_in):
        for particle in range(2):
            trial = position.copy()
            trial[particle] = (position[particle]
                               + diffusion * force_old[particle] * time_step
                               + rng.normal(0.0, 1.0, 2) * root_dt)
            log_new = log_psi(trial, alpha, beta, omega)
            force_new = quantum_force(trial, alpha, beta, omega)
            green = float(np.sum(
                0.5 * (force_old[particle] + force_new[particle])
                * (0.5 * diffusion * time_step
                   * (force_old[particle] - force_new[particle])
                   - trial[particle] + position[particle])))
            proposals += 1
            if green + 2.0 * (log_new - log_old) > math.log(rng.random()
                                                            + 1e-300):
                position, log_old, force_old = trial, log_new, force_new
                accepted += 1
        if cycle >= burn_in:
            e = local_energy(position, alpha, beta, omega)
            o = log_psi_gradient(position, alpha, beta, omega)
            energy_sum += e
            energy_squared += e * e
            gradient_sum += o
            gradient_energy_sum += o * e
            if keep_samples:
                samples[cycle - burn_in] = e

    mean_energy = energy_sum / n_cycles
    mean_gradient = gradient_sum / n_cycles
    mean_gradient_energy = gradient_energy_sum / n_cycles
    gradient = 2.0 * (mean_gradient_energy - mean_gradient * mean_energy)
    variance = energy_squared / n_cycles - mean_energy**2
    return dict(energy=mean_energy, gradient=gradient, variance=variance,
                acceptance=accepted / proposals, samples=samples)


# ---------------------------------------------------------------------------
#  Optimisers
# ---------------------------------------------------------------------------
def gradient_descent(alpha, beta, learning_rate=0.05, max_iter=40, tol=1e-4,
                     n_cycles=20000, time_step=0.5, seed=2024, verbose=False):
    """Plain steepest descent with a fixed learning rate.

    theta <- theta - eta * dE/dtheta.  Simple, and it works here because the
    surface is well conditioned; the history is returned so that the path can
    be plotted.
    """
    theta = np.array([alpha, beta], dtype=float)
    history = []
    for iteration in range(1, max_iter + 1):
        result = sample(theta[0], theta[1], n_cycles=n_cycles,
                        time_step=time_step,
                        rng=np.random.default_rng(seed + iteration))
        history.append((iteration, theta.copy(), result["energy"],
                        result["gradient"].copy(), result["variance"]))
        if verbose:
            print(f"   {iteration:3d}  alpha {theta[0]:.5f}  "
                  f"beta {theta[1]:.5f}  E {result['energy']:.6f}  "
                  f"|grad| {np.linalg.norm(result['gradient']):.5f}")
        if np.linalg.norm(result["gradient"]) < tol:
            break
        theta = theta - learning_rate * result["gradient"]
    return theta, history


def momentum_descent(alpha, beta, learning_rate=0.05, momentum=0.6,
                     max_iter=40, tol=1e-4, n_cycles=20000, time_step=0.5,
                     seed=2024):
    """Gradient descent with momentum, which damps the noise in the gradient.

    v <- gamma v + eta g,   theta <- theta - v.  With a stochastic gradient the
    averaging over successive steps is worth as much as the acceleration.
    """
    theta = np.array([alpha, beta], dtype=float)
    velocity = np.zeros(2)
    history = []
    for iteration in range(1, max_iter + 1):
        result = sample(theta[0], theta[1], n_cycles=n_cycles,
                        time_step=time_step,
                        rng=np.random.default_rng(seed + iteration))
        history.append((iteration, theta.copy(), result["energy"],
                        result["gradient"].copy(), result["variance"]))
        if np.linalg.norm(result["gradient"]) < tol:
            break
        velocity = momentum * velocity + learning_rate * result["gradient"]
        theta = theta - velocity
    return theta, history


def stochastic_reconfiguration(alpha, beta, learning_rate=0.2, max_iter=40,
                               tol=1e-4, n_cycles=20000, time_step=0.5,
                               regulariser=1e-3, seed=2024):
    """Natural-gradient descent, S^{-1} g instead of g.

    The metric of the variational manifold is the covariance of the
    logarithmic derivatives,

        S_kl = <O_k O_l> - <O_k><O_l> ,

    also called the quantum geometric tensor or, up to a factor, the quantum
    Fisher information.  Preconditioning the gradient with S^{-1} makes the
    step independent of how the parameters happen to be scaled, which is what
    plain gradient descent is sensitive to.  A small diagonal shift keeps S
    invertible.
    """
    theta = np.array([alpha, beta], dtype=float)
    history = []
    for iteration in range(1, max_iter + 1):
        energy, gradient, metric = _sample_with_metric(
            theta[0], theta[1], n_cycles=n_cycles, time_step=time_step,
            rng=np.random.default_rng(seed + iteration))
        history.append((iteration, theta.copy(), energy, gradient.copy(),
                        None))
        if np.linalg.norm(gradient) < tol:
            break
        shifted = metric + regulariser * np.eye(2)
        theta = theta - learning_rate * np.linalg.solve(shifted, gradient)
    return theta, history


# ---------------------------------------------------------------------------
#  Quasi-Newton methods: the secant condition, Broyden and BFGS
# ---------------------------------------------------------------------------
#  Newton's method needs the Jacobian (for F(x) = 0) or the Hessian (for
#  minimising f).  Both are expensive, and for a Monte Carlo energy the Hessian
#  is also noisy.  Quasi-Newton methods never form them: they *infer* curvature
#  from the pairs
#
#       s_k = x_{k+1} - x_k ,      y_k = grad_{k+1} - grad_k ,
#
#  which satisfy y_k = A s_k exactly when the function is quadratic with
#  Hessian A.  Imposing that one relation -- the secant condition -- and asking
#  for the smallest change to the current approximation in the Frobenius norm
#  determines the update completely.  That minimisation is a one-line Lagrange
#  problem, and it is where the identity ||X||_F^2 = Tr(X^T X) of chapter 1
#  earns its keep, through d Tr(X^T X)/dX = 2X.
# ---------------------------------------------------------------------------
def broyden_update(B, s, y):
    """Good Broyden rank-one update of a Jacobian/Hessian approximation.

        B <- B + (y - B s) s^T / (s^T s)

    the minimum-Frobenius-norm change to B satisfying B_new s = y.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    return B + np.outer(y - B @ s, s) / (s @ s)


def broyden_inverse_update(H, s, y):
    """Good Broyden update of the *inverse* approximation, H_new y = s.

        H <- H + (s - H y) y^T / (y^T y)

    Same derivation with the roles of s and y exchanged.  A step is then a
    matrix-vector product, p = -H F(x), with no linear system to solve.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    return H + np.outer(s - H @ y, y) / (y @ y)


def bfgs_inverse_update(H, s, y):
    """BFGS update of the inverse Hessian approximation.

        H <- V H V^T + rho s s^T ,   V = I - rho s y^T ,   rho = 1/(y^T s)

    Symmetric by construction, satisfies H_new y = s, and preserves positive
    definiteness provided the curvature condition y^T s > 0 holds.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    rho = 1.0 / (y @ s)
    V = np.eye(len(s)) - rho * np.outer(s, y)
    return V @ H @ V.T + rho * np.outer(s, s)


def bfgs_hessian_update(B, s, y):
    """BFGS update of the Hessian approximation itself, the direct form.

        B <- B - (B s)(B s)^T / (s^T B s) + y y^T / (y^T s)

    This is the inverse of `bfgs_inverse_update`, as the demonstration below
    verifies numerically; the two are related by Sherman-Morrison-Woodbury.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    Bs = B @ s
    return B - np.outer(Bs, Bs) / (s @ Bs) + np.outer(y, y) / (y @ s)


def dfp_inverse_update(H, s, y):
    """The DFP update, for contrast: the other classical rank-two formula.

        H <- H - (H y)(H y)^T / (y^T H y) + s s^T / (y^T s)

    DFP is the dual of BFGS -- apply the BFGS formula to B and invert, and DFP
    comes out.  It satisfies the same three requirements but is empirically
    less robust, which is why BFGS became the default.
    """
    s = np.asarray(s, dtype=float)
    y = np.asarray(y, dtype=float)
    Hy = H @ y
    return H - np.outer(Hy, Hy) / (y @ Hy) + np.outer(s, s) / (y @ s)


def frobenius_norm(A):
    """||A||_F, computed from the trace identity of chapter 1."""
    A = np.asarray(A, dtype=float)
    return math.sqrt(np.trace(A.T @ A))


def check_quasi_newton_algebra(n=5, n_trials=200, seed=11):
    """Verify, numerically, every claim made about the two updates.

    Returns a dictionary of worst-case residuals over `n_trials` random
    problems in `n` dimensions:

      * the Frobenius identity  ||A||_F^2 = Tr(A^T A)  and the vectorisation
        identity  vec(A).vec(B) = Tr(A^T B) ;
      * Broyden satisfies the secant condition B_new s = y, and no other
        matrix in the affine set {B + dB : dB s = y - B s} is closer to B in
        the Frobenius norm;
      * BFGS satisfies the inverse secant condition H_new y = s, stays
        symmetric, and stays positive definite when y^T s > 0 ;
      * the direct and inverse BFGS formulae are inverses of one another.
    """
    rng = np.random.default_rng(seed)
    worst = dict(frobenius=0.0, vectorisation=0.0, broyden_secant=0.0,
                 broyden_minimal=np.inf, bfgs_secant=0.0, bfgs_symmetry=0.0,
                 bfgs_inverse=0.0, dfp_secant=0.0)
    smallest_eigenvalue = np.inf

    for _ in range(n_trials):
        A = rng.normal(size=(n, n))
        Bm = rng.normal(size=(n, n))
        worst["frobenius"] = max(
            worst["frobenius"],
            abs(frobenius_norm(A)**2 - float(np.sum(A * A))))
        worst["vectorisation"] = max(
            worst["vectorisation"],
            abs(float(A.ravel() @ Bm.ravel()) - float(np.trace(A.T @ Bm))))

        s = rng.normal(size=n)
        y = rng.normal(size=n)

        # -- Broyden -----------------------------------------------------
        B = rng.normal(size=(n, n))
        Bnew = broyden_update(B, s, y)
        worst["broyden_secant"] = max(worst["broyden_secant"],
                                      float(np.abs(Bnew @ s - y).max()))
        # any competitor differs by Z(I - s s^T/s^T s), which annihilates s
        change = frobenius_norm(Bnew - B)
        projector = np.eye(n) - np.outer(s, s) / (s @ s)
        for _ in range(5):
            Z = rng.normal(size=(n, n))
            competitor = Bnew + Z @ projector
            assert np.abs(competitor @ s - y).max() < 1e-10
            worst["broyden_minimal"] = min(
                worst["broyden_minimal"],
                frobenius_norm(competitor - B) - change)

        # -- BFGS --------------------------------------------------------
        M = rng.normal(size=(n, n))
        H = M @ M.T + n * np.eye(n)          # symmetric positive definite
        if y @ s <= 0:                        # enforce the curvature condition
            y = -y
        if abs(y @ s) < 1e-8:
            continue
        Hnew = bfgs_inverse_update(H, s, y)
        worst["bfgs_secant"] = max(worst["bfgs_secant"],
                                   float(np.abs(Hnew @ y - s).max()))
        worst["bfgs_symmetry"] = max(worst["bfgs_symmetry"],
                                     float(np.abs(Hnew - Hnew.T).max()))
        smallest_eigenvalue = min(smallest_eigenvalue,
                                  float(np.linalg.eigvalsh(Hnew).min()))
        Bdirect = bfgs_hessian_update(np.linalg.inv(H), s, y)
        worst["bfgs_inverse"] = max(
            worst["bfgs_inverse"],
            float(np.abs(Bdirect @ Hnew - np.eye(n)).max()))

        Hdfp = dfp_inverse_update(H, s, y)
        worst["dfp_secant"] = max(worst["dfp_secant"],
                                  float(np.abs(Hdfp @ y - s).max()))

    worst["bfgs_smallest_eigenvalue"] = smallest_eigenvalue
    return worst


def wolfe_line_search(f, grad, x, p, f0=None, g0=None, alpha=1.0,
                      c1=1e-4, c2=0.9, max_steps=40):
    """A backtracking line search satisfying the Wolfe conditions.

        f(x + a p) <= f(x) + c1 a grad(x).p          (sufficient decrease)
        grad(x + a p).p >= c2 grad(x).p              (curvature)

    The second condition is what guarantees y^T s > 0 and therefore what keeps
    the BFGS matrix positive definite.  Returns the step length.
    """
    if f0 is None:
        f0 = f(x)
    if g0 is None:
        g0 = grad(x)
    slope = float(g0 @ p)
    if slope >= 0.0:
        return 0.0
    low, high = 0.0, np.inf
    for _ in range(max_steps):
        x_trial = x + alpha * p
        f_trial = f(x_trial)
        if f_trial > f0 + c1 * alpha * slope:
            high = alpha
        elif float(grad(x_trial) @ p) < c2 * slope:
            low = alpha
        else:
            return alpha
        alpha = 0.5 * (low + high) if np.isfinite(high) else 2.0 * low
    return alpha


def bfgs_minimise(f, grad, x0, tol=1e-8, max_iter=200, update="bfgs"):
    """Minimise f with a quasi-Newton method and a Wolfe line search.

    `update` selects the inverse-Hessian formula: "bfgs", "dfp" or "broyden".
    Returns (x, history) with history a list of (iteration, x, f, |grad|).
    """
    formula = dict(bfgs=bfgs_inverse_update, dfp=dfp_inverse_update,
                   broyden=broyden_inverse_update)[update]
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    g = grad(x)
    history = [(0, x.copy(), float(f(x)), float(np.linalg.norm(g)))]
    for iteration in range(1, max_iter + 1):
        if np.linalg.norm(g) < tol:
            break
        p = -H @ g
        step = wolfe_line_search(f, grad, x, p, f0=f(x), g0=g)
        if step == 0.0:
            H = np.eye(n)                     # restart on a bad direction
            continue
        s = step * p
        x_new = x + s
        g_new = grad(x_new)
        y = g_new - g
        if y @ s > 1e-12:                     # curvature condition
            H = formula(H, s, y)
        x, g = x_new, g_new
        history.append((iteration, x.copy(), float(f(x)),
                        float(np.linalg.norm(g))))
    return x, history


def exact_line_search(f, x, p, a0=1.0, tol=1e-12, max_expand=60):
    """Minimise f along x + a p for a > 0, by bracketing and golden section.

    Doubling from `a0` until the function starts to rise brackets the minimum;
    golden section then locates it.  Robust, derivative-free, and used here so
    that the steepest-descent comparison really is the textbook one.
    """
    p = np.asarray(p, dtype=float)
    phi = lambda a: f(x + a * p)
    f0 = phi(0.0)
    a, fa = a0, phi(a0)
    expansions = 0
    while fa < f0 and expansions < max_expand:      # push the bracket out
        a0, f0 = a, fa
        a *= 2.0
        fa = phi(a)
        expansions += 1
    if expansions == 0:                             # or pull it in
        while fa >= f0 and a > 1e-18:
            a *= 0.5
            fa = phi(a)
    return golden_section(phi, 0.0, 2.0 * a, tol=tol * max(1.0, abs(a)))


def steepest_descent_minimise(f, grad, x0, tol=1e-8, max_iter=200000):
    """Steepest descent with an exact line search, for comparison.

    This is the method whose convergence rate is (kappa-1)/(kappa+1) with
    kappa the condition number of the Hessian: even with the step chosen
    optimally at every iteration, the number of steps grows linearly with
    kappa.  That is the zig-zag, and it is what quasi-Newton methods remove.
    """
    x = np.array(x0, dtype=float)
    history = [(0, x.copy(), float(f(x)), float(np.linalg.norm(grad(x))))]
    for iteration in range(1, max_iter + 1):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            break
        step = exact_line_search(f, x, -g)
        if step * np.linalg.norm(g) < 1e-14:   # the line search has stalled
            break
        x = x - step * g
        history.append((iteration, x.copy(), float(f(x)),
                        float(np.linalg.norm(grad(x)))))
    return x, history


def broyden_root(F, x0, tol=1e-10, max_iter=100, use_inverse=True):
    """Solve F(x) = 0 by Broyden's method, starting from B_0 = I.

    This is the algorithm behind Broyden mixing in self-consistent field
    iterations: a fixed-point condition n = n[v(n)] is written as
    F(n) = n[v(n)] - n = 0 and the same rank-one update accelerates it.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    Fx = np.asarray(F(x), dtype=float)
    H = np.eye(n)
    B = np.eye(n)
    history = [(0, x.copy(), float(np.linalg.norm(Fx)))]
    for iteration in range(1, max_iter + 1):
        if np.linalg.norm(Fx) < tol:
            break
        p = -(H @ Fx) if use_inverse else -np.linalg.solve(B, Fx)
        x_new = x + p
        F_new = np.asarray(F(x_new), dtype=float)
        s, y = x_new - x, F_new - Fx
        if use_inverse:
            if abs(y @ y) > 1e-30:
                H = broyden_inverse_update(H, s, y)
        else:
            B = broyden_update(B, s, y)
        x, Fx = x_new, F_new
        history.append((iteration, x.copy(), float(np.linalg.norm(Fx))))
    return x, history


def newton_root(F, J, x0, tol=1e-10, max_iter=100):
    """Newton's method for F(x) = 0, for comparison with Broyden."""
    x = np.array(x0, dtype=float)
    Fx = np.asarray(F(x), dtype=float)
    history = [(0, x.copy(), float(np.linalg.norm(Fx)))]
    for iteration in range(1, max_iter + 1):
        if np.linalg.norm(Fx) < tol:
            break
        x = x - np.linalg.solve(np.asarray(J(x), dtype=float), Fx)
        Fx = np.asarray(F(x), dtype=float)
        history.append((iteration, x.copy(), float(np.linalg.norm(Fx))))
    return x, history


# ---------------------------------------------------------------------------
#  Derivative-free optimisation: line minimisation and Powell's method
# ---------------------------------------------------------------------------
GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))       # 1 - 1/phi = 0.381966...


def golden_section(phi, a, b, tol=1e-6, max_iter=200):
    """Minimise a unimodal phi on [a, b] by golden-section search.

    Each iteration discards a fixed fraction 1/phi of the bracket, so the
    interval shrinks by 0.618 per function evaluation.  Slower than Brent's
    parabolic interpolation but unconditionally robust, and it needs no
    derivatives at all.
    """
    c = b - (b - a) * (1.0 - GOLDEN)
    d = a + (b - a) * (1.0 - GOLDEN)
    fc, fd = phi(c), phi(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * (1.0 - GOLDEN)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * (1.0 - GOLDEN)
            fd = phi(d)
    return 0.5 * (a + b)


def line_minimise(f, x, direction, bracket=2.0, tol=1e-6):
    """Minimise f along x + a * direction, returning (a*, x + a* direction)."""
    phi = lambda a: f(x + a * np.asarray(direction, dtype=float))
    a = golden_section(phi, -bracket, bracket, tol=tol)
    return a, x + a * np.asarray(direction, dtype=float)


def powell_minimise(f, x0, tol=1e-8, max_iter=200, bracket=2.0, line_tol=1e-6):
    """Powell's direction-set method, written out.

    Start from the coordinate directions.  One iteration minimises along each
    in turn, records which gave the largest decrease, forms the net
    displacement d_new = x_n - x_0 -- an estimate of the direction of steepest
    overall progress -- and substitutes it for the direction that helped most.
    For a quadratic the directions become mutually conjugate and the method
    terminates in n cycles; no derivatives are ever used.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    directions = [np.eye(n)[i] for i in range(n)]
    fx = f(x)
    history = [(0, x.copy(), float(fx), 0)]
    evaluations = 0
    for iteration in range(1, max_iter + 1):
        x_start, f_start = x.copy(), fx
        best_decrease, best_index = 0.0, 0
        for index, direction in enumerate(directions):
            f_before = fx
            _, x = line_minimise(f, x, direction, bracket=bracket,
                                 tol=line_tol)
            fx = f(x)
            evaluations += 1
            if f_before - fx > best_decrease:
                best_decrease, best_index = f_before - fx, index
        new_direction = x - x_start
        norm = np.linalg.norm(new_direction)
        if norm > 1e-14:
            new_direction = new_direction / norm
            _, x = line_minimise(f, x, new_direction, bracket=bracket,
                                 tol=line_tol)
            fx = f(x)
            directions[best_index] = new_direction
        history.append((iteration, x.copy(), float(fx), evaluations))
        if abs(f_start - fx) <= tol * (abs(f_start) + abs(fx) + 1e-30):
            break
    return x, history


# ---------------------------------------------------------------------------
#  Quasi-Newton on the stochastic VMC gradient
# ---------------------------------------------------------------------------
def bfgs_descent(alpha, beta, step=1.0, max_iter=40, tol=1e-4, n_cycles=20000,
                 time_step=0.5, seed=2024, trust_radius=0.15, damping=True):
    """BFGS driven by the Monte Carlo gradient, with no line search.

    A line search is impossible here: the objective is a random variable, so
    "sufficient decrease" cannot be tested reliably.  Two safeguards replace
    it.  Powell damping modifies y so that the curvature condition holds even
    when the noisy gradient difference violates it, and a trust radius caps the
    step length.  Even so, this is the method that suffers most from noise --
    it infers curvature from *differences* of gradients, and differencing two
    noisy quantities amplifies the noise.
    """
    theta = np.array([alpha, beta], dtype=float)
    H = np.eye(2)
    result = sample(theta[0], theta[1], n_cycles=n_cycles, time_step=time_step,
                    rng=np.random.default_rng(seed + 1))
    g = result["gradient"].copy()
    history = [(1, theta.copy(), result["energy"], g.copy(), None)]
    n_skipped = 0
    for iteration in range(2, max_iter + 1):
        if np.linalg.norm(g) < tol:
            break
        p = -step * (H @ g)
        norm = np.linalg.norm(p)
        if norm > trust_radius:
            p = p * (trust_radius / norm)
        theta_new = theta + p
        result = sample(theta_new[0], theta_new[1], n_cycles=n_cycles,
                        time_step=time_step,
                        rng=np.random.default_rng(seed + iteration))
        g_new = result["gradient"].copy()
        s, y = theta_new - theta, g_new - g
        if damping:                            # Powell's damped update
            sy = float(s @ y)
            sBs = float(s @ np.linalg.solve(H, s))
            if sy < 0.2 * sBs:
                phi = 0.8 * sBs / (sBs - sy)
                y = phi * y + (1.0 - phi) * np.linalg.solve(H, s)
        if float(s @ y) > 1e-10:
            H = bfgs_inverse_update(H, s, y)
        else:
            n_skipped += 1
        theta, g = theta_new, g_new
        history.append((iteration, theta.copy(), result["energy"], g.copy(),
                        None))
    return theta, history, n_skipped


def powell_vmc(alpha, beta, n_cycles=6000, max_iter=6, seed=2024):
    """Powell's method applied directly to the noisy Monte Carlo energy.

    Included to make the point that derivative-free methods are the wrong tool
    here: the line minimisations chase Monte Carlo noise, and the number of
    energy evaluations is very much larger than the number of gradient steps a
    first-order method needs.
    """
    counter = dict(calls=0)

    def energy(theta):
        counter["calls"] += 1
        return sample(theta[0], theta[1], n_cycles=n_cycles,
                      rng=np.random.default_rng(seed + counter["calls"])
                      )["energy"]

    theta, history = powell_minimise(energy, [alpha, beta], tol=1e-5,
                                     max_iter=max_iter, bracket=0.4,
                                     line_tol=0.02)
    return theta, history, counter["calls"]


def _sample_with_metric(alpha, beta, n_cycles, time_step, rng, omega=1.0,
                        burn_in=2000):
    """As `sample`, but also accumulating S_kl = <O_k O_l> - <O_k><O_l>."""
    diffusion = 0.5
    root_dt = math.sqrt(time_step)
    position = rng.normal(0.0, 1.0, (2, 2)) * root_dt
    log_old = log_psi(position, alpha, beta, omega)
    force_old = quantum_force(position, alpha, beta, omega)

    energy_sum = 0.0
    gradient_sum = np.zeros(2)
    gradient_energy_sum = np.zeros(2)
    outer_sum = np.zeros((2, 2))

    for cycle in range(n_cycles + burn_in):
        for particle in range(2):
            trial = position.copy()
            trial[particle] = (position[particle]
                               + diffusion * force_old[particle] * time_step
                               + rng.normal(0.0, 1.0, 2) * root_dt)
            log_new = log_psi(trial, alpha, beta, omega)
            force_new = quantum_force(trial, alpha, beta, omega)
            green = float(np.sum(
                0.5 * (force_old[particle] + force_new[particle])
                * (0.5 * diffusion * time_step
                   * (force_old[particle] - force_new[particle])
                   - trial[particle] + position[particle])))
            if green + 2.0 * (log_new - log_old) > math.log(rng.random()
                                                            + 1e-300):
                position, log_old, force_old = trial, log_new, force_new
        if cycle >= burn_in:
            e = local_energy(position, alpha, beta, omega)
            o = log_psi_gradient(position, alpha, beta, omega)
            energy_sum += e
            gradient_sum += o
            gradient_energy_sum += o * e
            outer_sum += np.outer(o, o)

    mean_energy = energy_sum / n_cycles
    mean_gradient = gradient_sum / n_cycles
    gradient = 2.0 * (gradient_energy_sum / n_cycles
                      - mean_gradient * mean_energy)
    metric = outer_sum / n_cycles - np.outer(mean_gradient, mean_gradient)
    return mean_energy, gradient, metric


def sr_metric_report(alpha, beta, n_cycles=40000, time_step=0.5, seed=17):
    """The metric S, its spectrum, and what regularising it does to the step.

    S_kl = <O_k O_l> - <O_k><O_l> is the covariance of the logarithmic
    derivatives: the quantum geometric tensor, and up to a factor of four the
    quantum Fisher information.  It is positive semidefinite by construction --
    it is a covariance matrix -- and in practice nearly singular whenever two
    parameters do almost the same thing to the trial function.  The diagonal
    shift S -> S + lambda I interpolates between the natural gradient
    (lambda -> 0) and plain steepest descent (lambda -> infinity), and the
    table returned here shows that crossover.
    """
    energy, gradient, metric = _sample_with_metric(
        alpha, beta, n_cycles=n_cycles, time_step=time_step,
        rng=np.random.default_rng(seed))
    eigenvalues = np.linalg.eigvalsh(metric)
    rows = []
    for lam in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        direction = np.linalg.solve(metric + lam * np.eye(2), gradient)
        cosine = float(direction @ gradient
                       / (np.linalg.norm(direction) * np.linalg.norm(gradient)))
        rows.append((lam, float(np.linalg.norm(direction)),
                     math.degrees(math.acos(max(-1.0, min(1.0, cosine))))))
    return dict(energy=energy, gradient=gradient, metric=metric,
                eigenvalues=eigenvalues,
                condition=float(eigenvalues.max() / max(eigenvalues.min(),
                                                        1e-300)),
                regularisation=rows)


# ---------------------------------------------------------------------------
#  The production run: many walkers, vectorised
# ---------------------------------------------------------------------------
def production_run(alpha, beta, n_walkers=1000, n_steps=2000, time_step=0.5,
                   omega=1.0, rng=None, burn_in=200):
    """A long importance-sampled run with many independent walkers.

    Once the parameters are fixed, the only thing left is to accumulate
    statistics, and that parallelises perfectly: the walkers are independent,
    so the whole ensemble can be advanced with array operations.  Returns the
    walker-averaged local energy at each step -- a time series that is still
    autocorrelated, and therefore still needs resampling -- together with the
    grand total number of local-energy evaluations.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    diffusion = 0.5
    root_dt = math.sqrt(time_step)

    r = rng.normal(0.0, 1.0, (n_walkers, 2, 2)) * root_dt
    log_old = _log_psi_many(r, alpha, beta, omega)
    force_old = _quantum_force_many(r, alpha, beta, omega)

    series = np.empty(n_steps)
    accepted = 0
    proposals = 0

    for step in range(n_steps + burn_in):
        for particle in range(2):
            trial = r.copy()
            trial[:, particle] = (
                r[:, particle]
                + diffusion * force_old[:, particle] * time_step
                + rng.normal(0.0, 1.0, (n_walkers, 2)) * root_dt)
            log_new = _log_psi_many(trial, alpha, beta, omega)
            force_new = _quantum_force_many(trial, alpha, beta, omega)
            green = np.sum(
                0.5 * (force_old[:, particle] + force_new[:, particle])
                * (0.5 * diffusion * time_step
                   * (force_old[:, particle] - force_new[:, particle])
                   - trial[:, particle] + r[:, particle]), axis=1)
            take = (green + 2.0 * (log_new - log_old)
                    > np.log(rng.random(n_walkers) + 1e-300))
            r[take] = trial[take]
            log_old[take] = log_new[take]
            force_old[take] = force_new[take]
            accepted += int(take.sum())
            proposals += n_walkers
        if step >= burn_in:
            series[step - burn_in] = _local_energy_many(r, alpha, beta,
                                                        omega).mean()

    return dict(series=series, n_walkers=n_walkers, n_steps=n_steps,
                total_samples=n_walkers * n_steps,
                acceptance=accepted / proposals)


def _log_psi_many(r, alpha, beta, omega):
    r_squared = np.sum(r * r, axis=(1, 2))
    r12 = np.linalg.norm(r[:, 0] - r[:, 1], axis=1)
    return -0.5 * alpha * omega * r_squared + r12 / (1.0 + beta * r12)


def _local_energy_many(r, alpha, beta, omega):
    r_squared = np.sum(r * r, axis=(1, 2))
    r12 = np.linalg.norm(r[:, 0] - r[:, 1], axis=1)
    d = 1.0 / (1.0 + beta * r12)
    d2 = d * d
    return (0.5 * omega**2 * (1.0 - alpha**2) * r_squared
            + 2.0 * alpha * omega + 1.0 / r12
            + d2 * (alpha * omega * r12 - d2 + 2.0 * beta * d - 1.0 / r12))


def _quantum_force_many(r, alpha, beta, omega):
    separation = r[:, 0] - r[:, 1]
    r12 = np.linalg.norm(separation, axis=1)[:, None]
    d = 1.0 / (1.0 + beta * r12)
    drift = 2.0 * separation * d * d / r12
    force = np.empty_like(r)
    force[:, 0] = -2.0 * alpha * omega * r[:, 0] + drift
    force[:, 1] = -2.0 * alpha * omega * r[:, 1] - drift
    return force


# ---------------------------------------------------------------------------
#  Resampling, part 1: the exact variance of a correlated mean
# ---------------------------------------------------------------------------
#  The sample mean is a linear functional of the data, xbar = (1/N) 1^T x, so
#  its variance is exactly
#
#       Var(xbar) = (1/N^2) 1^T Sigma 1 ,
#
#  with Sigma the covariance matrix of the series.  Nothing is approximated
#  here; the entire difficulty of Monte Carlo error analysis is that Sigma is
#  not diagonal and is not known.  For a stationary series Sigma is Toeplitz,
#  Sigma_ij = C(|i-j|), and collecting the double sum by lag gives a formula
#  involving only the autocovariance function.  The three routines below
#  evaluate the same quantity three ways and are used to check each other.
# ---------------------------------------------------------------------------
def autocovariance(x, max_lag=None):
    """The autocovariance C(t) = Cov(x_k, x_{k+t}) for t = 0, 1, ..., max_lag.

    Uses the standard biased estimator, dividing by N rather than N-t: it has
    a smaller mean squared error and, more importantly, it guarantees that the
    implied covariance matrix is positive semidefinite, which the unbiased
    estimator does not.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n - 1
    d = x - x.mean()
    return np.array([float(np.dot(d[:n - t], d[t:]) / n)
                     for t in range(max_lag + 1)])


def summation_window(x, rule="first-negative", c=5.0, max_lag=None):
    """Where to truncate the sum over lags.

    The exact formula for Var(xbar) runs over all N-1 lags, but the *estimated*
    autocovariance at large lag is pure noise: each C(t) carries an error of
    order sigma^2/sqrt(N), and adding N of them accumulates an error larger
    than the answer.  Every practical estimate of tau therefore truncates, and
    the truncation is a genuine free parameter.  Two standard rules:

      "first-negative"  stop at the first t with rho(t) < 0.  Crude, cheap,
                        and what `correlation_time` in vmc.py uses.
      "sokal"           the self-consistent window of Sokal: the smallest W
                        with W >= c tau_int(W), conventionally c = 5.  It
                        balances the bias of truncating too early against the
                        noise of truncating too late.

    That this decision has to be made at all is the reason blocking exists.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 4, 5000)
    rho = autocovariance(x, max_lag=max_lag)
    rho = rho / rho[0]

    if rule == "first-negative":
        negative = np.nonzero(rho[1:] < 0.0)[0]
        return int(negative[0]) if len(negative) else max_lag
    if rule == "sokal":
        running = 0.5 + np.cumsum(rho[1:])
        for w in range(1, max_lag):
            if w >= c * running[w - 1]:
                return w
        return max_lag
    raise ValueError(f"unknown rule {rule!r}")


def variance_of_mean_exact(x, window=None, rule="first-negative"):
    """Var(xbar) evaluated several ways, as a consistency check.

    1. `quadratic_form`: (1/N^2) 1^T Sigma 1 with Sigma built explicitly from
       the estimated autocovariance at *every* lag -- the definition, O(N^2)
       in memory, and, as the demonstration shows, useless in practice: the
       noise in the large-lag estimates swamps the signal.
    2. `lag_sum`: the same double sum reorganised by lag and truncated at
       `window`,

           Var(xbar) = C(0)/N + (2/N^2) sum_{t=1}^{W} (N-t) C(t)
                     = (sigma^2/N) [ 1 + 2 sum_{t<=W} (1 - t/N) rho(t) ] ,

       which is exact for a stationary series of finite length N if W = N-1.
       The triangular weight (1 - t/N) is not a nuisance: a lag of t can occur
       in only N-t of the N^2 pairs.
    3. `large_n`: the large-N limit, (sigma^2/N) tau with
       tau = 1 + 2 sum_{t<=W} rho(t), dropping the triangular weight.  This is
       Eq. (12-tau) of chapter 12.

    Also returned are tau, the integrated autocorrelation time tau_int = tau/2
    used in much of the lattice literature, and the effective sample size.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if window is None:
        window = summation_window(x, rule=rule)
    window = max(1, min(window, n - 1))
    c = autocovariance(x, max_lag=window)
    lags = np.arange(window + 1)

    lag_sum = (c[0] + 2.0 * float(np.sum((1.0 - lags[1:] / n) * c[1:]))) / n
    tau = 1.0 + 2.0 * float(np.sum(c[1:] / c[0]))
    large_n = c[0] * tau / n

    quadratic = None
    if n <= 4000:                       # Sigma is N x N; keep it modest
        full = autocovariance(x)
        all_lags = np.arange(n)
        sigma = full[np.abs(np.subtract.outer(all_lags, all_lags))]
        quadratic = float(np.ones(n) @ sigma @ np.ones(n)) / n**2

    return dict(quadratic_form=quadratic, lag_sum=float(lag_sum),
                large_n=float(large_n), naive=float(c[0] / n),
                tau=tau, tau_int=0.5 * tau, n_eff=n / tau,
                window=window, sigma_squared=float(c[0]))


def blocked_covariance(x, levels=4, size=8):
    """The leading corner of the covariance matrix after k blocking steps.

    Blocking is a linear coarse-graining, and its effect on Sigma is to push
    the off-diagonal weight onto the diagonal: after enough transformations
    Sigma^(k) is approximately s_k^2 I, and the independent-sample formula
    s_k^2 / N_k becomes valid again.  Returned is, for each level, the first
    `size` rows and columns of the correlation matrix rho^(k), so that the
    off-diagonal elements can be watched shrinking towards zero.
    """
    x = np.asarray(x, dtype=float).copy()
    out = []
    for k in range(levels + 1):
        c = autocovariance(x, max_lag=size - 1)
        rho = c / c[0]
        lags = np.arange(size)
        out.append((k, len(x), float(c[0]),
                    rho[np.abs(np.subtract.outer(lags, lags))]))
        if len(x) % 2:
            x = x[:-1]
        x = 0.5 * (x[0::2] + x[1::2])
    return out


# ---------------------------------------------------------------------------
#  Resampling, part 2: blocking, bootstrap and jackknife
# ---------------------------------------------------------------------------
def blocking(x):
    """Flyvbjerg-Petersen blocking with Jonsson's automatic stopping.

    Repeatedly average neighbouring pairs.  Each transformation halves the
    number of samples and leaves the mean untouched, while the estimated
    variance of the mean grows until the blocks are longer than the
    correlation time, after which it plateaus.  Jonsson (Phys. Rev. E 98,
    043304 (2018)) turned the eyeball test for that plateau into a chi-square
    test on the remaining autocovariance, which is what is used here.

    Returns (mean, standard error, number of transformations used).
    """
    from scipy.stats import chi2

    x = np.asarray(x, dtype=float).copy()
    n = len(x)
    depth = int(math.floor(math.log2(n)))
    gamma = np.zeros(depth)
    sigma = np.zeros(depth)
    mean = x.mean()

    for i in range(depth):
        m = len(x)
        gamma[i] = float(np.sum((x[:m - 1] - x.mean())
                                * (x[1:] - x.mean())) / m)
        sigma[i] = float(x.var())
        x = 0.5 * (x[0::2] + x[1::2]) if m % 2 == 0 else \
            0.5 * (x[0:m - 1:2] + x[1:m:2])

    # Jonsson's test statistic, accumulated from the coarsest block back
    weights = (gamma / sigma)**2 * 2**np.arange(1, depth + 1)[::-1]
    statistic = np.cumsum(weights[::-1])[::-1]
    critical = chi2.ppf(0.95, np.arange(1, depth + 1))

    k = depth - 1
    for candidate in range(depth):
        if statistic[candidate] < critical[candidate]:
            k = candidate
            break
    return mean, math.sqrt(sigma[k] / 2**(depth - k)), k


def blocking_curve(x):
    """The naive error after each blocking transformation, for plotting."""
    x = np.asarray(x, dtype=float).copy()
    out = []
    while len(x) >= 8:
        out.append((len(x), float(x.std(ddof=1) / math.sqrt(len(x)))))
        if len(x) % 2:
            x = x[:-1]
        x = 0.5 * (x[0::2] + x[1::2])
    return out


def bootstrap(x, n_resamples=1000, rng=None, block=None):
    """Bootstrap error on the mean.

    Draw `n_resamples` samples of the same length, with replacement, and take
    the standard deviation of their means.  With correlated data the samples
    must be drawn in *blocks* rather than one at a time, otherwise the
    resampling destroys the correlations and reproduces the naive error; pass
    `block` to set the block length, or leave it None to use twice the
    correlation time.
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if block is None:
        block = max(1, int(2 * correlation_time(x)))
    n_blocks = n // block
    if n_blocks < 2:
        raise ValueError("series too short for this block length")
    blocks = x[:n_blocks * block].reshape(n_blocks, block)
    means = np.empty(n_resamples)
    for k in range(n_resamples):
        pick = rng.integers(0, n_blocks, n_blocks)
        means[k] = blocks[pick].mean()
    return float(x.mean()), float(means.std(ddof=1)), block


def jackknife(x, block=None):
    """Jackknife error on the mean, leaving out one block at a time.

    With n blocks and theta_(i) the estimate with block i removed,

        sigma^2 = (n-1)/n sum_i (theta_(i) - theta_bar)^2 .

    For the mean this is exact and reduces to the blocked standard error; it
    earns its keep for non-linear estimators, where the bootstrap and the
    jackknife are the only practical options.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if block is None:
        block = max(1, int(2 * correlation_time(x)))
    n_blocks = n // block
    blocks = x[:n_blocks * block].reshape(n_blocks, block).mean(axis=1)
    total = blocks.sum()
    leave_one_out = (total - blocks) / (n_blocks - 1)
    mean = leave_one_out.mean()
    variance = (n_blocks - 1) / n_blocks * np.sum((leave_one_out - mean)**2)
    return float(x.mean()), float(math.sqrt(variance)), block


# ---------------------------------------------------------------------------
#  Resampling, part 3: general estimators
# ---------------------------------------------------------------------------
#  Blocking gives the error on a *mean* and nothing else.  The bootstrap and
#  the jackknife take an arbitrary function of the data and return its error,
#  its bias and, for the bootstrap, its whole sampling distribution.  Both are
#  implemented below with a block argument, because on Monte Carlo data the
#  resampling unit must be a block: drawing single samples with replacement
#  produces series that are independent by construction, whatever the original
#  looked like, and reproduces the naive error exactly.
# ---------------------------------------------------------------------------
def _split_into_blocks(x, n_blocks, block):
    """Trim x to a whole number of contiguous blocks and reshape."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if block is None and n_blocks is None:
        n_blocks = 200
    if block is None:
        block = n // n_blocks
    n_blocks = n // block
    if n_blocks < 2:
        raise ValueError("series too short for this block length")
    trimmed = x[:n_blocks * block]
    return trimmed, trimmed.reshape((n_blocks, block) + x.shape[1:]), block


def block_bootstrap(x, estimator=None, n_resamples=1000, rng=None, block=None,
                    n_blocks=None, alpha=0.05):
    """The full bootstrap: distribution, standard error, bias and interval.

    `estimator` is any function of a one-dimensional array; it defaults to the
    mean.  `x` may also be a two-dimensional array of shape (N, k), in which
    case whole rows are resampled together and the estimator receives the
    (N, k) array -- this is how a ratio of averages, or any estimator of
    several jointly measured observables, is bootstrapped correctly.

    The resampling unit is a contiguous *block*, not a single sample, because
    resampling single points destroys the autocorrelation and reproduces the
    naive error exactly.  Give either `block` (the block length) or `n_blocks`
    (how many of them); the default is 200 blocks, which for a series of any
    reasonable length makes each block far longer than the correlation time.

    Returned:
      estimate    the estimator on the original data, theta_hat
      error       the bootstrap standard error, sd of the theta*_b
      bias        the bootstrap bias estimate, mean(theta*) - theta_hat
      corrected   theta_hat - bias, the bias-corrected estimate
      interval    the percentile interval at confidence 1 - alpha
      samples     the B bootstrap replicates themselves
    """
    if rng is None:
        rng = np.random.default_rng(2024)
    if estimator is None:
        estimator = np.mean
    trimmed, blocks, block = _split_into_blocks(x, n_blocks, block)
    count = len(blocks)

    estimate = float(estimator(trimmed))
    replicates = np.empty(n_resamples)
    for b in range(n_resamples):
        pick = rng.integers(0, count, count)
        replicates[b] = estimator(blocks[pick].reshape(trimmed.shape))

    bias = float(replicates.mean()) - estimate
    low, high = np.quantile(replicates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return dict(estimate=estimate, error=float(replicates.std(ddof=1)),
                bias=bias, corrected=estimate - bias,
                interval=(float(low), float(high)), block=block,
                n_blocks=count, samples=replicates)


def block_jackknife(x, estimator=None, block=None, n_blocks=None):
    """Jackknife error and bias for an arbitrary estimator.

    With n blocks, theta_(i) the estimate with block i deleted and theta_hat
    the estimate on all the data,

        sigma^2 = (n-1)/n sum_i (theta_(i) - mean(theta_(.)))^2 ,
        bias    = (n-1) ( mean(theta_(.)) - theta_hat ) .

    The factor (n-1) in the bias, which looks alarming, is exactly right: a
    bias of order 1/n changes by order 1/n^2 when one block is deleted, and
    multiplying by n-1 recovers it.  Deleting one block at a time rather than
    one sample keeps the correlations of the retained data intact.
    """
    if estimator is None:
        estimator = np.mean
    trimmed, blocks, block = _split_into_blocks(x, n_blocks, block)
    count = len(blocks)
    estimate = float(estimator(trimmed))

    shape = (-1,) + trimmed.shape[1:]
    partial = np.empty(count)
    for i in range(count):
        kept = np.concatenate((blocks[:i], blocks[i + 1:]))
        partial[i] = estimator(kept.reshape(shape))

    mean_partial = float(partial.mean())
    variance = (count - 1) / count * float(np.sum((partial
                                                   - mean_partial)**2))
    bias = (count - 1) * (mean_partial - estimate)
    return dict(estimate=estimate, error=math.sqrt(variance), bias=bias,
                corrected=estimate - bias, block=block, n_blocks=count,
                partial=partial)


def _demo():
    print("=" * 74)
    print("1. The gradient of the energy")
    print("=" * 74)
    print("The derivative of the energy with respect to a variational")
    print("parameter is itself a Monte Carlo average,")
    print("   dE/dtheta = 2 ( <O_theta E_L> - <O_theta><E_L> ),")
    print("with O_theta = d ln Psi_T / dtheta.  It costs nothing extra: the")
    print("same walk that measures the energy measures the gradient.")
    print()
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(40):
        r = rng.normal(0.0, 1.0, (2, 2))
        for alpha, beta in ((0.98, 0.40), (0.80, 0.25), (1.10, 0.60)):
            worst = max(worst, float(np.abs(
                log_psi_gradient(r, alpha, beta)
                - log_psi_gradient_fd(r, alpha, beta)).max()))
    print(f"   d ln Psi/d(alpha, beta), analytic against finite differences:")
    print(f"      largest discrepancy {worst:.1e}")
    print()
    print("   and the resulting energy gradient, against a finite difference")
    print("   of the energy itself:")
    print()
    print(f"{'alpha':>7s} {'beta':>7s} {'dE/dalpha':>21s} {'dE/dbeta':>21s}")
    for alpha, beta in ((0.90, 0.30), (1.00, 0.40), (1.05, 0.50)):
        analytic = sample(alpha, beta, n_cycles=40000,
                          rng=np.random.default_rng(99))["gradient"]
        h = 0.01
        numeric = np.empty(2)
        for k, (da, db) in enumerate(((h, 0.0), (0.0, h))):
            plus = sample(alpha + da, beta + db, n_cycles=40000,
                          rng=np.random.default_rng(99))["energy"]
            minus = sample(alpha - da, beta - db, n_cycles=40000,
                           rng=np.random.default_rng(99))["energy"]
            numeric[k] = (plus - minus) / (2.0 * h)
        print(f"{alpha:7.2f} {beta:7.2f} "
              f"{analytic[0]:10.5f} vs {numeric[0]:8.5f} "
              f"{analytic[1]:10.5f} vs {numeric[1]:8.5f}")
    print()
    print("   The two agree within the Monte Carlo noise.  Note how much")
    print("   cheaper the analytic route is: one run instead of two per")
    print("   parameter, and no cancellation of nearly equal numbers.")

    print()
    print("=" * 74)
    print("2. Gradient descent")
    print("=" * 74)
    print("Starting deliberately far from the minimum, at alpha = 0.85 and")
    print("beta = 0.20, with 6000 cycles per step -- short, noisy runs are")
    print("enough while we are still locating the minimum.")
    print()
    theta, history = gradient_descent(0.85, 0.20, learning_rate=0.05,
                                      max_iter=25, n_cycles=6000)
    print(f"{'iter':>5s} {'alpha':>9s} {'beta':>9s} {'energy':>12s} "
          f"{'|gradient|':>12s}")
    for iteration, point, energy, gradient, _ in history:
        if iteration <= 6 or iteration % 5 == 0 or iteration == len(history):
            print(f"{iteration:5d} {point[0]:9.5f} {point[1]:9.5f} "
                  f"{energy:12.6f} {np.linalg.norm(gradient):12.6f}")
    print()
    print(f"   converged to alpha = {theta[0]:.4f}, beta = {theta[1]:.4f} "
          f"in {len(history)} steps")

    print()
    print("=" * 74)
    print("3. Quasi-Newton algebra: the secant condition, Broyden and BFGS")
    print("=" * 74)
    print("Every claim the text makes about the two updates, checked on random")
    print("problems in five dimensions.  The Broyden update is the smallest")
    print("change in the Frobenius norm consistent with the secant condition,")
    print("so no competitor in the same affine set can be closer -- the")
    print("'minimality margin' below is the smallest excess ||dB'||_F -")
    print("||dB||_F over random competitors, and must not be negative.")
    print()
    checks = check_quasi_newton_algebra(n=5, n_trials=200)
    labels = (("frobenius", "||A||_F^2 - Tr(A^T A)"),
              ("vectorisation", "vec(A).vec(B) - Tr(A^T B)"),
              ("broyden_secant", "Broyden:  |B_new s - y|"),
              ("broyden_minimal", "Broyden:  minimality margin"),
              ("bfgs_secant", "BFGS:     |H_new y - s|"),
              ("bfgs_symmetry", "BFGS:     |H_new - H_new^T|"),
              ("bfgs_inverse", "BFGS:     |B_new H_new - I|"),
              ("dfp_secant", "DFP:      |H_new y - s|"))
    for key, label in labels:
        print(f"      {label:<34s} {checks[key]:.2e}")
    print(f"      BFGS: smallest eigenvalue of H_new "
          f"{checks['bfgs_smallest_eigenvalue']:.3e}  (positive definite)")

    print()
    print("   Broyden against Newton on a nonlinear system, F(x) = 0 with")
    print("   F = (x^2 + y^2 - 4, e^x + y - 1), starting from (1, 2).  Broyden")
    print("   never forms the Jacobian and still converges superlinearly:")
    print()

    def F(v):
        return np.array([v[0]**2 + v[1]**2 - 4.0,
                         math.exp(v[0]) + v[1] - 1.0])

    def J(v):
        return np.array([[2.0 * v[0], 2.0 * v[1]],
                         [math.exp(v[0]), 1.0]])

    _, newton_hist = newton_root(F, J, [1.0, 2.0])
    _, broyden_hist = broyden_root(F, [1.0, 2.0])
    print(f"{'iter':>6s} {'|F| Newton':>14s} {'|F| Broyden':>14s}")
    for k in range(max(len(newton_hist), len(broyden_hist))):
        a = f"{newton_hist[k][2]:14.3e}" if k < len(newton_hist) else " " * 14
        b = f"{broyden_hist[k][2]:14.3e}" if k < len(broyden_hist) \
            else " " * 14
        print(f"{k:6d} {a} {b}")
    print(f"   Newton needs {len(newton_hist)-1} iterations and "
          f"{len(newton_hist)-1} Jacobians;")
    print(f"   Broyden needs {len(broyden_hist)-1} and none at all.")

    print()
    print("=" * 74)
    print("4. Curvature matters: an ill-conditioned quadratic")
    print("=" * 74)
    print("f(x) = x^T A x / 2 - b^T x in six dimensions, A diagonal with")
    print("condition number kappa.  Iterations to bring |grad f| below 1e-6,")
    print("every method using the same exact or Wolfe line search.  Steepest")
    print("descent needs a number of iterations growing linearly with kappa --")
    print("the rate is (kappa-1)/(kappa+1) -- while the two rank-two updates")
    print("learn the curvature and are essentially insensitive to it.  The")
    print("rank-one Broyden update, which is neither symmetric nor positive")
    print("definite, sits in between: it is a root finder pressed into service")
    print("as a minimiser.  Powell needs no derivatives at all and terminates")
    print("in at most n cycles on a quadratic, as conjugacy demands.")
    print()
    print(f"{'kappa':>10s} {'steepest':>10s} {'BFGS':>8s} {'DFP':>8s} "
          f"{'Broyden':>9s} {'Powell':>8s}")
    for kappa in (1.0, 10.0, 100.0, 1000.0):
        n = 6
        diag = np.logspace(0.0, math.log10(kappa), n)
        b = np.ones(n)
        f = lambda x, d=diag, b=b: 0.5 * float(x @ (d * x)) - float(b @ x)
        g = lambda x, d=diag, b=b: d * x - b
        x0 = np.zeros(n)
        counts = []
        for update in ("bfgs", "dfp", "broyden"):
            _, hist = bfgs_minimise(f, g, x0, tol=1e-6, update=update,
                                    max_iter=2000)
            counts.append(len(hist) - 1)
        _, sd_hist = steepest_descent_minimise(f, g, x0, tol=1e-6)
        _, pw_hist = powell_minimise(f, x0, tol=1e-12)
        print(f"{kappa:10.0f} {len(sd_hist)-1:10d} {counts[0]:8d} "
              f"{counts[1]:8d} {counts[2]:9d} {len(pw_hist)-1:8d}")
    print()
    print("   And on Rosenbrock's banana, f = (1-x)^2 + 100(y-x^2)^2, which is")
    print("   not quadratic and has a curved valley:")
    rosen = lambda v: (1.0 - v[0])**2 + 100.0 * (v[1] - v[0]**2)**2
    rosen_grad = lambda v: np.array([
        -2.0 * (1.0 - v[0]) - 400.0 * v[0] * (v[1] - v[0]**2),
        200.0 * (v[1] - v[0]**2)])
    _, hist_bfgs = bfgs_minimise(rosen, rosen_grad, [-1.2, 1.0], tol=1e-8)
    _, hist_sd = steepest_descent_minimise(rosen, rosen_grad, [-1.2, 1.0],
                                           tol=1e-8)
    _, hist_pw = powell_minimise(rosen, [-1.2, 1.0], tol=1e-12, max_iter=500)
    print(f"      BFGS      {len(hist_bfgs)-1:6d} iterations, "
          f"f = {hist_bfgs[-1][2]:.3e}")
    print(f"      steepest  {len(hist_sd)-1:6d} iterations, "
          f"f = {hist_sd[-1][2]:.3e}")
    print(f"      Powell    {len(hist_pw)-1:6d} cycles,     "
          f"f = {hist_pw[-1][2]:.3e}")

    print()
    print("=" * 74)
    print("5. The metric of the variational manifold")
    print("=" * 74)
    print("S_kl = <O_k O_l> - <O_k><O_l>, sampled at alpha = 1.0, beta = 0.4.")
    print()
    report_sr = sr_metric_report(1.00, 0.40, n_cycles=40000)
    S = report_sr["metric"]
    print(f"      S = [[{S[0,0]:9.4f}, {S[0,1]:9.4f}],")
    print(f"           [{S[1,0]:9.4f}, {S[1,1]:9.4f}]]")
    print(f"      eigenvalues {report_sr['eigenvalues'][0]:.4e} and "
          f"{report_sr['eigenvalues'][1]:.4e}")
    print(f"      condition number {report_sr['condition']:.1f}")
    print()
    print("   The two parameters are far from equivalent.  The off-diagonal")
    print("   element is large -- alpha and beta move the state in overlapping")
    print("   directions -- and the two eigenvalues differ by an order of")
    print("   magnitude, so the manifold is stretched.  That anisotropy is")
    print("   exactly what plain gradient descent is blind to, and with more")
    print("   parameters it gets far worse.  Regularising S interpolates back")
    print("   towards steepest descent; 'angle' below is the angle between the")
    print("   SR direction and the bare gradient:")
    print()
    print(f"{'lambda':>10s} {'|S^-1 g|':>12s} {'angle (deg)':>13s}")
    for lam, length, angle in report_sr["regularisation"]:
        print(f"{lam:10.0e} {length:12.5f} {angle:13.2f}")

    print()
    print("=" * 74)
    print("6. Five optimisers compared on the quantum dot")
    print("=" * 74)
    print("All start from the same point and use the same number of samples")
    print("per step.  Momentum averages successive gradients, which is worth")
    print("as much as the acceleration when the gradient is noisy.  Stochastic")
    print("reconfiguration preconditions with the metric of the variational")
    print("manifold and takes the largest useful steps.  BFGS has to infer")
    print("curvature from differences of noisy gradients, which is precisely")
    print("what noise destroys; Powell does not use the gradient at all and")
    print("pays for it in energy evaluations.")
    print()
    print(f"{'method':>28s} {'steps':>7s} {'alpha':>9s} {'beta':>9s} "
          f"{'final energy':>14s}")
    for name, routine, kwargs in (
            ("gradient descent", gradient_descent,
             dict(learning_rate=0.05)),
            ("with momentum", momentum_descent,
             dict(learning_rate=0.03, momentum=0.6)),
            ("stochastic reconfiguration", stochastic_reconfiguration,
             dict(learning_rate=0.2))):
        theta, history = routine(0.85, 0.20, max_iter=25, n_cycles=6000,
                                 **kwargs)
        print(f"{name:>28s} {len(history):7d} {theta[0]:9.5f} "
              f"{theta[1]:9.5f} {history[-1][2]:14.6f}")
    theta, bfgs_hist, skipped = bfgs_descent(0.85, 0.20, step=1.0, max_iter=25,
                                             n_cycles=6000)
    print(f"{'BFGS, damped + trust radius':>28s} {len(bfgs_hist):7d} "
          f"{theta[0]:9.5f} {theta[1]:9.5f} {bfgs_hist[-1][2]:14.6f}")
    theta_raw, raw_hist, skipped_raw = bfgs_descent(
        0.85, 0.20, step=1.0, max_iter=25, n_cycles=6000, damping=False,
        trust_radius=np.inf)
    print(f"{'BFGS, no safeguards':>28s} {len(raw_hist):7d} "
          f"{theta_raw[0]:9.5f} {theta_raw[1]:9.5f} "
          f"{raw_hist[-1][2]:14.6f}")
    theta_p, powell_hist, calls = powell_vmc(0.85, 0.20, n_cycles=6000,
                                             max_iter=4)
    print(f"{'Powell, derivative free':>28s} {len(powell_hist)-1:7d} "
          f"{theta_p[0]:9.5f} {theta_p[1]:9.5f} {powell_hist[-1][2]:14.6f}")
    print()
    print("   With two well-scaled parameters every method works, and the")
    print("   ordering here is not the point; what the last three rows show is")
    print("   the cost of each.  BFGS needed the damped update and a trust")
    print(f"   radius (it rejected the curvature condition {skipped} times "
          f"with them and {skipped_raw} without),")
    print(f"   and Powell needed {calls} energy evaluations against the 25")
    print("   gradient evaluations of the first-order methods -- each of which")
    print("   also delivered the energy.  With thousands of parameters, a")
    print("   noisy gradient and a badly conditioned manifold, only stochastic")
    print("   reconfiguration and its variants survive.")

    print()
    print("=" * 74)
    print("7. The production run")
    print("=" * 74)
    print("With the parameters fixed, the only remaining task is statistics,")
    print("and that parallelises perfectly: the walkers are independent, so")
    print("the ensemble is advanced with array operations.")
    print()
    run = production_run(1.00, 0.40, n_walkers=1000, n_steps=2000,
                         time_step=0.5, rng=np.random.default_rng(2024))
    series = run["series"]
    print(f"   {run['n_walkers']} walkers x {run['n_steps']} steps "
          f"= {run['total_samples']:,} local-energy evaluations")
    print(f"   acceptance rate {run['acceptance']:.1%}")
    print(f"   mean energy {series.mean():.8f}")
    print(f"   correlation time of the walker-averaged series "
          f"{correlation_time(series):.2f}")

    print()
    print("=" * 74)
    print("8. The exact variance of a correlated mean")
    print("=" * 74)
    print("The sample mean is linear in the data, xbar = 1^T x / N, so its")
    print("variance is exactly (1/N^2) 1^T Sigma 1 with Sigma the covariance")
    print("matrix.  Nothing is approximated; the whole difficulty is that")
    print("Sigma is not diagonal.  For a stationary series Sigma is Toeplitz,")
    print("Sigma_ij = C(|i-j|), and collecting the double sum by lag gives")
    print()
    print("   Var(xbar) = C(0)/N + (2/N^2) sum_t (N-t) C(t)")
    print("             = (sigma^2/N) [1 + 2 sum_t (1 - t/N) rho(t)] ,")
    print()
    print("exact for finite N.  Dropping the triangular weight (1 - t/N) gives")
    print("the familiar (sigma^2/N) tau of chapter 12.  All three evaluated on")
    print("the production series:")
    print()
    exact = variance_of_mean_exact(series, rule="first-negative")
    sokal = variance_of_mean_exact(series, rule="sokal")
    print(f"   naive sigma^2/N                        "
          f"{math.sqrt(exact['naive']):.8f}")
    print(f"   lag sum, weight (1-t/N), W = {exact['window']:<4d}     "
          f"{math.sqrt(exact['lag_sum']):.8f}")
    print(f"   large-N limit (sigma^2/N) tau, same W  "
          f"{math.sqrt(exact['large_n']):.8f}")
    print(f"   the same with Sokal's window W = {sokal['window']:<4d} "
          f"{math.sqrt(sokal['lag_sum']):.8f}")
    if exact["quadratic_form"] is not None:
        print(f"   (1/N^2) 1^T Sigma 1, every lag         "
              f"{math.sqrt(exact['quadratic_form']):.8f}   <-- nonsense")
    print()
    print(f"   tau  = 1 + 2 sum rho(t)          {exact['tau']:.4f}")
    print(f"   tau_int = tau / 2                {exact['tau_int']:.4f}")
    print(f"   N_eff = N / tau                  {exact['n_eff']:.1f}"
          f"  out of {len(series)}")
    print()
    print("   The two tau conventions are a standard trap.  This book uses")
    print("   tau = 1 + 2 sum rho, so Var = tau sigma^2 / N; much of the")
    print("   lattice literature uses tau_int = 1/2 + sum rho = tau/2, so")
    print("   Var = 2 tau_int sigma^2 / N.  They are the same statement.")
    print()
    print("   Now the catch, and it is the reason blocking exists.  The exact")
    print("   formula runs over all N-1 lags, but each estimated C(t) carries")
    print("   an error of order sigma^2/sqrt(N), and adding N of them")
    print("   accumulates more noise than signal.  Summing them all does not")
    print("   give a slightly worse answer, it gives a meaningless one -- the")
    print("   full quadratic form above comes out *below* the naive error.")
    print("   So one truncates, and the answer depends on where:")
    print()
    print(f"{'window W':>10s} {'tau':>9s} {'error':>14s}")
    for W in (1, 2, 5, 10, 20, 50, 100, 500, len(series) - 1):
        e = variance_of_mean_exact(series, window=W)
        print(f"{W:10d} {e['tau']:9.3f} "
              f"{math.sqrt(max(e['lag_sum'], 0.0)):14.8f}")
    _, blocked_here, k_here = blocking(series)
    print(f"{'blocking':>10s} {'--':>9s} {blocked_here:14.8f}"
          f"   ({k_here} transformations, no free parameter)")
    print()
    print("   Between W = 1 and W = 500 the error wanders by ten per cent,")
    print("   and at W = N-1 it collapses.  Blocking asks no such question.")
    print()
    print("   Why blocking works: it is a linear coarse-graining, and its")
    print("   effect on Sigma is to move the off-diagonal weight onto the")
    print("   diagonal.  The leading correlations rho(1), rho(2), rho(3) after")
    print("   each transformation, for the badly tuned run where there is")
    print("   something to see:")
    print()
    bad = production_run(1.00, 0.40, n_walkers=400, n_steps=4000,
                         time_step=0.02, rng=np.random.default_rng(2024))
    print(f"{'level k':>8s} {'N_k':>8s} {'s_k^2':>12s} {'rho(1)':>9s} "
          f"{'rho(2)':>9s} {'rho(3)':>9s} {'s_k^2/N_k':>12s}")
    for k, nk, var_k, rho in blocked_covariance(bad["series"], levels=6):
        print(f"{k:8d} {nk:8d} {var_k:12.3e} {rho[0, 1]:9.4f} "
              f"{rho[0, 2]:9.4f} {rho[0, 3]:9.4f} "
              f"{math.sqrt(var_k / nk):12.8f}")
    print()
    print("   Sigma^(k) -> s_k^2 I, and s_k^2 / N_k -- the independent-sample")
    print("   formula, wrong at k = 0 -- climbs to the true error and stops.")
    print("   That plateau is the whole of the blocking method.")

    print()
    print("=" * 74)
    print("9. Resampling the error")
    print("=" * 74)
    print("The series is autocorrelated, so sigma/sqrt(N) is not the error.")
    print("Three standard ways of getting the right one, on the run above:")
    print()

    def report(x, label):
        naive = x.std(ddof=1) / math.sqrt(len(x))
        tau = correlation_time(x)
        _, error_b, k = blocking(x)
        _, error_boot, block_boot = bootstrap(
            x, n_resamples=1000, rng=np.random.default_rng(7))
        _, error_j, block_j = jackknife(x)
        print(f"   {label}")
        print(f"      correlation time tau         {tau:8.2f}")
        print(f"      naive sigma/sqrt(N)          {naive:.8f}")
        print(f"      sqrt(tau) x naive            "
              f"{naive*math.sqrt(tau):.8f}")
        print(f"      blocking                     {error_b:.8f}"
              f"   ({k} transformations)")
        print(f"      bootstrap, blocks of {block_boot:<5d}  "
              f"{error_boot:.8f}")
        print(f"      jackknife, blocks of {block_j:<5d}  {error_j:.8f}")
        print(f"      resampled / naive            "
              f"{error_b/naive:8.2f}")
        return error_b

    error_b = report(series, "well tuned, time step 0.5:")
    mean_b = series.mean()
    print()
    print("   Here the four honest estimates agree and the naive one is only")
    print("   modestly wrong, because the sampling is well tuned and tau is")
    print("   close to one.  That is not something to rely on.  The same")
    print("   calculation with a badly chosen time step:")
    print()
    report(bad["series"], "badly tuned, time step 0.02:")
    print()
    print("   Now the naive error understates the true one by a factor of")
    print("   four, and it does so silently.  Note that the three resampling")
    print("   estimates do not agree exactly either: blocking with Jonsson's")
    print("   stopping rule is deliberately conservative.  They agree on the")
    print("   order of magnitude, which is what matters, and all three are")
    print("   right where the naive estimate is wrong.")
    print()
    print("   the blocking curve for the well-tuned run:")
    print(f"{'samples':>10s} {'block':>8s} {'naive error':>14s}")
    for index, (n, error) in enumerate(blocking_curve(series)):
        if index % 2 == 0:
            print(f"{n:10d} {len(series)//n:8d} {error:14.8f}")

    print()
    print("=" * 74)
    print("10. Estimators that are not means")
    print("=" * 74)
    print("Blocking gives the error on a mean and on nothing else.  The")
    print("bootstrap and the jackknife take an arbitrary function of the data.")
    print("The natural test case in variational Monte Carlo is the variance of")
    print("the local energy: it is the zero-variance diagnostic -- it vanishes")
    print("when Psi_T is an exact eigenstate -- and no formula for its error")
    print("exists.  A single long chain, so that sigma^2 really is Var(E_L):")
    print()
    chain = sample(1.00, 0.40, n_cycles=200000, time_step=0.5,
                   rng=np.random.default_rng(31), keep_samples=True)
    e_local = chain["samples"]
    tau_chain = correlation_time(e_local)
    n_blocks = 200
    block = len(e_local) // n_blocks
    print(f"   {len(e_local):,} local energies, tau = {tau_chain:.2f}, cut")
    print(f"   into {n_blocks} blocks of {block}.  Each block is "
          f"{block/tau_chain:.0f} correlation times")
    print(f"   long, which is what makes the blocks exchangeable and the")
    print(f"   resampling legitimate.")
    print()
    print(f"{'estimator':>20s} {'value':>11s} {'bootstrap':>11s} "
          f"{'jackknife':>11s} {'blocking':>11s} {'jack bias':>11s}")
    for name, estimator in (("mean, E", np.mean),
                            ("variance sigma^2", np.var),
                            ("sigma", lambda a: float(np.std(a))),
                            ("ratio sigma / |E|",
                             lambda a: float(np.std(a) / abs(np.mean(a))))):
        boot = block_bootstrap(e_local, estimator=estimator, n_resamples=1000,
                               n_blocks=n_blocks,
                               rng=np.random.default_rng(11))
        jack = block_jackknife(e_local, estimator=estimator,
                               n_blocks=n_blocks)
        if estimator is np.mean:
            _, blocked, _ = blocking(e_local)
            blocked_text = f"{blocked:11.6f}"
        else:
            blocked_text = f"{'--':>11s}"
        print(f"{name:>20s} {boot['estimate']:11.6f} {boot['error']:11.6f} "
              f"{jack['error']:11.6f} {blocked_text} {jack['bias']:+11.2e}")
    print()
    print("   For the mean, all three agree -- the jackknife for a linear")
    print("   estimator is algebraically identical to the blocked standard")
    print("   error, and blocking is the same number again.  For everything")
    print("   below it, blocking has nothing to say, and the two resampling")
    print("   methods are the only route to an error bar at all.")
    print()
    print("   Bias.  The bootstrap and the jackknife both estimate it, the")
    print("   first as mean(theta*) - theta_hat and the second as")
    print("   (n-1)(mean(theta_(.)) - theta_hat).  There is a case with a")
    print("   known answer to test them on: the sample variance computed with")
    print("   1/N rather than 1/(N-1) is biased low by exactly -sigma^2/N.")
    print()
    boot_var = block_bootstrap(e_local, estimator=np.var, n_resamples=1000,
                               n_blocks=n_blocks,
                               rng=np.random.default_rng(11))
    jack_var = block_jackknife(e_local, estimator=np.var, n_blocks=n_blocks)
    print(f"      sigma^2, the 1/N estimator   {boot_var['estimate']:.8f}")
    print(f"      exact bias, -sigma^2 / N     "
          f"{-boot_var['estimate']/len(e_local):+.3e}")
    print(f"      jackknife bias estimate      {jack_var['bias']:+.3e}")
    print(f"      bootstrap bias estimate      {boot_var['bias']:+.3e}")
    print(f"      1/(N-1) estimator            "
          f"{float(np.var(e_local, ddof=1)):.8f}")
    print(f"      jackknife bias-corrected     {jack_var['corrected']:.8f}")
    print()
    print("   The jackknife gets the bias right to three digits with no")
    print("   knowledge of what it is estimating.  The bootstrap estimate is")
    print("   an order of magnitude too large -- not because the method is")
    print("   wrong but because a bias of 1e-8 is being read off B = 1000")
    print("   replicates each carrying an error of 1e-5, so the answer is")
    print("   noise.  Bias estimation needs a far larger B than error")
    print("   estimation does, and the jackknife, which is deterministic,")
    print("   does not have the problem at all.")
    print()
    print("   Note also the jackknife bias for the *mean* in the table above:")
    print("   it is 1e-13, that is, zero.  It has to be.  For a linear")
    print("   estimator every leave-one-out value is an exact rearrangement of")
    print("   the same numbers, so there is nothing for the correction to")
    print("   find.  Bias is a phenomenon of non-linear estimators.")
    print()
    print("   Confidence intervals.  The bootstrap gives the whole sampling")
    print("   distribution, not just its width, so a 95% interval is read off")
    print("   as the 2.5% and 97.5% quantiles of the replicates -- no")
    print("   normality assumed anywhere:")
    print()
    boot_mean = block_bootstrap(e_local, n_resamples=2000, n_blocks=n_blocks,
                                rng=np.random.default_rng(5))
    low, high = boot_mean["interval"]
    print(f"      E                    {boot_mean['estimate']:.6f}")
    print(f"      bootstrap error      {boot_mean['error']:.6f}")
    print(f"      95% percentile       [{low:.6f}, {high:.6f}]")
    print(f"      +/- 1.96 sigma       "
          f"[{boot_mean['estimate']-1.96*boot_mean['error']:.6f}, "
          f"{boot_mean['estimate']+1.96*boot_mean['error']:.6f}]")
    print(f"      skewness of the replicates "
          f"{float(np.mean((boot_mean['samples']-boot_mean['samples'].mean())**3) / boot_mean['samples'].std()**3):+.3f}")
    print()
    print("   Here the two intervals agree, because the mean of two hundred")
    print("   thousand samples is Gaussian by the central limit theorem.  For")
    print("   sigma, or for a ratio of averages, or for a fitted parameter,")
    print("   they need not, and the percentile interval is the honest one.")
    print()
    print("   Finally, the warning of exercise 3(c), made numerical.  Drawing")
    print("   single samples with replacement rather than blocks destroys the")
    print("   correlations and silently reproduces the naive error.  The")
    print("   effect is invisible on the chain above, where tau is close to")
    print("   one, so we use a deliberately sticky chain with a time step of")
    print("   0.02:")
    print()
    sticky = sample(1.00, 0.40, n_cycles=200000, time_step=0.02,
                    rng=np.random.default_rng(31), keep_samples=True)
    e_sticky = sticky["samples"]
    tau_sticky = correlation_time(e_sticky)
    naive_sticky = e_sticky.std(ddof=1) / math.sqrt(len(e_sticky))
    print(f"      tau = {tau_sticky:.1f}, so the honest error should be about")
    print(f"      sqrt(tau) = {math.sqrt(tau_sticky):.1f} times the naive one")
    print()
    print(f"{'block length':>16s} {'bootstrap error':>18s} {'/ naive':>10s}")
    for b in (1, 2, 8, 32, 128, 512, 2000):
        out = block_bootstrap(e_sticky, n_resamples=200, block=b,
                              rng=np.random.default_rng(3))
        print(f"{b:16d} {out['error']:18.8f} "
              f"{out['error']/naive_sticky:10.2f}")
    _, blocked_sticky, _ = blocking(e_sticky)
    print(f"{'blocking':>16s} {blocked_sticky:18.8f} "
          f"{blocked_sticky/naive_sticky:10.2f}")
    print()
    print("   At block length one the bootstrap returns the naive error to")
    print("   two digits, and it does so with every appearance of being a")
    print("   careful calculation.  The error climbs as the blocks lengthen")
    print("   and plateaus once they exceed the correlation time, which is")
    print("   the blocking plateau again in another guise.")

    print()
    print("=" * 74)
    print("11. The final answer")
    print("=" * 74)
    print(f"   E = {mean_b:.6f} +/- {error_b:.6f}")
    print(f"   exact (Taut)          {TAUT_ENERGY:.6f}")
    print(f"   CCSD, 42 orbitals     3.013626   (table 11.4)")
    print()
    gap = mean_b - TAUT_ENERGY
    print(f"   The energy sits {gap:.6f} above the exact answer, which is")
    print(f"   {gap/error_b:.0f} standard errors.  That gap is the two-parameter")
    print("   ansatz, not the sampling: it is what a better trial function,")
    print("   or diffusion Monte Carlo, would remove.  What this chapter has")
    print("   achieved is that the error bar is now small enough, and honest")
    print("   enough, for the gap to be visible at all.")


if __name__ == "__main__":
    _demo()

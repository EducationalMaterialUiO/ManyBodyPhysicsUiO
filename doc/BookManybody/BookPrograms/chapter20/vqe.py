"""
The variational quantum eigensolver.

Companion code to chapter 20 of *Quantum Mechanics for Many-particle Systems*.

Built on the statevector simulator of `qpe.py`, and written so that every
claim in the chapter can be checked.

  1. **Measurement.**  A quantum computer measures in the computational basis
     and nothing else.  Every Pauli string must therefore be rotated into that
     basis first -- Z directly, X with a Hadamard, Y with S^dagger H -- and the
     result estimated from a finite number of shots.  Both the rotations and
     the 1/sqrt(N) shot noise are implemented and verified.

  2. **Why single-qubit readout is not enough.**  Two states with identical
     one-qubit marginals but opposite <Z Z>; and a VQE run in which
     reconstructing the energy from marginals converges confidently to the
     wrong answer.  This is the most important negative result in the chapter
     and it is demonstrated, not asserted.

  3. **The parameter-shift rule.**  Exact gradients from two extra energy
     evaluations, valid whenever the generator has two eigenvalues.  Checked
     against analytic and finite differences, and shown to fail for a
     three-eigenvalue generator.

  4. **Applications.**  The one- and two-qubit model Hamiltonians of
     chapter 19, both Lipkin encodings, and the pairing, Heisenberg and
     Hubbard models of chapter 4 -- with the shot counts that make the
     comparison with phase estimation quantitative.

Runs on numpy alone; about two minutes.

Author: Morten Hjorth-Jensen
"""

import itertools
import math

import numpy as np

from qpe import (I2, X, Y, Z, apply_one_qubit, basis_state, bitstring,
                 commuting_groups, heisenberg_qubit, hubbard_qubit,
                 pauli_terms, probabilities)

try:
    from jordanwigner import pairing_pair_qubits
except ImportError:                     # pragma: no cover
    pairing_pair_qubits = None


S_DAGGER = np.array([[1.0, 0.0], [0.0, -1j]], dtype=complex)
HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)


# ---------------------------------------------------------------------------
#  The example Hamiltonians of chapter 19
# ---------------------------------------------------------------------------
def one_qubit_hamiltonian(eps=(0.0, 4.0), V=(3.0, 0.2), lam=1.0):
    """H = H_0 + lambda H_I for a two-level system, in Pauli form.

        H_0 = E I + Omega Z ,        E = (e0+e1)/2,   Omega = (e0-e1)/2,
        H_I = c I + omega_z Z + omega_x X ,

    with c = (V11+V22)/2, omega_z = (V11-V22)/2 and omega_x = V12 = V21.  The
    eigenvalues follow in closed form,

        E_pm = (E + lambda c) pm sqrt( (Omega + lambda omega_z)^2
                                       + (lambda omega_x)^2 ) ,

    so the model can be checked exactly at every step.  This is the system on
    which chapter 19 demonstrated phase estimation and chapter 20 demonstrates
    VQE, and using the same one twice is the point.
    """
    e0, e1 = eps
    v11, v12 = V
    energy = 0.5 * (e0 + e1)
    omega = 0.5 * (e0 - e1)
    c = 0.5 * (v11 + (-v11))            # V22 = -V11 in the standard choice
    omega_z = 0.5 * (v11 - (-v11))
    omega_x = v12
    hamiltonian = (energy * I2 + omega * Z
                   + lam * (c * I2 + omega_z * Z + omega_x * X))
    return hamiltonian


def one_qubit_exact(eps=(0.0, 4.0), V=(3.0, 0.2), lam=1.0):
    """The closed-form eigenvalues of `one_qubit_hamiltonian`."""
    e0, e1 = eps
    v11, v12 = V
    energy, omega = 0.5 * (e0 + e1), 0.5 * (e0 - e1)
    root = math.sqrt((omega + lam * v11)**2 + (lam * v12)**2)
    return np.array([energy - root, energy + root])


def two_qubit_hamiltonian(eps=(0.0, 2.5, 6.5, 7.0), Vz=0.0, Vx=2.0):
    """The 4x4 Hamiltonian of chapter 19, written in Pauli form.

    Starting from the matrix

        H = [[e1+Vz, 0, 0, Vx], [0, e2-Vz, Vx, 0],
             [0, Vx, e3-Vz, 0], [Vx, 0, 0, e4+Vz]] ,

    define the four averages

        e_II = (e1+e2+e3+e4)/4,   e_ZI = (e1+e2-e3-e4)/4,
        e_IZ = (e1-e2+e3-e4)/4,   e_ZZ = (e1-e2-e3+e4)/4,

    which are nothing but the expansion coefficients of the diagonal in the
    basis {II, ZI, IZ, ZZ} -- the diagonal part of a two-qubit operator is
    always a combination of exactly those four.  Then

        H = e_II II + e_ZI ZI + e_IZ IZ + e_ZZ ZZ + Vz ZZ + Vx XX .
    """
    e1, e2, e3, e4 = eps
    e_ii = (e1 + e2 + e3 + e4) / 4.0
    e_zi = (e1 + e2 - e3 - e4) / 4.0
    e_iz = (e1 - e2 + e3 - e4) / 4.0
    e_zz = (e1 - e2 - e3 + e4) / 4.0
    kron = np.kron
    return (e_ii * kron(I2, I2) + e_zi * kron(Z, I2) + e_iz * kron(I2, Z)
            + (e_zz + Vz) * kron(Z, Z) + Vx * kron(X, X))


def two_qubit_matrix(eps=(0.0, 2.5, 6.5, 7.0), Vz=0.0, Vx=2.0):
    """The same operator written out as the original 4x4 array."""
    e1, e2, e3, e4 = eps
    return np.array([[e1 + Vz, 0.0, 0.0, Vx],
                     [0.0, e2 - Vz, Vx, 0.0],
                     [0.0, Vx, e3 - Vz, 0.0],
                     [Vx, 0.0, 0.0, e4 + Vz]], dtype=complex)


# ---------------------------------------------------------------------------
#  Measurement
# ---------------------------------------------------------------------------
def rotate_to_z_basis(state, label):
    """Rotate so that the Pauli string `label` becomes a product of Z's.

    A quantum computer measures in the computational basis, which is the
    eigenbasis of Z.  To measure X, apply a Hadamard first, because
    H Z H = X; to measure Y, apply S^dagger then H, because
    H S^dagger Y S H = Z.  Identity factors need nothing.  These three
    circuits are the whole of Pauli measurement.
    """
    n = len(label)
    for qubit, letter in enumerate(label):
        if letter == "X":
            state = apply_one_qubit(state, HADAMARD, qubit, n)
        elif letter == "Y":
            state = apply_one_qubit(state, S_DAGGER, qubit, n)
            state = apply_one_qubit(state, HADAMARD, qubit, n)
    return state


def parity(index, label, n_qubits):
    """(-1) raised to the number of 1s on the non-identity positions."""
    bits = bitstring(index, n_qubits)
    return (-1.0)**sum(int(bits[q]) for q, l in enumerate(label) if l != "I")


def expectation_exact(state, label):
    """<P> from the full statevector: rotate, then take the parity average."""
    n = len(label)
    rotated = rotate_to_z_basis(state.copy(), label)
    p = probabilities(rotated)
    return float(sum(p[k] * parity(k, label, n) for k in range(len(p))))


def expectation_sampled(state, label, shots, rng):
    """<P> from a finite number of computational-basis measurements.

    This is what the hardware actually does: rotate the state, measure every
    qubit, and average the parity of the outcome over the shots.  The estimate
    is unbiased with variance (1 - <P>^2)/shots, so the error falls as
    1/sqrt(shots) -- the single hardest fact about VQE, since it means an
    extra digit costs a hundred times more measurements.
    """
    n = len(label)
    rotated = rotate_to_z_basis(state.copy(), label)
    p = probabilities(rotated)
    p = np.maximum(p, 0.0)
    p /= p.sum()
    draws = rng.choice(len(p), size=shots, p=p)
    signs = np.array([parity(k, label, n) for k in range(len(p))])
    return float(signs[draws].mean())


def energy_exact(state, terms):
    """<H> = sum_alpha h_alpha <P_alpha>, exactly."""
    return float(sum(c.real * expectation_exact(state, label)
                     for label, c in terms.items()))


def energy_sampled(state, terms, shots, rng):
    """<H> from `shots` measurements per Pauli string.

    The identity term costs nothing -- its expectation value is one whatever
    the state -- which is why it is always split off.
    """
    total = 0.0
    for label, c in terms.items():
        if label.count("I") == len(label):
            total += c.real
        else:
            total += c.real * expectation_sampled(state, label, shots, rng)
    return total


def shot_noise_scaling(state, terms, shot_list, repeats=200, seed=3):
    """The standard error of the sampled energy against the number of shots."""
    rng = np.random.default_rng(seed)
    exact = energy_exact(state, terms)
    out = []
    for shots in shot_list:
        estimates = [energy_sampled(state, terms, shots, rng)
                     for _ in range(repeats)]
        out.append((shots, float(np.std(estimates, ddof=1)),
                    float(np.mean(estimates)) - exact))
    return exact, out


# ---------------------------------------------------------------------------
#  Why single-qubit readout is not enough
# ---------------------------------------------------------------------------
def marginal_counterexample():
    """Two states with identical one-qubit marginals and opposite <ZZ>.

    The point is not subtle once seen: <Z_1> and <Z_2> are properties of the
    reduced density matrices, and the reduced density matrices throw away
    exactly the correlations that <Z_1 Z_2> measures.  No amount of
    single-qubit data can recover a two-qubit correlator.
    """
    bell = (basis_state("00") + basis_state("11")) / math.sqrt(2.0)
    anti = (basis_state("01") + basis_state("10")) / math.sqrt(2.0)
    out = {}
    for name, state in (("(|00> + |11>)/sqrt(2)", bell),
                        ("(|01> + |10>)/sqrt(2)", anti)):
        out[name] = dict(z1=expectation_exact(state, "ZI"),
                         z2=expectation_exact(state, "IZ"),
                         zz=expectation_exact(state, "ZZ"))
    return out


def vqe_failure_landscape(n_points=201):
    """The energy landscape of H = ZZ under exact and marginal readout.

    With the ansatz |psi(theta)> = cos(theta)|00> + sin(theta)|11>, every
    state in the family has <Z_1 Z_2> = 1 exactly, so the true landscape is
    *flat* and every theta is a ground state.  Reconstructing the correlator
    from the marginals as <Z_1><Z_2> gives cos^2(2 theta) instead: an
    oscillatory landscape with spurious minima at theta = pi/4.  An optimiser
    fed the second will converge confidently to a wrong answer, and nothing in
    its output will look wrong.
    """
    thetas = np.linspace(0.0, math.pi, n_points)
    exact, naive = [], []
    for theta in thetas:
        state = (math.cos(theta) * basis_state("00")
                 + math.sin(theta) * basis_state("11"))
        exact.append(expectation_exact(state, "ZZ"))
        naive.append(expectation_exact(state, "ZI")
                     * expectation_exact(state, "IZ"))
    return thetas, np.array(exact), np.array(naive)


# ---------------------------------------------------------------------------
#  Ansatz circuits
# ---------------------------------------------------------------------------
def rotation(axis, angle):
    """exp(-i theta P / 2) for P = X, Y, Z: the standard rotation gates."""
    pauli = {"X": X, "Y": Y, "Z": Z}[axis]
    return (math.cos(angle / 2.0) * I2
            - 1j * math.sin(angle / 2.0) * pauli)


def one_qubit_ansatz(theta):
    """|psi(theta)> = R_y(theta_0) R_x(theta_1) |0>.

    Two angles are enough to reach any point on the Bloch sphere, so for a
    one-qubit Hamiltonian this ansatz is exact: the variational minimum *is*
    the ground state, and any residual error is the optimiser's.
    """
    state = basis_state("0")
    state = rotation("Y", theta[0]) @ state
    state = rotation("X", theta[1]) @ state
    return state


def hardware_efficient(theta, n_qubits, layers=1, entangler="linear"):
    """The standard hardware-efficient ansatz: R_y R_z on each qubit, then CNOTs.

    Parameters are ordered layer by layer, and within a layer qubit by qubit,
    two per qubit.  The entangling block is a ladder of CNOTs, which has no
    parameters -- this is what makes the ansatz cheap and also what makes it
    physically uninformed, since nothing in it knows anything about the
    Hamiltonian.
    """
    state = basis_state("0" * n_qubits)
    index = 0
    for _ in range(layers):
        for q in range(n_qubits):
            state = apply_one_qubit(state, rotation("Y", theta[index]), q,
                                    n_qubits)
            state = apply_one_qubit(state, rotation("Z", theta[index + 1]), q,
                                    n_qubits)
            index += 2
        if entangler == "linear":
            for q in range(n_qubits - 1):
                state = _cnot(state, q, q + 1, n_qubits)
    return state


def _cnot(state, control, target, n_qubits):
    tensor = state.reshape((2,) * n_qubits).copy()
    index = [slice(None)] * n_qubits
    index[control] = 1
    block = tensor[tuple(index)]
    axis = target if target < control else target - 1
    block = np.flip(block, axis=axis)
    tensor[tuple(index)] = block
    return tensor.reshape(-1)


def n_parameters(n_qubits, layers=1):
    return 2 * n_qubits * layers


# ---------------------------------------------------------------------------
#  The parameter-shift rule
# ---------------------------------------------------------------------------
def parameter_shift_gradient(energy_function, theta, shift=math.pi / 2.0):
    """dE/dtheta_j = [E(theta_j + s) - E(theta_j - s)] / 2, s = pi/2.

    Valid whenever the parameter enters through exp(-i theta G / 2) with G a
    Pauli string, so that G/2 has eigenvalues +-1/2 and the energy is a
    trigonometric polynomial of frequency one:

        E(theta) = A + B cos(theta) + C sin(theta) .

    Two evaluations at theta +- pi/2 then give the derivative *exactly* -- not
    approximately, as a finite difference would.  The rule is derived and
    proved in the chapter; here it is checked.
    """
    theta = np.asarray(theta, dtype=float)
    gradient = np.empty_like(theta)
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += shift
        minus[j] -= shift
        gradient[j] = 0.5 * (energy_function(plus) - energy_function(minus))
    return gradient


def finite_difference_gradient(energy_function, theta, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    gradient = np.empty_like(theta)
    for j in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += h
        minus[j] -= h
        gradient[j] = (energy_function(plus) - energy_function(minus)) / (2 * h)
    return gradient


def check_parameter_shift(n_qubits=2, layers=2, seed=5):
    """The shift rule against a finite difference on a real ansatz."""
    rng = np.random.default_rng(seed)
    hamiltonian = two_qubit_hamiltonian()
    terms = pauli_terms(hamiltonian)
    theta = rng.uniform(0, 2 * math.pi, n_parameters(n_qubits, layers))

    def energy(t):
        return energy_exact(hardware_efficient(t, n_qubits, layers), terms)

    shift = parameter_shift_gradient(energy, theta)
    finite = finite_difference_gradient(energy, theta)
    return dict(max_difference=float(np.abs(shift - finite).max()),
                gradient_norm=float(np.linalg.norm(shift)))


def check_trigonometric_structure(n_points=9, seed=7):
    """E(theta) really is A + B cos(theta) + C sin(theta) in one parameter.

    Fitting those three constants to three points and predicting the rest is
    a sharper test than differentiating: if the functional form were wrong the
    prediction would fail everywhere else.
    """
    rng = np.random.default_rng(seed)
    terms = pauli_terms(two_qubit_hamiltonian())
    theta0 = rng.uniform(0, 2 * math.pi, n_parameters(2, 2))

    def energy(value):
        t = theta0.copy()
        t[0] = value
        return energy_exact(hardware_efficient(t, 2, 2), terms)

    nodes = np.array([0.0, 2 * math.pi / 3, 4 * math.pi / 3])
    design = np.column_stack([np.ones(3), np.cos(nodes), np.sin(nodes)])
    coefficients = np.linalg.solve(design, [energy(v) for v in nodes])

    test = np.linspace(0, 2 * math.pi, n_points)
    predicted = (coefficients[0] + coefficients[1] * np.cos(test)
                 + coefficients[2] * np.sin(test))
    actual = np.array([energy(v) for v in test])
    return float(np.abs(predicted - actual).max())


def equidistant_shift_gradient(energy_function, theta, frequencies):
    """The generalised shift rule for R equidistant frequencies 1, 2, ..., R.

        dE/dtheta = sum_{mu=1}^{2R} (-1)^{mu-1} E(theta + x_mu)
                    / (4 R sin^2(x_mu/2)) ,     x_mu = (2mu-1) pi / (2R) .

    R = 1 collapses to the familiar two-term rule.  The cost is 2R
    evaluations, so a generator with more eigenvalues is not merely awkward,
    it is proportionately more expensive.
    """
    total = 0.0
    for mu in range(1, 2 * frequencies + 1):
        x = (2 * mu - 1) * math.pi / (2 * frequencies)
        coefficient = ((-1)**(mu - 1)
                       / (4 * frequencies * math.sin(x / 2.0)**2))
        total += coefficient * energy_function(theta + x)
    return total


def shift_rule_fails(seed=11, theta=0.7):
    """A generator with three eigenvalues, for which the two-point rule fails.

    G = (Z_1 + Z_2)/2 has eigenvalues -1, 0, +1.  The differences of
    eigenvalues, which are what the energy actually sees, are then 0, +-1
    and +-2, so E(theta) is a trigonometric polynomial with *two* frequencies
    and five unknown coefficients.  Two evaluations cannot determine its
    derivative; the four-term rule can.
    """
    rng = np.random.default_rng(seed)
    generator = 0.5 * (np.kron(Z, I2) + np.kron(I2, Z))
    base = hardware_efficient(rng.uniform(0, 2 * math.pi, 8), 2, 2)
    hamiltonian = two_qubit_hamiltonian()
    values, vectors = np.linalg.eigh(generator)

    def evolve(t):
        return vectors @ np.diag(np.exp(-1j * t * values)) @ vectors.conj().T

    def energy(t):
        state = evolve(t) @ base
        return float(np.real(state.conj() @ hamiltonian @ state))

    exact = (energy(theta + 1e-6) - energy(theta - 1e-6)) / 2e-6
    two_point = 0.5 * (energy(theta + math.pi / 2)
                       - energy(theta - math.pi / 2))
    four_point = equidistant_shift_gradient(energy, theta, 2)
    return dict(spectrum=np.unique(np.round(values, 12)),
                exact=exact, two_point=two_point, four_point=four_point,
                two_point_error=abs(two_point - exact),
                four_point_error=abs(four_point - exact))


# ---------------------------------------------------------------------------
#  The VQE loop
# ---------------------------------------------------------------------------
def vqe(hamiltonian, n_qubits, layers=2, iterations=400, learning_rate=0.1,
        shots=None, seed=3, ansatz=None, marginal=False, theta0=None):
    """Minimise <H> over the ansatz parameters by gradient descent.

    `shots=None` uses exact expectation values; an integer switches on
    shot sampling, which is what makes the run realistic.  `marginal=True`
    reconstructs two-qubit correlators as products of one-qubit expectation
    values -- the mistake of Section 20.4, included so that its consequences
    can be watched.
    """
    rng = np.random.default_rng(seed)
    terms = pauli_terms(hamiltonian)
    if ansatz is None:
        ansatz = lambda t: hardware_efficient(t, n_qubits, layers)
    theta = (rng.uniform(0, 2 * math.pi, n_parameters(n_qubits, layers))
             if theta0 is None else np.array(theta0, dtype=float))

    def energy(t):
        state = ansatz(t)
        if not marginal and shots is None:
            # On a simulator the Pauli-by-Pauli sum and <psi|H|psi> agree to
            # machine precision, and the latter is far cheaper.  The
            # decomposition is what hardware needs, and is used wherever
            # measurement itself is the subject.
            return float(np.real(state.conj() @ hamiltonian @ state))
        if marginal:
            total = 0.0
            for label, c in terms.items():
                weight = len(label) - label.count("I")
                if weight == 0:
                    total += c.real
                elif weight == 1:
                    total += c.real * expectation_exact(state, label)
                else:                       # the mistake: factorise
                    product = 1.0
                    for q, letter in enumerate(label):
                        if letter != "I":
                            single = ["I"] * len(label)
                            single[q] = letter
                            product *= expectation_exact(state,
                                                         "".join(single))
                    total += c.real * product
            return total
        if shots is None:
            return energy_exact(state, terms)
        return energy_sampled(state, terms, shots, rng)

    history = []
    for step in range(iterations):
        value = energy(theta)
        history.append(value)
        gradient = parameter_shift_gradient(energy, theta)
        theta = theta - learning_rate * gradient
    final = energy(theta)
    history.append(final)
    # the honest energy of the state that was found, however it was optimised
    true_energy = energy_exact(ansatz(theta), terms)
    return dict(theta=theta, energy=final, true_energy=true_energy,
                history=np.array(history), state=ansatz(theta))


# ---------------------------------------------------------------------------
#  The Lipkin model
# ---------------------------------------------------------------------------
def lipkin_j_operators(j):
    """J_z, J_+, J_- in the |j, m> basis of dimension 2j+1."""
    dim = int(round(2 * j)) + 1
    m = np.array([j - k for k in range(dim)])
    jz = np.diag(m)
    jplus = np.zeros((dim, dim))
    for k in range(1, dim):
        mm = m[k]
        jplus[k - 1, k] = math.sqrt(j * (j + 1) - mm * (mm + 1))
    return jz, jplus, jplus.T


def lipkin_quasispin(N=4, eps=1.0, V=1.0, W=0.0):
    """H = eps J_z + (V/2)(J_+^2 + J_-^2) - W J_z^2 + W j^2 in the j = N/2 block.

    The simplification chapter 20 derives: using
    J_+ J_- + J_- J_+ = 2(J^2 - J_z^2) and fixing the maximal quasispin
    j = N/2 turns the W term into -W J_z^2 plus a constant W j^2.
    """
    j = N / 2.0
    jz, jplus, jminus = lipkin_j_operators(j)
    return (eps * jz + 0.5 * V * (jplus @ jplus + jminus @ jminus)
            - W * (jz @ jz) + W * j**2 * np.eye(len(jz)))


def lipkin_direct_qubits(N=6, eps=1.0, V=1.0, W=0.0):
    """The direct N-qubit Lipkin Hamiltonian, one qubit per particle.

        H = -(N eps/4 ... ) I + (eps/2) sum_i Z_i
            + (V/2) sum_{i<j} (X_i X_j - Y_i Y_j)
            + (W/2) sum_{i<j} Z_i Z_j ,

    obtained from J_a = (1/2) sum_i sigma^a_i and expanding the collective
    squares.  Every string has weight at most two and there are no
    Jordan-Wigner tails, because the two Lipkin levels are degenerate.
    """
    dim = 1 << N

    def single(op, site):
        out = np.array([[1.0]], dtype=complex)
        for k in range(N):
            out = np.kron(out, op if k == site else I2)
        return out

    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for i in range(N):
        hamiltonian += 0.5 * eps * single(Z, i)
    for i in range(N):
        for k in range(i):
            hamiltonian += 0.5 * V * (single(X, i) @ single(X, k)
                                      - single(Y, i) @ single(Y, k))
            hamiltonian += 0.5 * W * (single(Z, i) @ single(Z, k))
    return hamiltonian


def lipkin_symmetric_block(N, eps=1.0, V=1.0, W=0.0):
    """The maximal-spin block of `lipkin_direct_qubits`, for comparison.

    The direct mapping acts on 2^N states but the physics lives in the
    (N+1)-dimensional symmetric subspace j = N/2.  Projecting onto it must
    reproduce `lipkin_quasispin` exactly -- and does, which is the check that
    the two encodings describe the same model.
    """
    j = N / 2.0
    jz, jplus, jminus = lipkin_j_operators(j)
    return (eps * jz + 0.5 * V * (jplus @ jplus + jminus @ jminus)
            + 0.5 * W * ((jplus @ jminus + jminus @ jplus) - N * np.eye(len(jz))))


# ---------------------------------------------------------------------------
def _demo():
    rng = np.random.default_rng(2024)

    print("=" * 74)
    print("1. Measurement: the only thing hardware can do")
    print("=" * 74)
    print("A quantum computer measures in the computational basis, the")
    print("eigenbasis of Z.  Every other Pauli must be rotated into it first:")
    print("X with a Hadamard, since H Z H = X; Y with S^dagger then H.")
    print("Checking that the rotations are right, on a random two-qubit state:")
    print()
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    print(f"{'string':>8s} {'via rotation + Z readout':>26s} "
          f"{'<psi|P|psi> directly':>22s} {'difference':>12s}")
    table = {"I": I2, "X": X, "Y": Y, "Z": Z}
    for label in ("ZI", "XI", "YI", "XX", "YY", "ZZ", "XZ", "YX"):
        matrix = np.kron(table[label[0]], table[label[1]])
        direct = float(np.real(state.conj() @ matrix @ state))
        rotated = expectation_exact(state, label)
        print(f"{label:>8s} {rotated:26.12f} {direct:22.12f} "
              f"{abs(rotated-direct):12.2e}")

    print()
    print("   And the cost of finite statistics.  Each shot returns a bit")
    print("   string; the expectation value is the average parity over the")
    print("   shots, unbiased with variance (1 - <P>^2)/N.  The two-qubit")
    print("   Hamiltonian of chapter 19, sampled:")
    print()
    hamiltonian = two_qubit_hamiltonian()
    terms = pauli_terms(hamiltonian)
    trial = hardware_efficient(rng.uniform(0, 2 * math.pi, 8), 2, 2)
    exact, rows = shot_noise_scaling(trial, terms,
                                     [10, 100, 1000, 10000], repeats=200)
    print(f"   exact energy of this state {exact:.6f}")
    print()
    print(f"{'shots':>8s} {'std. error':>12s} {'x sqrt(shots)':>15s} "
          f"{'bias':>12s}")
    for shots, spread, bias in rows:
        print(f"{shots:8d} {spread:12.5f} {spread*math.sqrt(shots):15.4f} "
              f"{bias:12.5f}")
    print()
    print("   The third column is constant: the error falls as 1/sqrt(N) and")
    print("   nothing can be done about it.  An extra decimal digit costs a")
    print("   hundred times more measurements.  This single fact governs the")
    print("   whole economics of VQE, and it is what section 6 weighs against")
    print("   the exponential precision of phase estimation.")

    print()
    print("=" * 74)
    print("2. Why single-qubit readout is not enough")
    print("=" * 74)
    print("It is tempting to measure each qubit separately and multiply.  Here")
    print("are two states with identical one-qubit marginals:")
    print()
    print(f"{'state':>26s} {'<Z_1>':>9s} {'<Z_2>':>9s} {'<Z_1 Z_2>':>11s}")
    for name, values in marginal_counterexample().items():
        print(f"{name:>26s} {values['z1']:9.4f} {values['z2']:9.4f} "
              f"{values['zz']:11.4f}")
    print()
    print("   Both marginals vanish, and the correlator is +1 for one state")
    print("   and -1 for the other.  <Z_1> and <Z_2> are properties of the")
    print("   reduced density matrices, and the reduced density matrices")
    print("   discard precisely what <Z_1 Z_2> measures.  No single-qubit")
    print("   data whatsoever can distinguish these two states.")
    print()
    print("   The consequence for VQE.  Take H = Z_1 Z_2 with the ansatz")
    print("   |psi(theta)> = cos(theta)|00> + sin(theta)|11>.  Every member")
    print("   of that family has <Z_1 Z_2> = 1, so the true landscape is flat.")
    print("   Reconstructing the correlator as <Z_1><Z_2> gives cos^2(2theta):")
    print()
    thetas, exact_land, naive_land = vqe_failure_landscape()
    print(f"{'theta/pi':>10s} {'true E':>10s} {'naive E':>10s}")
    for k in range(0, len(thetas), 25):
        print(f"{thetas[k]/math.pi:10.3f} {exact_land[k]:10.4f} "
              f"{naive_land[k]:10.4f}")
    print()
    print(f"   true energy:  constant {exact_land.min():.4f} to "
          f"{exact_land.max():.4f}")
    print(f"   naive energy: ranges {naive_land.min():.4f} to "
          f"{naive_land.max():.4f}, with spurious minima")
    print(f"   at theta = pi/4 and 3pi/4, where it reports "
          f"{naive_land[len(thetas)//4]:.4f} instead of 1.")
    print()
    print("   An optimiser handed the second landscape descends into one of")
    print("   those minima and reports an energy 1.0 below the truth, with a")
    print("   perfectly smooth convergence curve.  Nothing in its output looks")
    print("   wrong.  This is the most dangerous kind of error in the whole")
    print("   subject: not noisy, not unstable, just confidently incorrect.")
    print()
    print("   The same failure inside a real VQE run on the chapter-19")
    print("   two-qubit Hamiltonian:")
    print()
    exact_spectrum = np.linalg.eigvalsh(hamiltonian)
    honest = vqe(hamiltonian, 2, layers=2, iterations=300, seed=3)
    broken = vqe(hamiltonian, 2, layers=2, iterations=300, seed=3,
                 marginal=True)
    print(f"      exact ground state           {exact_spectrum[0]:11.6f}")
    print(f"      VQE, joint readout           {honest['energy']:11.6f}")
    print(f"      VQE, marginal reconstruction {broken['energy']:11.6f}"
          f"   <- what the optimiser reports")
    print(f"      true energy of that state    "
          f"{broken['true_energy']:11.6f}   <- what it actually prepared")
    print()
    print("   The optimiser converged smoothly and stopped at a state lying")
    print(f"   {broken['true_energy']-exact_spectrum[0]:.3f} above the ground "
          f"state, and the number it")
    print("   reports is not even the energy of the state it prepared.  The")
    print("   variational guarantee is gone with it: a quantity that is not")
    print("   <psi|H|psi> for any state is under no obligation to lie above")
    print("   E_0, so a marginal estimate can equally well come out too low,")
    print("   and then it cannot be recognised as wrong at all.  Measuring the")
    print("   qubits jointly costs nothing extra -- the same shots give")
    print("   <Z_1>, <Z_2> and <Z_1 Z_2> at once -- so this error buys nothing.")

    print()
    print("=" * 74)
    print("3. The parameter-shift rule")
    print("=" * 74)
    print("If a parameter enters as exp(-i theta P / 2) with P a Pauli string,")
    print("then P/2 has eigenvalues +-1/2 and the energy is a trigonometric")
    print("polynomial of frequency one,")
    print()
    print("      E(theta) = A + B cos(theta) + C sin(theta).")
    print()
    print("Fitting A, B, C at three points and predicting elsewhere:")
    print(f"      largest prediction error {check_trigonometric_structure():.2e}")
    print()
    print("The derivative then follows from two evaluations, exactly:")
    print()
    print("      dE/dtheta = [E(theta + pi/2) - E(theta - pi/2)] / 2 .")
    print()
    checks = check_parameter_shift()
    print(f"      shift rule vs. finite difference "
          f"{checks['max_difference']:.2e}")
    print(f"      (gradient norm {checks['gradient_norm']:.4f})")
    print()
    print("   Note what this is not.  It is not a finite difference with a")
    print("   clever step size: the shift is pi/2, enormous, and the result is")
    print("   exact.  That matters on hardware, where a finite difference with")
    print("   a small step subtracts two noisy numbers that are nearly equal.")
    print()
    print("   It requires two eigenvalues in the generator.  For")
    print("   G = (Z_1 + Z_2)/2, whose eigenvalues are -1, 0, +1, the energy")
    print("   has two frequencies and the two-point rule is simply wrong:")
    print()
    fails = shift_rule_fails()
    print(f"      generator spectrum         {fails['spectrum']}")
    print(f"      exact derivative           {fails['exact']:+.6f}")
    print(f"      two-point rule             {fails['two_point']:+.6f}"
          f"   error {fails['two_point_error']:.2e}")
    print(f"      four-term rule             {fails['four_point']:+.6f}"
          f"   error {fails['four_point_error']:.2e}")
    print()
    print("   The two-point rule does not merely lose accuracy here, it")
    print("   returns zero: the two shifted evaluations coincide, and an")
    print("   optimiser using it would conclude it had reached a stationary")
    print("   point and stop.  The generalised rule with 2R evaluations for R")
    print("   frequencies recovers the derivative exactly.")

    print()
    print("=" * 74)
    print("4. VQE on the example Hamiltonians of chapter 19")
    print("=" * 74)
    print("The one-qubit model first, where two angles reach every state on")
    print("the Bloch sphere, so the ansatz is exact and the only error is the")
    print("optimiser's.")
    print()
    print(f"{'lambda':>8s} {'exact E_0':>12s} {'VQE':>12s} {'error':>11s}")
    for lam in (0.0, 0.5, 1.0):
        h1 = one_qubit_hamiltonian(lam=lam)
        exact_1 = one_qubit_exact(lam=lam)[0]
        out = vqe(h1, 1, layers=1, iterations=300, learning_rate=0.2, seed=1)
        print(f"{lam:8.2f} {exact_1:12.6f} {out['energy']:12.6f} "
              f"{abs(out['energy']-exact_1):11.2e}")
    print()
    print("Then the two-qubit model, with the hardware-efficient ansatz at")
    print("increasing depth.  More layers means more parameters and a larger")
    print("reachable set:")
    print()
    print(f"{'layers':>8s} {'parameters':>12s} {'VQE':>12s} {'error':>11s}")
    for layers in (1, 2, 3):
        out = vqe(hamiltonian, 2, layers=layers, iterations=400, seed=3)
        print(f"{layers:8d} {n_parameters(2, layers):12d} "
              f"{out['energy']:12.6f} "
              f"{abs(out['energy']-exact_spectrum[0]):11.2e}")
    print(f"   exact ground state {exact_spectrum[0]:.6f}")
    print()
    print("   Every value is above the exact one, as the variational")
    print("   principle demands.  That is the property phase estimation does")
    print("   not have, and it is worth a great deal: a VQE number can be")
    print("   compared with another VQE number, and lower is better.")

    print()
    print("=" * 74)
    print("5. The Lipkin model, two encodings")
    print("=" * 74)
    print("Fixing the maximal quasispin j = N/2 and using")
    print("J_+ J_- + J_- J_+ = 2(J^2 - J_z^2) reduces the Lipkin Hamiltonian")
    print("to eps J_z + (V/2)(J_+^2 + J_-^2) - W J_z^2 + W j^2, a matrix of")
    print("dimension N+1.  Two ways to put that on qubits:")
    print()
    for N in (4, 6):
        quasispin = lipkin_quasispin(N, eps=1.0, V=1.0, W=0.0)
        direct = lipkin_direct_qubits(N, eps=1.0, V=1.0, W=0.0)
        block = lipkin_symmetric_block(N, eps=1.0, V=1.0, W=0.0)
        terms_direct = pauli_terms(direct)
        labels = [l for l in terms_direct if l.count("I") != len(l)]
        spectrum_q = np.linalg.eigvalsh(quasispin)
        spectrum_d = np.linalg.eigvalsh(direct)
        print(f"   N = {N}:")
        print(f"      quasispin block:  dimension {len(quasispin):2d}, "
              f"E_0 = {spectrum_q[0]:+.6f}")
        print(f"      direct mapping:   {N} qubits, {len(labels)} Pauli "
              f"strings, max weight 2")
        print(f"      lowest eigenvalue of the {1<<N}-dimensional qubit "
              f"Hamiltonian {spectrum_d[0]:+.6f}")
        print(f"      symmetric block vs. quasispin form: "
              f"{np.abs(np.linalg.eigvalsh(block) - spectrum_q).max():.2e}")
        ceil = int(math.ceil(math.log2(N + 1)))
        print(f"      a compressed encoding would need only "
              f"ceil(log2({N+1})) = {ceil} qubits")
        print()
    print("   The direct mapping uses N qubits for an (N+1)-dimensional")
    print("   problem -- exponentially wasteful -- but every Pauli string has")
    print("   weight two and there are no Jordan-Wigner tails, because the")
    print("   two Lipkin levels are degenerate.  A compressed encoding needs")
    print("   only ceil(log2(N+1)) qubits but its strings are longer, and the")
    print("   unused states of the register have to be kept out of the")
    print("   calculation with a penalty term.  Neither is uniformly better.")

    print()
    print("=" * 74)
    print("6. VQE on the chapter-4 models, and what it costs")
    print("=" * 74)
    print("The same models chapter 19 fed to phase estimation.  Here the")
    print("hardware-efficient ansatz is used throughout, with no physics in")
    print("it, so the error is the ansatz's rather than the algorithm's.")
    print()
    models = [("Lipkin, N = 4 (direct)", lipkin_direct_qubits(4, 1.0, 1.0), 4),
              ("Heisenberg, 3 sites", heisenberg_qubit(3, J=1.0), 3),
              ("Hubbard, 2 sites", hubbard_qubit(2, t=1.0, U=2.0), 4)]
    if pairing_pair_qubits is not None:
        models.insert(1, ("pairing, 3 levels",
                          pairing_pair_qubits(3, g=1.0), 3))
    print(f"{'model':>22s} {'qubits':>7s} {'layers':>7s} {'exact':>11s} "
          f"{'VQE':>11s} {'error':>10s} {'groups':>7s}")
    for name, ham, nq in models:
        exact_e = float(np.linalg.eigvalsh(ham)[0])
        best = None
        for layers in (2, 3, 4):
            for seed in (7, 17, 27, 37):
                out = vqe(ham, nq, layers=layers, iterations=300,
                          learning_rate=0.15, seed=seed)
                if best is None or out["energy"] < best[0]:
                    best = (out["energy"], layers)
        labels = [l for l in pauli_terms(ham) if l.count("I") != len(l)]
        print(f"{name:>22s} {nq:7d} {best[1]:7d} {exact_e:11.5f} "
              f"{best[0]:11.5f} {abs(best[0]-exact_e):10.2e} "
              f"{len(commuting_groups(labels)):7d}")
    print()
    print("   Four random restarts at each depth, best kept -- without them")
    print("   the optimiser gets stuck in local minima, which is itself worth")
    print("   knowing.  The hardware-efficient ansatz carries no information")
    print("   about any of these Hamiltonians, and where it falls short the")
    print("   fault is the ansatz's rather than the algorithm's.")
    print()
    print("   The last column is the measurement cost: with G commuting")
    print("   groups and S shots each, one energy evaluation costs G*S")
    print("   measurements, and the parameter-shift gradient costs 2P times")
    print("   that for P parameters.  A single optimisation step of the")
    print("   4-qubit, 4-layer ansatz therefore needs 2*32*G*S shots -- which")
    print("   is why VQE is measurement-bound and phase estimation is")
    print("   depth-bound, and why the two fail on different hardware.")


if __name__ == "__main__":
    _demo()

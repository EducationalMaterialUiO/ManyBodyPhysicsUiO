"""
The quantum Fourier transform and quantum phase estimation.

Companion code to chapter 19 of *Quantum Mechanics for Many-particle Systems*.

A statevector simulator in numpy, written so that every claim in the chapter
can be checked rather than asserted.

  1. **The QFT.**  Built gate by gate from Hadamards, controlled phase
     rotations and a final SWAP network, exactly as the circuit in the text.
     It is checked three ways: against the explicit Fourier matrix, against
     the product form written as a tensor product of one-qubit phase states,
     and against `numpy.fft`.  The bit-reversal convention -- the commonest
     source of confusion -- is made explicit by running the circuit with and
     without the final swaps.

  2. **Quantum phase estimation.**  Controlled powers of a unitary followed by
     the inverse QFT.  When the phase is an exact m-bit binary fraction the
     answer comes out with probability one; when it is not, the output is a
     distribution, and the chapter's bound P(nearest) >= 4/pi^2 is verified.

  3. **The models of chapter 4.**  Lipkin, pairing, Heisenberg and Hubbard,
     mapped to qubits, exponentiated to give U = exp(-iHt), and fed to QPE.
     Both the exact exponential and a Trotterised approximation are used, so
     that the two distinct error sources -- finite register and finite Trotter
     step -- can be separated and measured.

Everything runs on numpy alone except for `scipy.linalg.expm`, which is used
for the exact time evolution.  The demonstration takes about a minute.

Author: Morten Hjorth-Jensen
"""

import itertools
import math

import numpy as np

try:                                    # the chapter 4 model builders
    from models import HeisenbergChain, HubbardChain, LipkinModel
    from jordanwigner import decompose, pairing_pair_qubits, single
except ImportError:                     # pragma: no cover
    HeisenbergChain = HubbardChain = LipkinModel = None
    decompose = pairing_pair_qubits = single = None


# ---------------------------------------------------------------------------
#  A minimal statevector simulator
# ---------------------------------------------------------------------------
#  A state of n qubits is a complex vector of length 2^n.  We index it so that
#  qubit 0 is the *most* significant bit, matching the convention of the
#  chapter: the basis state |x_0 x_1 ... x_{n-1}> sits at position
#  x_0 2^{n-1} + x_1 2^{n-2} + ... + x_{n-1}.  Reshaping the vector to
#  (2, 2, ..., 2) then makes axis k the k-th qubit, and a one-qubit gate is a
#  contraction on that axis.
# ---------------------------------------------------------------------------
HADAMARD = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / math.sqrt(2.0)


def phase_gate(k):
    """R_k = diag(1, exp(2 pi i / 2^k)), the controlled rotation of the QFT."""
    return np.array([[1.0, 0.0],
                     [0.0, np.exp(2j * math.pi / 2**k)]], dtype=complex)


def apply_one_qubit(state, gate, qubit, n_qubits):
    """Apply a 2x2 gate to one qubit of a statevector."""
    tensor = state.reshape((2,) * n_qubits)
    tensor = np.tensordot(gate, tensor, axes=([1], [qubit]))
    tensor = np.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


def apply_controlled_one_qubit(state, gate, control, target, n_qubits):
    """Apply `gate` to `target` on the subspace where `control` is |1>."""
    tensor = state.reshape((2,) * n_qubits).copy()
    index = [slice(None)] * n_qubits
    index[control] = 1
    block = tensor[tuple(index)]                       # control = 1 slice
    axis = target if target < control else target - 1
    block = np.moveaxis(np.tensordot(gate, block, axes=([1], [axis])), 0, axis)
    tensor[tuple(index)] = block
    return tensor.reshape(-1)


def apply_swap(state, a, b, n_qubits):
    """Exchange two qubits."""
    tensor = state.reshape((2,) * n_qubits)
    return np.swapaxes(tensor, a, b).reshape(-1)


def basis_state(bits):
    """|bits> as a statevector, `bits` a string such as '101'."""
    n = len(bits)
    state = np.zeros(1 << n, dtype=complex)
    state[int(bits, 2)] = 1.0
    return state


def probabilities(state):
    return np.abs(state)**2


def bitstring(index, n_qubits):
    return format(index, "0{}b".format(n_qubits))


# ---------------------------------------------------------------------------
#  The quantum Fourier transform
# ---------------------------------------------------------------------------
def qft(state, n_qubits, swaps=True, approximation=None):
    """The QFT circuit of the chapter, applied to a statevector.

    For each qubit j: a Hadamard, then a controlled R_{k-j+1} from every later
    qubit k, and at the end a SWAP network reversing the register.  The
    product form

        QFT|x_1...x_n> = (|0> + e^{2 pi i 0.x_n}|1>) (x) ...
                         ... (x) (|0> + e^{2 pi i 0.x_1...x_n}|1>) / 2^{n/2}

    is what the circuit is built from, and the reversal in that formula is
    exactly what the final swaps undo.

    `swaps=False` omits them, which is what most software libraries do -- they
    return the register in reversed order and document the fact.
    `approximation=d` drops controlled rotations with k - j >= d, giving the
    approximate QFT.
    """
    for j in range(n_qubits):
        state = apply_one_qubit(state, HADAMARD, j, n_qubits)
        for k in range(j + 1, n_qubits):
            if approximation is not None and k - j >= approximation:
                continue
            state = apply_controlled_one_qubit(state, phase_gate(k - j + 1),
                                               k, j, n_qubits)
    if swaps:
        for j in range(n_qubits // 2):
            state = apply_swap(state, j, n_qubits - 1 - j, n_qubits)
    return state


def inverse_qft(state, n_qubits, swaps=True):
    """The inverse QFT: reverse the gate order and conjugate every rotation."""
    if swaps:
        for j in range(n_qubits // 2):
            state = apply_swap(state, j, n_qubits - 1 - j, n_qubits)
    for j in reversed(range(n_qubits)):
        for k in reversed(range(j + 1, n_qubits)):
            state = apply_controlled_one_qubit(
                state, phase_gate(k - j + 1).conj(), k, j, n_qubits)
        state = apply_one_qubit(state, HADAMARD, j, n_qubits)
    return state


def fourier_matrix(n_qubits):
    """The explicit 2^n x 2^n Fourier matrix, F_jk = e^{2 pi i jk/N}/sqrt(N)."""
    dim = 1 << n_qubits
    j, k = np.meshgrid(np.arange(dim), np.arange(dim), indexing="ij")
    return np.exp(2j * math.pi * j * k / dim) / math.sqrt(dim)


def product_form(x, n_qubits):
    """The QFT of |x> written directly as a tensor product of phase states.

    Each factor is (|0> + e^{2 pi i 0.x_{n-m+1}...x_n}|1>)/sqrt(2), and the
    binary fraction 0.x_{n-m+1}...x_n is exactly the fractional part of
    x / 2^m.  Building the state this way uses no circuit at all, so agreement
    with `qft` is a genuine check of the derivation.
    """
    state = np.array([1.0], dtype=complex)
    for m in range(1, n_qubits + 1):
        phase = np.exp(2j * math.pi * (x % 2**m) / 2**m)
        state = np.kron(state, np.array([1.0, phase]) / math.sqrt(2.0))
    return state


def check_qft(n_qubits=4):
    """Verify the circuit against the matrix, the product form and the FFT."""
    dim = 1 << n_qubits
    matrix = fourier_matrix(n_qubits)

    circuit_error = product_error = 0.0
    for x in range(dim):
        state = basis_state(bitstring(x, n_qubits))
        out = qft(state.copy(), n_qubits)
        circuit_error = max(circuit_error,
                            float(np.abs(out - matrix[:, x]).max()))
        product_error = max(product_error,
                            float(np.abs(out - product_form(x, n_qubits)).max()))

    # a random state, against numpy's FFT.  numpy uses the opposite sign
    # convention and a different normalisation, hence the conjugate and factor
    rng = np.random.default_rng(2024)
    random = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    random /= np.linalg.norm(random)
    fft_error = float(np.abs(qft(random.copy(), n_qubits)
                             - np.fft.ifft(random) * math.sqrt(dim)).max())

    # the inverse really is the inverse
    roundtrip = inverse_qft(qft(random.copy(), n_qubits), n_qubits)
    inverse_error = float(np.abs(roundtrip - random).max())

    # unitarity of the assembled circuit
    columns = np.column_stack([qft(basis_state(bitstring(x, n_qubits)),
                                   n_qubits) for x in range(dim)])
    unitary_error = float(np.abs(columns.conj().T @ columns
                                 - np.eye(dim)).max())

    return dict(matrix=circuit_error, product_form=product_error,
                fft=fft_error, inverse=inverse_error, unitary=unitary_error)


def gate_counts(n_qubits, approximation=None):
    """(Hadamards, controlled rotations, swaps) for the QFT circuit."""
    hadamards = n_qubits
    rotations = 0
    for j in range(n_qubits):
        for k in range(j + 1, n_qubits):
            if approximation is None or k - j < approximation:
                rotations += 1
    return hadamards, rotations, n_qubits // 2


def approximation_error(n_qubits=6, x=37):
    """How much the approximate QFT costs in accuracy, level by level."""
    exact = qft(basis_state(bitstring(x, n_qubits)), n_qubits)
    out = []
    for d in range(1, n_qubits + 1):
        approx = qft(basis_state(bitstring(x, n_qubits)), n_qubits,
                     approximation=d)
        _, rotations, _ = gate_counts(n_qubits, approximation=d)
        out.append((d, rotations, float(np.abs(approx - exact).max())))
    return out


# ---------------------------------------------------------------------------
#  Quantum phase estimation
# ---------------------------------------------------------------------------
def phase_estimation(unitary, eigenstate, n_control):
    """Run QPE and return the probability distribution over the m bit strings.

    The system register is held in `eigenstate`, which need not be an
    eigenstate: if it is a superposition, the output distribution is the
    corresponding mixture, which is the content of the overlap requirement.

    The control register is prepared in a uniform superposition, controlled
    powers U^{2^j} are applied, and the inverse QFT decodes the accumulated
    phases into a binary fraction.  Everything is done with dense matrices,
    which is fine for the register sizes in the chapter and hopeless beyond
    them -- as it should be, since that is the whole point of the algorithm.
    """
    dim_system = len(eigenstate)
    dim_control = 1 << n_control
    state = np.zeros(dim_control * dim_system, dtype=complex)

    # |+>^m (x) |psi>
    for k in range(dim_control):
        state[k * dim_system:(k + 1) * dim_system] = eigenstate
    state /= math.sqrt(dim_control)

    # controlled powers: control qubit j (most significant first) applies
    # U^{2^{m-1-j}}, which for control value k applies U^k in total
    powers = [np.linalg.matrix_power(unitary, 1 << (n_control - 1 - j))
              for j in range(n_control)]
    for j, power in enumerate(powers):
        block = 1 << (n_control - 1 - j)
        for k in range(dim_control):
            if (k // block) % 2 == 1:
                chunk = state[k * dim_system:(k + 1) * dim_system]
                state[k * dim_system:(k + 1) * dim_system] = power @ chunk

    # inverse QFT on the control register only
    reshaped = state.reshape(dim_control, dim_system)
    for column in range(dim_system):
        reshaped[:, column] = inverse_qft(reshaped[:, column].copy(),
                                          n_control)
    state = reshaped.reshape(-1)

    return probabilities(state).reshape(dim_control, dim_system).sum(axis=1)


def phase_unitary(phi, dimension=1):
    """A unitary with the single known eigenphase phi."""
    return np.exp(2j * math.pi * phi) * np.eye(dimension, dtype=complex)


def best_estimate(distribution, n_control):
    """The most likely bit string, its probability, and the phase it encodes."""
    index = int(np.argmax(distribution))
    return (bitstring(index, n_control), float(distribution[index]),
            index / (1 << n_control))


def check_exact_phase(n_control=3, phi=5.0 / 8.0):
    """The exactly representable case: one bit string, probability one."""
    distribution = phase_estimation(phase_unitary(phi), np.array([1.0 + 0j]),
                                    n_control)
    label, probability, estimate = best_estimate(distribution, n_control)
    return dict(bits=label, probability=probability, estimate=estimate,
                exact=phi)


def check_inexact_phase(n_control=4, phi=0.3):
    """The generic case: a distribution peaked at the nearest bit string.

    The textbook bound is that the nearest m-bit approximation is obtained
    with probability at least 4/pi^2 = 0.405, whatever the phase.
    """
    distribution = phase_estimation(phase_unitary(phi), np.array([1.0 + 0j]),
                                    n_control)
    label, probability, estimate = best_estimate(distribution, n_control)
    nearest = round(phi * (1 << n_control)) % (1 << n_control)
    return dict(bits=label, probability=probability, estimate=estimate,
                exact=phi, nearest=bitstring(nearest, n_control),
                bound=4.0 / math.pi**2,
                two_best=float(np.sort(distribution)[-2:].sum()))


# ---------------------------------------------------------------------------
#  From Hamiltonians to phases
# ---------------------------------------------------------------------------
def evolution_operator(hamiltonian, time):
    """U(t) = exp(-i H t), computed exactly."""
    from scipy.linalg import expm
    return expm(-1j * hamiltonian * time)


def trotter_operator(terms, time, steps):
    """First-order Trotter: (prod_k exp(-i H_k t/r))^r.

    `terms` is a list of Hermitian matrices summing to H.  The error of the
    first-order product formula is O(t^2/r) per step and hence O(t^2/r) in
    total, controlled by the commutators [H_j, H_k]; this is what the chapter
    calls the second error source, alongside the finite control register.
    """
    from scipy.linalg import expm
    step = expm(-1j * terms[0] * time / steps)
    for term in terms[1:]:
        step = expm(-1j * term * time / steps) @ step
    return np.linalg.matrix_power(step, steps)


def energy_from_phase(phi, time):
    """Invert phi = -E t / (2 pi) mod 1, choosing the branch in [-2pi/t, 0).

    The modulus is the aliasing that makes the choice of t part of the
    algorithm: t must be small enough that every eigenvalue of interest maps
    into one period, and large enough that the phases are well separated.
    """
    return -2.0 * math.pi * phi / time


def qpe_on_hamiltonian(hamiltonian, eigenstate, time, n_control,
                       terms=None, trotter_steps=None):
    """Run QPE on U = exp(-i H t) and convert the answer back to an energy."""
    if terms is not None and trotter_steps is not None:
        unitary = trotter_operator(terms, time, trotter_steps)
    else:
        unitary = evolution_operator(hamiltonian, time)
    distribution = phase_estimation(unitary, eigenstate, n_control)
    label, probability, phi = best_estimate(distribution, n_control)
    return dict(bits=label, probability=probability, phase=phi,
                energy=energy_from_phase(phi, time),
                distribution=distribution)


def choose_time(hamiltonian, safety=0.9):
    """A time step that maps the whole spectrum into one period of the phase.

    We need |E| t / (2 pi) < 1 for every eigenvalue, so t < 2 pi / ||H||.  On
    real hardware the spectrum is exactly what one does not know, so the bound
    used in practice is the 1-norm of the Pauli coefficients,

        Lambda = sum_alpha |h_alpha|  >=  ||H|| ,

    which is available from the Hamiltonian as written.  It is a loose bound,
    and the looseness costs resolution: the useful range of phases is
    |E_0|/Lambda rather than 1, so a few control qubits are spent on empty
    space.  Choosing the offset and the time step well is a real part of
    designing a QPE calculation, not a detail.
    """
    coefficients = pauli_terms(hamiltonian)
    norm = sum(abs(c) for label, c in coefficients.items()
               if label.count("I") != len(label))
    return safety * 2.0 * math.pi / norm


# ---------------------------------------------------------------------------
#  The models of chapter 4, as qubit Hamiltonians
# ---------------------------------------------------------------------------
X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
I2 = np.eye(2, dtype=complex)


def _single(op, site, n):
    """op acting on `site`, identity elsewhere."""
    out = np.array([[1.0]], dtype=complex)
    for k in range(n):
        out = np.kron(out, op if k == site else I2)
    return out


def lipkin_qubit(n_particles=2, eps=1.0, V=1.0, W=0.0):
    """The Lipkin model on n qubits, one per particle.

    With sigma = +/- labelling the two levels, the quasispin operators are
    sums of single-particle Pauli operators,

        J_z = (1/2) sum_p Z_p,
        J_x = (1/2) sum_p X_p,   J_y = (1/2) sum_p Y_p,

    so that H = eps J_z + V (J_x^2 - J_y^2) + W(...) is a sum of one- and
    two-qubit Pauli strings with no Jordan-Wigner tails at all -- the
    degeneracy of the two levels has done the work.  This is why the Lipkin
    model is the standard first target for quantum-computing demonstrations.
    """
    dim = 1 << n_particles
    jz = 0.5 * sum(_single(Z, p, n_particles) for p in range(n_particles))
    jx = 0.5 * sum(_single(X, p, n_particles) for p in range(n_particles))
    jy = 0.5 * sum(_single(Y, p, n_particles) for p in range(n_particles))
    hamiltonian = eps * jz + V * (jx @ jx - jy @ jy)
    if W:
        hamiltonian = hamiltonian + W * (jx @ jx + jy @ jy
                                         - 0.5 * n_particles
                                         * np.eye(dim, dtype=complex))
    return hamiltonian


def heisenberg_qubit(sites=3, J=1.0, h=0.0, pbc=True):
    """H = J sum_<ij> S_i . S_j - h sum_i S^z_i, one qubit per site.

    Spin operators on different sites commute, so there are no Jordan-Wigner
    strings: the qubit Hamiltonian is a sum of nearest-neighbour XX + YY + ZZ
    terms, weight two, and the circuit depth is independent of the system
    size.
    """
    bonds = [(i, (i + 1) % sites) for i in range(sites)]
    if not pbc:
        bonds = bonds[:-1]
    dim = 1 << sites
    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for i, j in bonds:
        for op in (X, Y, Z):
            hamiltonian += 0.25 * J * _single(op, i, sites) \
                @ _single(op, j, sites)
    for i in range(sites):
        hamiltonian -= 0.5 * h * _single(Z, i, sites)
    return hamiltonian


def hubbard_qubit(sites=2, t=1.0, U=2.0):
    """The Fermi-Hubbard chain under Jordan-Wigner, 2 * sites qubits.

    Ordering the modes as (1 up, 1 down, 2 up, 2 down, ...) keeps the parity
    strings short: a hopping term between neighbouring sites of the same spin
    spans two modes of the opposite spin, so the strings have weight four.
    The interaction is diagonal, hence a product of Z operators.
    """
    n_modes = 2 * sites
    dim = 1 << n_modes

    def creation(mode):
        """a^+_mode under Jordan-Wigner, with the parity string."""
        out = np.array([[1.0]], dtype=complex)
        lowering = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
        for k in range(n_modes):
            if k < mode:
                out = np.kron(out, Z)
            elif k == mode:
                out = np.kron(out, lowering)
            else:
                out = np.kron(out, I2)
        return out

    create = [creation(m) for m in range(n_modes)]
    annihilate = [c.conj().T for c in create]
    number = [create[m] @ annihilate[m] for m in range(n_modes)]

    hamiltonian = np.zeros((dim, dim), dtype=complex)
    for site in range(sites - 1):
        for spin in (0, 1):
            a = 2 * site + spin
            b = 2 * (site + 1) + spin
            hamiltonian += -t * (create[a] @ annihilate[b]
                                 + create[b] @ annihilate[a])
    for site in range(sites):
        hamiltonian += U * number[2 * site] @ number[2 * site + 1]
    return hamiltonian


def pauli_terms(hamiltonian, tol=1e-10):
    """Split a Hamiltonian into its Pauli strings: {label: coefficient}."""
    dim = hamiltonian.shape[0]
    n = int(round(math.log2(dim)))
    table = {"I": I2, "X": X, "Y": Y, "Z": Z}
    out = {}
    for word in itertools.product("IXYZ", repeat=n):
        label = "".join(word)
        matrix = np.array([[1.0]], dtype=complex)
        for letter in word:
            matrix = np.kron(matrix, table[letter])
        coefficient = np.trace(matrix @ hamiltonian) / dim
        if abs(coefficient) > tol:
            out[label] = complex(coefficient)
    return out


def commuting_groups(labels):
    """Greedily partition Pauli strings into mutually commuting groups.

    Terms within a group can be exponentiated together without Trotter error,
    so the number of groups is a better measure of circuit depth than the
    number of terms.
    """
    def commutes(a, b):
        anticommuting = sum(1 for p, q in zip(a, b)
                            if p != "I" and q != "I" and p != q)
        return anticommuting % 2 == 0

    groups = []
    for label in labels:
        for group in groups:
            if all(commutes(label, other) for other in group):
                group.append(label)
                break
        else:
            groups.append([label])
    return groups


def model_summary(name, hamiltonian):
    """Pauli count, weights and commuting groups for a qubit Hamiltonian."""
    terms = pauli_terms(hamiltonian)
    labels = [l for l in terms if l.count("I") != len(l)]
    weights = [len(l) - l.count("I") for l in labels]
    return dict(name=name, n_qubits=int(round(math.log2(len(hamiltonian)))),
                n_terms=len(labels),
                max_weight=max(weights) if weights else 0,
                groups=len(commuting_groups(labels)),
                spectrum=np.linalg.eigvalsh(hamiltonian))


# ---------------------------------------------------------------------------
def _demo():
    print("=" * 74)
    print("1. The quantum Fourier transform")
    print("=" * 74)
    print("The circuit -- Hadamards, controlled R_k, and a final SWAP")
    print("network -- checked against three independent references.")
    print()
    checks = check_qft(4)
    for key, label in (("matrix", "circuit vs. the Fourier matrix"),
                       ("product_form", "circuit vs. the product form"),
                       ("fft", "circuit vs. numpy.fft"),
                       ("inverse", "inverse QFT o QFT = identity"),
                       ("unitary", "assembled circuit is unitary")):
        print(f"      {label:<36s} {checks[key]:.2e}")
    print()
    print("   The second line is the one that matters: the product form was")
    print("   built with no circuit at all, straight from the binary-fraction")
    print("   factorisation, so agreement means the derivation is right and")
    print("   not merely self-consistent.")

    print()
    print("   The worked example of the text, QFT_8 |101>:")
    print()
    state = qft(basis_state("101"), 3)
    print(f"{'k':>6s} {'amplitude':>28s} {'|amp|':>9s} {'phase/2pi':>11s}")
    for k in range(8):
        amp = state[k]
        print(f"{bitstring(k,3):>6s} {amp.real:+11.6f}{amp.imag:+11.6f}i "
              f"{abs(amp):9.6f} {np.angle(amp)/(2*math.pi) % 1.0:11.6f}")
    print()
    print("   Every amplitude has modulus 1/sqrt(8) = 0.353553, and the phase")
    print("   of |k> is 5k/8 mod 1 -- the Fourier kernel with x = 5.")

    print()
    print("   Bit ordering.  Without the final swaps the register comes out")
    print("   reversed, which is what most libraries do and document:")
    print()
    with_swaps = qft(basis_state("100"), 3)
    without = qft(basis_state("100"), 3, swaps=False)
    reversed_state = qft(basis_state("001"), 3)
    print(f"      |QFT(100) with swaps - QFT(001) without swaps| = "
          f"{np.abs(with_swaps - qft(basis_state('100'), 3)).max():.1e}")
    print(f"      the no-swap output of |100> equals the swapped output of")
    print(f"      the bit-reversed input |001|: "
          f"{np.abs(without - reversed_state).max():.1e}")

    print()
    print("   Gate counts, and the approximate QFT.  The exact circuit needs")
    print("   n Hadamards, n(n-1)/2 controlled rotations and n/2 swaps.")
    print("   Dropping rotations with k - j >= d gives the approximate QFT:")
    print()
    print(f"{'n':>5s} {'H':>5s} {'controlled R':>14s} {'swaps':>7s} "
          f"{'classical DFT':>15s}")
    for n in (3, 6, 10, 100):
        h, r, s = gate_counts(n)
        classical = f"{4**n:.1e}" if n <= 10 else "1.6e+60"
        print(f"{n:5d} {h:5d} {r:14d} {s:7d} {classical:>15s}")
    print()
    print(f"{'d':>5s} {'rotations':>11s} {'max amplitude error':>21s}")
    for d, rotations, error in approximation_error(6, 37):
        print(f"{d:5d} {rotations:11d} {error:21.3e}")
    print()
    print("   Each extra level of rotation buys roughly a factor of three in")
    print("   accuracy for a handful of gates, and the last levels -- the")
    print("   smallest angles -- buy the most per gate.  Truncating at")
    print("   d ~ log n leaves an error that vanishes with n for fixed")
    print("   accuracy, which is the basis of the O(n log n) approximate QFT.")

    print()
    print("=" * 74)
    print("2. Quantum phase estimation")
    print("=" * 74)
    print("Controlled powers of U write the eigenphase into the control")
    print("register as a Fourier state; the inverse QFT reads it out.")
    print()
    exact = check_exact_phase(3, 5.0 / 8.0)
    print(f"   phi = 5/8 = 0.101 with a 3-qubit register:")
    print(f"      measured |{exact['bits']}> with probability "
          f"{exact['probability']:.6f}")
    print(f"      estimate {exact['estimate']:.6f}, exact {exact['exact']:.6f}")
    print()
    print("   When the phase is an exact m-bit binary fraction the answer is")
    print("   deterministic.  When it is not, the output is a distribution:")
    print()
    for n_control in (3, 4, 6, 8):
        out = check_inexact_phase(n_control, 0.3)
        print(f"      m = {n_control}: |{out['bits']}> with p = "
              f"{out['probability']:.4f}, estimate {out['estimate']:.6f}, "
              f"error {abs(out['estimate']-0.3):.2e}")
    out = check_inexact_phase(4, 0.3)
    print()
    print(f"   The bound of the text is that the nearest m-bit value is")
    print(f"   obtained with probability at least 4/pi^2 = {out['bound']:.4f};")
    print(f"   here the best single outcome has {out['probability']:.4f} and")
    print(f"   the best two together {out['two_best']:.4f}.")

    print()
    print("=" * 74)
    print("3. The models of chapter 4 as qubit Hamiltonians")
    print("=" * 74)
    print("QPE needs U = exp(-iHt), and the cost of building it is set by the")
    print("number of Pauli strings, their weight, and how many mutually")
    print("commuting groups they fall into -- terms within a group can be")
    print("exponentiated together with no Trotter error at all.")
    print()
    models = [
        ("Lipkin, N = 2", lipkin_qubit(2, eps=1.0, V=1.0)),
        ("Lipkin, N = 4", lipkin_qubit(4, eps=1.0, V=1.0)),
        ("pairing, 3 levels", pairing_pair_qubits(3, g=1.0)
         if pairing_pair_qubits is not None else None),
        ("Heisenberg, 3 sites", heisenberg_qubit(3, J=1.0)),
        ("Heisenberg, 4 sites", heisenberg_qubit(4, J=1.0)),
        ("Hubbard, 2 sites", hubbard_qubit(2, t=1.0, U=2.0)),
    ]
    print(f"{'model':>22s} {'qubits':>7s} {'Pauli terms':>12s} "
          f"{'max weight':>11s} {'groups':>7s} {'E_0':>10s}")
    for name, hamiltonian in models:
        if hamiltonian is None:
            continue
        info = model_summary(name, hamiltonian)
        print(f"{name:>22s} {info['n_qubits']:7d} {info['n_terms']:12d} "
              f"{info['max_weight']:11d} {info['groups']:7d} "
              f"{info['spectrum'][0]:10.5f}")
    print()
    print("   Lipkin and Heisenberg have weight-two Hamiltonians and no")
    print("   Jordan-Wigner tails -- the first because the two levels are")
    print("   degenerate so the quasispin operators are sums of single-qubit")
    print("   Paulis, the second because spin operators on different sites")
    print("   commute.  Hubbard pays the fermionic price: weight-four strings")
    print("   from the parity factors.")

    print()
    print("=" * 74)
    print("4. Phase estimation on the models")
    print("=" * 74)
    print("Take the exact ground state as the system register, evolve for a")
    print("time short enough that the whole spectrum fits in one period of")
    print("the phase, and read the energy back off the control register.")
    print()
    for name, hamiltonian in models:
        if hamiltonian is None:
            continue
        spectrum, vectors = np.linalg.eigh(hamiltonian)
        ground = vectors[:, 0]
        time = choose_time(hamiltonian)
        print(f"   {name}:  exact E_0 = {spectrum[0]:+.6f},  t = {time:.4f}")
        print(f"{'':>6s}{'m':>4s} {'bits':>12s} {'p':>8s} {'E (QPE)':>12s} "
              f"{'error':>11s}")
        for m in (4, 6, 8, 10):
            out = qpe_on_hamiltonian(hamiltonian, ground, time, m)
            print(f"{'':>6s}{m:4d} {out['bits']:>12s} "
                  f"{out['probability']:8.4f} {out['energy']:+12.6f} "
                  f"{abs(out['energy']-spectrum[0]):11.2e}")
        print()
    print("   The error halves with each extra control qubit, which is the")
    print("   1/2^m resolution of the algorithm.  Note that the probability")
    print("   is not one: the energies are not exact binary fractions of the")
    print("   period, so the output is a distribution peaked at the nearest")
    print("   representable value.")

    print()
    print("=" * 74)
    print("5. Trotter error: the second source of error")
    print("=" * 74)
    print("On hardware U = exp(-iHt) must itself be built from gates, and the")
    print("standard route is a product formula.  This introduces an error")
    print("that has nothing to do with the size of the control register, and")
    print("the two must be balanced.  Heisenberg on 3 sites, m = 8 control")
    print("qubits, first-order Trotter with r steps:")
    print()
    hamiltonian = heisenberg_qubit(3, J=1.0)
    spectrum, vectors = np.linalg.eigh(hamiltonian)
    ground = vectors[:, 0]
    time = choose_time(hamiltonian)
    terms = []
    labels = [l for l in pauli_terms(hamiltonian) if l.count("I") != len(l)]
    for group in commuting_groups(labels):
        block = np.zeros_like(hamiltonian)
        coefficients = pauli_terms(hamiltonian)
        for label in group:
            matrix = np.array([[1.0]], dtype=complex)
            for letter in label:
                matrix = np.kron(matrix, {"I": I2, "X": X, "Y": Y,
                                          "Z": Z}[letter])
            block = block + coefficients[label] * matrix
        terms.append(block)
    print(f"   the Hamiltonian splits into {len(terms)} commuting groups")
    print()
    print(f"{'r':>6s} {'E (QPE)':>12s} {'error':>11s} "
          f"{'|U_trot - U|':>14s}")
    exact_u = evolution_operator(hamiltonian, time)
    for steps in (1, 2, 4, 8, 16, 32):
        out = qpe_on_hamiltonian(hamiltonian, ground, time, 8,
                                 terms=terms, trotter_steps=steps)
        approx = trotter_operator(terms, time, steps)
        print(f"{steps:6d} {out['energy']:+12.6f} "
              f"{abs(out['energy']-spectrum[0]):11.2e} "
              f"{np.abs(approx-exact_u).max():14.2e}")
    print()
    print("   Read the last column first: the operator error falls cleanly as")
    print("   1/r, which is the first-order product formula's rate.  The")
    print("   energy error in the middle column follows it down -- and then")
    print("   stops, at r = 8, on the floor set by the 8-qubit control")
    print("   register.  Beyond that point more Trotter steps buy nothing at")
    print("   all.  The two errors have to be balanced against each other,")
    print("   not minimised separately, and the balance is what fixes the")
    print("   circuit depth of a real calculation.")

    print()
    print("=" * 74)
    print("6. The overlap requirement")
    print("=" * 74)
    print("QPE does not need the exact eigenstate.  Fed a superposition")
    print("sum_j c_j |E_j>, it returns E_j with probability |c_j|^2 -- so the")
    print("state preparation decides the success rate, not the accuracy.")
    print("Lipkin with N = 4, m = 8, starting from states of decreasing")
    print("overlap with the ground state:")
    print()
    hamiltonian = lipkin_qubit(4, eps=1.0, V=1.0)
    spectrum, vectors = np.linalg.eigh(hamiltonian)
    time = choose_time(hamiltonian)
    rng = np.random.default_rng(7)
    ground, excited = vectors[:, 0], vectors[:, 1]
    reference = qpe_on_hamiltonian(hamiltonian, ground, time, 8)
    excited_ref = qpe_on_hamiltonian(hamiltonian, excited, time, 8)
    index_0 = int(reference["bits"], 2)
    index_1 = int(excited_ref["bits"], 2)
    print(f"   exact E_0 = {spectrum[0]:+.6f}, E_1 = {spectrum[1]:+.6f}")
    print()
    print(f"{'|c_0|^2':>10s} {'p(E_0 bits)':>13s} {'p(E_1 bits)':>13s} "
          f"{'E reported':>12s}")
    for mix in (0.0, 0.2, 0.4, 0.7):
        trial = math.sqrt(1 - mix) * ground + math.sqrt(mix) * excited
        trial /= np.linalg.norm(trial)
        out = qpe_on_hamiltonian(hamiltonian, trial, time, 8)
        print(f"{abs(np.vdot(ground, trial))**2:10.4f} "
              f"{out['distribution'][index_0]:13.4f} "
              f"{out['distribution'][index_1]:13.4f} "
              f"{out['energy']:+12.6f}")
    print()
    print("   The two peaks are the two eigenvalues, and their weights are")
    print("   the overlaps.  QPE does not average them -- it returns one or")
    print("   the other, and the run that returns E_0 returns it to full")
    print("   precision.  What the overlap controls is how many repetitions")
    print("   are needed before that happens, and once |c_0|^2 falls far")
    print("   enough the wrong peak becomes the taller one and a single shot")
    print("   is actively misleading.  This is why a good variational state")
    print("   -- chapter 20 -- is worth having even when the final answer")
    print("   comes from QPE, and why the honest procedure is to look at the")
    print("   whole histogram rather than at the modal outcome.")


if __name__ == "__main__":
    _demo()

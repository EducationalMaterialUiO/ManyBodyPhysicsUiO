#!/usr/bin/env python3
"""Reproducible spinless 1D Kohn--Sham teaching example.

Hamiltonian: h0 = -1/2 d²/dx² + x²/2; w(x-y)=1/sqrt((x-y)²+a²).
Atomic/oscillator units; N spinless particles occupy N distinct spatial states.
The exchange-only local functional is constructed from this uniform 1D gas.
No physical data, HF results, or exact interacting energies are fitted.

Dependencies: numpy, scipy, matplotlib. Run this file directly. All generated
files are placed beside the script, or in the directory supplied by --output-dir.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from numpy.polynomial.hermite import hermval
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import eigh
from scipy.special import k0


class UniformExchange:
    """Interpolate exchange energy per length and differentiate that interpolant.

    Fourier transform of the interaction is 2 K0(a|q|). For spinless particles,
    k_F=pi*n and Q=2*k_F, so
      e_x(n) = -1/(2*pi²) integral_0^Q (Q-q) K0(aq) dq,
      v_x(n) = -1/pi integral_0^Q K0(aq) dq.
    The logarithmic singularity of K0 at zero is integrable. Adaptive quadrature
    on t=q/Q avoids the large finite real-space cutoff of the original example.
    Hermite interpolation uses both quantities; its own derivative enters SCF.
    """

    def __init__(self, a=0.5, nmax=3.0, ntable=601):
        self.a, self.nmax, self.ntable = a, nmax, ntable
        self.nodes = np.linspace(0.0, np.sqrt(nmax), ntable) ** 2
        energy = np.zeros(ntable)
        slope = np.zeros(ntable)
        for j, n in enumerate(self.nodes[1:], 1):
            Q = 2.0 * np.pi * n
            first = quad(lambda t: (1-t)*k0(a*Q*t), 0.0, 1.0,
                         epsabs=2e-12, epsrel=2e-12)[0]
            second = quad(lambda t: k0(a*Q*t), 0.0, 1.0,
                          epsabs=2e-12, epsrel=2e-12)[0]
            energy[j] = -Q*Q*first/(2*np.pi**2)
            slope[j] = -Q*second/np.pi
        self.spline = CubicHermiteSpline(self.nodes, energy, slope,
                                        extrapolate=False)

    def _check(self, n):
        n = np.asarray(n)
        if np.any(n < -1e-13) or np.any(n > self.nmax):
            raise ValueError("Density is outside the exchange table")
        return np.maximum(n, 0.0)

    def energy_density(self, n):
        return self.spline(self._check(n))

    def potential(self, n):
        return self.spline(self._check(n), 1)


def simpson_weights(n, h):
    if n % 2 != 1:
        raise ValueError("Simpson quadrature requires an odd number of grid points")
    w = np.ones(n)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return w*h/3.0


def solve(N, exchange, nbasis=8, L=8.0, ngrid=401, alpha=0.3,
          density_tol=1e-10, energy_tol=1e-12, maxiter=500,
          initial="oscillator"):
    """Galerkin SCF in the first nbasis harmonic-oscillator functions.

    Quadrature-orthogonalize the sampled basis so that N is conserved to machine
    precision in the finite calculation. h0 is transformed by the same change
    of basis. All local, Hartree, and exchange terms use the same weights.
    Direct energy is evaluated from the output orbitals/density at EVERY step;
    the eigenvalue double-counting identity is checked only at convergence.
    """
    if not 0 < N <= nbasis:
        raise ValueError("Require 0 < N <= nbasis for spinless occupations")
    x = np.linspace(-L, L, ngrid)
    weights = simpson_weights(ngrid, x[1]-x[0])
    phi = np.empty((ngrid, nbasis))
    for j in range(nbasis):
        coef = np.zeros(j+1)
        coef[-1] = 1.0
        phi[:, j] = (hermval(x, coef)*np.exp(-x*x/2)
                      /np.sqrt(2.0**j*math.factorial(j)*np.sqrt(np.pi)))
    S = phi.T @ (weights[:, None]*phi)
    s, U = eigh(S)
    if s.min() < 1e-10:
        raise ValueError("Sampled oscillator basis is linearly dependent")
    X = (U/np.sqrt(s)) @ U.T
    B = phi @ X
    h0 = X.T @ np.diag(np.arange(nbasis)+0.5) @ X
    K = 1.0/np.sqrt((x[:, None]-x[None, :])**2+exchange.a**2)

    def density(C):
        return np.sum(np.abs(B @ C[:, :N])**2, axis=1)

    def potential_matrix(v):
        return B.T @ ((weights*v)[:, None]*B)

    def hartree(n):
        return K @ (weights*n)

    def components(C, n):
        occupied = C[:, :N]
        one = float(np.trace(occupied.T @ h0 @ occupied))
        EH = float(0.5*np.dot(weights*n, hartree(n)))
        Ex = float(np.dot(weights, exchange.energy_density(n)))
        return one, EH, Ex

    _, C = eigh(h0)
    n = density(C)
    if initial == "broadened":
        n = np.exp(-x*x/2)
        n *= N/np.dot(weights, n)
    history = []
    last_energy = np.inf
    for iteration in range(1, maxiter+1):
        F = h0 + potential_matrix(hartree(n)+exchange.potential(n))
        eps, C = eigh(F)
        nout = density(C)
        residual = nout-n
        one, EH, Ex = components(C, nout)
        energy = one+EH+Ex
        history.append(dict(iteration=iteration, energy=energy,
                            density_max=float(np.max(np.abs(residual))),
                            density_L1=float(np.dot(weights, np.abs(residual))),
                            energy_change=float(abs(energy-last_energy))))
        if history[-1]["density_max"] < density_tol and abs(energy-last_energy) < energy_tol:
            break
        # Convex mixing preserves positivity and the particle number.
        n = (1-alpha)*n + alpha*nout
        last_energy = energy
    else:
        raise RuntimeError("SCF failed to converge")

    n = nout
    vh, vx = hartree(n), exchange.potential(n)
    Ffinal = h0+potential_matrix(vh+vx)
    occupied = C[:, :N]
    one, EH, Ex = components(C, n)
    direct_energy = one+EH+Ex
    # eps were calculated from the previous input density, now agreeing to tol.
    eigenvalue_energy = float(np.sum(eps[:N])-EH+Ex-np.dot(weights*n, vx))
    orbital_residual = float(np.linalg.norm(Ffinal @ occupied-occupied*eps[:N]))
    identity_error = abs(direct_energy-eigenvalue_energy)
    P = occupied @ occupied.T
    ortho = float(np.linalg.norm(occupied.T @ occupied-np.eye(N)))
    results = dict(N=N, nbasis=nbasis, L=L, ngrid=ngrid, alpha=alpha,
                   initial=initial, iterations=iteration, energy=direct_energy,
                   one_body=one, hartree=EH, exchange=Ex,
                   hartree_plus_exchange=EH+Ex,
                   particles=float(np.dot(weights, n)),
                   peak_density=float(n.max()),
                   overlap_error=float(np.linalg.norm(S-np.eye(nbasis))),
                   orthogonality_error=ortho,
                   density_max_residual=history[-1]["density_max"],
                   density_L1_residual=history[-1]["density_L1"],
                   orbital_residual=orbital_residual,
                   commutator_residual=float(np.linalg.norm(Ffinal @ P-P @ Ffinal)),
                   eigenvalue_energy=eigenvalue_energy,
                   energy_identity_error=identity_error,
                   edge_density=float(max(n[0], n[-1])),
                   occupied_eigenvalues=eps[:N].tolist())
    assert abs(results["particles"]-N) < 2e-12
    assert ortho < 2e-12
    assert identity_error < 2e-9
    assert orbital_residual < 2e-9
    return results, x, n, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ex = UniformExchange()
    results = {"model": dict(spin="spinless", softening=0.5,
                              external_potential="x^2/2", units="oscillator / atomic",
                              exchange="1D uniform-gas exchange; no correlation",
                              exchange_table=ex.ntable), "runs": {}}
    saved = {}
    for N in (1, 2, 4):
        result, x, n, history = solve(N, ex)
        key = f"N{N}_basis8_grid401"
        results["runs"][key] = result
        saved[N] = (x, n, history)
        print(key, json.dumps(result, sort_keys=True), flush=True)
    for key, kwargs in (
        ("N4_basis8_grid801", dict(ngrid=801)),
        ("N4_basis8_L10_grid501", dict(L=10, ngrid=501)),
        ("N4_basis12_L10_grid501", dict(nbasis=12, L=10, ngrid=501)),
        ("N4_basis16_L10_grid501", dict(nbasis=16, L=10, ngrid=501)),
        ("N4_basis8_broadened_start", dict(initial="broadened")),
    ):
        result, _, _, _ = solve(4, ex, **kwargs)
        results["runs"][key] = result
        print(key, json.dumps(result, sort_keys=True), flush=True)
    refined_ex = UniformExchange(ntable=1201)
    result, _, _, _ = solve(4, refined_ex)
    results["runs"]["N4_basis8_exchange1201"] = result
    print("N4_basis8_exchange1201", json.dumps(result, sort_keys=True), flush=True)

    # An independent density-space directional derivative check of EH+Ex.
    x, n, history = saved[4]
    w = simpson_weights(x.size, x[1]-x[0])
    K = 1/np.sqrt((x[:, None]-x[None, :])**2+ex.a**2)
    eta = n*(np.cos(x)-np.dot(w*n, np.cos(x))/4)
    def eint(d):
        return 0.5*np.dot(w*d, K @ (w*d))+np.dot(w, ex.energy_density(d))
    delta = 1e-5
    analytic = float(np.dot(w*eta, K @ (w*n)+ex.potential(n)))
    numerical = float((eint(n+delta*eta)-eint(n-delta*eta))/(2*delta))
    results["functional_derivative_check"] = dict(analytic=analytic, central_difference=numerical,
                                                 absolute_error=abs(analytic-numerical), delta=delta)
    sample_n = np.array([0.05, 0.2, 0.4, 0.8, 1.5, 2.5])
    results["uniform_gas_samples"] = [dict(n=float(z), epsilon_x=float(ex.energy_density(z)/z),
                                          v_x=float(ex.potential(z))) for z in sample_n]
    results["numerical_caveat"] = (
        "All energies evaluate an approximate exchange-only functional in a finite basis. "
        "SCF convergence establishes a stationary self-consistent solution, not a global minimum. "
        "Grid refinement and a second starting density test numerical behavior; basis and functional errors remain distinct.")
    (args.output_dir/"ks_model_results.json").write_text(json.dumps(results, indent=2)+"\n")
    np.savetxt(args.output_dir/"ks_densities.dat", np.column_stack([saved[1][0], saved[1][1], saved[2][1], saved[4][1]]),
               header="x n_N1 n_N2 n_N4", comments="")
    np.savetxt(args.output_dir/"ks_scf_N4.dat", np.array([[h["iteration"], h["energy"], h["density_max"], h["density_L1"]] for h in history]),
               header="iteration energy residual_max residual_L1", comments="")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    for N, color in [(1,"#164b79"), (2,"#c57c20"), (4,"#a62835")]:
        ax[0].plot(saved[N][0], saved[N][1], color=color, label=f"N={N}")
    ax[0].set(xlabel="$x$", ylabel="$n(x)$", xlim=(-5,5))
    ax[0].legend(frameon=False)
    ax[1].semilogy([h["iteration"] for h in history], [h["density_max"] for h in history], color="#164b79")
    ax[1].set(xlabel="SCF iteration", ylabel="$\\max_x|n_{\\mathrm{out}}-n_{\\mathrm{in}}|$")
    ax[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output_dir/"ks_density_scf.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir/"ks_density_scf.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved numerical results and density/SCF plots in", args.output_dir)


if __name__ == "__main__":
    main()

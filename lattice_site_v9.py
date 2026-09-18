"""
OTOC spectrum simulation. Author: Hongyi Xiong

This is a code for complete eigenvalue decomposition of OTOCs on simple spin chains. 
There are currently minimal optimization for exact diagonalization which may be improved 
on in the future.

The spin chain models created are: Kicked-Ising Floquet system, XXZ Heisenberg chain, time-
indepdent Ising Hamiltonian, and models for Haar evolution. All tractable models are equipped
with randomized spatial fields to break any undesired translational / periodicity symmetry.

Observables so defined are fixed on 1 or 2 sites. When calculating terms of the form A(t)B, A
and B can be freely chosen as any observables defined on any site in the chain. 2-site observables
are defined so that the left-most site is specified.

The code can be parallelized in time, site, and realizations. Multiple realizations should be
ran to ensure smoothness over spatial randomness.

Due to legacy reasons the chain models are only defined on even number of qubits. These are specified
through L = N/2, where N is the length of the chain (and hence the number of qubits)

FILE STRUCTURE
Output produces real_XXX.npy, which contains a timeseries of eigenvalues across all requested sites.
U(1) conservation law of magnetization is built into the XXZ Heisenberg chain. For non-infinite
temperture states, an additional weight file is stored, as weights_XXX.npy for each corresponding
realization.

"""

import argparse
import os
import numpy as np
from scipy.linalg import expm
from scipy.stats import unitary_group
from tqdm import tqdm

# Common operators
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
P0 = np.diag([1, 0]).astype(complex)
P1 = np.diag([0, 1]).astype(complex)
_obs_rng = np.random.default_rng(0)

# Helper function for a random, dense, observable
def _random_hermitian(dim, rng, traceless=True):
    a = rng.random(size=(dim, dim)) + 1j * rng.random(size=(dim, dim))
    h = (a + a.conj().T) / 2
    return h - np.trace(h) / dim * np.eye(dim) if traceless else h

# Dictionary of observables. (OBS, #SUPPORT)
OBSERVABLES = {
    "sx": (X, 1), "sy": (Y, 1), "sz": (Z, 1),
    "sr": (_random_hermitian(2, _obs_rng, True), 1),
    # These entries are replaced by make_observables(alpha).
    "z+x": (Z, 1), "z+y": (Z, 1), "x+y": (X, 1),
    "sgx": (np.kron(P0, X), 2), 
    "sgy": (np.kron(P0, Y), 2),
    "sgz": (np.kron(P0, Z), 2), 
    "sgx_d": (np.kron(P1, X), 2),
    "sgy_d": (np.kron(P1, Y), 2), 
    "sgz_d": (np.kron(P1, Z), 2),
    "sR": (_random_hermitian(4, _obs_rng), 2),
    "sX": (np.kron(X, X), 2), 
    "sY": (np.kron(Y, Y), 2),
    "sZ": (np.kron(Z, Z), 2),
    "sXZ": (np.kron(X, Z), 2),
    "sc": (np.array([[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]], complex), 2),
    "sc_005": (np.kron(I2, X) + 0.05 * np.kron(X, X), 2),
    "sc_020": (np.kron(I2, X) + 0.2 * np.kron(X, X), 2),
    "sc_095": (np.kron(I2, X) + 0.95 * np.kron(X, X), 2),
    "sc_010": (np.kron(I2, X) + 0.1 * np.kron(X, X), 2),
    "O0": (0.5 * np.kron(X, X) + 0.5 * np.kron(Y, Y), 2),
    "O2": (0.5 * np.kron(X, X) - 0.5 * np.kron(Y, Y), 2),
}

# Helper function for custom-built observables
def make_observables(alpha):
    """Return observables, with Pauli sums normalized to unitary operators."""
    observables = OBSERVABLES.copy()
    beta1 = np.cos(alpha * np.pi/2)
    beta2 = np.sin(alpha * np.pi/2)
    observables.update({
        "z+x": (beta1 * Z + beta2 * X, 1),
        "z+y": (beta1 * Z + beta2 * Y, 1),
        "x+y": (beta1 * X + beta2 * Y, 1),
    })
    return observables

# Arguments
def parse_args():
    p = argparse.ArgumentParser(description="OTOC spectrum simulation")
    p.add_argument("--ortho_obs", choices=["Y", "N", "H"], default="N")
    p.add_argument("--alpha", type=float, default=0.1, help="Weight of the second Pauli in sum observables")
    for name, default in [("hx", .9), ("hz", .8), ("h2", .9), ("g", .2), ("tau", 1.), ("delta", 1.5)]:
        p.add_argument("--" + name, type=float, default=default, help="Chain parameters")
    p.add_argument("--doder", "--disorder", dest="disorder", choices=["X", "Z"], default="Z")
    p.add_argument("--L", type=int, default=5, help="Half-length of the chain")
    p.add_argument("-D", "--Dynamics", choices=["Ising", "Kicked", "GOE", "UHaar", "MHaar", "Circuit", "Heisenberg", "shell-haar"], default="Ising")
    p.add_argument("-P", "--Periodic", choices=["Y", "N"], default="N")
    for name, default in [("tstart", 0.), ("tend", 5.)]: 
        p.add_argument("--" + name, type=float, default=default)
    p.add_argument("--s_start", type=int, default=1, help="Site-scanning start loc (inclusive)")
    p.add_argument("--s_end", type=int, help="Site-scanning end loc (inclusive)")
    p.add_argument("--t_type", type=str, choices=["Linear", "Log"], default="Linear", help="Spacings for time array")
    p.add_argument("--N", type=float, default=1, help="Discretization timestep")
    p.add_argument("--n_reals", type=int, default=1, help="Number of realizations")
    p.add_argument("--real_start", type=int, default=0, help="Start with realization #") 
    p.add_argument("--real_end", type=int, help="End on realization #")
    p.add_argument("--obsA", choices=OBSERVABLES, default="sx", help="Observable A")
    p.add_argument("--obsB", choices=OBSERVABLES, default="sx", help="Observable B")
    p.add_argument("--output_dir", default="results", help="Output Directory")
    p.add_argument("--n2", type=int, default=5, help="Fix the location of observable A")
    p.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    p.add_argument("--mu", type=float, default=0.0, help="Thermal parameter in rho = exp(-mu M) / Z")
    p.add_argument("--beta", type=float, default=0.0, help="Thermal parameter in rho = exp(-beta H) / Z")
    p.add_argument("--m_operator", choices=["total_z"], default="total_z", help="Conserved operator used to construct the density matrix")
    p.add_argument(
        "--energy_density", type=float, default=None,
        help="Micro-canonical shell center in energy per site (Ising only)",
    )
    p.add_argument(
        "--energy_width", type=float, default=None,
        help="Micro-canonical shell full width in energy density (Ising only)",
    )

    return p.parse_args()

# Helper function to expand support of local gates onto the full Hilbert space
def local_op(op, support, site, nsites):
    if support not in (1, 2) or site < 0 or site + support > nsites: 
        raise ValueError("Invalid local operator placement")
    return np.kron(np.kron(np.eye(2 ** site, dtype=complex), op), np.eye(2 ** (nsites - site - support), dtype=complex))

def precompute_site_ops(a, b, na, nb, nsites):
    return ([local_op(X, 1, i, nsites) for i in range(nsites)],
            [local_op(Y, 1, i, nsites) for i in range(nsites)],
            [local_op(Z, 1, i, nsites) for i in range(nsites)],
            [local_op(a, na, i, nsites) for i in range(nsites-na+1)],
            [local_op(b, nb, i, nsites) for i in range(nsites-nb+1)])

# Ising Hamiltonian terms (also applied to the kicked Ising chain)
def _ising_terms(args, rng, xops, zops):
    zero = np.zeros_like(xops[0])
    n = len(xops)
    # Interaction term
    hzz = sum((-args.tau * zops[i] @ zops[i+1] for i in range(n-1)), start=zero.copy())
    if args.Periodic == "Y": 
        hzz += -args.tau * zops[0] @ zops[-1]

    # Magnetic fields
    hx = sum((-args.tau * args.hx * x for x in xops), start=zero.copy())
    hz = sum((-args.tau * args.hz * z for z in zops), start=zero.copy())

    # Parity-breaking disorder
    disorder = rng.normal(size=n)
    ops = zops if args.disorder == "Z" else xops
    hd = sum((args.tau * args.g * ds * op for ds, op in zip(disorder, ops)), start=zero.copy())
    return hzz, hx, hz, hd

# XXZ Heisenberg chain
def _heisenberg_terms(args, rng, xops, yops, zops):
    zero = np.zeros_like(xops[0])
    n = len(xops)
    # Interaction term
    hSS = sum((-args.tau * (args.delta * zops[i] @ zops[i+1] + xops[i] @ xops[i+1] + yops[i] @ yops[i+1]) for i in range(n-1)), start=zero.copy())
    if args.Periodic == "Y": 
        hSS += -args.tau * (args.delta * zops[0] @ zops[-1] + xops[0] @ xops[-1] + yops[0] @ yops[-1])

    hz  = sum((-args.tau * args.hz * z  * (-1)**(i) for (i, z) in enumerate(zops)), start=zero.copy())

    h2  = sum((-args.tau * args.h2 * z  * (-1)**(i) for (i, z) in enumerate(zops)), start=zero.copy())

    disorder = rng.normal(size=n)
    ops = zops
    hd = sum((args.tau * args.g * ds * op for ds, op in zip(disorder, ops)), start=zero.copy())
    return hSS, h2, hz, hd

def magnetization_indices(n_sites, magnetization):
    """Return computational-basis indices with total magnetization M."""
    if (n_sites + magnetization) % 2 != 0:
        return np.array([], dtype=int)
    n_up = (n_sites + magnetization) // 2
    if n_up < 0 or n_up > n_sites:
        return np.array([], dtype=int)
    return np.array([state for state in range(2 ** n_sites) if state.bit_count() == n_up], dtype=int)


def restrict_to_magnetization(matrix, n_sites, magnetization):
    indices = magnetization_indices(n_sites, magnetization)
    return matrix[np.ix_(indices, indices)]


def magnetization_block_haar(n_sites, rng):
    """Return a unitary with independent Haar blocks at fixed magnetization."""
    dim = 2 ** n_sites
    matrix = np.zeros((dim, dim), dtype=complex)
    for magnetization in range(-n_sites, n_sites + 1, 2):
        indices = magnetization_indices(n_sites, magnetization)
        block = unitary_group.rvs(len(indices), random_state=rng)
        matrix[np.ix_(indices, indices)] = block
    return matrix


def _energy_shell_labels(values, nsites, energy_width):
    """Return zero-centered finite-width energy-density shell labels."""
    if nsites <= 0 or energy_width <= 0:
        raise ValueError("Require nsites > 0 and energy_width > 0")
    energy_density = np.real(values) / nsites
    return np.floor(energy_density / energy_width + 0.5).astype(int)


def shell_haar_unitary(hamiltonian, nsites, energy_width, rng, return_metadata=False):
    """Construct independent Haar blocks in finite-width H-energy shells.

    Returns the shell-Haar unitary, shell labels in ascending-H eigenbasis
    order, and the corresponding block dimensions.
    """
    energies, vectors = np.linalg.eigh(hamiltonian)
    labels = _energy_shell_labels(energies, nsites, energy_width)
    shell_unitary = np.zeros_like(hamiltonian, dtype=complex)
    dimensions = []
    for shell in np.unique(labels):
        indices = np.flatnonzero(labels == shell)
        block = unitary_group.rvs(len(indices), random_state=rng)
        shell_unitary[np.ix_(indices, indices)] = block
        dimensions.append((int(shell), int(len(indices))))
    unitary = vectors @ shell_unitary @ vectors.conj().T
    if return_metadata:
        return unitary, {
            "hamiltonian_energies": energies,
            "hamiltonian_vectors": vectors,
            "shell_labels": labels,
            "shell_dimensions": dimensions,
            "shell_center_energies": np.asarray(
                [shell * energy_width * nsites for shell, _ in dimensions],
                dtype=float,
            ),
        }
    return unitary, labels, dimensions


# Random 2-qubit circuit gate
def _rand_gate(rng):
    g = np.zeros((4, 4), complex)
    g[0, 0] = np.exp(2j*np.pi*rng.random())
    g[3, 3] = np.exp(2j*np.pi*rng.random())
    g[1:3, 1:3] = unitary_group.rvs(2, random_state=rng)
    return g

# Circuit construction
def _circuit(rng, L):
    even = _rand_gate(rng)
    for _ in range(L-1): 
        even = np.kron(even, _rand_gate(rng))
    odd = I2
    for _ in range(L-1): 
        odd = np.kron(odd, _rand_gate(rng))
    return np.kron(odd, I2) @ even

# Building evolution
def build_dynamics(args, dim, rng, xops, yops, zops, return_metadata=False):

    if args.Dynamics in ("Ising", "Kicked", "shell-haar"):
        hzz, hx, hz, hd = _ising_terms(args, rng, xops, zops)
        if args.Dynamics == "Ising": 
            return hzz + hx + hz + hd, False
        elif args.Dynamics == "shell-haar":
            hamiltonian = hzz + hx + hz + hd
            if return_metadata:
                dynamics, shell_metadata = shell_haar_unitary(
                    hamiltonian, len(xops), args.energy_width, rng,
                    return_metadata=True,
                )
                dimensions = shell_metadata["shell_dimensions"]
            else:
                dynamics, _, dimensions = shell_haar_unitary(
                    hamiltonian, len(xops), args.energy_width, rng
                )
            print(
                f"shell-Haar: width={args.energy_width}, shells={len(dimensions)}, "
                f"block dimensions={[size for _, size in dimensions]}",
                flush=True,
            )
            if return_metadata:
                return dynamics, True, shell_metadata
            return dynamics, True
        else:
            if args.disorder == "Z":
                return expm(-1j*(hzz+hz+hd)) @ expm(-1j*hx), True
            else:
                return expm(-1j*(hzz+hz)) @ expm(-1j*(hx+hd)), True
    elif args.Dynamics == "Heisenberg":
        hSS, h2, hz, hd = _heisenberg_terms(args, rng, xops, yops, zops)
        if args.disorder == "Z":
            return expm(-1j*(hSS+hz+hd)) @ expm(-1j*h2), True
        else:
            return expm(-1j*(hSS+hz)) @ expm(-1j*(h2+hd)), True
    elif args.Dynamics == "GOE":
        raw = rng.normal(size=(dim, dim))
        return (raw + raw.T) / np.sqrt(2 * dim), False
    
    elif args.Dynamics == "UHaar": 
        return unitary_group.rvs(dim=dim, random_state=rng), True
    elif args.Dynamics == "MHaar":
        return magnetization_block_haar(int(np.log2(dim)), rng), True
    else:
        return _circuit(rng, args.L), True

def _diagonalize(dynamics, unitary):
    if unitary:
        values, vectors = np.linalg.eig(dynamics)
        return values, vectors, np.linalg.inv(vectors), "unitary"
    else:
        values, vectors = np.linalg.eigh(dynamics)
        return values, vectors, vectors.conj().T, "hamiltonian"

def _phase_product(a, b, values, t, mode):
    if mode == "hamiltonian":
        phase = np.exp(-1j * values * t)
    else:
        phase = values**t
    return (a * (np.conjugate(phase)[:, None] * phase[None, :])) @ b

def _build_conserved_operator(m_operator, zops):
    if m_operator == "total_z":
        return sum(zops, start=np.zeros_like(zops[0]))
    raise ValueError(f"Unsupported conserved operator: {m_operator}")


def _thermal_density_matrix(m_operator, mu, zops):
    M = _build_conserved_operator(m_operator, zops)
    rho = expm(-mu * M)
    rho /= np.trace(rho)
    return M, rho


def _validate_energy_shell_args(args):
    energy_density = getattr(args, "energy_density", None)
    energy_width = getattr(args, "energy_width", None)
    beta = getattr(args, "beta", 0.0)
    if beta != 0.0 and args.Dynamics not in ("Ising", "shell-haar"):
        raise ValueError("Nonzero beta is supported only for Ising and shell-haar dynamics")
    if beta != 0.0 and energy_density is not None:
        raise ValueError("Nonzero beta cannot be combined with --energy_density")
    if args.Dynamics == "shell-haar":
        if energy_width is None:
            raise ValueError("shell-haar requires --energy_width")
        if energy_width <= 0:
            raise ValueError("Require --energy_width > 0")
        if energy_density is not None and args.mu != 0.0:
            raise ValueError("Micro-canonical energy shells cannot be combined with nonzero --mu")
        return False
    supplied = (energy_density is not None, energy_width is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("--energy_density and --energy_width must be supplied together")
    if energy_width is not None and energy_width <= 0:
        raise ValueError("Require --energy_width > 0")
    if not any(supplied):
        return False
    if args.Dynamics != "Ising":
        raise ValueError("Micro-canonical energy shells are supported only for -D Ising")
    if args.mu != 0.0:
        raise ValueError("Micro-canonical energy shells cannot be combined with nonzero --mu")
    return True


def _microcanonical_weights(values, nsites, energy_density, energy_width, realization=None):
    """Return normalized eigenstate weights for an energy-density shell."""
    target_energy = energy_density * nsites
    half_width = energy_width * nsites / 2.0
    lower = target_energy - half_width
    upper = target_energy + half_width
    selected = (np.real(values) >= lower) & (np.real(values) <= upper)
    count = int(np.count_nonzero(selected))
    if count == 0:
        nearest = np.sort(np.abs(np.real(values) - target_energy))[:3]
        nearest_energies = np.real(values)[np.argsort(np.abs(np.real(values) - target_energy))[:3]]
        label = f" for realization {realization}" if realization is not None else ""
        raise ValueError(
            f"Energy shell is empty{label}: target energy {target_energy:.12g}, "
            f"interval [{lower:.12g}, {upper:.12g}], nearest energies "
            f"{nearest_energies.tolist()} (distances {nearest.tolist()})"
        )
    weights = np.zeros(len(values), dtype=float)
    weights[selected] = 1.0 / count
    return weights, target_energy, lower, upper


def _microcanonical_projector(
    energies, vectors, nsites, energy_density, energy_width, realization=None
):
    """Return a normalized micro-canonical projector in the original basis."""
    weights, target_energy, lower, upper = _microcanonical_weights(
        energies, nsites, energy_density, energy_width, realization
    )
    projector = vectors @ np.diag(weights) @ vectors.conj().T
    return projector, int(np.count_nonzero(weights)), target_energy, lower, upper


def _shell_haar_thermal_density(shell_metadata, nsites, energy_width, beta):
    """Build a coarse-grained beta-H density matrix for shell-Haar dynamics."""
    labels = shell_metadata["shell_labels"]
    vectors = shell_metadata["hamiltonian_vectors"]
    shells, dimensions = np.unique(labels, return_counts=True)
    shell_energies = shells * energy_width * nsites
    shell_factors = np.exp(-beta * shell_energies)
    partition_function = np.sum(dimensions * shell_factors)
    state_weights = np.empty(len(labels), dtype=float)
    for shell, factor in zip(shells, shell_factors):
        state_weights[labels == shell] = factor / partition_function
    rho = vectors @ np.diag(state_weights) @ vectors.conj().T
    return rho, shell_energies, dimensions, shell_factors, partition_function


def _check_conservation(M, dynamics, unitary, tolerance=1e-7):
    if unitary:
        residual = dynamics @ M @ dynamics.conj().T - M
    else:
        residual = dynamics @ M - M @ dynamics
    scale = max(1.0, np.linalg.norm(M))
    relative_error = np.linalg.norm(residual) / scale
    if relative_error > tolerance:
        kind = "unitary conjugation" if unitary else "commutator"
        raise ValueError(
            f"Selected m_operator is not conserved: {kind} relative error "
            f"is {relative_error:.3e} (tolerance {tolerance:.1e})"
        )


def run_realization(
    args, dynamics, unitary, rho, aops, b0, times, site_start,
    nsites=None, realization=None, return_shell_count=False,
    analysis_rho=None, analysis_shell_count=None,
):
    """Compute correlator eigenvalues and their spectral coefficients.

    For normal correlators the weights are real and nonnegative.  For
    non-normal correlators they are biorthogonal spectral coefficients and
    may be signed or complex.
    """
    values, P, Pinv, mode = _diagonalize(dynamics, unitary)
    shell_count = None
    energy_density = getattr(args, "energy_density", None)
    energy_width = getattr(args, "energy_width", None)
    if analysis_rho is not None:
        rho_eigenbasis = Pinv @ analysis_rho @ P
        shell_count = analysis_shell_count
    elif energy_density is not None:
        if mode != "hamiltonian":
            raise ValueError("Micro-canonical energy shells require Hamiltonian dynamics")
        if nsites is None:
            raise ValueError("nsites is required for a micro-canonical ensemble")
        shell_weights, target_energy, lower, upper = _microcanonical_weights(
            values, nsites, energy_density, energy_width, realization
        )
        shell_count = int(np.count_nonzero(shell_weights))
        rho_eigenbasis = np.diag(shell_weights)
        print(
            f"Energy shell: target={target_energy:.8g}, interval="
            f"[{lower:.8g}, {upper:.8g}], selected={shell_count}", flush=True,
        )
    else:
        if args.beta ==0.0:
            rho_eigenbasis = Pinv @ rho @ P
        elif args.beta !=0.0 and args.Dynamics == "Ising" and mode == "hamiltonian":
            rho_thermal = np.diag(np.exp(-args.beta * values))
            rho_eigenbasis = rho_thermal / np.sum(np.exp(-args.beta * values))
        else:
            raise ValueError("Nonzero beta is only supported for Hamiltonian dynamics")
        
    b = Pinv @ b0 @ P
    correlators = np.empty((len(aops), len(times), dynamics.shape[0]), complex)
    thermal_weights = np.empty_like(correlators)
    for i, a0 in enumerate(aops):
        print(f"Computing n1={site_start + i}", flush=True)
        a = Pinv @ a0 @ P
        if args.ortho_obs == "Y": 
            a -= np.diag(np.diag(a))
        elif args.ortho_obs == "H": 
            a = np.diag(np.diag(a))
        for j, t in enumerate(tqdm(times, leave=False)): 
            product = _phase_product(a, b, values, t, mode)
            lambdas, vectors = np.linalg.eig(product)
            correlators[i, j, :] = lambdas
            inverse_vectors = np.linalg.inv(vectors)
            weights = np.diag(inverse_vectors @ rho_eigenbasis @ vectors)
            thermal_weights[i, j, :] = weights / np.sum(weights)
    maximum_imaginary = np.max(np.abs(np.imag(thermal_weights)))
    minimum_real = np.min(np.real(thermal_weights))
    total_weights = np.sum(thermal_weights, axis=2)
    maximum_normalization_error = np.max(np.abs(total_weights - 1.0))
    print(
        f"Weight diagnostics: max |Im(w)| = {maximum_imaginary:.3e}, "
        f"min Re(w) = {minimum_real:.3e}, "
        f"max |sum(w) - 1| = {maximum_normalization_error:.3e}",
        flush=True,
    )
    if return_shell_count:
        return correlators, thermal_weights, shell_count
    return correlators, thermal_weights


def main():
    args = parse_args()
    _validate_energy_shell_args(args)
    shell_haar_mode = args.Dynamics == "shell-haar"
    analysis_shell_mode = args.energy_density is not None
    if not 0 <= args.alpha <= 1:
        raise ValueError("Require 0 <= alpha <= 1")
    observables = make_observables(args.alpha)
    nsites = 2*args.L
    dim = 2**nsites

    if args.L < 1 or args.N <= 0 or args.tend < args.tstart: 
        raise ValueError("Require L >= 1, N > 0, tend >= tstart")
    a, na = observables[args.obsA]
    b, nb = observables[args.obsB]
    if args.s_end is None:
        send = nsites
    else:
        send = args.s_end

    if not 1 <= args.s_start <= send <= nsites-na+1: 
        raise ValueError("Invalid A site range")
    if not 1 <= args.n2 <= nsites-nb+1: 
        raise ValueError("Invalid B site")
    if args.real_end is None:
        rend = args.n_reals
    else:
        rend = min(args.real_end, args.n_reals)
    if args.real_start < 0 or rend < args.real_start: 
        raise ValueError("Invalid realization range")
    if args.t_type=="Linear":
        times = np.arange(args.tstart, args.tend, args.N)
    else:
        times_log = np.logspace(np.log10(args.tstart), np.log10(args.tend), int(args.N)).astype(int)
        times = []
        for i in range(1, len(times_log)):
            if times_log[i]!=times_log[i-1]:
                times.append(times_log[i-1])
        times = np.asarray(times)

    # Info Diagnostics
    print("\n" + "=" * 64, flush=True)
    print("OTOC spectrum simulation", flush=True)
    print("=" * 64, flush=True)
    print(f"Observables: A = {args.obsA} ({na}-site), B = {args.obsB} ({nb}-site)", flush=True)
    print(f"Sum-observable alpha: {args.alpha} (second-Pauli weight)", flush=True)
    print(f"B location: site {args.n2}", flush=True)
    print(f"A sites: {args.s_start} through {send}", flush=True)
    print(f"Chain: {nsites} sites ({args.L} unit cells), Hilbert-space dimension = {dim}", flush=True)
    print(f"Dynamics: {args.Dynamics}", flush=True)
    print(f"Thermal parameter: mu = {args.mu}, M = {args.m_operator}", flush=True)
    print(f"Thermal parameter: beta = {args.beta}", flush=True)
    if analysis_shell_mode:
        print(
            f"Micro-canonical shell: energy density = {args.energy_density}, "
            f"width = {args.energy_width}", flush=True,
        )
    if args.Dynamics in ("Ising", "Kicked", "Heisenberg", "shell-haar"):
        print(
            f"  tau = {args.tau}, hx = {args.hx}, hz = {args.hz}, h2 = {args.h2}, delta = {args.delta}, "
            f"disorder strength = {args.g}, disorder axis = {args.disorder}, "
            f"boundary conditions = {'periodic' if args.Periodic == 'Y' else 'open'}",
            flush=True,
        )
        if args.Dynamics == "Kicked":
            print("  evolution: kicked (integer time steps)", flush=True)
        elif args.Dynamics == "Heisenberg":
            print("  evolution: kicked Heisenberg (using h2)", flush=True)
        elif args.Dynamics == "shell-haar":
            print("  evolution: shell-Haar unitary in Ising energy shells", flush=True)
        else:
            print("  evolution: continuous Hamiltonian", flush=True)
    elif args.Dynamics == "GOE":
        print("  real-symmetric GOE Hamiltonian", flush=True)
    elif args.Dynamics == "UHaar":
        print("  Haar-random unitary", flush=True)
    elif args.Dynamics == "MHaar":
        print("  independent Haar-random unitaries in each magnetization block", flush=True)
    elif args.Dynamics == "Circuit":
        print("  random brickwork circuit", flush=True)
    if args.t_type=="Linear":
        print(f"Time range: {args.tstart} to {args.tend} with step {args.N}", flush=True)
    else:
        print(f"Time range: {args.tstart} to {args.tend} with Log scale and {args.N} time points", flush=True)
        print(times, flush=True)
    print(f"Realizations: {args.real_start} through {rend - 1}", flush=True)
    if args.ortho_obs != "N":
        description = "orthogonal/off-diagonal component" if args.ortho_obs == "Y" else "parallel/diagonal component"
        print(f"A decomposition: {description}", flush=True)
    print("=" * 64 + "\n", flush=True)

    # Simulations

    os.makedirs(args.output_dir, exist_ok=True)

    xops, yops, zops, aops, bops = precompute_site_ops(a, b, na, nb, nsites)
    b0 = bops[args.n2-1]

    M, rho = _thermal_density_matrix(args.m_operator, args.mu, zops)

    shell_counts = []
    shell_thermal_partitions = []
    for real in range(args.real_start, rend):
        dynamics_result = build_dynamics(
            args, dim, np.random.default_rng(args.seed+real), xops, yops, zops,
            return_metadata=shell_haar_mode,
        )
        if shell_haar_mode:
            dynamics, unitary, shell_metadata = dynamics_result
        else:
            dynamics, unitary = dynamics_result
        if args.mu != 0.0:
            _check_conservation(M, dynamics, unitary)
        analysis_rho = None
        analysis_shell_count = None
        if shell_haar_mode and args.beta != 0.0:
            (
                analysis_rho, shell_energies, shell_dimensions,
                shell_factors, partition_function,
            ) = _shell_haar_thermal_density(
                shell_metadata, nsites, args.energy_width, args.beta
            )
            shell_thermal_partitions.append(
                (shell_energies, shell_dimensions, shell_factors, partition_function)
            )
            print(
                f"shell-Haar beta ensemble: beta={args.beta}, "
                f"shell energies={shell_energies.tolist()}, "
                f"partition function={partition_function:.8g}",
                flush=True,
            )
        if shell_haar_mode and args.energy_density is not None:
            (
                analysis_rho, analysis_shell_count, target_energy,
                lower, upper,
            ) = _microcanonical_projector(
                shell_metadata["hamiltonian_energies"],
                shell_metadata["hamiltonian_vectors"],
                nsites, args.energy_density, args.energy_width, real,
            )
            print(
                f"Analysis shell: target={target_energy:.8g}, interval="
                f"[{lower:.8g}, {upper:.8g}], selected={analysis_shell_count}",
                flush=True,
            )
        result, weights, shell_count = run_realization(
            args, dynamics, unitary, rho, aops[args.s_start-1:send],
            b0, times, args.s_start, nsites=nsites, realization=real,
            return_shell_count=True,
            analysis_rho=analysis_rho,
            analysis_shell_count=analysis_shell_count,
        )
        filename = os.path.join(args.output_dir, f"real_{real:03d}.npy")
        np.save(filename, result)
        print(f"Saved realization {real} to {filename}", flush=True)
        if args.mu != 0.0 or analysis_shell_mode or args.beta != 0.0:
            weights_filename = os.path.join(args.output_dir, f"weights_{real:03d}.npy")
            np.save(weights_filename, weights)
            print(f"Saved weights {real} to {weights_filename}", flush=True)
        if analysis_shell_mode:
            shell_counts.append(shell_count)

    metadata = dict(vars(args), times=times)
    if analysis_shell_mode:
        metadata.update(
            shell_target_energy=args.energy_density * nsites,
            shell_width_energy=args.energy_width * nsites,
            shell_selected_counts=np.asarray(shell_counts, dtype=int),
        )
    if shell_haar_mode and args.beta != 0.0:
        partition_metadata = np.empty(len(shell_thermal_partitions), dtype=object)
        partition_metadata[:] = shell_thermal_partitions
        metadata["shell_haar_beta_partitions"] = partition_metadata
    np.savez(os.path.join(args.output_dir, "parameters.npz"), **metadata)

if __name__ == "__main__": 
    main()

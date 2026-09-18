"""
OTOC spectrum simulation for Ising, kicked, random-matrix, and circuit dynamics.
Ryan (Hongyi Xiong)

"""
import argparse
import os
import numpy as np
from scipy.linalg import eigvals, expm
from scipy.stats import unitary_group
from tqdm import tqdm

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
P0 = np.diag([1, 0]).astype(complex)
P1 = np.diag([0, 1]).astype(complex)
_obs_rng = np.random.default_rng(0)

def _random_hermitian(dim, rng, traceless=True):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    h = (a + a.conj().T) / 2
    return h - np.trace(h) / dim * np.eye(dim) if traceless else h

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
    "sc": (np.array([[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]], complex), 2),
}


def make_observables(alpha):
    """Return observables, with Pauli sums normalized to unitary operators."""
    observables = OBSERVABLES.copy()
    beta = np.sqrt(1 - alpha**2)
    observables.update({
        "z+x": (beta * Z + alpha * X, 1),
        "z+y": (beta * Z + alpha * Y, 1),
        "x+y": (beta * X + alpha * Y, 1),
    })
    return observables

# Arguments
def parse_args():
    p = argparse.ArgumentParser(description="OTOC spectrum simulation")
    p.add_argument("--ortho_obs", choices=["Y", "N", "H"], default="N")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Weight of the second Pauli in sum observables")
    for name, default in [("hx", .9), ("hz", .8), ("h2", .9), ("g", .2), ("tau", 1.), ("delta", 1.5)]:
        p.add_argument("--" + name, type=float, default=default)

    p.add_argument("--doder", "--disorder", dest="disorder", choices=["X", "Z"], default="Z")
    p.add_argument("--L", type=int, default=5)
    p.add_argument("-D", "--Dynamics", choices=["Ising", "Kicked", "GOE", "UHaar", "Circuit", "Heisenberg"], default="Ising")
    p.add_argument("-P", "--Periodic", choices=["Y", "N"], default="N")

    for name, default in [("tstart", 0.), ("tend", 5.)]: 
        p.add_argument("--" + name, type=float, default=default)

    p.add_argument("--s_start", type=int, default=1)
    p.add_argument("--s_end", type=int)

    p.add_argument("--N", type=float, default=1, help="Discretization timestep")
    p.add_argument("--n_reals", type=int, default=1)
    p.add_argument("--real_start", type=int, default=0) 
    p.add_argument("--real_end", type=int)
    p.add_argument("--obsA", choices=OBSERVABLES, default="sx")
    p.add_argument("--obsB", choices=OBSERVABLES, default="sx")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--n2", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)

    return p.parse_args()

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
def build_dynamics(args, dim, rng, xops, yops, zops):

    if args.Dynamics in ("Ising", "Kicked"):
        hzz, hx, hz, hd = _ising_terms(args, rng, xops, zops)
        if args.Dynamics == "Ising": 
            return hzz + hx + hz + hd, False
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

def run_realization(args, dynamics, unitary, aops, b0, times, site_start):
    values, P, Pinv, mode = _diagonalize(dynamics, unitary)
    b = Pinv @ b0 @ P
    correlators = np.empty((len(aops), len(times), dynamics.shape[0]), complex)
    for i, a0 in enumerate(aops):
        print(f"Computing n1={site_start + i}", flush=True)
        a = Pinv @ a0 @ P
        if args.ortho_obs == "Y": 
            a -= np.diag(np.diag(a))
        elif args.ortho_obs == "H": 
            a = np.diag(np.diag(a))
        for j, t in enumerate(tqdm(times, leave=False)): 
            correlators[i, j, :] = eigvals(_phase_product(a, b, values, t, mode), check_finite=False)
    return correlators


def main():
    args = parse_args()
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
    
    times = np.arange(args.tstart, args.tend, args.N)

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
    if args.Dynamics in ("Ising", "Kicked", "Heisenberg"):
        print(
            f"  tau = {args.tau}, hx = {args.hx}, hz = {args.hz}, h2 = {args.h2}, delta = {args.delta}"
            f"disorder strength = {args.g}, disorder axis = {args.disorder}, "
            f"boundary conditions = {'periodic' if args.Periodic == 'Y' else 'open'}",
            flush=True,
        )
        if args.Dynamics == "Kicked":
            print("  evolution: kicked (integer time steps)", flush=True)
        elif args.Dynamics == "Heisenberg":
            print("  evolution: kicked Heisenberg (using h2)", flush=True)
        else:
            print("  evolution: continuous Hamiltonian", flush=True)
    elif args.Dynamics == "GOE":
        print("  real-symmetric GOE Hamiltonian", flush=True)
    elif args.Dynamics == "UHaar":
        print("  Haar-random unitary", flush=True)
    elif args.Dynamics == "Circuit":
        print("  random brickwork circuit", flush=True)
    print(f"Time range: {args.tstart} to {args.tend} with step {args.N}", flush=True)
    print(f"Realizations: {args.real_start} through {rend - 1}", flush=True)
    if args.ortho_obs != "N":
        description = "orthogonal/off-diagonal component" if args.ortho_obs == "Y" else "parallel/diagonal component"
        print(f"A decomposition: {description}", flush=True)
    print("=" * 64 + "\n", flush=True)

    # Simulations

    os.makedirs(args.output_dir, exist_ok=True)

    xops, yops, zops, aops, bops = precompute_site_ops(a, b, na, nb, nsites)
    b0 = bops[args.n2-1]

    np.savez(os.path.join(args.output_dir, "parameters.npz"), **vars(args), times=times)

    for real in range(args.real_start, rend):
        dynamics, unitary = build_dynamics(args, dim, np.random.default_rng(args.seed+real), xops, yops, zops)
        result = run_realization(
            args, dynamics, unitary, aops[args.s_start-1:send], b0, times, args.s_start
        )
        filename = os.path.join(args.output_dir, f"real_{real:03d}.npy")
        np.save(filename, result); print(f"Saved realization {real} to {filename}", flush=True)

if __name__ == "__main__": 
    main()

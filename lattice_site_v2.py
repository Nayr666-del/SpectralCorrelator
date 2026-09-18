import numpy as np
from scipy.linalg import eigvals, expm
import os
import argparse
from tqdm import tqdm


# Gates
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

# Pauli observables
sx = np.array([[0, 1],
               [1, 0]], dtype=complex)
sz = np.array([[1, 0],
               [0, -1]], dtype=complex)
sy = np.array([[0, -1.j],
               [1.j, 0]], dtype=complex)

observables = {
    "sx": sx,
    "sy": sy,
    "sz": sz,
}


# Ising parameters

def parse_args():
    parser = argparse.ArgumentParser(description="Ising Hamiltonian parameters")
    parser.add_argument("--hx", type=float, default=-1.05, help="Transverse field strength")
    parser.add_argument("--hz", type=float, default=0.5, help="Longitudinal field strength")
    parser.add_argument("--g", type=float, default=0.2, help="Disorder strength")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling strength")
    parser.add_argument("--L", type=int, default=5, help="Half the length of the chain")
    parser.add_argument("-D", "--Dynamics", default="Ising", choices=["Ising", "Kicked", "GOE", "GUE"], help="Dynamics type (default: Ising)")

    parser.add_argument("--tstart", type=float, default=0.0, help="Start time for simulation")
    parser.add_argument("--tend", type=float, default=5.0, help="End time for simulation")
    parser.add_argument("--N", type=int, default=5, help="Number of discretization steps between integer timesteps")

    parser.add_argument("--n_reals", type=int, default=1, help="Number of realizations for disorder averaging")
    parser.add_argument("--real_start", type=int, default=0, help="Starting index for realizations")
    parser.add_argument("--real_end", type=int, default=None, help="Ending index for realizations")

    parser.add_argument("--obsA", type=str, default="sx", help="Observable to compute correlators for (sx, sy, sz)")
    parser.add_argument("--obsB", type=str, default="sx", help="Observable to compute correlators for (sx, sy, sz)")

    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for the spectrum state")
    parser.add_argument("--n2", type=int, default=5, help="Site index for operator B0")

    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    return parser.parse_args()


# Small helper: place a 2x2 operator on one site of a 2L-site chain

def local_op(op, site, nqubits):
    return np.kron(
        np.kron(np.eye(2 ** site, dtype=complex), op),
        np.eye(2 ** (nqubits - site - 1), dtype=complex),
    )


# Precompute all single-site operators once

def precompute_site_ops(obsA, obsB, nqubits):
    X_ops = [local_op(X, i, nqubits) for i in range(nqubits)]
    Z_ops = [local_op(Z, i, nqubits) for i in range(nqubits)]
    A_ops = [local_op(obsA, i, nqubits) for i in range(nqubits)]
    B_ops = [local_op(obsB, i, nqubits) for i in range(nqubits)]
    return X_ops, Z_ops, A_ops, B_ops


# Ising Hamiltonian

def Ising_Hamiltonian(L, dim, rng, J, hx, hz, g, X_ops=None, Z_ops=None):
    nqubits = 2 * L

    if X_ops is None or Z_ops is None:
        X_ops, Z_ops, _, _ = precompute_site_ops(X, Z, nqubits)

    HZZ = np.zeros((dim, dim), dtype=complex)
    HX = np.zeros((dim, dim), dtype=complex)
    HZ = np.zeros((dim, dim), dtype=complex)
    HD = np.zeros((dim, dim), dtype=complex)

    for i in range(nqubits - 1):
        HZZ += -J * (Z_ops[i] @ Z_ops[i + 1])

    for i in range(nqubits):
        HX += -hx * X_ops[i]
        HZ += -hz * Z_ops[i]

    if g != 0:
        disorder = rng.normal(size=nqubits)
        for i in range(nqubits):
            HD += g * disorder[i] * Z_ops[i]

    H = HZZ + HX + HZ + HD
    return H

def Kicked_Ising_Unitary(L, dim, rng, J, hx, hz, g, X_ops=None, Z_ops=None):
    nqubits = 2 * L

    if X_ops is None or Z_ops is None:
        X_ops, Z_ops, _, _ = precompute_site_ops(X, Z, nqubits)

    HZZ = np.zeros((dim, dim), dtype=complex)
    HX = np.zeros((dim, dim), dtype=complex)
    HZ = np.zeros((dim, dim), dtype=complex)
    HD = np.zeros((dim, dim), dtype=complex)

    for i in range(nqubits - 1):
        HZZ += -J * (Z_ops[i] @ Z_ops[i + 1])

    for i in range(nqubits):
        HX += -hx * X_ops[i]
        HZ += -hz * Z_ops[i]

    if g != 0:
        disorder = rng.normal(size=nqubits)
        for i in range(nqubits):
            HD += g * disorder[i] * Z_ops[i]

    U = expm(-1j * (HZZ + HD + HZ)) @ expm(-1j * (HX))
    return U


def phase_scaled_product(Aeig, C, eigenvalues, t):
    # Equivalent to Aeig * exp(1j * deltaE * t) @ C, but avoids building deltaE explicitly.
    u = np.exp(1j * eigenvalues * t)
    v = np.conjugate(u)
    AX = Aeig * (u[:, None] * v[None, :])
    M = AX @ C
    return M

def unitary_phase_scaled_product(Aeig, C, eigenvalues, t):
    s = eigenvalues ** t
    AX = Aeig * (np.conjugate(s)[:, None] * s[None, :])
    M = AX @ C
    return M


def main():
    args = parse_args()

    hx = args.hx
    hz = args.hz
    g = args.g
    J = args.J
    L = args.L
    tstart = args.tstart
    tend = args.tend
    N = args.N
    n_reals = args.n_reals
    obsA = observables[args.obsA]
    obsB = observables[args.obsB]
    output_dir = args.output_dir
    n2 = args.n2
    dynamics_type = args.Dynamics
    if n2 < 1 or n2 > 2 * L:
        raise ValueError(f"n2 must be between 1 and {2 * L}, got {n2}")
    seed = args.seed
    real_start = args.real_start
    real_end = n_reals if args.real_end is None else min(args.real_end, n_reals)

    os.makedirs(output_dir, exist_ok=True)

    dim = 2 ** (2 * L)
    if dynamics_type == "Kicked":
        N = 1  # For kicked dynamics, we only need integer time steps
    times = np.arange(tstart, tend, 1 / N)
    nqubits = 2 * L

    # Precompute all site-local operators once
    X_ops, Z_ops, A_ops, B_ops = precompute_site_ops(obsA, obsB, nqubits)

    # B0 is fixed for the whole run
    B0 = B_ops[n2 - 1]

    np.savez(
        os.path.join(output_dir, "parameters.npz"),
        L=L,
        hx=hx,
        hz=hz,
        g=g,
        J=J,
        times=times,
        obsA=args.obsA,
        obsB=args.obsB,
        n_reals=n_reals,
        real_start=real_start,
        real_end=real_end,
        dynamics_type=dynamics_type,
        seed=seed,
        n2=n2,
    )
    if dynamics_type == "Ising":
        print(f"Running Ising dynamics with L={L}, hx={hx}, hz={hz}, g={g}, J={J}, n2={n2}")
        for real in range(real_start, real_end):
            # One RNG per realization => same H_Ising across all sites in this realization
            rng = np.random.default_rng(seed + real)

            H_Ising = Ising_Hamiltonian(L, dim, rng, J, hx, hz, g, X_ops=X_ops, Z_ops=Z_ops)

            eigenvalues, P = np.linalg.eigh(H_Ising)
            Pdag = P.conj().T

            # Precompute B in the eigenbasis once per realization
            C = Pdag @ B0 @ P

            spectrum_state = np.zeros((nqubits, len(times), dim), dtype=complex)

            for n1 in range(1, nqubits + 1):
                print(f"Realization {real}, A on site n1 = {n1}", flush=True)

                A0 = A_ops[n1 - 1]
                Aeig = Pdag @ A0 @ P

                for t_idx, t in enumerate(tqdm(times, leave=False)):
                    M = phase_scaled_product(Aeig, C, eigenvalues, t)
                    evals_B = eigvals(M, check_finite=False)
                    spectrum_state[n1 - 1, t_idx, :] = evals_B

            # Save after each realization so you do not lose work if the job stops
            filename = os.path.join(output_dir, f"real_{real:03d}.npy")
            np.save(filename, spectrum_state)
            print(f"Saved realization {real} to {filename}", flush=True)
    elif dynamics_type == "Kicked":
        print(f"Running Kicked Ising dynamics with L={L}, hx={hx}, hz={hz}, g={g}, J={J}, n2={n2}")
        for real in range(real_start, real_end):
            rng = np.random.default_rng(seed + real)

            U_Kicked = Kicked_Ising_Unitary(L, dim, rng, J, hx, hz, g, X_ops=X_ops, Z_ops=Z_ops)

            eigenvalues, P = np.linalg.eig(U_Kicked)
            Pinv = np.linalg.inv(P)  # For non-Hermitian matrices, use the inverse instead of conjugate transpose

            # Precompute B in the eigenbasis once per realization
            C = Pinv @ B0 @ P

            spectrum_state = np.zeros((nqubits, len(times), dim), dtype=complex)

            for n1 in range(1, nqubits + 1):
                print(f"Realization {real}, A on site n1 = {n1}", flush=True)

                A0 = A_ops[n1 - 1]
                Aeig = Pinv @ A0 @ P

                for t_idx, t in enumerate(tqdm(times, leave=False)):
                    M = unitary_phase_scaled_product(Aeig, C, eigenvalues, t)
                    evals_B = eigvals(M, check_finite=False)
                    spectrum_state[n1 - 1, t_idx, :] = evals_B

            # Save after each realization so you do not lose work if the job stops
            filename = os.path.join(output_dir, f"real_{real:03d}.npy")
            np.save(filename, spectrum_state)
            print(f"Saved realization {real} to {filename}", flush=True)

if __name__ == "__main__":
    main()

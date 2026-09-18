import numpy as np
from scipy.linalg import eigvals
import os
import argparse
from tqdm import tqdm


# Gates

X = np.array([[0, 1],
              [1, 0]])
Z = np.array([[1, 0],
              [0, -1]])

# Pauli observables

sx = np.array([[0, 1],[1, 0]])
sz = np.array([[1, 0],[0, -1]])
sy = np.array([[0, -1.j],[1.j, 0]], dtype=complex)

observables = {
    "sx": sx,
    "sy": sy,
    "sz": sz
}


# Ising parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Ising Hamiltonian parameters")
    parser.add_argument("--hx", type=float, default=-1.05, help="Transverse field strength")
    parser.add_argument("--hz", type=float, default=0.5, help="Longitudinal field strength")
    parser.add_argument("--g", type=float, default=0.2, help="Disorder strength")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling strength")
    parser.add_argument("--L", type=int, default=5, help="Half the length of the chain")

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

# Ising Hamiltonian

def Ising_Hamiltonian(L, dim, rng, J, hx, hz, g):
    # Construct coupling term
    HZZ = np.zeros((dim, dim), dtype=complex)
    HX = np.zeros((dim, dim), dtype=complex)
    HZ = np.zeros((dim, dim), dtype=complex)
    HD = np.zeros((dim, dim), dtype=complex)
    for i in range(2*L-1):
        HZZ += -J * np.kron(np.kron(np.kron(np.eye(2**(i)), Z), Z), np.eye(2**(2*L-i-2)))
    for i in range(2*L):
        HX += -hx * np.kron(np.kron(np.eye(2**i), X), np.eye(2**(2*L-i-1)))
        HZ += -hz * np.kron(np.kron(np.eye(2**i), Z), np.eye(2**(2*L-i-1)))
    for i in range(2*L):
        HD += g * rng.normal() * np.kron(np.kron(np.eye(2**i), X), np.eye(2**(2*L-i-1)))

    H = HZZ + HX + HZ + HD
    return H

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
    if n2 < 1 or n2 > 2*L:
        raise ValueError(f"n2 must be between 1 and {2*L}, got {n2}")
    seed = args.seed
    real_start = args.real_start
    real_end = n_reals if args.real_end is None else min(args.real_end, n_reals)

    os.makedirs(output_dir, exist_ok=True)

    dim = 2**(2*L)
    times = np.arange(tstart, tend, 1/N)



    # B0 is fixed for the whole run
    B0 = np.kron(
        np.kron(np.eye(2 ** (n2 - 1)), obsB),
        np.eye(dim // (2 ** n2))
    )

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
        seed=seed,
        n2=n2,
    )

    for real in range(real_start, real_end):
        # one RNG per realization => same H_Ising across all sites in this realization
        rng = np.random.default_rng(seed + real)

        H_Ising =Ising_Hamiltonian(L, dim, rng, J, hx, hz, g)
        eigenvalues, P = np.linalg.eigh(H_Ising)
        Pdag = P.conj().T
        deltaE = eigenvalues[:, None] - eigenvalues[None, :]

        # precompute C once per realization
        C = Pdag @ B0 @ P

        for n1 in range(1, 2*L + 1):
                spectrum_state = np.zeros(
                    (int(2*L), len(times), dim),
                    dtype=complex
                )
            print(f"Realization {real}, A on site n1 = {n1}", flush=True)

            A0 = np.kron(
                np.kron(np.eye(2 ** (n1 - 1)), obsA),
                np.eye(dim // (2 ** n1))
            )
            Aeig = Pdag @ A0 @ P

            for t in tqdm(range(len(times))):
                phase_matrix = np.exp(1j * deltaE * times[t])
                AX = Aeig * phase_matrix
                M = AX @ C
                evals_B = eigvals(M, check_finite=False)
                spectrum_state[n1 - 1, t, :] = evals_B

        # save after each realization so you do not lose work if the job stops
        filename = os.path.join(output_dir, f"real_{real:03d}.npy")
        np.save(filename, spectrum_state)
        print(f"Saved realization {real} to {filename}", flush=True)

if __name__ == "__main__":
    main()
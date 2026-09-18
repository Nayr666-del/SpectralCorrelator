import numpy as np
from scipy.linalg import eigvals, expm
from scipy.stats import unitary_group
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

# Pauli on two sites
sX = np.kron(sx, sx)
sY = np.kron(sy, sy)
sZ = np.kron(sz, sz)

# Projection
sp = np.array([[1, 0],
               [0, 0]], dtype=complex)
spd = np.array([[0, 0],
                [0, 1]], dtype=complex)

# Custom
s1 = np.array([[0.5, 0.5j],
               [-0.5j, 0.5]], dtype=complex)

sc = np.kron(s1, sy)
sc_no = 2
# Non-unitary observables
# Random traceless hermitian single-site observable
rng_obs = np.random.default_rng(0)

sr_gen = rng_obs.random((2,2)) + 1j*rng_obs.random((2,2))
sr_trace = (sr_gen + sr_gen.conj().T)/2
sr = sr_trace - np.trace(sr_trace)/2 * np.eye(2)

# Gell-Mann observables
sgz = np.kron(sp, sz)
sgz_d = np.kron(spd, sz)

sgx = np.kron(sp, sx)
sgx_d = np.kron(spd, sx)

sgy = np.kron(sp, sy)
sgy_d = np.kron(spd, sy)

sR_gen = rng_obs.random((4,4)) + 1j*rng_obs.random((4,4))
sR_trace = (sR_gen + sR_gen.conj().T) / 2
sR = sR_trace - np.trace(sR_trace) / 4 * np.eye(4)

observables = {
    "sx": (sx, 1),
    "sy": (sy, 1),
    "sz": (sz, 1),
    "sr": (sr, 1),
    "sgx": (sgx, 2),
    "sgy": (sgy, 2),
    "sgz": (sgz, 2),
    "sgx_d": (sgx_d, 2),
    "sgy_d": (sgy_d, 2),
    "sgz_d": (sgz_d, 2),
    "sR": (sR, 2),
    "sX": (sX, 2),
    "sY": (sY, 2),
    "sZ": (sZ, 2),
    "sc": (sc, sc_no),
}


# Ising parameters

def parse_args():
    parser = argparse.ArgumentParser(description="Ising Hamiltonian parameters")
    parser.add_argument("--ortho_obs", type=str, default="N", choices=["Y", "N", "H"], help="Compute the Hamiltonian-orthogonal component")
    parser.add_argument("--hx", type=float, default=0.9, help="Transverse field strength")
    parser.add_argument("--hz", type=float, default=0.8, help="Longitudinal field strength")
    parser.add_argument("--g", type=float, default=0.2, help="Disorder strength")
    parser.add_argument("--tau", type=float, default=1.0, help="Coupling strength")
    parser.add_argument("--doder", type=str, default="Z", choices=["X", "Z"], help="Disorder type for Kicked Ising (default: Z)")
    parser.add_argument("--L", type=int, default=5, help="Half the length of the chain")
    parser.add_argument("-D", "--Dynamics", default="Ising", choices=["Ising", "Kicked", "GOE", "UHaar"], help="Dynamics type (default: Ising)")
    parser.add_argument("-P", "--Periodic", default="N", choices=["Y", "N"], help="Periodic chain toggle (defulat: N)")

    parser.add_argument("--tstart", type=float, default=0.0, help="Start time for simulation")
    parser.add_argument("--tend", type=float, default=5.0, help="End time for simulation")
    parser.add_argument("--s_start", type=int, default=1, help="Starting lattice site")
    parser.add_argument("--s_end", type=int, default=None, help="Ending lattice site")
    parser.add_argument("--N", type=float, default=5, help="Number of discretization steps between integer timesteps")

    parser.add_argument("--n_reals", type=int, default=1, help="Number of realizations for disorder averaging")
    parser.add_argument("--real_start", type=int, default=0, help="Starting index for realizations")
    parser.add_argument("--real_end", type=int, default=None, help="Ending index for realizations")

    parser.add_argument("--obsA", type=str, default="sx", help="Observable to compute correlators for (sx, sy, sz)")
    parser.add_argument("--obsB", type=str, default="sx", help="Observable to compute correlators for (sx, sy, sz)")

    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for the spectrum state")
    parser.add_argument("--n2", type=int, default=5, help="Site index for operator B0")

    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    return parser.parse_args()


# Small helper: place a 2x2 operator (or 4x4) on one site of a 2L-site chain

def local_op(op, n, site, nqubits):
    if n==1:
        return np.kron(
            np.kron(np.eye(2 ** site, dtype=complex), op),
            np.eye(2 ** (nqubits - site - 1), dtype=complex),
        )
    elif n==2:
        return np.kron(
            np.kron(np.eye(2 ** site, dtype=complex), op),
            np.eye(2 ** (nqubits - site - 2), dtype=complex),
        )


# Precompute all single-site operators once

def precompute_site_ops(obsA, obsB, nA, nB, nqubits):
    X_ops = [local_op(X, 1, i, nqubits) for i in range(nqubits)]
    Z_ops = [local_op(Z, 1, i, nqubits) for i in range(nqubits)]
    A_ops = [local_op(obsA, nA, i, nqubits) for i in range(nqubits + 1 - nA)]
    return X_ops, Z_ops, A_ops


# Ising Hamiltonian

def Ising_Hamiltonian_Z_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=None, Z_ops=None):
    nqubits = 2 * L

    if X_ops is None or Z_ops is None:
        X_ops, Z_ops, _, _ = precompute_site_ops(X, Z, 1, 1, nqubits)

    HZZ = np.zeros((dim, dim), dtype=complex)
    HX = np.zeros((dim, dim), dtype=complex)
    HZ = np.zeros((dim, dim), dtype=complex)
    HD = np.zeros((dim, dim), dtype=complex)

    for i in range(nqubits - 1):
        HZZ += -tau * (Z_ops[i] @ Z_ops[i + 1])
    if periodic=="Y":
        HZZ += -tau * (Z_ops[0] @ Z_ops[-1])
    for i in range(nqubits):
        HX += -tau * hx * X_ops[i]
        HZ += -tau * hz * Z_ops[i]

    if g != 0:
        disorder = rng.normal(size=nqubits)
        for i in range(nqubits):
            HD += tau * g * disorder[i] * Z_ops[i]

    H = HZZ + HX + HZ + HD
    return H

def Ising_Hamiltonian_X_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=None, Z_ops=None):
    nqubits = 2 * L

    if X_ops is None or Z_ops is None:
        X_ops, Z_ops, _, _ = precompute_site_ops(X, Z, 1, 1, nqubits)

    HZZ = np.zeros((dim, dim), dtype=complex)
    HX = np.zeros((dim, dim), dtype=complex)
    HZ = np.zeros((dim, dim), dtype=complex)
    HD = np.zeros((dim, dim), dtype=complex)

    for i in range(nqubits - 1):
        HZZ += -tau * (Z_ops[i] @ Z_ops[i + 1])
    if periodic=="Y":
        HZZ += -tau * (Z_ops[0] @ Z_ops[-1])
    for i in range(nqubits):
        HX += -tau * hx * X_ops[i]
        HZ += -tau * hz * Z_ops[i]

    if g != 0:
        disorder = rng.normal(size=nqubits)
        for i in range(nqubits):
            HD += tau * g * disorder[i] * X_ops[i]

    H = HZZ + HX + HZ + HD
    return H

def Kicked_Ising_Unitary_X_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=None, Z_ops=None):
    nqubits = 2 * L

    if X_ops is None or Z_ops is None:
        X_ops, Z_ops, _, _ = precompute_site_ops(X, Z, 1, 1, nqubits)

    HZZ = np.zeros((dim, dim), dtype=complex)
    HX = np.zeros((dim, dim), dtype=complex)
    HZ = np.zeros((dim, dim), dtype=complex)
    HD = np.zeros((dim, dim), dtype=complex)

    for i in range(nqubits - 1):
        HZZ += -tau * (Z_ops[i] @ Z_ops[i + 1])
    if periodic=="Y":
        HZZ += -tau * (Z_ops[0] @ Z_ops[-1])
    for i in range(nqubits):
        HX += -tau * hx * X_ops[i]
        HZ += -tau * hz * Z_ops[i]

    if g != 0:
        disorder = rng.normal(size=nqubits)
        for i in range(nqubits):
            HD += tau * g * disorder[i] * X_ops[i]

    U = expm(-1j * (HZZ + HZ)) @ expm(-1j * (HX + HD))
    return U


def Kicked_Ising_Unitary_Z_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=None, Z_ops=None):
    nqubits = 2 * L

    if X_ops is None or Z_ops is None:
        X_ops, Z_ops, _, _ = precompute_site_ops(X, Z, 1, 1, nqubits)

    HZZ = np.zeros((dim, dim), dtype=complex)
    HX = np.zeros((dim, dim), dtype=complex)
    HZ = np.zeros((dim, dim), dtype=complex)
    HD = np.zeros((dim, dim), dtype=complex)

    for i in range(nqubits - 1):
        HZZ += -tau * (Z_ops[i] @ Z_ops[i + 1])
    if periodic=="Y":
        HZZ += -tau * (Z_ops[0] @ Z_ops[-1])
    for i in range(nqubits):
        HX += -tau * hx * X_ops[i]
        HZ += -tau * hz * Z_ops[i]

    if g != 0:
        disorder = rng.normal(size=nqubits)
        for i in range(nqubits):
            HD += tau * g * disorder[i] * Z_ops[i]

    U = expm(-1j * (HZZ + HD + HZ)) @ expm(-1j * (HX))
    return U

def GOE_Hamiltonian(dim, rng):
    H0 = rng.normal(size=(dim, dim))
    H = (H0 + H0.T) / np.sqrt(2 * dim)
    return H

def Haar_Unitary(dim, rng):
    U_Haar = unitary_group.rvs(dim=dim, random_state=rng)
    return U_Haar

def phase_scaled_product(Aeig, eigenvalues, t):
    # Equivalent to Aeig * exp(1j * deltaE * t) @ C, but avoids building deltaE explicitly.
    u = np.exp(1j * eigenvalues * t)
    v = np.conjugate(u)
    AX = Aeig * (u[:, None] * v[None, :])
    return AX

def unitary_phase_scaled_product(Aeig, eigenvalues, t):
    s = eigenvalues ** t
    AX = Aeig * (np.conjugate(s)[:, None] * s[None, :])
    return AX

def ETH_Elements(M, eigenvalues):
    N = len(eigenvalues)
    Ann = np.diag(M)
    i, j = np.where(~np.eye(N, dtype=bool))

    Amn = M[i, j]
    qe_diff = eigenvalues[i] - eigenvalues[j]
    return Ann, Amn, qe_diff

def main():
    args = parse_args()
    ortho_obs = args.ortho_obs
    hx = args.hx
    hz = args.hz
    g = args.g
    tau = args.tau
    L = args.L
    tstart = args.tstart
    tend = args.tend
    s_start = args.s_start
    s_end = args.s_end
    N = args.N
    n_reals = args.n_reals
    obsA, nA = observables[args.obsA]
    obsB, nB = observables[args.obsB]
    periodic = args.Periodic
    output_dir = args.output_dir
    n2 = args.n2
    dynamics_type = args.Dynamics
    if n2 < 1 or n2 > 2 * L - nB + 1:
        raise ValueError(f"n2 must be between 1 and {2 * L - nB + 1}, got {n2}")
    seed = args.seed
    real_start = args.real_start
    real_end = n_reals if args.real_end is None else min(args.real_end, n_reals)

    os.makedirs(output_dir, exist_ok=True)

    dim = 2 ** (2 * L)

    if dynamics_type == "Kicked": # For kicked dynamics, we only need integer time steps
        N = max(1, int(N))
        times = np.arange(tstart, tend, N)
    else:
        times = np.arange(tstart, tend, N)
    
    if s_end is None:
        s_end = 2 * L 
    elif s_end < s_start:
        raise ValueError("s_end must be larger than s_start")
    elif ((s_end >  2 * L) or (s_end < 1)) or ((s_start > 2 * L) or (s_start < 1)):
        raise ValueError("s_end and s_start must be within valid range from 1 to 2L")
    
    if s_end > 2 * L - nA + 1:
        raise ValueError(f"Using {str(nA)}-site observable obsA = {obsA}. Can only iterate up to site {str(2*L - nA + 1)}")

      

    nqubits = s_end - s_start + 1 # Take to mean the range of sites needed. e.g. 4 ~ 6 is 3 sites

    # Precompute all site-local operators once
    X_ops, Z_ops, A_ops = precompute_site_ops(obsA, obsB, nA, nB, 2 * L)


    np.savez(
        os.path.join(output_dir, "parameters.npz"),
        L=L,
        hx=hx,
        hz=hz,
        g=g,
        ortho=ortho_obs,
        tau=tau,
        times=times,
        obsA=args.obsA,
        obsB=args.obsB,
        n_reals=n_reals,
        real_start=real_start,
        real_end=real_end,
        dynamics_type=dynamics_type,
        disorder=args.doder,
        periodic=periodic,
        seed=seed,
        n2=n2,
        s_start=s_start,
        s_end=s_end,
    )


    print(f"ObsA = {args.obsA}  ObsB = {args.obsB}", flush=True)
    if periodic=="Y":
        print("Periodic Ising Chain")
    if dynamics_type == "Ising" and args.doder == "Z":
        print(f"Running Z-disorder Ising dynamics with L={L}, hx={hx}, hz={hz}, g={g}, tau={tau}, n2={n2}")
        for real in range(real_start, real_end):
            # Array for storage of Energy / Quasi-energy
            ETH_Matrix_Diagonal = np.zeros((nqubits, dim), dtype=complex)
            ETH_Matrix_Off = np.zeros((nqubits, dim**2 - dim), dtype=complex)
            # One RNG per realization => same H_Ising across all sites in this realization
            rng = np.random.default_rng(seed + real)
            H_Ising = Ising_Hamiltonian_Z_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=X_ops, Z_ops=Z_ops)
            eigenvalues, P = np.linalg.eigh(H_Ising)
            Pdag = P.conj().T

            # Evaluate at the late-time steps
            T_eval = times[-1]

            for n1 in range(s_start, s_end + 1):
                print(f"Realization {real}, A on site n1 = {n1}", flush=True)

                A0 = A_ops[n1 - 1]
                Aeig = Pdag @ A0 @ P

                M = phase_scaled_product(Aeig, eigenvalues, T_eval)
                Ann, Amn, qe_diff = ETH_Elements(M, eigenvalues)
                ETH_Matrix_Diagonal[n1 - s_start, :] = Ann
                ETH_Matrix_Off[n1 - s_start, :] = Amn

            # Save after each realization so you do not lose work if the job stops
            filename = os.path.join(output_dir, f"real_{real:03d}.npz")
            np.savez(filename, 
                     Diagonal=ETH_Matrix_Diagonal,
                     Off_Diagonal=ETH_Matrix_Off,
                     Eigenvalues=eigenvalues,
                     Eigen_Diff=qe_diff)
            print(f"Saved realization {real} to {filename}", flush=True)
    elif dynamics_type == "Ising" and args.doder == "X":
        print(f"Running X-disorder Ising dynamics with L={L}, hx={hx}, hz={hz}, g={g}, tau={tau}, n2={n2}")
        for real in range(real_start, real_end):
            # Array for storage of Energy / Quasi-energy
            ETH_Matrix_Diagonal = np.zeros((nqubits, dim), dtype=complex)
            ETH_Matrix_Off = np.zeros((nqubits, dim**2 - dim), dtype=complex)
            # One RNG per realization => same H_Ising across all sites in this realization
            rng = np.random.default_rng(seed + real)
            H_Ising = Ising_Hamiltonian_X_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=X_ops, Z_ops=Z_ops)
            eigenvalues, P = np.linalg.eigh(H_Ising)
            Pdag = P.conj().T

            # Evaluate at the late-time steps
            T_eval = times[-1]

            for n1 in range(s_start, s_end + 1):
                print(f"Realization {real}, A on site n1 = {n1}", flush=True)

                A0 = A_ops[n1 - 1]
                Aeig = Pdag @ A0 @ P

                M = phase_scaled_product(Aeig, eigenvalues, T_eval)
                Ann, Amn, qe_diff = ETH_Elements(M, eigenvalues)
                ETH_Matrix_Diagonal[n1 - s_start, :] = Ann
                ETH_Matrix_Off[n1 - s_start, :] = Amn

            # Save after each realization so you do not lose work if the job stops
            filename = os.path.join(output_dir, f"real_{real:03d}.npz")
            np.savez(filename, 
                     Diagonal=ETH_Matrix_Diagonal,
                     Off_Diagonal=ETH_Matrix_Off,
                     Eigenvalues=eigenvalues,
                     Eigen_Diff=qe_diff)
            print(f"Saved realization {real} to {filename}", flush=True)
    elif dynamics_type == "Kicked" and args.doder == "Z":
        print(f"Running Z-disorder Kicked Ising dynamics with L={L}, hx={hx}, hz={hz}, g={g}, tau={tau}, n2={n2}")
        for real in range(real_start, real_end):
             # Array for storage of Energy / Quasi-energy
            ETH_Matrix_Diagonal = np.zeros((nqubits, dim), dtype=complex)
            ETH_Matrix_Off = np.zeros((nqubits, dim**2 - dim), dtype=complex)
            # One RNG per realization => same H_Ising across all sites in this realization
            rng = np.random.default_rng(seed + real)
            H_Ising = Kicked_Ising_Unitary_Z_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=X_ops, Z_ops=Z_ops)
            eigenvalues, P = np.linalg.eig(H_Ising)
            Pdag = P.conj().T

            tol = np.linalg.norm(P.conj().T @ P - np.eye(P.shape[0]))
            print(f"Tolerence: {tol}", flush=True)

            # Evaluate at the late-time steps
            T_eval = times[-1]

            for n1 in range(s_start, s_end + 1):
                print(f"Realization {real}, A on site n1 = {n1}", flush=True)

                A0 = A_ops[n1 - 1]
                Aeig = Pdag @ A0 @ P

                qe = -np.angle(eigenvalues)
                M = unitary_phase_scaled_product(Aeig, eigenvalues, T_eval)
                Ann, Amn, qe_diff = ETH_Elements(M, qe)
                ETH_Matrix_Diagonal[n1 - s_start, :] = Ann
                ETH_Matrix_Off[n1 - s_start, :] = Amn

            # Save after each realization so you do not lose work if the job stops
            filename = os.path.join(output_dir, f"real_{real:03d}.npz")
            np.savez(filename, 
                     Diagonal=ETH_Matrix_Diagonal,
                     Off_Diagonal=ETH_Matrix_Off,
                     Eigenvalues=qe,
                     Eigen_Diff=qe_diff)
            print(f"Saved realization {real} to {filename}", flush=True)
    elif dynamics_type == "Kicked" and args.doder == "X":
        print(f"Running X-disorder Kicked Ising dynamics with L={L}, hx={hx}, hz={hz}, g={g}, tau={tau}, n2={n2}")
        for real in range(real_start, real_end):
             # Array for storage of Energy / Quasi-energy
            ETH_Matrix_Diagonal = np.zeros((nqubits, dim), dtype=complex)
            ETH_Matrix_Off = np.zeros((nqubits, dim**2 - dim), dtype=complex)
            # One RNG per realization => same H_Ising across all sites in this realization
            rng = np.random.default_rng(seed + real)
            H_Ising = Kicked_Ising_Unitary_X_Disorder(L, dim, rng, tau, hx, hz, g, periodic, X_ops=X_ops, Z_ops=Z_ops)
            eigenvalues, P = np.linalg.eig(H_Ising)
            Pdag = P.conj().T

            tol = np.linalg.norm(P.conj().T @ P - np.eye(P.shape[0]))
            print(f"Tolerence: {tol}", flush=True)

            # Evaluate at the late-time steps
            T_eval = times[-1]

            for n1 in range(s_start, s_end + 1):
                print(f"Realization {real}, A on site n1 = {n1}", flush=True)

                A0 = A_ops[n1 - 1]
                Aeig = Pdag @ A0 @ P

                qe = -np.angle(eigenvalues)
                M = unitary_phase_scaled_product(Aeig, eigenvalues, T_eval)
                Ann, Amn, qe_diff = ETH_Elements(M, qe)
                ETH_Matrix_Diagonal[n1 - s_start, :] = Ann
                ETH_Matrix_Off[n1 - s_start, :] = Amn

            # Save after each realization so you do not lose work if the job stops
            filename = os.path.join(output_dir, f"real_{real:03d}.npz")
            np.savez(filename, 
                     Diagonal=ETH_Matrix_Diagonal,
                     Off_Diagonal=ETH_Matrix_Off,
                     Eigenvalues=qe,
                     Eigen_Diff=qe_diff)
            print(f"Saved realization {real} to {filename}", flush=True)

            

if __name__ == "__main__":
    main()

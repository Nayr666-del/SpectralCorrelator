import numpy as np
from glob import glob
from scipy.linalg import eigvals, expm
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

def file_load_params(file):
    params = np.load(f"{file}/parameters.npz")
    return params

def file_load_data(file, times, n_reals, s_start, s_end, L, kmax=10):
    dim = 2 ** (2 * L)

    # Load file names once
    paths = sorted(glob(f"{file}/real_*.npz"))




    arrays = []

    for real in range(n_reals):
        spectrum_state_cluster = np.load(paths[real])
        arrays.append(spectrum_state_cluster)
        for k in range(1, kmax+1):
            correlators_spectrum_cluster[:, :, k-1, real] = np.sum(spectrum_state_cluster**k, axis=2) / dim

    all_eigs = np.concatenate(arrays, axis=-1)

    return correlators_spectrum_cluster, all_eigs

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis Parameters")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for Analysis")
    parser.add_argument("--kmax", type=int, default=4, help="maximum k-OTOC")
    parser.add_argument("--n_plot", type=int, default=None, help="Site to plot")
    parser.add_argument("--dim", type=int, default=1, help="Dim to plot")
    parser.add_argument("--UHaar", type=str, default=None, help="Is UHaar available for L1 computation?")
    return parser.parse_args()



def main():
    args = parse_args()
    output_dir = args.output_dir
    data_dir = args.data_dir
    kmax = args.kmax
    n_plot = args.n_plot
    dim = args.dim
    UHaar = args.UHaar

    params = file_load_params(data_dir)
    times = params["times"]
    n_reals = params["n_reals"]
    L = params["L"]
    n2 = params["n2"]
    s_start = params['s_start']
    s_end = params['s_end']
    
    

    if n_plot is None:
        n_plot = n2
    elif ((n_plot <= 1) or (2 * L <= n_plot)):
        raise ValueError(f"n_plot must be between 1 and {2 * L}, got {n_plot}")

    

    correlators, eigs = file_load_data(data_dir, times, n_reals, s_start, s_end, L, kmax)
    if dim == 2: 
        spectrum_2D(eigs, times, L, output_dir, n_plot, s_start)
        np.savez(
                os.path.join(output_dir, "parameters.npz"), 
                **{k: params[k] for k in params.files}
            )
        np.save(os.path.join(output_dir, "correlators.npy"), correlators)
        radial_data = eigs[n_plot-s_start, -1, :]
        np.save(os.path.join(output_dir, "radial.npy"), radial_data)
        if UHaar is not None:
            eigs_final = file_load_UHaar_data(UHaar, n_reals)
            L1_Distance = spectral_distance_2D(eigs, L, 31, times, eigs_final, s_start, s_end)
            np.save(os.path.join(output_dir, "L1_Distance.npy"), L1_Distance)
        return 0
    chi_square_distance, KL_distance, L1_norm = spectral_distance(eigs, L, times)
    spectrum_1D(eigs, times, L, output_dir)
    river_diagram(eigs, times, n_plot, output_dir)
    density_evolution(eigs, times, n_plot, output_dir)
    


    np.savez(
        os.path.join(output_dir, "parameters.npz"), 
        **{k: params[k] for k in params.files}
    )
    np.save(os.path.join(output_dir, "correlators.npy"), correlators)
    np.save(os.path.join(output_dir, "chi_square_distance.npy"), chi_square_distance)
    np.save(os.path.join(output_dir, "KL_distance.npy"), KL_distance)
    np.save(os.path.join(output_dir, "L1_norm.npy"), L1_norm)
    print("Analysis saved.", flush=True)

if __name__ == "__main__":
    main()




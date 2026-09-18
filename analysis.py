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

def file_load_UHaar_data(file, n_reals):

    paths = sorted(glob(f"{file}/real_*.npy"))

    arrays = []

    for real in range(n_reals):
        spectrum_state_cluster = np.load(paths[real])
        arrays.append(spectrum_state_cluster)

    all_eigs = np.concatenate(arrays, axis=-1)

    return all_eigs

def file_load_data(file, times, n_reals, s_start, s_end, L, kmax=10):
    dim = 2 ** (2 * L)

    # Load file names once
    paths = sorted(glob(f"{file}/real_*.npy"))



    correlators_spectrum_cluster = np.zeros(
        (int(s_end - s_start + 1), len(times), kmax, n_reals),
        dtype=complex
    )

    arrays = []

    for real in range(n_reals):
        spectrum_state_cluster = np.load(paths[real])
        arrays.append(spectrum_state_cluster)
        for k in range(1, kmax+1):
            correlators_spectrum_cluster[:, :, k-1, real] = np.sum(spectrum_state_cluster**k, axis=2) / dim

    all_eigs = np.concatenate(arrays, axis=-1)

    return correlators_spectrum_cluster, all_eigs

def spectral_distance(eigenvalues, L, times):
    uniform_dist = 0.5
    eps=1e-12

    chi_square_distance = np.zeros((int(2*L), len(times)))
    KL_distance = np.zeros((int(2*L), len(times)))
    L1_norm = np.zeros((int(2*L), len(times)))
    for i in range(int(2*L)):
        for j in range(len(times)):
            hist, edges = np.histogram(np.angle(eigenvalues[i, j, :]) / np.pi, bins=100, range=(-1, 1), density=True)
            dx = np.diff(edges)
            p = np.clip(hist, eps, None)
            chi_square_distance[i, j] = np.sqrt(np.sum((hist - uniform_dist)**2 * dx))
            L1_norm[i, j] = np.sum(np.abs(hist - uniform_dist) * dx)
            KL_distance[i, j] = np.sum(p * dx * np.log(p / uniform_dist))

    return chi_square_distance, KL_distance, L1_norm

def spectral_distance_2D(eigenvalues, L, bins, times, eig_final, s_start, s_end):
    L1_norm = np.zeros((int(2*L), len(times)))
    site_range = s_end - s_start + 1

    xbins = np.linspace(-1.1, 1.1, bins)
    ybins = np.linspace(-1.1, 1.1, bins)
    L1_Distance = np.zeros((site_range, len(times)))

    for i in range(site_range):
        x0, y0 = np.real(eig_final[i, 0, :]), np.imag(eig_final[i, 0, :])
        for j in range(len(times)):
            x1, y1 = np.real(eigenvalues[i, j, :]), np.imag(eigenvalues[i, j, :])
            H0, _, _ = np.histogram2d(x0, y0, bins=[xbins, ybins], density=False)
            H1, _, _ = np.histogram2d(x1, y1, bins=[xbins, ybins], density=False)
            P0 = H0 / np.sum(H0)
            P1 = H1 / np.sum(H1)
            L1_Distance[i, j] = np.sum(np.abs(P1 - P0))
    return L1_Distance

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis Parameters")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory for Analysis")
    parser.add_argument("--kmax", type=int, default=4, help="maximum k-OTOC")
    parser.add_argument("--n_plot", type=int, default=None, help="Site to plot")
    parser.add_argument("--dim", type=int, default=1, help="Dim to plot")
    parser.add_argument("--UHaar", type=str, default=None, help="Is UHaar available for L1 computation?")
    return parser.parse_args()

def spectrum_1D(eigenvalues, times, L, output_dir):
    fig, axes = plt.subplots(2, int(L), figsize=(int(2*L),5), sharex=True, sharey=True)

    def update(time_index):
        for ax in axes.flat:
            ax.clear()

        for i, ax, in enumerate(axes.flat):
            ax.hist(
                np.angle(eigenvalues[i, time_index, :]) / np.pi,
                bins=200,
                range=(-1, 1),
                histtype='step',
                density=True,
                color='blue'
            )
            ax.set_title(f"n = {i+1}")
            ax.set_ylim(0, 1.5)
        fig.suptitle(f"T={times[time_index]}", fontsize=16)

        for ax in axes[-1]:
            ax.set_xlabel(r"$\phi/\pi$")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"$P(\phi)$")
        fig.tight_layout()
    
    anim = FuncAnimation(fig, update, frames=len(times), interval=200, repeat=True)
    anim.save(os.path.join(output_dir, "spectrum.mp4"), writer="ffmpeg", dpi=100)
    plt.close(fig)
    print(f"Animation saved.", flush=True)


def spectrum_2D(eigenvalues, times, L, output_dir, n_plot, s_start):
    fig, ax = plt.subplots(figsize=(8, 8))
    size = 1
    # Initial frame
    id = 0
    x = np.real(eigenvalues[n_plot-s_start, id, :])
    y = np.imag(eigenvalues[n_plot-s_start, id, :])

    H, _, _ = np.histogram2d(
        x, y,
        bins=300,
        range=[[-size, size], [-size, size]],
        density=True
    )

    H = gaussian_filter(H, sigma=1)

    im = ax.imshow(
        H.T,
        origin="lower",
        extent=[-size, size, -size, size],
        cmap="magma",
        aspect="equal",
        vmin=0,
        vmax=2
    )

    plt.colorbar(im, ax=ax, label="Smoothed density")
    ax.set_xlabel(r"Re($\lambda$)")
    ax.set_ylabel(r"Im($\lambda$)")

    def update(frame):
        x = np.real(eigenvalues[n_plot-s_start, frame, :])
        y = np.imag(eigenvalues[n_plot-s_start, frame, :])

        H, _, _ = np.histogram2d(
            x, y,
            bins=300,
            range=[[-size, size], [-size, size]],
            density=True
        )

        H = gaussian_filter(H, sigma=1)

        im.set_data(H.T)
        ax.set_title(f"T = {times[frame]}")

        return (im,)

    anim = FuncAnimation(fig, update, frames=len(times), interval=200, repeat=True)
    anim.save(os.path.join(output_dir, "spectrum.mp4"), writer="ffmpeg", dpi=100)

def river_diagram(eigenvalues, times, n_plot, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    phi = np.angle(eigenvalues)
    for i in range(len(times)):
        ax.plot(times[i]*np.ones(len(phi[n_plot -1 , i, :])), phi[n_plot - 1, i, :] / np.pi, 'b.', ms=0.4, alpha=0.3)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\phi/\pi$")
    ax.set_xlim(0, times[-1])
    ax.set_ylim(-1.02, 1.02)
    ax.set_title(f"{n_plot}")
    fig.savefig(os.path.join(output_dir, f"riverplot_{n_plot}.png"))
    plt.close(fig)
    

def density_evolution(eigenvalues, times, n_plot, output_dir):
    idx = n_plot - 1

    phi = np.angle(eigenvalues[idx]) / np.pi      # shape = (ntimes, neigs)

    nbins = 200
    bins = np.linspace(-1, 1, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    density = np.zeros((nbins, len(times)))

    for i in range(len(times)):
        hist, _ = np.histogram(phi[i], bins=bins, density=True)
        density[:, i] = hist

    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(
        density,
        origin="lower",
        aspect="auto",
        extent=[times[0], times[-1], -1, 1],
        cmap="magma",
        vmin=0.0,
        vmax=1.5,
    )

    fig.colorbar(im, ax=ax, label=r"$P(\phi)$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\phi/\pi$")
    ax.set_title(f"Site {n_plot}")

    fig.savefig(os.path.join(output_dir, f"density_{n_plot}.png"))
    plt.close(fig)

def radial_density_log(z, nbins=100, rmin=None, rmax=None):
    """
    Radial density rho(r) using logarithmic bins.

    Returns
    -------
    r : geometric bin centers
    rho : angle-averaged density
    p : radial density, p(r)=2*pi*r*rho(r)
    """

    r = np.abs(z)
    r = r[r > 0]          # log requires r>0

    if rmin is None:
        rmin = r.min()
    if rmax is None:
        rmax = r.max()

    edges = np.geomspace(rmin, rmax, nbins + 1)

    counts, _ = np.histogram(r, bins=edges)

    N = len(r)

    # Exact annulus areas
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    rho = counts / (N * area)

    rc = np.sqrt(edges[:-1] * edges[1:])   # geometric centers
    p = 2 * np.pi * rc * rho

    return rc, rho, p


def radial_density(z, nbins=100, rmax=None):
    """
    Estimate the angle-averaged density rho(r) from samples z.

    Returns
    -------
    r : bin centers
    rho : angle-averaged density, normalized so that
          2*pi*int rho(r)*r dr = 1
    p : radial probability density = 2*pi*r*rho
    """
    r = np.abs(z)

    if rmax is None:
        rmax = r.max()

    edges = np.linspace(0, rmax, nbins + 1)
    counts, _ = np.histogram(r, bins=edges)

    dr = np.diff(edges)
    rc = 0.5 * (edges[:-1] + edges[1:])

    N = len(z)

    # Area of each annulus
    area = np.pi * (edges[1:]**2 - edges[:-1]**2)

    # Estimate of the 2D density
    rho = counts / (N * area)

    # Radial pdf
    p = 2 * np.pi * rc * rho

    return rc, rho, p



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
    #river_diagram(eigs, times, n_plot, output_dir)
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




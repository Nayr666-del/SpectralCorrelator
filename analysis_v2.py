"""Analysis utilities for lattice_site_v7 output.

The input files contain one array per realization with shape
``(sites, times, eigenvalues)``.  This script keeps the 1D phase spectrum
separate from the 2D complex-plane spectrum.
"""

import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
import numpy as np


# Load simulation parameters and realization arrays.
def load_data(data_dir):
    params = np.load(os.path.join(data_dir, "parameters.npz"), allow_pickle=True)
    paths = sorted(glob(os.path.join(data_dir, "real_*.npy")))
    if not paths:
        raise FileNotFoundError(f"No realization files found in {data_dir}")

    n_reals = int(params["n_reals"])
    if n_reals > len(paths):
        raise ValueError(f"Expected {n_reals} realizations, found {len(paths)}")
    arrays = [np.load(path) for path in paths[:n_reals]]
    return params, arrays


# Compute the k-th spectral moments and average them over realizations.
def averaged_correlators(realizations, kmax):
    first = realizations[0]
    nsites, ntimes, dimension = first.shape
    correlators = np.empty((nsites, ntimes, kmax), dtype=complex)
    normalization = dimension

    for k in range(1, kmax + 1):
        moments = [np.sum(array ** k, axis=2) / normalization for array in realizations]
        correlators[:, :, k - 1] = np.mean(moments, axis=0)
    return correlators


# Save a phase histogram animation for a selected 1D site.
def save_spectrum_1d(eigenvalues, times, site_index, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    def update(frame):
        ax.clear()
        phase = np.angle(eigenvalues[site_index, frame]) / np.pi
        ax.hist(phase, bins=200, range=(-1, 1), density=True, histtype="step")
        ax.set(xlim=(-1, 1), ylim=(0, 1.5), xlabel=r"$\phi/\pi$",
               ylabel=r"$P(\phi)$", title=f"T = {times[frame]}")

    animation = FuncAnimation(fig, update, frames=len(times), interval=200)
    animation.save(output_path, writer="ffmpeg", dpi=100)
    plt.close(fig)


# Save a complex-plane density animation for a selected 2D site.
def save_spectrum_2d(eigenvalues, times, site_index, output_path,
                     size=1, vmin=0, vmax=2):
    fig, ax = plt.subplots(figsize=(8, 8))
    image = None
    

    def update(frame):
        nonlocal image
        values = eigenvalues[site_index, frame]
        density, _, _ = np.histogram2d(
            values.real, values.imag, bins=300,
            range=[[-size, size], [-size, size]], density=True
        )
        density = gaussian_filter(density, sigma=1)
        if image is None:
            image = ax.imshow(density.T, origin="lower",
                              extent=(-size, size, -size, size),
                              cmap="magma", aspect="equal", vmin=vmin, vmax=vmax)
            fig.colorbar(image, ax=ax, label="Smoothed density")
            ax.set(xlabel=r"Re($\lambda$)", ylabel=r"Im($\lambda$)")
        else:
            image.set_data(density.T)
        ax.set_title(f"T = {times[frame]}")
        return (image,)

    animation = FuncAnimation(fig, update, frames=len(times), interval=200)
    animation.save(output_path, writer="ffmpeg", dpi=100)
    plt.close(fig)


def model_gxgx(r):
    """Radial density for the gx-gx model."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = 1 / (4 * np.pi * r * (1 - r) ** 2)
    return np.where((r > 0) & (r < 0.5), density, 0.0)


def model_ccx2(r):
    """Radial density for the cc-x2 model."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = (1 / (4 * np.pi * r)
                   * (1 / (1 - r) ** 2 + 4 / (4 - r) ** 2))
    return np.where((r >= 1.6) & (r <= 2.5), density, 0.0)




# Compute the 1D distance from the uniform phase distribution.
def l1_norm_1d(eigenvalues):
    result = np.empty(eigenvalues.shape[:2])
    uniform = 0.5
    for site in range(eigenvalues.shape[0]):
        for time in range(eigenvalues.shape[1]):
            histogram, edges = np.histogram(
                np.angle(eigenvalues[site, time]) / np.pi,
                bins=100, range=(-1, 1), density=True
            )
            result[site, time] = np.sum(np.abs(histogram - uniform) * np.diff(edges))
    return result


def l1_distance_to_model_2d(x, y, model_func, xbins, ybins, p0=0.5):
    """
    L1 distance between empirical 2D probabilities and a radial model.

    ``model_func(r)`` is the continuous radial density rho(r), normalized
    according to ``2 * pi * integral(rho(r) * r dr) = 1 - p0``.  A Dirac mass
    of weight ``p0`` is placed in the bin containing the origin.  The model is
    evaluated at Cartesian bin centers and converted to probability mass using
    each bin's area.
    """
    if not 0 <= p0 <= 1:
        raise ValueError("p0 must be between 0 and 1")

    H_data, _, _ = np.histogram2d(x, y, bins=[xbins, ybins], density=False)
    data_count = np.sum(H_data)
    if data_count == 0:
        raise ValueError("No data points fall inside the supplied bins")
    P_data = H_data / data_count

    xc = 0.5 * (xbins[:-1] + xbins[1:])
    yc = 0.5 * (ybins[:-1] + ybins[1:])
    X, Y = np.meshgrid(xc, yc, indexing="ij")

    dx = np.diff(xbins)[:, None]
    dy = np.diff(ybins)[None, :]
    area = dx * dy

    radii = np.sqrt(X ** 2 + Y ** 2)
    model_vals = model_func(radii)
    model_vals = np.clip(model_vals, 0, None)

    P_model = model_vals * area
    if np.sum(P_model) == 0 and p0 == 0:
        raise ValueError("The radial model has no support inside the supplied bins")

    origin_x = np.searchsorted(xbins, 0, side="right") - 1
    origin_y = np.searchsorted(ybins, 0, side="right") - 1
    if not (0 <= origin_x < P_model.shape[0] and 0 <= origin_y < P_model.shape[1]):
        raise ValueError("The supplied bins must contain the origin")
    P_model[origin_x, origin_y] += p0

    return np.sum(np.abs(P_data - P_model))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze lattice_site_v7 output")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--dim", type=int, choices=(1, 2), default=1)
    parser.add_argument("--n_plot", type=int, default=None,
                        help="1-based site for spectrum and radial.npy")
    parser.add_argument("--size", type=float, default=1,
                        help="Half-width of the 2D histogram extent")
    parser.add_argument("--vmin", type=float, default=0,
                        help="Minimum value for the 2D spectrum color scale")
    parser.add_argument("--vmax", type=float, default=2,
                        help="Maximum value for the 2D spectrum color scale")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.kmax < 1:
        raise ValueError("kmax must be positive")
    if args.size <= 0:
        raise ValueError("size must be positive")
    if args.vmin >= args.vmax:
        raise ValueError("vmin must be smaller than vmax")

    params, realizations = load_data(args.data_dir)
    times = params["times"]
    nsites = realizations[0].shape[0]
    s_start = int(params["s_start"])
    selected_site = args.n_plot if args.n_plot is not None else int(params["n2"])
    if not s_start <= selected_site < s_start + nsites:
        raise ValueError(f"n_plot must be between {s_start} and {s_start + nsites - 1}")
    site_index = selected_site - s_start

    os.makedirs(args.output_dir, exist_ok=True)
    correlators = averaged_correlators(realizations, args.kmax)
    eigenvalues = np.concatenate(realizations, axis=2)
    radial = eigenvalues[site_index, -1]

    np.savez(
        os.path.join(args.output_dir, "parameters.npz"),
        **{key: params[key] for key in params.files}
    )
    np.save(os.path.join(args.output_dir, "correlators.npy"), correlators)
    np.save(os.path.join(args.output_dir, "radial.npy"), radial)
    if args.dim == 1:
        l1_norm = l1_norm_1d(eigenvalues)
        save_spectrum_1d(eigenvalues, times, site_index,
                         os.path.join(args.output_dir, "spectrum.mp4"))
    else:
        # l1_norm = l1_norm_2d_placeholder(eigenvalues)
        save_spectrum_2d(eigenvalues, times, site_index,
                         os.path.join(args.output_dir, "spectrum.mp4"),
                         size=args.size, vmin=args.vmin,
                         vmax=args.vmax)
    #np.save(os.path.join(args.output_dir, "L1_norm.npy"), l1_norm)
    print(f"Saved analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

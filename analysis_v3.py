"""Analysis utilities for lattice_site_v8 output.

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


# Load simulation parameters, eigenvalues, and spectral weights.
def load_data(data_dir):
    params = np.load(os.path.join(data_dir, "parameters.npz"), allow_pickle=True)
    paths = sorted(glob(os.path.join(data_dir, "real_*.npy")))
    if not paths:
        raise FileNotFoundError(f"No realization files found in {data_dir}")

    n_reals = int(params["n_reals"])
    if n_reals > len(paths):
        raise ValueError(f"Expected {n_reals} realizations, found {len(paths)}")
    arrays = [np.load(path) for path in paths[:n_reals]]
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays[1:]):
        raise ValueError("Eigenvalue arrays have inconsistent shapes")
    weight_paths = [
        os.path.join(data_dir, f"weights_{index:03d}.npy")
        for index in range(n_reals)
    ]
    existing_weights = [os.path.exists(path) for path in weight_paths]
    if any(existing_weights) and not all(existing_weights):
        raise ValueError("Weight files are incomplete for the requested realizations")

    weights = []
    for array, path in zip(arrays, weight_paths):
        if all(existing_weights):
            weight_array = np.load(path)
            if weight_array.shape != array.shape:
                raise ValueError(
                    f"Weight shape {weight_array.shape} does not match "
                    f"eigenvalue shape {array.shape}"
                )
            if not np.all(np.isfinite(weight_array)):
                raise ValueError(f"Non-finite weights found in {path}")
            weight_sum = np.sum(weight_array, axis=2)
            # if not np.allclose(weight_sum, 1.0, atol=1e-3):
            #     print(f"Weight sum max in {path}: {np.abs(weight_sum).max()})", flush=True)
            #     raise ValueError(f"Weights in {path} are not normalized")
        else:
            dimension = array.shape[2]
            weight_array = np.full(array.shape, 1 / dimension, dtype=float)
        weights.append(weight_array)

    return params, arrays, weights


# Compute the k-th spectral moments and average them over realizations.
def averaged_correlators(realizations, weights, kmax):
    first = realizations[0]
    nsites, ntimes, dimension = first.shape
    correlators = np.empty((nsites, ntimes, kmax), dtype=complex)

    for k in range(1, kmax + 1):
        moments = [
            np.sum(weight * array ** k, axis=2)
            for array, weight in zip(realizations, weights)
        ]
        correlators[:, :, k - 1] = np.mean(moments, axis=0)
    return correlators


def _real_weights(weights, context, require_nonnegative=False):
    imaginary_size = np.max(np.abs(np.imag(weights)))
    if imaginary_size > 1e-10:
        raise ValueError(
            f"{context} requires real weights, but max |Im(w)| = "
            f"{imaginary_size:.3e}"
        )
    real_weights = np.real(weights)
    if require_nonnegative and np.min(real_weights) < -1e-10:
        raise ValueError(
            f"{context} requires nonnegative weights, but min Re(w) = "
            f"{np.min(real_weights):.3e}"
        )
    return real_weights


# Save a phase histogram animation for a selected 1D site.
def save_spectrum_1d(eigenvalues, weights, times, site_index, output_path):
    weights = _real_weights(weights, "1D spectrum plotting")
    fig, ax = plt.subplots(figsize=(8, 5))

    def update(frame):
        ax.clear()
        phase = np.angle(eigenvalues[site_index, frame]) / np.pi
        frame_weights = weights[site_index, frame]
        ax.hist(
            phase,
            bins=200,
            range=(-1, 1),
            weights=frame_weights,
            density=True,
            histtype="step",
        )
        ax.set(xlim=(-1, 1), ylim=(0, 1.5), xlabel=r"$\phi/\pi$",
               ylabel=r"$P(\phi)$", title=f"T = {times[frame]}")

    animation = FuncAnimation(fig, update, frames=len(times), interval=200)
    animation.save(output_path, writer="ffmpeg", dpi=100)
    plt.close(fig)


# Save a complex-plane density animation for a selected 2D site.
def save_spectrum_2d(eigenvalues, weights, times, site_index, output_path,
                     size=1, vmin=0, vmax=2):
    weights = _real_weights(weights, "2D spectrum plotting")
    fig, ax = plt.subplots(figsize=(8, 8))
    image = None
    

    def update(frame):
        nonlocal image
        values = eigenvalues[site_index, frame]
        frame_weights = weights[site_index, frame]
        density, _, _ = np.histogram2d(
            values.real,
            values.imag,
            bins=300,
            range=[[-size, size], [-size, size]],
            weights=frame_weights,
            density=True,
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
def l1_norm_1d(eigenvalues, weights):
    weights = _real_weights(weights, "L1 distance", require_nonnegative=True)
    result = np.empty(eigenvalues.shape[:2])
    uniform = 0.5
    for site in range(eigenvalues.shape[0]):
        for time in range(eigenvalues.shape[1]):
            histogram, edges = np.histogram(
                np.angle(eigenvalues[site, time]) / np.pi,
                bins=100,
                range=(-1, 1),
                weights=weights[site, time],
                density=True,
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
    parser = argparse.ArgumentParser(description="Analyze weighted lattice spectra")
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

    params, realizations, realization_weights = load_data(args.data_dir)
    times = params["times"]
    nsites = realizations[0].shape[0]
    s_start = int(params["s_start"])
    selected_site = args.n_plot if args.n_plot is not None else int(params["n2"])
    if not s_start <= selected_site < s_start + nsites:
        raise ValueError(f"n_plot must be between {s_start} and {s_start + nsites - 1}")
    site_index = selected_site - s_start

    os.makedirs(args.output_dir, exist_ok=True)
    correlators = averaged_correlators(realizations, realization_weights, args.kmax)
    eigenvalues = np.concatenate(realizations, axis=2)
    weights = np.concatenate(
        [weight / len(realization_weights) for weight in realization_weights],
        axis=2,
    )
    radial = eigenvalues[site_index, -1]
    radial_weights = weights[site_index, -1]

    np.savez(
        os.path.join(args.output_dir, "parameters.npz"),
        **{key: params[key] for key in params.files}
    )
    np.save(os.path.join(args.output_dir, "correlators.npy"), correlators)
    np.save(os.path.join(args.output_dir, "radial.npy"), radial)
    np.save(os.path.join(args.output_dir, "radial_weights.npy"), radial_weights)

    maximum_imaginary = np.max(np.abs(np.imag(weights)))
    weights_are_real = maximum_imaginary <= 1e-3
    if not weights_are_real:
        print(
            "Skipping weighted spectrum plot and L1 analysis: "
            f"max |Im(w)| = {maximum_imaginary:.3e}",
        )
    elif args.dim == 1:
        real_weights = np.real(weights)
        save_spectrum_1d(
            eigenvalues,
            real_weights,
            times,
            site_index,
            os.path.join(args.output_dir, "spectrum.mp4"),
        )
        if np.min(real_weights) < -1e-3:
            print(
                "Skipping L1 analysis: signed weights are not a "
                "probability distribution."
            )
        else:
            l1_norm = l1_norm_1d(eigenvalues, real_weights)
            np.save(os.path.join(args.output_dir, "L1_norm.npy"), l1_norm)
    else:
        save_spectrum_2d(
            eigenvalues,
            np.real(weights),
            times,
            site_index,
            os.path.join(args.output_dir, "spectrum.mp4"),
            size=args.size,
            vmin=args.vmin,
            vmax=args.vmax,
        )
    print(f"Saved analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

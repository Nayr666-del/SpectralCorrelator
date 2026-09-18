"""Analysis utilities for lattice_site_v8 output.

The input files contain one array per realization with shape
``(sites, times, eigenvalues)``.  This script keeps the 1D phase spectrum
separate from the 2D complex-plane spectrum.
"""

import argparse
import os
import warnings
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.special import comb
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

def averaged_cumulants(realizations, weights, kmax):
    first = realizations[0]
    nsites, ntimes, dimension = first.shape
    cumulants_2 = np.empty((nsites, ntimes), dtype=complex)
    cumulants_4 = np.empty((nsites, ntimes), dtype=complex)
    moments1 = np.array([
        np.sum(weight * array, axis=2)
        for array, weight in zip(realizations, weights)
    ])

    moments2 = np.array([
        np.sum(weight * array ** 2, axis=2)
        for array, weight in zip(realizations, weights)
    ])

    cumulants_2 = np.mean(moments1, axis=0)
    cumulants_4 = np.mean(moments2 - 2 * moments1**2, axis=0)
    return cumulants_2, cumulants_4


def _real_weights(weights, context, require_nonnegative=False):
    imaginary_size = np.max(np.abs(np.imag(weights)))
    # if imaginary_size > 1e-10:
    #     raise ValueError(
    #         f"{context} requires real weights, but max |Im(w)| = "
    #         f"{imaginary_size:.3e}"
    #     )
    real_weights = np.real(weights)
    # if require_nonnegative and np.min(real_weights) < -1e-10:
    #     raise ValueError(
    #         f"{context} requires nonnegative weights, but min Re(w) = "
    #         f"{np.min(real_weights):.3e}"
    #     )
    return real_weights


# Save a phase histogram animation for a selected 1D site.
def save_spectrum_1d(eigenvalues, weights, times, site_index, output_path,
                     interval=200):
    weights = _real_weights(weights, "1D spectrum plotting")
    fig, ax = plt.subplots(figsize=(8, 5))

    def update(frame):
        ax.clear()
        phase = np.angle(eigenvalues[site_index, frame]) / np.pi
        frame_weights = weights[site_index, frame]
        ax.hist(
            phase,
            bins=201,
            range=(-1, 1),
            weights=frame_weights,
            density=True,
            histtype="step",
        )
        ax.set(xlim=(-1, 1), ylim=(0, 1.5), xlabel=r"$\phi/\pi$",
               ylabel=r"$P(\phi)$", title=f"T = {times[frame]}")

    animation = FuncAnimation(fig, update, frames=len(times), interval=interval)
    animation.save(output_path, writer="ffmpeg", dpi=100)
    plt.close(fig)


# Save a complex-plane density animation for a selected 2D site.
def save_spectrum_2d(eigenvalues, weights, times, site_index, output_path,
                     size=1, vmin=0, vmax=2, interval=200):
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

    animation = FuncAnimation(fig, update, frames=len(times), interval=interval)
    animation.save(output_path, writer="ffmpeg", dpi=100)
    plt.close(fig)

### 2D models
def model_gxgx(r):
    """Radial density for the gx-gx model."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = 1 / (4 * np.pi * r * (1 - r) ** 2)
    return np.where((r > 0) & (r <= 0.5), density, 0.0)


def model_ccx2(r):
    """Radial density for the cc-x2 model."""
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        density = (1 / (4 * np.pi * r)
                   * (1 / (1 - r) ** 2 + 4 / (4 - r) ** 2))
    return np.where((r >= 1.6) & (r <= 2.5), density, 0.0)



### 1D models
def model_uniform(phi):
    return np.full_like(np.asarray(phi, dtype=float), 0.5)

def phase_model_s(theta, phi1, a1):
    if not(2 * a1 < np.abs(theta) < np.pi):
        return 0
    else:
        num = np.sqrt((phi1 - np.cos(theta / 2)**2))
        denom = np.pi * np.sin(np.abs(theta) / 2)
        return num / denom

def gapAngles(phi):
    phi = phi * np.pi
    dim = 2**(2*L)
    result = np.zeros_like(phi)
    Z = np.array([[1, 0],
              [0, -1]], dtype=complex)
    if mu != 0.0:
        M = np.zeros((2**(2*L), 2**(2*L)), dtype=complex)
        for i in range(2 * L):
            Z_loc = np.kron(np.eye(2**i), np.kron(Z, np.eye(2**(2*L-i-1))))
            M += Z_loc
        Zsum = np.real(np.trace(np.exp(-mu * M))) 
    else:
        Zsum = dim
    atom = 0

    for m in range(2*L +1):
        am = comb(2*L-1, m-1)
        dm = comb(2*L, m)
        bm = comb(2*L-1, m)
        rm = min(am, bm) / dm
        alpha = am / dm
        beta = am / dm
        phi1 = 4 * alpha * (1-alpha) 
        a1 = np.arccos(np.sqrt(phi1))
        atom += dm / Zsum * np.exp(-mu * (2*m - 2 * L)) * np.abs(am - bm) / dm
        result += dm / Zsum * np.exp(-mu * (2*m - 2 * L)) * np.array([phase_model_s(theta, phi1, a1) for theta in phi])

    result = np.asarray(result)

    return result * np.pi/2

MODEL_REGISTRY = {
    "uniform": model_uniform,
    "gpAngles": gapAngles,
    "gxgx": model_gxgx,
    "ccx2": model_ccx2,
}


def _validate_density(values, context="model"):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(f"{context} must return finite, nonnegative values")
    return values


def _weighted_histogram_1d(values, weights, nbins, domain):
    edges = np.linspace(domain[0], domain[1], nbins + 1)
    hist, _ = np.histogram(values, bins=edges, weights=weights, density=False)
    total = np.sum(hist)
    if total <= 1e-8:
        raise ValueError("No weighted data points fall inside the 1D domain")
    return hist / total, edges


def l1_norm_1d(eigenvalues, weights, p, model=None, nbins=100, domain=(-1, 1),
               smooth_sigma=0.0, p0=0.0):
    """Return per-site/time L1 distances for phase densities."""

    if nbins < 1 or domain[0] >= domain[1]:
        raise ValueError("nbins must be positive and domain must be increasing")
    weights = _real_weights(weights, "L1 distance", require_nonnegative=True)

    result = np.empty(eigenvalues.shape[:2])
    atom_result = np.empty(eigenvalues.shape[:2]) if p0 > 0 else None

    centers = np.linspace(domain[0], domain[1], nbins, endpoint=False)
    width = (domain[1] - domain[0]) / nbins
    centers += width / 2

    if model is None:
        model_pdf = np.full(nbins, 1 / (domain[1] - domain[0]))
    elif callable(model):
        model_pdf = _validate_density(model(centers), "1D analytic model")
        print("1D analytic model", flush=True)
    elif isinstance(model, tuple) and len(model) == 2:
        model_pdf = np.interp(centers, model[0], model[1], left=0, right=0)
        model_pdf = _validate_density(model_pdf, "1D model")
    elif isinstance(model, tuple) and len(model) == 3:
        model_pdf, _ = _weighted_histogram_1d(model[0], model[1], nbins, domain)
        model_pdf = model_pdf / width
    else:
        raise ValueError("1D model must be callable, (x, pdf), (samples, weights), or None")
    for site in range(eigenvalues.shape[0]):
        for time in range(eigenvalues.shape[1]):
            values = eigenvalues[site, time]
            w = weights[site, time]
            atom_mask = np.abs(values - 1) < 1e-10
            smooth_data = values[~atom_mask] if p0 > 0 else values
            smooth_weight = w[~atom_mask] if p0 > 0 else w
            try:
                mass, _ = _weighted_histogram_1d(
                    np.angle(smooth_data) / np.pi,
                    smooth_weight, nbins, domain)
            except ValueError:
                print("ValueError", flush=True)
                mass = 0
            if smooth_sigma > 0:
                mass = gaussian_filter(mass, smooth_sigma, mode="constant")
                mass /= np.sum(mass)
            mass = mass * (1-p0)
            result[site, time] = np.sum(np.abs(mass / width - model_pdf)**p * width)**(1/p)

            if atom_result is not None:
                atom_mass = np.sum(w[atom_mask])
                atom_result[site, time] = abs(atom_mass - p0)
    if atom_result is None:
        return result
    return result, atom_result


def _edges(value, size):
    if isinstance(value, str) and "," in value:
        value = [float(item) for item in value.split(",")]
    if np.isscalar(value):
        if int(value) < 1:
            raise ValueError("bin count must be positive")
        return np.linspace(-size, size, int(value) + 1)
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or len(result) < 2 or np.any(np.diff(result) <= 0):
        raise ValueError("bin edges must be a strictly increasing 1D array")
    return result


def _model_grid_2d(model, xbins, ybins):
    xc = (xbins[:-1] + xbins[1:]) / 2
    yc = (ybins[:-1] + ybins[1:]) / 2
    X, Y = np.meshgrid(xc, yc, indexing="ij")
    area = np.diff(xbins)[:, None] * np.diff(ybins)[None, :]
    if callable(model):
        values = np.asarray(model(np.sqrt(X ** 2 + Y ** 2)), dtype=float)
        if values.shape != X.shape:
            raise ValueError("radial model returned the wrong shape")
        values = _validate_density(values.ravel(), "2D radial model").reshape(values.shape)
        P = values * area
    else:
        P = np.asarray(model, dtype=float)
        if P.shape != X.shape:
            raise ValueError("2D model grid shape does not match supplied bins")
        if not np.all(np.isfinite(P)) or np.any(P < 0):
            raise ValueError("2D model grid must be finite and nonnegative")
    # if p0 and callable(model):
    #     ix = np.searchsorted(xbins, 0, side="right") - 1
    #     iy = np.searchsorted(ybins, 0, side="right") - 1
    #     if not (0 <= ix < P.shape[0] and 0 <= iy < P.shape[1]):
    #         raise ValueError("supplied 2D bins must contain the origin")
    #     P[ix, iy] += p0
    return P


def l1_norm_2d(eigenvalues, weights, p, model=None, xbins=200, ybins=200,
               p0=0.0, size=1.0, smooth_sigma=0.0):
    if not 0 <= p0 <= 1 or smooth_sigma < 0:
        raise ValueError("p0 must be between 0 and 1 and smooth_sigma nonnegative")
    xbins, ybins = _edges(xbins, size), _edges(ybins, size)
    weights = _real_weights(weights, "2D L1 distance", require_nonnegative=True)

    filter_sigma = (smooth_sigma, smooth_sigma)
    model_grid = None if model is None else model

    if model_grid is not None and not (callable(model_grid) or isinstance(model_grid, tuple)):
        model_grid = _model_grid_2d(model_grid, xbins, ybins)
    smooth_result = np.empty(eigenvalues.shape[:2])
    atom_result = np.empty(eigenvalues.shape[:2]) if p0 > 0 else None
    area = np.diff(xbins)[:, None] * np.diff(ybins)[None, :]
    for site in range(eigenvalues.shape[0]):
        for time in range(eigenvalues.shape[1]):
            values = eigenvalues[site, time]
            frame_weights = weights[site, time]
            atom_mask = np.abs(values) <= 1e-10
            continuous_values = values[~atom_mask] if p0 > 0 else values
            continuous_weights = frame_weights[~atom_mask] if p0 > 0 else frame_weights
            H, _, _ = np.histogram2d(continuous_values.real,
                                     continuous_values.imag,
                                     bins=[xbins, ybins], weights=continuous_weights)
            if np.sum(H) <= 1e-8:
                raise ValueError("No weighted continuous 2D data points fall inside supplied bins")
            P_data = H / np.sum(H)

            if smooth_sigma > 0:
                P_data = gaussian_filter(P_data, filter_sigma, mode="constant")
                P_data /= np.sum(P_data)

            if model_grid is None:
                P_model = np.full_like(P_data, 1 / P_data.size)
            elif callable(model_grid):
                P_model = _model_grid_2d(model_grid, xbins, ybins)
                P_model /= np.sum(P_model)
            elif isinstance(model_grid, tuple): # Comparing against emperical (Haar models)
                P_model = _weighted_histogram_2d(
                    model_grid, xbins, ybins, filter_sigma
                )
            else:
                P_model = model_grid

            if smooth_sigma > 0 and not isinstance(model_grid, tuple):
                P_model = gaussian_filter(P_model, filter_sigma, mode="constant")
                P_model /= np.sum(P_model)
            smooth_result[site, time] = np.sum(np.abs(P_data / area - P_model / area)**p * area)**(1/p) * (1-p0)

            if atom_result is not None:
                atom_mass = np.sum(frame_weights[atom_mask])
                atom_result[site, time] = abs(atom_mass - p0)
    if atom_result is None:
        return smooth_result
    return smooth_result, atom_result


def _weighted_histogram_2d(model, xbins, ybins, filter_sigma):
    samples, sample_weights = model
    H, _, _ = np.histogram2d(np.real(samples), np.imag(samples),
                             bins=[xbins, ybins], weights=sample_weights)
    if np.sum(H) <= 1e-8:
        raise ValueError("No empirical model points fall inside supplied bins")
    H /= np.sum(H)
    if any(sigma > 0 for sigma in filter_sigma):
        H = gaussian_filter(H, filter_sigma, mode="constant")
        H /= np.sum(H)
    return H


def load_empirical_model(path):
    if os.path.isdir(path):
        radial = np.load(os.path.join(path, "radial.npy"))
        weights_path = os.path.join(path, "radial_weights.npy")
        if os.path.exists(weights_path):
            radial_weights = np.load(weights_path)
        else:
            warnings.warn(
                "No empirical sample weights found; using uniform weights.",
                UserWarning,
            )
            radial_weights = np.ones(radial.shape, dtype=float)
    else:
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if "radial" not in loaded:
                raise ValueError("empirical npz must contain radial")
            radial = loaded["radial"]
            if "radial_weights" in loaded:
                radial_weights = loaded["radial_weights"]
            else:
                warnings.warn(
                    "No empirical sample weights found; using uniform weights.",
                    UserWarning,
                )
                radial_weights = np.ones(radial.shape, dtype=float)
        else:
            radial, radial_weights = loaded, np.ones(loaded.shape, dtype=float)
            warnings.warn(
                "No empirical sample weights found; using uniform weights.",
                UserWarning,
            )
    if radial.shape != radial_weights.shape:
        raise ValueError("empirical eigenvalues and weights must have matching shapes")
    radial_weights = _real_weights(radial_weights, "empirical model", True)
    # Empirical files may preserve site/time dimensions (for example, the
    # analysis output stores radial.npy with shape (1, eigenvalues)).  Treat
    # all entries as samples when building the comparison distribution.
    return np.asarray(radial).reshape(-1), np.asarray(radial_weights).reshape(-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze weighted lattice spectra. Examples: --dim 1 --model_type analytic --model_name uniform; "
                    "--dim 2 --model_type empirical --model_file reference_results")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--dim", type=int, choices=(1, 2), default=1)
    parser.add_argument("--model_type", choices=("none", "analytic", "empirical"), default="none")
    parser.add_argument("--model_name", default=None,
                        help="Registered analytic model: uniform, gxgx, or ccx2")
    parser.add_argument("--model_file", default=None,
                        help="Empirical directory or npz/npy file")
    parser.add_argument("--compare_file", dest="model_file", help=argparse.SUPPRESS)
    parser.add_argument("--p0", type=float, default=0.0,
                        help="Origin mass for analytic radial models")
    parser.add_argument("--xbins", default=200,
                        help="2D x bin count or comma-separated edges")
    parser.add_argument("--ybins", default=200,
                        help="2D y bin count or comma-separated edges")
    parser.add_argument("--nbins1d", type=int, default=100)
    parser.add_argument("--smooth_sigma", type=float, default=0.0,
                        help="Automatic empirical Gaussian smoothing in bins. in grid units")
    parser.add_argument("--n_plot", type=int, default=None,
                        help="1-based site for spectrum and radial.npy")
    parser.add_argument("--size", type=float, default=1,
                        help="Half-width of the 2D histogram extent")
    parser.add_argument("--vmin", type=float, default=0,
                        help="Minimum value for the 2D spectrum color scale")
    parser.add_argument("--vmax", type=float, default=2,
                        help="Maximum value for the 2D spectrum color scale")
    parser.add_argument("--video_interval", type=int, default=200,
                        help="Delay between video frames in milliseconds")
    parser.add_argument("--p", type=int, default=1,
                        help="L_p norm measure. Default set to 1 for L1 norm")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.kmax < 1:
        raise ValueError("kmax must be positive")
    if args.size <= 0:
        raise ValueError("size must be positive")
    if args.vmin >= args.vmax:
        raise ValueError("vmin must be smaller than vmax")
    if args.video_interval <= 0:
        raise ValueError("video_interval must be positive")
    if args.smooth_sigma < 0:
        raise ValueError("smooth_sigma must be nonnegative")
    if args.model_type == "analytic" and args.model_name not in MODEL_REGISTRY:
        raise ValueError("analytic model_name must be one of: " + ", ".join(MODEL_REGISTRY))
    if args.model_type == "empirical" and not args.model_file:
        raise ValueError("empirical model_type requires --model_file")

    params, realizations, realization_weights = load_data(args.data_dir)
    times = params["times"]
    nsites = realizations[0].shape[0]
    global L
    global mu 
    L = params["L"]
    try:
        mu = params["mu"]
    except:
        mu = 0.0
    try:
        s_start = int(params["s_start"])
    except:
        s_start = 1
    selected_site = args.n_plot if args.n_plot is not None else int(params["n2"])
    if not s_start <= selected_site < s_start + nsites:
        raise ValueError(f"n_plot must be between {s_start} and {s_start + nsites - 1}")
    site_index = selected_site - s_start

    os.makedirs(args.output_dir, exist_ok=True)
    correlators = averaged_correlators(realizations, realization_weights, args.kmax)
    k2, k4 = averaged_cumulants(realizations, realization_weights, args.kmax)
    eigenvalues = np.concatenate(realizations, axis=2)
    weights = np.concatenate(
        [weight / len(realization_weights) for weight in realization_weights],
        axis=2,
    )
    # Optional at this point
    time_indices = [-1]#[5, 40, -1]#[4, 10, -1]
    radial = eigenvalues[site_index, time_indices]
    radial_weights = weights[site_index, time_indices]

    np.savez(
        os.path.join(args.output_dir, "parameters.npz"),
        **{key: params[key] for key in params.files}
    )
    np.save(os.path.join(args.output_dir, "correlators.npy"), correlators)
    np.save(os.path.join(args.output_dir, "k2.npy"), k2)
    np.save(os.path.join(args.output_dir, "k4.npy"), k4)
    np.save(os.path.join(args.output_dir, "radial.npy"), radial)
    np.save(os.path.join(args.output_dir, "radial_weights.npy"), radial_weights)

    maximum_imaginary = np.max(np.abs(np.imag(weights)))
    weights_are_real = True # maximum_imaginary <= 1e-3
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
            interval=args.video_interval,
        )
        # if np.min(real_weights) < -1e-10:
        #     print(
        #         "Skipping L1 analysis: signed weights are not a "
        #         "probability distribution."
        #     )
        # else:
        model = None
        if args.model_type == "analytic":
            model = MODEL_REGISTRY[args.model_name]
        elif args.model_type == "empirical":
            empirical, empirical_weights = load_empirical_model(args.model_file)
            model = (np.angle(empirical) / np.pi, empirical_weights, "empirical")
        distances = l1_norm_1d(eigenvalues, real_weights, args.p, model=model,
                                nbins=args.nbins1d, smooth_sigma=args.smooth_sigma
                                if args.model_type == "empirical" else 0, p0=args.p0)
        if args.p0 > 0.0:
            l1_norm, atom_norm = distances
            np.save(os.path.join(args.output_dir, "L1_norm_1d.npy"), l1_norm)
            np.save(os.path.join(args.output_dir, "L1_atom_1d.npy"), atom_norm)
        else:
            np.save(os.path.join(args.output_dir, "L1_norm_1d.npy"), distances)
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
            interval=args.video_interval,
        )
        if weights_are_real and args.model_type != "none":
            real_weights = np.real(weights)
            # if np.min(real_weights) < -1e-10:
            #     raise ValueError("2D L1 analysis requires nonnegative weights")
            model = None
            if args.model_type == "analytic":
                model = MODEL_REGISTRY[args.model_name]
            elif args.model_type == "empirical":
                empirical, empirical_weights = load_empirical_model(args.model_file)
                model = (empirical, empirical_weights)
            distances = l1_norm_2d(
                eigenvalues, real_weights, args.p, model=model,
                xbins=args.xbins, ybins=args.ybins, p0=args.p0,
                size=args.size, smooth_sigma=args.smooth_sigma
            )
            if args.p0 > 0:
                smooth_distance, atom_distance = distances
                np.save(os.path.join(args.output_dir, "L1_norm_2d.npy"), smooth_distance)
                np.save(os.path.join(args.output_dir, "L1_norm_2d_atom.npy"), atom_distance)
            else:
                np.save(os.path.join(args.output_dir, "L1_norm_2d.npy"), distances)
    print(f"Saved analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    main()

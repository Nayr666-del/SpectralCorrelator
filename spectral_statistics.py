"""Level spacings and adjacent-gap ratios for the models in lattice_site_v9."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.special import ndtr

import lattice_site_v9 as lattice
from combine_spectral_statistics import (
    FORMAT_VERSION, MODELS, STATISTICS, atomic_savez, encode_metadata,
    validate_metadata,
)


MODEL_PARAMETER_NAMES = {
    "L", "hx", "hz", "h2", "g", "tau", "delta", "disorder", "Periodic",
    "seed", "magnetization", "central_fraction", "unfolding", "bandwidth_scale",
}


# Check model overrides without allowing scheduling or output-path changes.
def validate_model_parameters(parameters):
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError("Model parameters must be an object keyed by model name")
    for model, overrides in parameters.items():
        if model not in MODELS:
            raise ValueError(f"Unknown model in model parameters: {model}")
        if not isinstance(overrides, dict):
            raise ValueError(f"Parameters for {model} must be an object")
        unknown = overrides.keys() - MODEL_PARAMETER_NAMES
        if unknown:
            raise ValueError(f"Unknown parameters for {model}: {sorted(unknown)}")
    return parameters


# Read the same model settings for local runs and individual array workers.
def load_model_parameters(filename):
    with open(filename, encoding="utf-8") as stream:
        parameters = json.load(stream)
    if parameters is None:
        raise ValueError("Model parameter file must contain an object")
    return validate_model_parameters(parameters)


# Calculate ratios without replacing genuine degeneracies by small gaps.
def spacing_ratios(spacings, circular=False):
    spacings = np.asarray(spacings, dtype=float)
    if circular:
        left = spacings
        right = np.roll(spacings, -1)
    else:
        left = spacings[:-1]
        right = spacings[1:]
    denominator = np.maximum(left, right)
    ratios = np.full(denominator.shape, np.nan, dtype=float)
    np.divide(np.minimum(left, right), denominator, out=ratios, where=denominator > 0)
    return ratios


# Evaluate the Gaussian KDE cumulative count with bounded temporary storage.
def unfold_levels(energies, selected, bandwidth_scale=1.0):
    energies = np.asarray(energies, dtype=float)
    selected = np.asarray(selected, dtype=float)
    bandwidth = np.std(energies, ddof=1) * energies.size ** (-1 / 5) * bandwidth_scale
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("KDE unfolding requires a positive finite bandwidth")
    counts = np.empty(selected.size, dtype=float)
    chunk_size = max(1, 1_000_000 // energies.size)
    for start in range(0, selected.size, chunk_size):
        stop = start + chunk_size
        kernels = (selected[start:stop, None] - energies[None, :]) / bandwidth
        ndtr(kernels, out=kernels)
        counts[start:stop] = kernels.sum(axis=1)
    return counts


# Calculate Ising statistics using raw gaps for r and optionally unfolded s.
def hamiltonian_statistics(eigenvalues, central_fraction=0.8, unfolding="kde", bandwidth_scale=1.0):
    if not 0 < central_fraction <= 1:
        raise ValueError("Require 0 < central_fraction <= 1")
    if unfolding not in ("kde", "mean") or not np.isfinite(bandwidth_scale) or bandwidth_scale <= 0:
        raise ValueError("Require unfolding kde|mean and bandwidth_scale > 0")
    energies = np.sort(np.asarray(eigenvalues, dtype=float))
    if energies.ndim != 1 or not np.all(np.isfinite(energies)):
        raise ValueError("Energies must be a finite one-dimensional array")
    retained = int(energies.size * central_fraction)
    if retained < 3:
        raise ValueError("Require at least three retained levels")
    start = (energies.size - retained) // 2
    selected = energies[start:start + retained]
    raw = np.diff(selected)
    if raw.mean() <= 0:
        raise ValueError("Spectrum has zero mean spacing")
    spacings = raw.copy()
    if unfolding == "kde":
        spacings = np.diff(unfold_levels(energies, selected, bandwidth_scale))
    if not np.all(np.isfinite(spacings)) or spacings.mean() <= 0 or np.any(spacings < 0):
        raise ValueError("Unfolding did not produce valid positive-mean spacings")
    return {"r": spacing_ratios(raw), "s": spacings / spacings.mean(), "s_raw": raw}


# Include the circular gap and ratio across the quasienergy branch cut.
def floquet_statistics(eigenvalues):
    eigenvalues = np.asarray(eigenvalues, dtype=complex)
    if eigenvalues.ndim != 1 or eigenvalues.size < 3 or not np.all(np.isfinite(eigenvalues)):
        raise ValueError("Require at least three finite Floquet eigenvalues")
    if np.any(np.abs(eigenvalues) == 0):
        raise ValueError("Floquet eigenvalues must be nonzero")
    phases = np.sort(np.mod(-np.angle(eigenvalues), 2 * np.pi))
    raw = np.diff(np.concatenate((phases, [phases[0] + 2 * np.pi])))
    return {"r": spacing_ratios(raw, circular=True), "s": raw / raw.mean(), "s_raw": raw}


# Record common run settings identically in every shard and combined file.
def _metadata(args, model):
    for name in ("L", "seed", "magnetization"):
        if type(getattr(args, name)) is not int:
            raise ValueError(f"{name} must be an integer for {model}")
    if args.L < 1:
        raise ValueError(f"Require L >= 1 for {model}")
    values = vars(args).copy()
    values["model"] = model
    for name in ("real_start", "real_end", "save_mode", "output_dir"):
        values.pop(name)
    values["format_version"] = FORMAT_VERSION
    dimension = 2 ** (2 * args.L)
    if model == "Heisenberg":
        nsites = 2 * args.L
        if abs(args.magnetization) > nsites or (nsites - args.magnetization) % 2 != 0:
            raise ValueError("Invalid magnetization sector for sum(Z_i)")
        dimension = math.comb(nsites, (nsites - args.magnetization) // 2)
    values["dimension"] = dimension
    validate_metadata(values)
    return values


# Build only the full-space Pauli operators needed by the selected model.
def _site_operators(nsites, model):
    xops = [lattice.local_op(lattice.X, 1, i, nsites) for i in range(nsites)]
    yops = []
    if model == "Heisenberg":
        yops = [lattice.local_op(lattice.Y, 1, i, nsites) for i in range(nsites)]
    zops = [lattice.local_op(lattice.Z, 1, i, nsites) for i in range(nsites)]
    return xops, yops, zops


# Keep each realization's diagonalization independent of scheduling and output.
def _calculate_model(args, model, metadata):
    nsites = 2 * args.L
    xops, yops, zops = _site_operators(nsites, model)
    dynamics_args = SimpleNamespace(**vars(args), Dynamics=model)
    samples = {name: [] for name in STATISTICS}
    r_offsets = [0]
    s_offsets = [0]
    for real in range(args.real_start, args.real_end):
        dynamics, unitary = lattice.build_dynamics(
            dynamics_args, 2 ** nsites, np.random.default_rng(args.seed + real),
            xops, yops, zops,
        )
        if model == "Heisenberg":
            # v9 counts one-bits as up; sum(Z_i) assigns those bits -1.
            dynamics = lattice.restrict_to_magnetization(dynamics, nsites, -args.magnetization)
        if unitary:
            eigenvalues = np.linalg.eigvals(dynamics)
        else:
            eigenvalues = np.linalg.eigvalsh(dynamics)
        del dynamics
        if unitary:
            stats = floquet_statistics(eigenvalues)
        else:
            stats = hamiltonian_statistics(
                eigenvalues, args.central_fraction, args.unfolding, args.bandwidth_scale,
            )
        del eigenvalues
        if args.save_mode == "shards":
            filename = Path(args.output_dir) / model / f"real_{real:06d}.npz"
            atomic_savez(filename, **stats, realization_id=np.asarray(real), metadata=encode_metadata(metadata))
        for name in STATISTICS:
            samples[name].append(stats[name])
        r_offsets.append(r_offsets[-1] + stats["r"].size)
        s_offsets.append(s_offsets[-1] + stats["s"].size)
        undefined = int(np.count_nonzero(np.isnan(stats["r"])))
        finite = stats["r"][np.isfinite(stats["r"])]
        mean_r = float("nan")
        if finite.size:
            mean_r = float(finite.mean())
        print(
            f"{model} realization={real} dimension={metadata['dimension']} "
            f"r={stats['r'].size} s={stats['s'].size} mean_r={mean_r:.8g} undefined_r={undefined}",
            flush=True,
        )
    result = {name: np.concatenate(samples[name]) for name in STATISTICS}
    result.update(
        realization_ids=np.arange(args.real_start, args.real_end, dtype=np.int64),
        r_offsets=np.asarray(r_offsets, dtype=np.int64),
        s_offsets=np.asarray(s_offsets, dtype=np.int64),
        metadata=encode_metadata(metadata),
    )
    return result


# Run selected models and optionally publish combined archives or shards.
def calculate_statistics(
    model="all", *, L=5, hx=0.9, hz=0.8, h2=0.9, g=0.2, tau=1.0,
    delta=1.5, disorder="Z", Periodic="N", n_reals=100, seed=1,
    magnetization=0, central_fraction=0.8, unfolding="kde", bandwidth_scale=1.0,
    real_start=0, real_end=None, save_mode="combined", output_dir=None,
    model_parameters=None,
):
    """Return model-keyed flat r/s/s_raw arrays, offsets, IDs and JSON metadata.

    n_reals describes the complete run; [real_start, real_end) selects its IDs.
    output_dir=None performs no file writes. Shard mode requires output_dir.
    model_parameters maps model names to overrides of the shared parameters.
    """
    args = SimpleNamespace(**locals())
    del args.model_parameters
    overrides = validate_model_parameters(model_parameters)
    if model not in (*MODELS, "all"):
        raise ValueError(f"Unknown model: {model}")
    for name in ("L", "n_reals", "seed", "magnetization", "real_start"):
        if type(getattr(args, name)) is not int:
            raise ValueError(f"{name} must be an integer")
    if real_end is None:
        args.real_end = n_reals
    if type(args.real_end) is not int or not 0 <= real_start < args.real_end <= n_reals:
        raise ValueError("Require 0 <= real_start < real_end <= n_reals")
    if save_mode not in ("combined", "shards"):
        raise ValueError("save_mode must be combined or shards")
    if save_mode == "shards" and output_dir is None:
        raise ValueError("Shard mode requires output_dir")
    selected = MODELS
    if model != "all":
        selected = (model,)
    model_args = {
        name: SimpleNamespace(**{**vars(args), **overrides.get(name, {})})
        for name in selected
    }
    metadata = {name: _metadata(model_args[name], name) for name in selected}
    results = {}
    for name in selected:
        result = _calculate_model(model_args[name], name, metadata[name])
        if output_dir is not None and save_mode == "combined":
            atomic_savez(Path(output_dir) / f"statistics_{name}.npz", **result)
        results[name] = result
    return results


# Expose v9 parameter names alongside realization and unfolding controls.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-D", "--Dynamics", dest="model", choices=(*MODELS, "all"), default="all")
    parser.add_argument("--L", type=int, default=5, help="Half the number of spin sites")
    for name, default in (("hx", 0.9), ("hz", 0.8), ("h2", 0.9), ("g", 0.2), ("tau", 1.0), ("delta", 1.5)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--disorder", "--doder", choices=("X", "Z"), default="Z")
    parser.add_argument("-P", "--Periodic", choices=("Y", "N"), default="N")
    parser.add_argument("--n_reals", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--magnetization", type=int, default=0, help="Heisenberg sector of sum(Z_i)")
    parser.add_argument("--central_fraction", type=float, default=0.8, help="Central fraction of Ising levels")
    parser.add_argument("--unfolding", choices=("kde", "mean"), default="kde", help="Ising spacing normalization")
    parser.add_argument("--bandwidth_scale", type=float, default=1.0, help="Multiplier of Scott's KDE bandwidth")
    parser.add_argument("--real_start", type=int, default=0)
    parser.add_argument("--real_end", type=int)
    parser.add_argument("--save_mode", choices=("combined", "shards"), default="combined")
    parser.add_argument("--output_dir", default="results/spectral_statistics")
    parser.add_argument("--model_parameters", help="JSON file of per-model overrides; these take precedence over shared options")
    args = parser.parse_args()
    try:
        if args.model_parameters is not None:
            args.model_parameters = load_model_parameters(args.model_parameters)
        calculate_statistics(**vars(args))
    except (ValueError, OSError, np.linalg.LinAlgError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

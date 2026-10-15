"""Validate and pool independent spectral-statistics realization files."""

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


MODELS = ("Ising", "Kicked", "Heisenberg")
FORMAT_VERSION = 1
STATISTICS = ("r", "s", "s_raw")


# Publish a complete archive without exposing partially written output.
def atomic_savez(filename, **arrays):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=filename.parent, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **arrays)
        os.replace(temporary, filename)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


# Store metadata as a scalar JSON string, requiring no pickle on loading.
def encode_metadata(metadata):
    return np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False))


# Check metadata independently of the calculation module and SciPy.
def validate_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    required = {
        "format_version", "model", "L", "hx", "hz", "h2", "g", "tau",
        "delta", "disorder", "Periodic", "seed", "n_reals", "magnetization",
        "central_fraction", "unfolding", "bandwidth_scale", "dimension",
    }
    if not required.issubset(metadata):
        raise ValueError(f"Missing metadata fields: {sorted(required - metadata.keys())}")
    for name in ("format_version", "L", "seed", "n_reals", "magnetization", "dimension"):
        if type(metadata[name]) is not int:
            raise ValueError(f"Metadata {name} must be an integer")
    if metadata["format_version"] != FORMAT_VERSION or metadata["model"] not in MODELS:
        raise ValueError("Unsupported format version or model")
    if metadata["L"] < 1 or metadata["seed"] < 0 or metadata["n_reals"] < 1:
        raise ValueError("Require L >= 1, seed >= 0, and n_reals >= 1")
    for name in ("hx", "hz", "h2", "g", "tau", "delta", "central_fraction", "bandwidth_scale"):
        if type(metadata[name]) not in (int, float) or not math.isfinite(metadata[name]):
            raise ValueError(f"Metadata {name} must be finite and numeric")
    if metadata["disorder"] not in ("X", "Z") or metadata["Periodic"] not in ("Y", "N"):
        raise ValueError("Invalid disorder axis or boundary conditions")
    if metadata["unfolding"] not in ("kde", "mean"):
        raise ValueError("Invalid unfolding method")
    if not 0 < metadata["central_fraction"] <= 1 or metadata["bandwidth_scale"] <= 0:
        raise ValueError("Require 0 < central_fraction <= 1 and bandwidth_scale > 0")
    nsites = 2 * metadata["L"]
    dimension = 2 ** nsites
    if metadata["model"] == "Heisenberg":
        magnetization = metadata["magnetization"]
        if abs(magnetization) > nsites or (nsites - magnetization) % 2 != 0:
            raise ValueError("Invalid magnetization sector for sum(Z_i)")
        dimension = math.comb(nsites, (nsites - magnetization) // 2)
    if metadata["dimension"] != dimension:
        raise ValueError("Analyzed dimension does not match model parameters")
    retained = dimension
    if metadata["model"] == "Ising":
        retained = int(dimension * metadata["central_fraction"])
    if retained < 3:
        raise ValueError("Require at least three retained levels")


# Validate one realization's arrays while preserving undefined ratios.
def validate_arrays(arrays, metadata):
    dimension = metadata["dimension"]
    s_count = dimension
    r_count = dimension
    if metadata["model"] == "Ising":
        retained = int(dimension * metadata["central_fraction"])
        s_count = retained - 1
        r_count = retained - 2
    for name in STATISTICS:
        values = arrays[name]
        if values.ndim != 1 or values.dtype.kind != "f":
            raise ValueError(f"{name} must be a one-dimensional floating-point array")
        expected = s_count
        if name == "r":
            expected = r_count
        if values.size != expected:
            raise ValueError(f"Invalid {name} shape: expected ({expected},), got {values.shape}")
        if np.any(np.isinf(values)) or np.any(values < 0):
            raise ValueError(f"Invalid values in {name}")
        if name != "r" and not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite spacings in {name}")
    if np.any(arrays["r"] > 1):
        raise ValueError("Ratios must lie in [0, 1] or be NaN")
    if not np.isclose(arrays["s"].mean(), 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("Normalized spacings must have mean one")
    if arrays["s_raw"].mean() <= 0:
        raise ValueError("Raw spacings must have positive mean")
    if metadata["model"] != "Ising":
        if not np.isclose(arrays["s_raw"].sum(), 2 * np.pi, rtol=1e-10, atol=1e-12):
            raise ValueError("Circular raw spacings must sum to 2*pi")


# Load a shard with explicit checks and filename context on failure.
def load_shard(filename):
    try:
        with np.load(filename, allow_pickle=False) as archive:
            required = {"metadata", "realization_id", *STATISTICS}
            if not required.issubset(archive.files):
                raise ValueError("Missing shard fields")
            encoded = archive["metadata"]
            if encoded.shape != () or encoded.dtype.kind != "U":
                raise ValueError("Metadata must be a scalar JSON string")
            metadata = json.loads(encoded.item())
            validate_metadata(metadata)
            identity = archive["realization_id"]
            if identity.shape != () or identity.dtype.kind not in "iu":
                raise ValueError("realization_id must be a scalar integer")
            realization_id = int(identity)
            if not 0 <= realization_id < metadata["n_reals"]:
                raise ValueError("Realization ID outside the configured range")
            arrays = {name: archive[name] for name in STATISTICS}
            validate_arrays(arrays, metadata)
        return realization_id, metadata, arrays
    except Exception as error:
        raise ValueError(f"Cannot load shard {filename}: {error}") from error


# Combine exactly the requested realizations, reading one shard at a time.
def combine_statistics(input_dir, output, n_reals):
    if type(n_reals) is not int or n_reals < 1:
        raise ValueError("Require n_reals >= 1")
    files = sorted(Path(input_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"No realization files found in {input_dir}")
    reference = None
    records = {}
    for filename in files:
        realization_id, metadata, arrays = load_shard(filename)
        if reference is None:
            reference = metadata
        if metadata != reference:
            raise ValueError(f"Incompatible metadata in {filename}")
        if realization_id in records:
            raise ValueError(f"Duplicate realization ID {realization_id}")
        records[realization_id] = (filename, arrays["r"].size, arrays["s"].size)
        del arrays
    expected = set(range(n_reals))
    actual = set(records)
    if actual != expected:
        raise ValueError(
            f"Realization IDs do not match: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    if reference["n_reals"] != n_reals:
        raise ValueError("Requested n_reals does not match shard metadata")
    if Path(output).resolve() in {record[0].resolve() for record in records.values()}:
        raise ValueError("Output must not overwrite an input shard")
    ids = np.arange(n_reals, dtype=np.int64)
    r_offsets = np.concatenate(([0], np.cumsum([records[i][1] for i in ids])))
    s_offsets = np.concatenate(([0], np.cumsum([records[i][2] for i in ids])))
    combined = {
        "r": np.empty(int(r_offsets[-1]), dtype=float),
        "s": np.empty(int(s_offsets[-1]), dtype=float),
        "s_raw": np.empty(int(s_offsets[-1]), dtype=float),
    }
    for i in ids:
        identity, metadata, arrays = load_shard(records[i][0])
        if identity != i or metadata != reference:
            raise ValueError(f"Shard changed while combining: {records[i][0]}")
        combined["r"][r_offsets[i]:r_offsets[i + 1]] = arrays["r"]
        for name in ("s", "s_raw"):
            combined[name][s_offsets[i]:s_offsets[i + 1]] = arrays[name]
        del arrays
    combined.update(
        realization_ids=ids, r_offsets=r_offsets, s_offsets=s_offsets,
        metadata=encode_metadata(reference),
    )
    atomic_savez(output, **combined)
    return combined


# Expose aggregation without requiring the simulation dependencies.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n_reals", type=int, required=True)
    args = parser.parse_args()
    try:
        result = combine_statistics(args.input_dir, args.output, args.n_reals)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(f"Saved {args.n_reals} realizations to {args.output}; r={result['r'].size}, s={result['s'].size}")


if __name__ == "__main__":
    main()

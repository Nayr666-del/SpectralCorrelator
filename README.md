# Quantum Information Scrambling from a Spectral Perspective

This repository studies thermalization and scrambling through the spectra of
correlation operators on spin chains. `lattice_site_v9.py` generates correlator
spectra; `analysis_v4.py` processes those spectra; `spectral_statistics.py`
calculates level spacings and adjacent-gap ratios of the underlying models.

## Dependencies

Use Python 3; the cluster scripts load `python/3.13.9`.

| Component | Dependencies |
| --- | --- |
| `lattice_site_v9.py`, `spectral_statistics.py`, numerical tests | NumPy, SciPy, tqdm |
| `combine_spectral_statistics.py` | NumPy |
| `submit_spectral_statistics.py` | Python standard library |
| `analysis_v4.py` | NumPy, SciPy, Matplotlib; FFmpeg for video output |
| Slurm workflows | Bash, Slurm, and the cluster's environment-module setup |

Install the Python packages in your selected environment:

```bash
python3 -m pip install numpy scipy tqdm matplotlib
```

FFmpeg is a separate executable; the existing analysis jobs load the cluster's
`ffmpeg/8.0.1` module. Local simulation and combination runs do not require Slurm.

## Running `lattice_site_v9.py`

For each disorder realization, the script constructs a Hamiltonian or Floquet
unitary, diagonalizes it, and computes the eigenvalues of `A(t) B` at the
requested sites and times. It uses dense matrices and full diagonalization.
The chain has **`2 * L` spin sites**, with Hilbert-space dimension `2 ** (2 * L)`.

A small, eight-site Ising run:

```bash
python3 lattice_site_v9.py -D Ising -P N --L 4 \
    --hx 0.9 --hz 0.8 --g 0.2 --tau 1.0 --disorder Z \
    --obsA sz --obsB sz --n2 4 --s_start 1 --s_end 8 \
    --tstart 0 --tend 5 --N 1 --n_reals 2 --seed 1 \
    --output_dir results/ising_v9
```

This evaluates A on sites 1 through 8, fixes B on site 4, and saves spectra at
times `0, 1, 2, 3, 4` for realizations 0 and 1.

### Dynamics and physical parameters

Select one model with `-D` / `--Dynamics`:

| Value | Construction |
| --- | --- |
| `Ising` (default) | Static nearest-neighbor ZZ Hamiltonian, uniform X/Z fields, and spatial Gaussian disorder |
| `Kicked` | Kicked Ising unitary: X-field kick followed by the ZZ/Z-field step; disorder joins the step for its selected axis |
| `Heisenberg` | Kicked XXZ chain with staggered Z fields `hz` and `h2`, and Z disorder |
| `GOE` | Real symmetric Gaussian random Hamiltonian |
| `UHaar` | Haar-random unitary on the full Hilbert space |
| `MHaar` | Independent Haar-random blocks at fixed total Z magnetization |
| `Circuit` | Two layers of random nearest-neighbor gates conserving total Z magnetization |
| `shell-haar` | Independent Haar-random blocks in finite-width energy-density shells of the Ising Hamiltonian |

| Option | Default | Meaning |
| --- | --- | --- |
| `--L` | `5` | Half the number of spin sites |
| `-P` / `--Periodic` | `N` | `Y` includes the closing interaction bond; `N` uses open boundaries for Ising/XXZ models |
| `--hx`, `--hz` | `0.9`, `0.8` | Ising transverse/longitudinal fields; Heisenberg uses `hz` as a staggered Z field |
| `--h2` | `0.9` | Heisenberg staggered Z kick strength |
| `--delta` | `1.5` | XXZ anisotropy |
| `--g` | `0.2` | Amplitude of independent, standard-normal site disorder |
| `--tau` | `1.0` | Multiplies every Ising/XXZ Hamiltonian term, including both Floquet steps |
| `--disorder` / `--doder` | `Z` | Ising disorder axis, `X` or `Z` |

Heisenberg disorder is always Z-directed. Its axis option instead selects the
Floquet step containing the disorder. `hx` is unused for Heisenberg, and `h2` and
`delta` are unused for Ising. The random-matrix/circuit modes ignore the chain
couplings. The circuit uses its own open gate arrangement regardless of `-P`.

Heisenberg conserves total Z magnetization, but **v9 calculates correlators in the
full Hilbert space**. Fixed-sector diagonalization is provided separately by
`spectral_statistics.py`.

### Observables, sites, and times

| Option | Default | Meaning |
| --- | --- | --- |
| `--obsA`, `--obsB` | `sx`, `sx` | Observable names; see groups below |
| `--s_start`, `--s_end` | `1`, last site | Inclusive range of A's starting sites |
| `--n2` | `5` | Fixed starting site of **B** |
| `--ortho_obs` | `N` | Use full A (`N`), remove its diagonal (`Y`), or retain only its diagonal (`H`), in the dynamics eigenbasis |
| `--alpha` | `0.1` | Mixing parameter in `[0, 1]` for `z+x`, `z+y`, and `x+y` |
| `--t_type` | `Linear` | Time sampling: `Linear` or `Log` |
| `--tstart`, `--tend` | `0`, `5` | Time-range endpoints |
| `--N` | `1` | Linear time step; in `Log` mode, number of initial log-spaced samples |

Sites are **1-based**. For a two-site observable, the specified site is its left
endpoint and must be at most `2 * L - 1`. Set `--s_end` explicitly when A has
two-site support; its default is the last spin site. For small chains, also set
`--n2` explicitly so B fits.

Observable groups:

- Single-site: `sx`, `sy`, `sz`, and the fixed random Hermitian observable `sr`.
- Single-site Pauli sums: `z+x`, `z+y`, `x+y`; the coefficients are
  `cos(alpha * pi / 2)` and `sin(alpha * pi / 2)`.
- Two-site: `sgx`, `sgy`, `sgz`, their `_d` variants, `sR`, `sX`, `sY`, `sZ`,
  `sXZ`, `sc`, `sc_005`, `sc_010`, `sc_020`, `sc_095`, `O0`, and `O2`.
  Their exact matrices are in `OBSERVABLES` in `lattice_site_v9.py`.

Linear times use `np.arange(tstart, tend, N)`: **`tend` is exclusive**. For one
snapshot at time 49, use `--tstart 49 --tend 50 --N 1`. Use integer times for
Floquet evolution. Log mode requires a positive starting time; it truncates samples
to integers, removes repeats, and omits the final generated point in the current
implementation. Use enough samples for a nonempty time array and inspect the
saved `times` array for the exact schedule. `--N` does not specify the chain size.

### Ensembles and energy shells

The default `--mu 0 --beta 0` uses the infinite-temperature density matrix.

| Options | Behavior |
| --- | --- |
| `--mu VALUE --m_operator total_z` | Uses `rho = exp(-mu * sum(Z_i)) / Z`; nonzero `mu` requires conservation of total Z, checked for every realization |
| `-D Ising --beta VALUE` | Uses the exact Ising Gibbs state `rho = exp(-beta * H) / Z`; leave `mu=0` for this ensemble |
| `-D Ising --energy_density E --energy_width W` | Uniform microcanonical ensemble over energies satisfying `E - W/2 <= energy/(2*L) <= E + W/2` |
| `-D shell-haar --energy_width W` | Constructs zero-centered energy-density shells of width W and independent Haar blocks within them |
| `-D shell-haar --energy_width W --beta VALUE` | Weights each state using its shell-center energy, with normalization accounting for shell dimensions |

`energy_width` must be positive. For Ising microcanonical runs, supply both shell
options and keep `mu=beta=0`; an empty shell raises an error. For `shell-haar`,
`energy_width` is required, and optional `energy_density` selects a microcanonical
analysis window using the underlying Ising energies. That window also requires
`mu=beta=0`. Nonzero `beta` is supported only for Ising and `shell-haar`.

### Realizations and Slurm Python commands

`--n_reals` defaults to 1. `--real_start` is inclusive (default 0), and
`--real_end` is exclusive (default `n_reals`, capped at `n_reals` if supplied).
The model RNG uses `seed + realization_id`, with `--seed 1` by default.
Use different output directories for different parameter sets.

The following are the **Python invocations to place inside an existing Slurm
array script**, after loading Python and setting BLAS threads to the allocated
CPU count. They assume array IDs `0..99` and write each task's results to a
separate scratch directory, including its own `parameters.npz`. Copy the results
to persistent storage as in the existing job scripts; when collecting them into
one dataset, retain one shared parameter archive rather than copying task
archives concurrently.

Kicked Ising, two-site observables, one snapshot per realization:

```bash
python3 lattice_site_v9.py -D Kicked -P N --L 6 \
    --hx 0.9 --hz 0.8 --tau 0.65 --g 0.2 --disorder Z \
    --obsA sc --obsB sc --n2 6 --s_start 6 --s_end 6 --ortho_obs N \
    --tstart 49 --tend 50 --N 1 --n_reals 100 \
    --real_start "$SLURM_ARRAY_TASK_ID" --real_end "$((SLURM_ARRAY_TASK_ID + 1))" \
    --seed 1 --output_dir "/scratch/$USER/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_kicked"
```

Kicked Heisenberg, with a magnetization-weighted ensemble:

```bash
python3 lattice_site_v9.py -D Heisenberg -P N --L 6 \
    --hz 0.8 --h2 0.461538 --delta 1.5 --tau 0.65 --g 0.2 --disorder Z \
    --obsA sz --obsB sz --n2 6 --s_start 6 --s_end 6 \
    --mu 0.5 --m_operator total_z --tstart 0 --tend 50 --N 1 --n_reals 100 \
    --real_start "$SLURM_ARRAY_TASK_ID" --real_end "$((SLURM_ARRAY_TASK_ID + 1))" \
    --seed 1 --output_dir "/scratch/$USER/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_heisenberg"
```

Static Ising, with a Gibbs ensemble:

```bash
python3 lattice_site_v9.py -D Ising -P N --L 6 \
    --hx 1.2 --hz 0.5 --tau 0.75 --g 0.2 --disorder Z \
    --obsA sz --obsB sz --n2 6 --s_start 6 --s_end 6 \
    --beta 0.7 --mu 0 --tstart 0 --tend 50 --N 1 --n_reals 100 \
    --real_start "$SLURM_ARRAY_TASK_ID" --real_end "$((SLURM_ARRAY_TASK_ID + 1))" \
    --seed 1 --output_dir "/scratch/$USER/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_ising_beta"
```

For an Ising microcanonical run, replace `--beta 0.7` with
`--beta 0 --energy_density 0 --energy_width 0.5`. For shell-Haar evolution, use
`-D shell-haar --energy_width 0.5`; optionally add either `--beta VALUE` or
`--energy_density VALUE` with the ensemble restrictions above.

### Output files and analysis

Files are written under `--output_dir` (default `results`):

| File | Content |
| --- | --- |
| `real_000.npy`, `real_001.npy`, ... | Complex eigenvalues of `A(t) B`, with shape `(A sites, times, 2 ** (2 * L))` |
| `weights_000.npy`, ... | Matching spectral coefficients, saved when `mu != 0`, `beta != 0`, or a microcanonical window is selected |
| `parameters.npz` | Parsed arguments and the actual `times` array; shell runs include selection counts and relevant shell metadata |

The first array axis follows `s_start..s_end`, and the second follows `times`.
Spectral coefficients are normalized to sum to one. For non-normal correlators
they may be signed or complex; they are not necessarily probabilities.

`analysis_v4.py` reads `parameters.npz`, realization spectra, and available
weights. Its products include `correlators.npy`, `k2.npy`, `k4.npy`,
`radial.npy`, `radial_weights.npy`, a copied `parameters.npz`, spectrum videos,
and optional `L1_norm_1d.npy` / `L1_norm_2d.npy` comparisons. Run
`python3 analysis_v4.py --help` for analysis options.

## Level spacings and gap ratios

`spectral_statistics.py` returns `r`, mean-one `s`, and raw gaps `s_raw` for
`Ising`, `Kicked`, `Heisenberg`, or `all`. It reuses v9's model constructors.
Heisenberg selects a sector of `sum(Z_i)` with `--magnetization` (default 0).
Full matrices are still constructed before restriction.

Ising defaults to the central 80% of levels and Gaussian-KDE unfolding;
`--central_fraction` and `--bandwidth_scale` control these choices.
`--unfolding mean --central_fraction 1` uses simple mean normalization of all
energy gaps. Floquet models include the closing circular gap and periodic
ratios. Ratios always use raw gaps; undefined zero/zero ratios remain `NaN`.

Local calculation, saving one archive per model:

```bash
python3 spectral_statistics.py --Dynamics all --L 4 --n_reals 100 \
    --hx 0.9 --hz 0.8 --h2 0.9 --g 0.2 --tau 1 --delta 1.5 \
    --output_dir results/spacings
```

For different parameters in each model, edit `spectral_parameters.example.json`
or create your own JSON file with the same model names. For example:

```json
{
  "Ising": {"hx": 1.2, "hz": 0.5, "tau": 0.75},
  "Kicked": {"hx": 0.9, "hz": 0.8, "tau": 0.65},
  "Heisenberg": {"hz": 0.8, "h2": 0.461538, "delta": 1.5, "tau": 0.65}
}
```

```bash
python3 spectral_statistics.py --Dynamics all --L 4 --n_reals 100 \
    --model_parameters spectral_parameters.example.json --output_dir results/spacings
```

Entries override shared command-line settings **only for their model**; omitted
values retain the shared settings/defaults. You may also vary `L`, `g`,
`disorder`, `Periodic`, `seed`, magnetization, and unfolding options per model.
Scheduling options such as `n_reals`, realization ranges, and output paths remain
shared. Output metadata records the effective parameters of each model.

The Python API is `calculate_statistics(model="Heisenberg", L=4, n_reals=100)`;
it returns a model-keyed dictionary and writes no files unless `output_dir` is
specified. `n_reals` describes the complete run, and `[real_start, real_end)`
selects its realization IDs with reproducible seeds `seed + realization_id`.
For multiple models, pass the same mapping directly as
`calculate_statistics(model="all", L=4, model_parameters={"Ising": {"tau": 0.75}, "Kicked": {"tau": 0.65}})`.

For one realization per Slurm array task:

```bash
python3 spectral_statistics.py --Dynamics Heisenberg --L 6 --n_reals 100 \
    --real_start "$SLURM_ARRAY_TASK_ID" --real_end "$((SLURM_ARRAY_TASK_ID + 1))" \
    --save_mode shards --output_dir results/spacings/shards
```

Combine all expected IDs after the jobs finish:

```bash
python3 combine_spectral_statistics.py \
    --input_dir results/spacings/shards/Heisenberg \
    --output results/spacings/statistics_Heisenberg.npz --n_reals 100
```

Keep each model/parameter set in its own shard directory and combined output
outside that directory. The merger rejects missing/duplicate IDs and incompatible
or malformed files. Combined archives contain flat `r`, `s`, `s_raw` arrays,
`realization_ids`, `r_offsets`, `s_offsets` (also for `s_raw`), and JSON `metadata`.
Load them with `np.load(path, allow_pickle=False)`; use `np.nanmean(r)` if needed.

For automatic submission and dependent aggregation, run
`bash spectral_statistics.sh submit --L 6 --tau 0.65`. It defaults to 100
realizations per model, 300 tasks, and 10 concurrent workers. The job file follows
the other Slurm scripts: edit its `#SBATCH` header for worker resources, its run
variables for model selection/output, and its Python command for shared physics
defaults. The launcher calculates the array range from the model/realization
counts and reads the concurrency limit from the header's `%10` suffix.
You can also set `SPECTRAL_MODELS`, `SPECTRAL_N_REALS`, `SPECTRAL_CPUS`,
`SPECTRAL_MEM`, `SPECTRAL_TIME`, `SPECTRAL_CONCURRENCY`, and
`SPECTRAL_OUTPUT_ROOT`. The script also exposes partition, Python-module, and
merge-resource settings. Each run saves `retry.sh` (pass failed array IDs) and
`merge.sh` for manual recovery. Do not change source code between submission and
retries; each worker still needs memory for the full matrices and solver.
Launch through `bash spectral_statistics.sh submit ...` so the Python helper
`submit_spectral_statistics.py` prepares the run directory, configuration
snapshot, worker array, and single dependent merge job. The `submit`, `worker`,
and `merge` modes and existing environment overrides remain available.
To use different parameters per model, submit with
`bash spectral_statistics.sh submit --L 6 --model_parameters spectral_parameters.example.json`
or set `SPECTRAL_MODEL_PARAMETERS` to the JSON path. Submission saves a copy as
`model_parameters.json` in the run directory; workers and retries use that copy,
so later edits to the original configuration do not change queued jobs.

## Validation

```bash
python3 -m unittest -v test_spectral_statistics test_lattice_site_v9
```

Tests include 100-realization sequential/batch comparisons and simulated Slurm
execution; they do not submit cluster jobs.

"""Submit spectral-statistics workers and a dependent merge job to Slurm."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile


MODELS = ("Ising", "Kicked", "Heisenberg")
SCHEDULING_OPTIONS = {
    "-D", "--Dynamics", "--n_reals", "--real_start", "--real_end",
    "--save_mode", "--output_dir",
}


# Read resource defaults from the editable Slurm job header.
def slurm_defaults(script):
    defaults = {}
    for line in Path(script).read_text().splitlines():
        if line.startswith("#SBATCH "):
            arguments = shlex.split(line[len("#SBATCH "):])
            if len(arguments) == 1 and "=" in arguments[0]:
                name, value = arguments[0].split("=", 1)
                defaults[name] = value
    return defaults


# Separate per-model configuration from ordinary shared physics options.
def physics_options(arguments, model_parameters):
    physics = []
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        option = argument.split("=", 1)[0]
        if option in SCHEDULING_OPTIONS:
            raise ValueError(f"Set scheduling options using SPECTRAL_* variables, not {argument}")

        if option == "--model_parameters":
            if "=" in argument:
                model_parameters = argument.split("=", 1)[1]
            else:
                index += 1
                if index == len(arguments):
                    raise ValueError("--model_parameters requires a JSON file")
                model_parameters = arguments[index]
            if not model_parameters:
                raise ValueError("--model_parameters requires a JSON file")
        else:
            physics.append(argument)
        index += 1
    return physics, model_parameters


# Preserve spaces and literal arguments in generated helper scripts.
def write_command(filename, arguments):
    command = shlex.join([str(argument) for argument in arguments])
    Path(filename).write_text(f"#!/bin/bash\nset -eu\nexec {command}\n")


# Validate Slurm's parsable job ID, including optional cluster suffixes.
def submit_job(arguments):
    result = subprocess.run(["sbatch", *arguments], check=True, capture_output=True, text=True)
    identity = result.stdout.strip().split(";", 1)[0]
    if not identity.isascii() or not identity.isdigit():
        raise ValueError(f"Unexpected sbatch response: {result.stdout.strip()}")
    return identity


# Snapshot settings before launching independent workers and aggregation.
def submit(args):
    script = Path(args.script).resolve()
    models = args.models.split()
    if not models or len(set(models)) != len(models) or any(model not in MODELS for model in models):
        raise ValueError("Select unique model names from Ising, Kicked, Heisenberg")
    if args.n_reals < 1:
        raise ValueError("n_reals must be a positive integer")
    defaults = slurm_defaults(script)
    partition = os.environ.get("SPECTRAL_PARTITION", defaults.get("--partition", "medium"))
    walltime = os.environ.get("SPECTRAL_TIME", defaults.get("--time", "1-00:00:00"))
    memory = os.environ.get("SPECTRAL_MEM", defaults.get("--mem", "30G"))
    cpus = int(os.environ.get("SPECTRAL_CPUS", defaults.get("--cpus-per-task", "1")))
    array_limit = defaults.get("--array", "0-299%10").partition("%")[2] or "10"
    concurrency = int(os.environ.get("SPECTRAL_CONCURRENCY", array_limit))
    if cpus < 1 or concurrency < 1:
        raise ValueError("CPUS and CONCURRENCY must be positive integers")
    physics, model_parameters = physics_options(args.physics, args.model_parameters)
    if model_parameters and not Path(model_parameters).is_file():
        raise ValueError(f"Cannot read model parameter file: {model_parameters}")
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    label = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ_")
    run_dir = Path(tempfile.mkdtemp(prefix=label, dir=output_root))
    (run_dir / "logs").mkdir()
    (run_dir / "shards").mkdir()
    if model_parameters:
        snapshot = run_dir / "model_parameters.json"
        shutil.copyfile(model_parameters, snapshot)
        physics.extend(("--model_parameters", str(snapshot)))
    payload = [
        str(script.parent), str(run_dir), str(args.n_reals), args.python_module,
        args.python_bin, str(len(models)), *models, "--", *physics,
    ]
    worker_options = [
        "--parsable", f"--partition={partition}", f"--time={walltime}",
        "--ntasks=1", f"--cpus-per-task={cpus}", f"--mem={memory}",
        f"--chdir={script.parent}", "--job-name=spectral-statistics",
        f"--output={run_dir}/logs/spectral-%A_%a.out",
        f"--error={run_dir}/logs/spectral-%A_%a.err",
    ]
    write_command(run_dir / "merge.sh", [str(script), "merge", *payload])
    retry_arguments = shlex.join([*worker_options, str(script), "worker", *payload])
    retry_command = 'sbatch --array="${1:?Specify failed task IDs, e.g. 2,7-9}" ' + retry_arguments
    (run_dir / "retry.sh").write_text(f"#!/bin/bash\nset -eu\nexec {retry_command}\n")
    print(f"Run directory: {run_dir}", flush=True)
    task_count = len(models) * args.n_reals
    array_id = submit_job([
        *worker_options, f"--array=0-{task_count - 1}%{concurrency}",
        str(script), "worker", *payload,
    ])
    (run_dir / "array_job_id").write_text(array_id + "\n")
    print(f"Array job: {array_id} ({task_count} tasks)", flush=True)
    # The generated merge script has no worker-array #SBATCH directives.
    merge_id = submit_job([
        "--parsable", f"--partition={partition}",
        f"--time={os.environ.get('SPECTRAL_MERGE_TIME', '00:30:00')}",
        "--ntasks=1", "--cpus-per-task=1",
        f"--mem={os.environ.get('SPECTRAL_MERGE_MEM', '4G')}",
        f"--chdir={script.parent}", "--job-name=spectral-merge",
        f"--output={run_dir}/logs/merge-%j.out", f"--error={run_dir}/logs/merge-%j.err",
        f"--dependency=afterok:{array_id}", str(run_dir / "merge.sh"),
    ])
    (run_dir / "merge_job_id").write_text(merge_id + "\n")
    print(f"Merge job: {merge_id}", flush=True)
    return run_dir


# Keep submission dependent only on Python's standard library.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--n_reals", type=int, required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--python_module", required=True)
    parser.add_argument("--python_bin", required=True)
    parser.add_argument("--model_parameters", default="")
    parser.add_argument("physics", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.physics and args.physics[0] == "--":
        args.physics = args.physics[1:]
    try:
        submit(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        details = str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr:
            details = error.stderr.strip()
        parser.exit(1, f"Error: {details}\n")


if __name__ == "__main__":
    main()

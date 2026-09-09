"""
Environment/reproducibility recording framework versions,
hardware, and seeds, logged alongside every experiment's results.
"""

# json is used to save experiment results in a structured .json file.
import json

# platform provides information about the operating system/platform.
import platform

# random controls Python's built-in random number generator.
import random

# dataclass utilities are imported here for possible structured
# configuration/data objects.
from dataclasses import dataclass, field

# NumPy has its own random number generator, so we need to seed it too.
import numpy as np

# PyTorch provides the model, tensors, GPU detection, and PyTorch RNG.
import torch


def set_all_seeds(seed: int = 42):
    """
    Set random seeds across the main libraries used by the project.

    The goal is reproducibility.

    If the same seed and experimental setup are used, we try to
    make the experiment produce the same or very similar results.
    """

    # Set the seed for Python's built-in random module.
    random.seed(seed)

    # Set the seed for NumPy's random number generator.
    np.random.seed(seed)

    # Set the seed for PyTorch operations running on the CPU.
    torch.manual_seed(seed)

    # Check whether a CUDA-compatible GPU is available.
    if torch.cuda.is_available():

        # Set the seed for random operations on all available GPUs.
        #
        # manual_seed_all() is used instead of only seeding GPU 0
        # because the project could potentially use multiple GPUs.
        torch.cuda.manual_seed_all(seed)


def record_environment() -> dict:
    """
    Snapshot of hardware + exact library versions
    for the reproducibility log.

    This records the environment in which the experiment was run.

    That is important because model results can depend on:
        - Python version
        - PyTorch version
        - Transformers version
        - datasets version
        - GPU
        - CUDA version
    """

    # Import the versions of the Hugging Face libraries used
    # by the project.
    import transformers
    import datasets

    # Create a dictionary containing the environment information.
    env = {

        # Record the Python version.
        "python_version": platform.python_version(),

        # Record the operating-system/platform information.
        "platform": platform.platform(),

        # Record the exact installed PyTorch version.
        "torch_version": torch.__version__,

        # Record the exact Transformers version.
        "transformers_version": transformers.__version__,

        # Record the exact Hugging Face datasets version.
        "datasets_version": datasets.__version__,

        # Record whether CUDA/GPU support is available.
        "cuda_available": torch.cuda.is_available(),
    }

    # Only try to record GPU-specific information if CUDA
    # is actually available.
    if torch.cuda.is_available():

        # Record the name of the GPU being used.
        #
        # get_device_name(0) means GPU device 0.
        env["gpu_name"] = torch.cuda.get_device_name(0)

        # Record the CUDA version that PyTorch was built against.
        env["cuda_version"] = torch.version.cuda

    # Return the complete environment dictionary.
    return env


def log_experiment(
    name: str,
    config: dict,
    results: dict,
    out_dir: str = "results"
):
    """
    Save a full JSON record and maintain a CSV whose columns
    are the union of fields from all experiments.

    This prevents heterogeneous experiments
    (e.g. pruning vs. distillation) from breaking CSV logging.
    """

    # These imports are only needed when this function is called.
    import csv
    import os

    # Create the output directory if it does not already exist.
    #
    # exist_ok=True means no error is raised if the directory
    # already exists.
    os.makedirs(out_dir, exist_ok=True)

    # Combine all information into one experiment record.
    #
    # The record contains:
    #
    # experiment name
    # + experiment configuration
    # + experiment results
    # + environment information
    #
    # The ** syntax expands each dictionary into the new dictionary.
    record = {
        "experiment": name,
        **config,
        **results,
        **record_environment()
    }

    # Construct the path for this experiment's individual JSON file.
    #
    # Example:
    #     results/bert_baseline.json
    json_path = os.path.join(
        out_dir,
        f"{name}.json"
    )

    # Open/create the JSON file for writing.
    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        # Save the experiment record as formatted JSON.
        #
        # indent=2 makes the file human-readable.
        #
        # default=str allows values that JSON does not normally
        # understand to be converted to strings instead of causing
        # the logging process to fail.
        json.dump(
            record,
            f,
            indent=2,
            default=str
        )

    # Construct the path to the combined CSV file.
    #
    # All experiments will be stored in:
    #
    #     results/metrics.csv
    csv_path = os.path.join(
        out_dir,
        "metrics.csv"
    )

    # Start with an empty list of previous experiment rows.
    rows = []

    # Check whether metrics.csv already exists.
    if os.path.isfile(csv_path):

        # If it exists, open it and load all existing rows.
        with open(
            csv_path,
            newline="",
            encoding="utf-8"
        ) as f:

            # DictReader converts each CSV row into a dictionary.
            rows = list(csv.DictReader(f))

    # Add the current experiment's record to the rows.
    rows.append(record)

    # Keep track of every unique column name that appears
    # across all experiments.
    fieldnames = []

    # Look through every experiment record.
    for row in rows:

        # Look through every field in that experiment.
        for key in row:

            # Only add a column if it has not already been added.
            if key not in fieldnames:
                fieldnames.append(key)

    # Rewrite the CSV using the complete set of columns.
    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        # Create a CSV writer using the union of all field names.
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        # Write the column names/header.
        writer.writeheader()

        # Write every experiment row.
        writer.writerows(rows)

    # Return the complete record so the caller can also use it
    # immediately in Python.
    return record
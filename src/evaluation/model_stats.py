"""Environment/reproducibility recording framework versions,
hardware, and seeds, logged alongside every experiment's results."""
import json
import platform
import random
from dataclasses import dataclass, field

import numpy as np
import torch


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def record_environment() -> dict:
    """Snapshot of hardware + exact library versions for the reproducibility log."""
    import transformers
    import datasets

    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(0)
        env["cuda_version"] = torch.version.cuda
    return env


def log_experiment(name: str, config: dict, results: dict, out_dir: str = "results"):
    """Save a full JSON record and maintain a CSV whose columns are the union
    of fields from all experiments. This prevents heterogeneous experiments
    (e.g. pruning vs. distillation) from breaking CSV logging.
    """
    import csv
    import os

    os.makedirs(out_dir, exist_ok=True)
    record = {"experiment": name, **config, **results, **record_environment()}

    json_path = os.path.join(out_dir, f"{name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    csv_path = os.path.join(out_dir, "metrics.csv")
    rows = []
    if os.path.isfile(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    rows.append(record)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return record

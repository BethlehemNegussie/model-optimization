"""Shared metrics task performance, size, parameters, sparsity, compression.

Kept in one place so every technique (quantization / pruning / distillation /
low-rank / combined) is scored identically -- this is what makes the
before/after and cross-technique comparisons fair.
"""
import os
import tempfile

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true, y_pred) -> dict:
    """Task performance-> accuracy + macro-F1 ."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def count_parameters(model: torch.nn.Module) -> dict:
    """Total vs. effective (non-zero) parameter count.

    For a densely-parameterized model these are equal. For a pruned model
    (mask-based, unstructured) or an MoE-style model, `nonzero` reflects the
    parameters that actually do work.
    """
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum().item()) for p in model.parameters())
    return {"total_parameters": total, "nonzero_parameters": nonzero}


def compute_sparsity(model: torch.nn.Module) -> float:
    """Percentage of zero-valued parameters."""
    total = sum(p.numel() for p in model.parameters())
    zeros = sum(int((p == 0).sum().item()) for p in model.parameters())
    return 100.0 * zeros / total if total else 0.0


def measure_serialized_size_mb(model: torch.nn.Module) -> float:
    """Actual serialized checkpoint size on disk, in MB.

    Uses a real save/stat/delete round-trip rather than estimating from dtype
    * numel, because that estimate silently misses things like packed
    int4 storage, quantization metadata, or structural pruning that changes
    tensor shapes rather than just zeroing values.
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        path = f.name
    try:
        torch.save(model.state_dict(), path)
        size_bytes = os.path.getsize(path)
    finally:
        os.remove(path)
    return size_bytes / (1024 ** 2)


def compression_ratio(baseline_size_mb: float, optimized_size_mb: float) -> float:
    """Baseline size / optimized size."""
    return baseline_size_mb / optimized_size_mb if optimized_size_mb else float("inf")


def performance_retention(baseline_metric: float, optimized_metric: float) -> float:
    """Optimized performance relative to baseline, as a percentage."""
    return 100.0 * optimized_metric / baseline_metric if baseline_metric else 0.0


def peak_memory_usage_mb() -> float:
    """Peak CUDA memory usage in MB, if a GPU is available. Returns NaN on CPU-only runs since
    torch does not track peak RSS as reliably across platforms."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return float("nan")

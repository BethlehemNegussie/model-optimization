"""
Shared metrics for task performance, size, parameters, sparsity, compression.

Kept in one place so every technique (quantization / pruning / distillation /
low-rank / combined) is scored identically -- this is what makes the
before/after and cross-technique comparisons fair.
"""

# os is used to inspect the size of the temporary saved model file.
import os

# tempfile lets us create a temporary file for measuring the actual
# serialized model size.
import tempfile

# NumPy is imported here for numerical operations.
import numpy as np

# PyTorch is used to inspect model parameters and GPU memory.
import torch

# These functions calculate standard classification metrics.
from sklearn.metrics import accuracy_score, f1_score


def compute_classification_metrics(y_true, y_pred) -> dict:
    """
    Task performance -> accuracy + macro-F1.

    y_true = the actual/ground-truth labels.
    y_pred = the labels predicted by the model.

    These metrics tell us whether optimization caused
    the model's task performance to decrease.
    """

    return {

        # Accuracy = number of correct predictions /
        #            total number of predictions.
        #
        # Example:
        # 95 correct out of 100 -> accuracy = 0.95.
        #
        # float() converts the result into a normal Python float
        # so it can easily be saved in JSON/CSV.
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),

        # Macro-F1 calculates the F1 score independently for
        # each class and then takes the average.
        #
        # This gives every class equal importance, regardless
        # of how many examples that class has.
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro"
            )
        ),
    }


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Total vs. effective (non-zero) parameter count.

    Total parameters:
        How many parameter values exist in the model.

    Non-zero parameters:
        How many of those parameter values are actually non-zero.

    For a normal dense model these are usually equal.

    For a pruned model, many weights may be zero, so the
    non-zero count shows how many parameter values remain active.
    """

    # Count every parameter element in the model.
    #
    # p.numel() = number of elements in a tensor.
    #
    # Example:
    #     weight shape = [100, 200]
    #     numel() = 20,000
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    # Count the parameters whose values are not zero.
    #
    # (p != 0) creates a Boolean tensor:
    #
    #     True  -> parameter is non-zero
    #     False -> parameter is zero
    #
    # .sum() counts the True values.
    #
    # int() converts the result to a normal Python integer.
    nonzero = sum(
        int((p != 0).sum().item())
        for p in model.parameters()
    )

    # Return both values so they can be logged and compared.
    return {
        "total_parameters": total,
        "nonzero_parameters": nonzero
    }


def compute_sparsity(model: torch.nn.Module) -> float:
    """
    Percentage of zero-valued parameters.

    Sparsity tells us what percentage of the model's parameters
    are currently zero.

    Formula:

        sparsity =
            (number of zero parameters / total parameters) * 100
    """

    # Count all parameter elements.
    total = sum(
        p.numel()
        for p in model.parameters()
    )

    # Count parameters whose value is exactly zero.
    zeros = sum(
        int((p == 0).sum().item())
        for p in model.parameters()
    )

    # Convert the zero count into a percentage.
    #
    # The `if total else 0.0` prevents division by zero
    # if a model somehow contains no parameters.
    return (
        100.0 * zeros / total
        if total
        else 0.0
    )


def measure_serialized_size_mb(model: torch.nn.Module) -> float:
    """
    Actual serialized checkpoint size on disk, in MB.

    Instead of estimating model size mathematically, this function
    actually saves the model and measures the resulting file.

    This is important because different optimization techniques
    can store models differently.

    For example:
        - quantization can use packed representations
        - INT4 can have special storage formats
        - metadata can add overhead
        - structural pruning can change tensor shapes
    """

    # Create a temporary .pt file.
    #
    # delete=False is used because we need to save the model to the
    # file and then inspect its size ourselves.
    with tempfile.NamedTemporaryFile(
        suffix=".pt",
        delete=False
    ) as f:

        # Save the path so we can use it after the temporary
        # file object is closed.
        path = f.name

    try:

        # Actually serialize the model's state dictionary.
        #
        # state_dict() contains the model's learned parameters
        # and buffers.
        torch.save(
            model.state_dict(),
            path
        )

        # Get the exact file size in bytes.
        size_bytes = os.path.getsize(path)

    finally:

        # Delete the temporary checkpoint after measuring it.
        #
        # This prevents temporary model files from accumulating.
        os.remove(path)

    # Convert bytes to megabytes.
    #
    # 1024 ** 2 = 1,048,576 bytes per MB.
    return size_bytes / (1024 ** 2)


def compression_ratio(
    baseline_size_mb: float,
    optimized_size_mb: float
) -> float:
    """
    Baseline size / optimized size.

    A higher compression ratio means the optimized model
    is smaller relative to the original.

    Example:

        baseline = 400 MB
        optimized = 100 MB

        compression = 400 / 100 = 4x
    """

    # Divide the original model size by the optimized size.
    #
    # If optimized_size_mb is zero, return infinity instead
    # of causing a division-by-zero error.
    return (
        baseline_size_mb / optimized_size_mb
        if optimized_size_mb
        else float("inf")
    )


def performance_retention(
    baseline_metric: float,
    optimized_metric: float
) -> float:
    """
    Optimized performance relative to baseline, as a percentage.

    Formula:

        retention =
            optimized performance / baseline performance * 100

    Example:

        baseline accuracy = 0.95
        optimized accuracy = 0.94

        retention = 0.94 / 0.95 * 100
                  ≈ 98.95%
    """

    # Calculate how much of the original performance remains.
    #
    # If baseline_metric is zero, return 0 rather than
    # dividing by zero.
    return (
        100.0 * optimized_metric / baseline_metric
        if baseline_metric
        else 0.0
    )


def peak_memory_usage_mb() -> float:
    """
    Peak CUDA memory usage in MB, if a GPU is available.

    This tells us the maximum amount of GPU memory allocated
    by PyTorch during the experiment.

    On CPU-only systems, this function returns NaN because
    PyTorch does not provide the same reliable peak-memory
    measurement across different CPU platforms.
    """

    # Check whether a CUDA-compatible GPU is available.
    if torch.cuda.is_available():

        # Get the maximum amount of GPU memory allocated by PyTorch.
        #
        # torch.cuda.max_memory_allocated()
        # returns bytes.
        #
        # Divide by 1024^2 to convert bytes -> MB.
        return (
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

    # If there is no GPU, return NaN ("Not a Number") rather
    # than pretending that GPU memory usage is zero.
    return float("nan")
"""
Latency/throughput benchmarking, built for experimental fairness.

Before timing, repeated runs, and a fixed, documented environment
are used so that different models can be compared under the same
conditions.
"""

# time is used to measure how long each forward pass takes.
import time

# dataclass makes it easy to define a structured object for storing
# all the benchmark results.
from dataclasses import dataclass, asdict

# PyTorch is used for model execution, device handling, and CUDA
# synchronization.
import torch


@dataclass
class LatencyResult:
    """
    Container for all latency and throughput measurements.

    Instead of returning many separate values, the benchmark
    returns one LatencyResult object containing all measurements.
    """

    # Average time required for one complete model forward pass
    # on the tested batch, measured in milliseconds.
    mean_ms_per_batch: float

    # Standard deviation of the measured batch latency.
    # This tells us how much the timings vary between runs.
    std_ms_per_batch: float

    # Median latency.
    #
    # p50 means 50th percentile:
    # 50% of measured runs are at or below this value.
    p50_ms: float

    # 95th-percentile latency.
    #
    # p95 tells us a "slower typical case":
    # 95% of runs are at or below this latency.
    p95_ms: float

    # Number of individual samples processed per second.
    throughput_samples_per_sec: float

    # Number of samples processed in each batch.
    batch_size: int

    # Number of warm-up runs that were performed and discarded.
    n_warmup: int

    # Number of actual timing runs.
    n_runs: int

    # Device used for benchmarking, for example "cuda" or "cpu".
    device: str

    def as_dict(self):
        """
        Convert the dataclass into a normal Python dictionary.

        This makes the benchmark result easier to save as JSON,
        CSV, or combine with other experiment metrics.
        """

        # asdict() converts every field of the dataclass into
        # a dictionary key/value pair.
        return asdict(self)


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    sample_batch: dict,
    n_warmup: int = 10,
    n_runs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> LatencyResult:
    """
    Time repeated forward passes on a fixed sample batch.

    Warm-up runs are discarded because the first CUDA executions
    can have one-time overhead such as kernel initialization.

    CUDA synchronization is used because CUDA operations are
    asynchronous. Without synchronization, the CPU clock could
    stop before the GPU has actually finished computing.
    """

    # Disable gradient computation for benchmarking.
    #
    # We only care about inference speed, so calculating gradients
    # would add unnecessary computation and memory usage.
    #
    # The @torch.no_grad() decorator above already handles this
    # for the entire function.

    # Put the model in evaluation mode and move it to the
    # selected device.
    #
    # eval() disables training-specific behavior such as dropout.
    model.eval().to(device)

    # Move every tensor in the sample batch to the same device
    # as the model.
    #
    # Example:
    #     CPU tensors -> GPU tensors if device="cuda"
    batch = {
        k: v.to(device)
        for k, v in sample_batch.items()
    }

    # Get the number of samples in the batch.
    #
    # next(iter(batch.values())) gets the first tensor in the
    # batch dictionary, and .shape[0] is its batch dimension.
    batch_size = next(iter(batch.values())).shape[0]

    # ---------------------------------------------------------
    # WARM-UP PHASE
    # ---------------------------------------------------------

    # Run the model several times before measuring it.
    #
    # These runs are NOT included in the benchmark results.
    #
    # Why?
    # The first GPU executions may include one-time overhead
    # such as CUDA kernel initialization.
    for _ in range(n_warmup):

        # Perform a normal forward pass.
        _ = model(**batch)

    # CUDA operations are asynchronous.
    #
    # Synchronize after warm-up so that all warm-up work has
    # completely finished before timing starts.
    if device == "cuda":
        torch.cuda.synchronize()

    # This list will store the duration of every timed run.
    times_ms = []

    # ---------------------------------------------------------
    # ACTUAL TIMING PHASE
    # ---------------------------------------------------------

    # Repeat the forward pass n_runs times.
    #
    # More repetitions make the timing statistics more reliable.
    for _ in range(n_runs):

        # Make sure any previous GPU operations have finished
        # before starting the timer.
        #
        # Without this, the CPU might start timing while previous
        # GPU work is still running.
        if device == "cuda":
            torch.cuda.synchronize()

        # Start a high-resolution timer.
        start = time.perf_counter()

        # Perform the actual inference/forward pass.
        _ = model(**batch)

        # Again synchronize CUDA before stopping the timer.
        #
        # This is critical because GPU operations are asynchronous.
        if device == "cuda":
            torch.cuda.synchronize()

        # Calculate elapsed time:
        #
        # perf_counter() - start = seconds
        #
        # * 1000 converts seconds -> milliseconds.
        times_ms.append(
            (time.perf_counter() - start) * 1000
        )

    # Sort all measured latencies from fastest to slowest.
    times_ms.sort()

    # Calculate the arithmetic mean latency.
    mean_ms = sum(times_ms) / len(times_ms)

    # Calculate the standard deviation.
    #
    # This measures how much individual timing results vary
    # around the mean.
    std_ms = (
        sum(
            (t - mean_ms) ** 2
            for t in times_ms
        )
        / len(times_ms)
    ) ** 0.5

    # Calculate the 50th percentile, or median.
    #
    # Since times_ms is sorted, the middle value represents
    # approximately the p50 latency.
    p50 = times_ms[len(times_ms) // 2]

    # Calculate the 95th percentile.
    #
    # This looks at a point near the slower end of the timing
    # distribution.
    p95 = times_ms[
        int(len(times_ms) * 0.95)
    ]

    # Calculate throughput.
    #
    # mean_ms / 1000 converts milliseconds -> seconds.
    #
    # batch_size / seconds_per_batch gives:
    #
    #     samples per second
    throughput = batch_size / (mean_ms / 1000)

    # Package all measurements into a LatencyResult object.
    return LatencyResult(
        mean_ms_per_batch=mean_ms,
        std_ms_per_batch=std_ms,
        p50_ms=p50,
        p95_ms=p95,
        throughput_samples_per_sec=throughput,
        batch_size=batch_size,
        n_warmup=n_warmup,
        n_runs=n_runs,
        device=device,
    )


@torch.no_grad()
def evaluate_classifier(
    model,
    dataloader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    input_keys=("input_ids", "attention_mask")
):
    """
    Run a model over a full dataloader and collect predictions/labels
    for fixed-evaluation-condition baseline measurement.

    This function is different from benchmark_latency():

        evaluate_classifier()
            -> measures classification performance

        benchmark_latency()
            -> measures inference speed
    """

    # Put the model into evaluation mode and move it to the
    # requested device.
    model.eval().to(device)

    # These lists will store predictions and ground-truth labels
    # from every batch.
    all_preds, all_labels = [], []

    # Go through the complete evaluation dataset.
    for batch in dataloader:

        # Select only the inputs that the model expects.
        #
        # For BERT this is normally:
        #     input_ids
        #     attention_mask
        #
        # The `if k in batch` check makes the function tolerant
        # of batches that do not contain every possible input key.
        inputs = {
            k: batch[k].to(device)
            for k in input_keys
            if k in batch
        }

        # Get the ground-truth labels.
        #
        # Some datasets use "labels", while others use "label",
        # so the function supports both conventions.
        labels = (
            batch["labels"]
            if "labels" in batch
            else batch["label"]
        )

        # Run the model on this batch.
        outputs = model(**inputs)

        # Take the class with the highest predicted logit.
        #
        # outputs.logits has roughly this shape:
        #
        #     [batch_size, number_of_classes]
        #
        # argmax(dim=-1) selects the class with the largest
        # score for each sample.
        #
        # .cpu() moves the predictions back to CPU.
        #
        # .tolist() converts the tensor into a Python list.
        preds = (
            outputs.logits
            .argmax(dim=-1)
            .cpu()
            .tolist()
        )

        # Add this batch's predictions to the complete list.
        all_preds.extend(preds)

        # Add this batch's true labels to the complete list.
        all_labels.extend(labels.tolist())

    # Return every prediction and corresponding true label.
    #
    # These can then be passed into metrics.py to calculate
    # accuracy and macro-F1.
    return all_preds, all_labels
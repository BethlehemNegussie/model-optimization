"""Latency/throughput benchmarking, built for experimental fairness 
 before timing, repeated runs, and a fixed, documented environment.
"""
import time
from dataclasses import dataclass, asdict

import torch


@dataclass
class LatencyResult:
    mean_ms_per_batch: float
    std_ms_per_batch: float
    p50_ms: float
    p95_ms: float
    throughput_samples_per_sec: float
    batch_size: int
    n_warmup: int
    n_runs: int
    device: str

    def as_dict(self):
        return asdict(self)


@torch.no_grad()
def benchmark_latency(
    model: torch.nn.Module,
    sample_batch: dict,
    n_warmup: int = 10,
    n_runs: int = 50,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> LatencyResult:
    """Time repeated forward passes on a fixed sample batch.

    - Warm-up runs are discarded (first CUDA calls pay a one-off kernel/JIT
      cost that would otherwise bias the mean upward).
    - Timing uses torch.cuda.synchronize() around each run so async CUDA
      kernels are actually finished before the clock stops.
    """
    model.eval().to(device)
    batch = {k: v.to(device) for k, v in sample_batch.items()}
    batch_size = next(iter(batch.values())).shape[0]

    for _ in range(n_warmup):
        _ = model(**batch)
    if device == "cuda":
        torch.cuda.synchronize()

    times_ms = []
    for _ in range(n_runs):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = model(**batch)
        if device == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000)

    times_ms.sort()
    mean_ms = sum(times_ms) / len(times_ms)
    std_ms = (sum((t - mean_ms) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    p50 = times_ms[len(times_ms) // 2]
    p95 = times_ms[int(len(times_ms) * 0.95)]
    throughput = batch_size / (mean_ms / 1000)

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
def evaluate_classifier(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu",
                          input_keys=("input_ids", "attention_mask")):
    """Run a model over a full dataloader and collect predictions/labels for
    fixed-evaluation-condition baseline measurement."""
    model.eval().to(device)
    all_preds, all_labels = [], []
    for batch in dataloader:
        inputs = {k: batch[k].to(device) for k in input_keys if k in batch}
        labels = batch["labels"] if "labels" in batch else batch["label"]
        outputs = model(**inputs)
        preds = outputs.logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels

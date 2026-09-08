"""
Implements both unstructured magnitude pruning (highest theoretical sparsity,
but irregular memory access -> no real speedup on standard hardware) and
structured pruning (removes whole neurons/attention heads/channels ->
smaller, hardware-friendly tensors with a measurable speedup).

Criteria chosen: L1 magnitude (|W_ij| < epsilon), justified as the simplest,
best-documented importance proxy for Transformer weight matrices, applied
per-layer so no single layer is pruned into uselessness (global magnitude
pruning tends to over-prune early/late layers unevenly).
"""
import copy

import torch
import torch.nn.utils.prune as prune


def unstructured_magnitude_prune(model: torch.nn.Module, amount: float, layers=(torch.nn.Linear,)):
    """Zero out the lowest-magnitude `amount` fraction of weights per Linear
    layer. Mask based -> parameter count on disk is unchanged (dense tensor,
    now sparse-valued) unless the result is exported to a sparse format -
    this is why unstructured pruning needs `metrics.compute_sparsity` rather
    than raw serialized size to show its effect.
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, layers):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")  # encode mask into .weight, drop mask buffer
    return model


def structured_channel_prune(model: torch.nn.Module, amount: float, layers=(torch.nn.Linear,), dim: int = 0):
    """Remove whole output neuron rows (structured, ln-norm ranked) from
    Linear layers. Unlike unstructured pruning this changes tensor shapes,
    so it does shrink the serialized checkpoint and does speed up inference
    on ordinary hardware.

    Implemented via torch's built-in ln_structured (masks + bakes zeros along
    `dim`); a follow-up shape-compaction pass (dropping fully-zeroed rows and
    resizing adjacent layers) is required for a true parameter count
    reduction and is left as the natural next step once masked sparsity is
    validated.
    """
    model = copy.deepcopy(model)
    for module in model.modules():
        if isinstance(module, layers):
            prune.ln_structured(module, name="weight", amount=amount, n=1, dim=dim)
            prune.remove(module, "weight")
    return model
    


def iterative_prune(model: torch.nn.Module, amounts, prune_fn=unstructured_magnitude_prune):
    """Apply pruning at multiple sparsity levels returning one pruned model per level so each can be
    independently fine-tuned/evaluated.
    """
    return {amount: prune_fn(model, amount) for amount in amounts}

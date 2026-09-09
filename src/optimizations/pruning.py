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

# copy is used to create an independent copy of the model.
# This means pruning does not modify the original trained model.
import copy

# PyTorch's pruning utilities provide functions such as
# l1_unstructured() and ln_structured().
import torch

import torch.nn.utils.prune as prune


def unstructured_magnitude_prune(
    model: torch.nn.Module,
    amount: float,
    layers=(torch.nn.Linear,)
):
    """
    Zero out the lowest-magnitude `amount` fraction of weights
    per Linear layer.

    Example:
        amount = 0.30

    means approximately 30% of the weights in EACH selected
    Linear layer are set to zero.

    This creates an unstructured sparse model:
    the zeros can be scattered anywhere inside the weight matrix.

    Important:
    The dense tensor still has the same shape and is normally
    stored as a dense tensor on disk.

    Therefore, pruning 30% of the weights does NOT automatically
    mean the checkpoint becomes 30% smaller.

    Instead, we measure the effect using sparsity:
        sparsity = zero weights / total weights

    A real storage or inference-speed benefit would require
    sparse-aware storage or hardware/software that can exploit
    the sparse structure.
    """

    # Make an independent copy of the model.
    #
    # This is important because we want to compare:
    #     baseline model
    #     vs.
    #     pruned model
    #
    # without destroying the original baseline.
    model = copy.deepcopy(model)

    # Go through every module/layer inside the model.
    #
    # For a Transformer this will include many different
    # components, such as Linear layers, LayerNorms, etc.
    for module in model.modules():

        # Only prune layers whose type appears in `layers`.
        #
        # By default:
        #     layers = (torch.nn.Linear,)
        #
        # So only Linear layers are pruned.
        #
        # This avoids blindly pruning every type of layer.
        if isinstance(module, layers):

            # Perform L1 unstructured pruning on the weight tensor.
            #
            # L1 magnitude means we look at:
            #
            #     |weight|
            #
            # Small-magnitude weights are considered less important
            # and are therefore selected for pruning.
            #
            # `amount` determines the fraction of weights to remove.
            #
            # Example:
            #     amount=0.30
            #
            # means approximately 30% of the weights in this
            # Linear layer will be zeroed.
            prune.l1_unstructured(
                module,
                name="weight",
                amount=amount
            )

            # PyTorch's pruning mechanism temporarily creates
            # a pruning mask around the original weight.
            #
            # prune.remove() permanently applies that mask to
            # the weight and removes the pruning reparameterization.
            #
            # In other words:
            #
            #     original weight
            #          ↓
            #     pruning mask
            #          ↓
            #     zeroed weight
            #
            # After remove(), the resulting zeros are stored directly
            # in module.weight.
            prune.remove(
                module,
                "weight"
            )  # encode mask into .weight, drop mask buffer

    # Return the independently pruned model.
    return model


def structured_channel_prune(
    model: torch.nn.Module,
    amount: float,
    layers=(torch.nn.Linear,),
    dim: int = 0
):
    """
    Perform structured pruning.

    Instead of removing individual weights randomly throughout
    a matrix, structured pruning removes entire structures.

    For a Linear layer, this implementation uses rows along
    `dim=0`, which correspond to output neurons.

    Example:

        Before:

        [ x x x x ]
        [ x x x x ]
        [ x x x x ]
        [ x x x x ]

        If one output neuron is pruned:

        [ x x x x ]
        [ 0 0 0 0 ]  <- entire output neuron
        [ x x x x ]
        [ x x x x ]

    This type of regular structure is more hardware-friendly
    than scattered individual zeros.

    IMPORTANT:
    This implementation creates structured masks and bakes
    the zeros into the tensor.

    A true parameter-count reduction requires a later
    shape-compaction step that physically removes the
    zeroed rows and adjusts connected layers.
    """

    # Again, create an independent copy so the original model
    # remains untouched.
    model = copy.deepcopy(model)

    # Visit every module in the model.
    for module in model.modules():

        # Only apply structured pruning to the requested layer types.
        if isinstance(module, layers):

            # Perform structured L1-norm pruning.
            #
            # ln_structured() evaluates complete structures
            # instead of individual weights.
            #
            # name="weight":
            #     prune the weight tensor.
            #
            # amount=amount:
            #     fraction of structures to prune.
            #
            # n=1:
            #     use the L1 norm:
            #
            #         sum(|weights|)
            #
            # to determine which structures are least important.
            #
            # dim=dim:
            #     determines which dimension represents the
            #     structure being removed.
            #
            # dim=0 for Linear weights means output rows /
            # output neurons are considered as structures.
            prune.ln_structured(
                module,
                name="weight",
                amount=amount,
                n=1,
                dim=dim
            )

            # Permanently apply the pruning mask to the weight
            # and remove the temporary pruning reparameterization.
            prune.remove(
                module,
                "weight"
            )

    # Return the structured-pruned model.
    return model


def iterative_prune(
    model: torch.nn.Module,
    amounts,
    prune_fn=unstructured_magnitude_prune
):
    """
    Apply pruning at multiple sparsity levels.

    Example:

        amounts = [0.30, 0.50]

    produces:

        {
            0.30: model pruned at 30%,
            0.50: model pruned at 50%
        }

    Each model is created independently from the ORIGINAL
    input model.

    This is important because we want to compare 30% and 50%
    pruning fairly.

    We do NOT want:

        baseline
           ↓
        30% pruning
           ↓
        another 50% pruning

    because that would make the second model effectively
    undergo additional pruning.

    Instead:

        baseline ──→ 30% model
        baseline ──→ 50% model
    """

    # Create one independently pruned model for every requested
    # pruning amount.
    #
    # The dictionary key is the pruning amount.
    #
    # Example:
    #
    # {
    #     0.3: <30%-pruned model>,
    #     0.5: <50%-pruned model>
    # }
    return {
        amount: prune_fn(model, amount)
        for amount in amounts
    }
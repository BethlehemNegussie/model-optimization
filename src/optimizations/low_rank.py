"""
Low-Rank Factorization 

Factors a target Linear layer's weight matrix W (out x in) via truncated
SVD into W ~= U_r @ diag(S_r) @ V_r^T, then replaces the single Linear layer
with two smaller Linear layers (in -> rank -> out). Parameter count drops
from (out * in) to rank * (in + out), which is a net win whenever
rank < (out * in) / (out + in).

"""
import copy

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Drop-in replacement for nn.Linear, implemented as two smaller matmuls."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)
        self.rank = rank

    def forward(self, x):
        return self.up(self.down(x))


def factorize_linear(layer: nn.Linear, rank: int) -> LowRankLinear:
    """SVD-factorize one Linear layer's weight into a LowRankLinear at the
    given rank, initialized so the factored layer's output initially matches
    the original layer's output before any fine-tuning.
    """
    max_rank = min(layer.in_features, layer.out_features)
    if not 1 <= rank <= max_rank:
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
    W = layer.weight.detach()  # shape: (out_features, in_features)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)

    U_r = U[:, :rank]                       # (out, rank)
    S_r = S[:rank]                          # (rank,)
    Vh_r = Vh[:rank, :]                     # (rank, in)

    lr_layer = LowRankLinear(layer.in_features, layer.out_features, rank, bias=layer.bias is not None)
    lr_layer.down.weight.data = Vh_r                              # (rank, in)
    lr_layer.up.weight.data = U_r * S_r.unsqueeze(0)               # (out, rank)
    if layer.bias is not None:
        lr_layer.up.bias.data = layer.bias.data.clone()

    return lr_layer


def apply_low_rank_to_bert_ffn(model, rank: int, target_substrings=("intermediate.dense", "output.dense")):
    """Walk a BERT model and replace every FFN Linear layer whose fully
    qualified name contains one of `target_substrings` with a LowRankLinear
    at the given rank. Returns the modified model and a list of replaced
    layer names for the reproducibility log.
    """
    model = copy.deepcopy(model)
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(s in name for s in target_substrings):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, factorize_linear(module, rank))
            replaced.append((name, module.in_features, module.out_features, rank))
    return model, replaced


def parameter_savings(in_features: int, out_features: int, rank: int) -> dict:
    """Compare original vs. factored parameter count for a chosen rank
    """
    original = in_features * out_features
    factored = rank * (in_features + out_features)
    return {
        "original_params": original,
        "factored_params": factored,
        "reduction_pct": 100.0 * (1 - factored / original) if original else 0.0,
        "breakeven_rank": (in_features * out_features) // (in_features + out_features),
    }

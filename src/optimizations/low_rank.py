"""
Low-Rank Factorization 

Factors a target Linear layer's weight matrix W (out x in) via truncated
SVD into W ~= U_r @ diag(S_r) @ V_r^T, then replaces the single Linear layer
with two smaller Linear layers (in -> rank -> out). Parameter count drops
from (out * in) to rank * (in + out), which is a net win whenever
rank < (out * in) / (out + in).

"""
"""
Singular Value Decomposition decomposes weight W into U sum v^T, we don't need 
all the components, we can only keep only the most important r(rank) components(truncated SVD)
"""
import copy #we don't want to modify the original BERT model unexpectedly

import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    """Drop-in replacement for nn.Linear, implemented as two smaller matmuls."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__() #This initializes the underlying nn.Module
        self.down = nn.Linear(in_features, rank, bias=False) #the first smaller Linear layer
        #the representation is compressed into a lower-dimensional space
        self.up = nn.Linear(rank, out_features, bias=bias)
        self.rank = rank #store rank

    def forward(self, x): #fwd pass, mathematically equivalent to multiplying by two smaller matrices
        return self.up(self.down(x))


def factorize_linear(layer: nn.Linear, rank: int) -> LowRankLinear:
    """SVD-factorize one Linear layer's weight into a LowRankLinear at the
    given rank, initialized so the factored layer's output initially matches
    the original layer's output before any fine-tuning.
    """
    #Take an existing trained Linear layer and replace it with a low-rank approximation
    max_rank = min(layer.in_features, layer.out_features) #rank of a matrix cannot exceed its smaller dimension
    if not 1 <= rank <= max_rank:#prevents invalid values
        raise ValueError(f"rank must be in [1, {max_rank}], got {rank}")
    W = layer.weight.detach()  # shape: (out_features, in_features), extracts the trained weight matrix
    #detach because we're taking the already-trained weights and decomposing them
    #Perform SVD
    U, S, Vh = torch.linalg.svd(W, full_matrices=False) #W ≈ U × S × Vᵀ

    U_r = U[:, :rank]                       # (out, rank), We keep the first rank columns of U
    S_r = S[:rank]                          # (rank,), Keep the first 128 singular values
    Vh_r = Vh[:rank, :]                     # (rank, in), keep only the first 128 components

    #replacement layer
    #create  input → rank → output  with the same input/output dimensions as the original layer  
    lr_layer = LowRankLinear(layer.in_features, layer.out_features, rank, bias=layer.bias is not None)
    lr_layer.down.weight.data = Vh_r                              # (rank, in)
    lr_layer.up.weight.data = U_r * S_r.unsqueeze(0)               # (out, rank)
    #The original Linear layer may have a bias. We preserve it by putting it on the up layer.
    # So we're not throwing away the original bias
    if layer.bias is not None:
        lr_layer.up.bias.data = layer.bias.data.clone()

    return lr_layer

#function determines which BERT layers get factorized
def apply_low_rank_to_bert_ffn(model, rank: int, target_substrings=("intermediate.dense", "output.dense")):
    """Walk a BERT model and replace every FFN Linear layer whose fully
    qualified name contains one of `target_substrings` with a LowRankLinear
    at the given rank. Returns the modified model and a list of replaced
    layer names for the reproducibility log.
    """
    model = copy.deepcopy(model) #creating an independent copy used when comparing experiments fairly
    replaced = [] ##record which layers were modified
    for name, module in list(model.named_modules()): #named_modules() gives us all the modules inside BERT
        if isinstance(module, nn.Linear) and any(s in name for s in target_substrings):
            # checks Is this actually a Linear layer? and Does its name contain intermidiate.dense, if factorize it
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model#locate the module containing that Linear layer
            setattr(parent, child_name, factorize_linear(module, rank))#Replace the Linear layer
            #record layer name, input dimension, output dimension, rank        
            replaced.append((name, module.in_features, module.out_features, rank))
    return model, replaced # we get modified BERT and list of factorized layers

#calculates whether the factorization actually saves parameters
def parameter_savings(in_features: int, out_features: int, rank: int) -> dict:
    """Compare original vs. factored parameter count for a chosen rank
    """
    original = in_features * out_features #original weight matrix
    factored = rank * (in_features + out_features)
    return {
        "original_params": original,
        "factored_params": factored,
        #What percentage of parameters did we remove?
        "reduction_pct": 100.0 * (1 - factored / original) if original else 0.0,
        #calculates approximately the largest rank where factorization still saves parameters
        "breakeven_rank": (in_features * out_features) // (in_features + out_features),
        
        #The condition is:  rank × (in + out) < in × out  therefore rank < (in × out) / (in + out) 
        # if your rank is above the break-even point, low-rank factorization 
       # may actually use more parameters than the original layer 
    }

from __future__ import annotations

import torch.nn as nn

from src.models.hetero_lora_linear import HeteroLoRALinear
from src.models.inject import get_parent_module, should_wrap


def discover_lora_targets(model, target_modules):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_wrap(name, target_modules):
            out_features, in_features = module.weight.shape
            targets.append((name, out_features, in_features))
    return targets


def inject_hetero_lora(
    model,
    target_modules,
    rank,
    alpha=16.0,
    dropout=0.0,
    init_scale=0.01,
    rank_by_layer=None,
):
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_wrap(name, target_modules):
            layer_rank = int(rank_by_layer.get(name, rank)) if rank_by_layer else int(rank)
            parent, child_name = get_parent_module(model, name)
            setattr(
                parent,
                child_name,
                HeteroLoRALinear(
                    module,
                    rank=layer_rank,
                    alpha=alpha,
                    dropout=dropout,
                    init_scale=init_scale,
                    layer_name=name,
                ),
            )
            replaced.append(name)
    if not replaced:
        raise RuntimeError(f"No modules matched target_modules={target_modules}")
    return replaced

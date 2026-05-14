import torch.nn as nn

from src.models.florg_linear import FLoRGLinear
from src.models.lora_linear import LoRALinear


def get_parent_module(model, module_name):
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def should_wrap(name, target_modules):
    return any(name.endswith(t) or f".{t}" in name for t in target_modules)


def inject_florg(model, target_modules, rank, alpha, dropout, init_scale):
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_wrap(name, target_modules):
            parent, child_name = get_parent_module(model, name)
            setattr(
                parent,
                child_name,
                FLoRGLinear(module, rank=rank, alpha=alpha, dropout=dropout, init_scale=init_scale, layer_name=name),
            )
            replaced.append(name)
    if not replaced:
        raise RuntimeError(f"No modules matched target_modules={target_modules}")
    return replaced


def inject_lora(model, target_modules, rank, alpha, dropout):
    replaced = []
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and should_wrap(name, target_modules):
            parent, child_name = get_parent_module(model, name)
            setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout, layer_name=name))
            replaced.append(name)
    if not replaced:
        raise RuntimeError(f"No modules matched target_modules={target_modules}")
    return replaced

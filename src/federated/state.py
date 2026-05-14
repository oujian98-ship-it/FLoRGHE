import torch

from src.models.florg_linear import FLoRGLinear
from src.models.lora_linear import LoRALinear


def get_florg_state(model):
    return {name: module.get_A().cpu() for name, module in model.named_modules() if isinstance(module, FLoRGLinear)}


@torch.no_grad()
def set_florg_state(model, state):
    for name, module in model.named_modules():
        if isinstance(module, FLoRGLinear):
            module.set_A(state[name].to(module.A.device))


def get_lora_state(model):
    return {
        name: {"lora_A": module.lora_A.detach().cpu().clone(), "lora_B": module.lora_B.detach().cpu().clone()}
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }

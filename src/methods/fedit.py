from __future__ import annotations

import torch

from src.models.inject import inject_lora
from src.models.lora_linear import LoRALinear
from src.models.model_utils import (
    average_classifier_states,
    freeze_non_adapter_params,
    get_classifier_state,
    set_classifier_state,
    tensor_tree_numel,
)


class FedITMethod:
    name = "fedit"

    def __init__(self, cfg):
        self.cfg = cfg

    def inject(self, model):
        return inject_lora(
            model,
            self.cfg.model.target_modules,
            self.cfg.adapter.rank,
            self.cfg.adapter.alpha,
            self.cfg.adapter.dropout,
        )

    def freeze(self, model):
        freeze_non_adapter_params(model, self.cfg.model.train_classifier_head)

    def get_state(self, model):
        state = {
            name: {"lora_A": module.lora_A.detach().cpu().clone(), "lora_B": module.lora_B.detach().cpu().clone()}
            for name, module in model.named_modules()
            if isinstance(module, LoRALinear)
        }
        if self.cfg.model.train_classifier_head:
            state["__classifier_head__"] = get_classifier_state(model)
        return state

    @torch.no_grad()
    def set_state(self, model, state):
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                module.lora_A.copy_(state[name]["lora_A"].to(module.lora_A.device, module.lora_A.dtype))
                module.lora_B.copy_(state[name]["lora_B"].to(module.lora_B.device, module.lora_B.dtype))
        if self.cfg.model.train_classifier_head:
            set_classifier_state(model, state.get("__classifier_head__", {}))

    def aggregate(self, client_states, prev_state):
        out = {}
        for layer in [k for k in prev_state if k != "__classifier_head__"]:
            out[layer] = {
                "lora_A": sum(cs[layer]["lora_A"] for cs in client_states) / len(client_states),
                "lora_B": sum(cs[layer]["lora_B"] for cs in client_states) / len(client_states),
            }
        if self.cfg.model.train_classifier_head:
            out["__classifier_head__"] = average_classifier_states([cs["__classifier_head__"] for cs in client_states])
        return out

    def communication(self, state, num_clients):
        one_direction = tensor_tree_numel(state)
        upload = num_clients * one_direction
        download = num_clients * one_direction
        return {"upload": upload, "download": download, "total": upload + download}

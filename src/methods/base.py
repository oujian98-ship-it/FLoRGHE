from __future__ import annotations

from src.models.inject import inject_florg, inject_lora
from src.models.model_utils import freeze_non_adapter_params
from src.methods.fedit import FedITMethod
from src.methods.federa import FeDeRAMethod
from src.methods.ffa_lora import FFALoRAMethod
from src.methods.florg import FLoRGMethod


class MethodBase:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg

    def inject(self, model):
        raise NotImplementedError

    def freeze(self, model):
        freeze_non_adapter_params(model, self.cfg.model.train_classifier_head)

    def get_state(self, model):
        raise NotImplementedError

    def set_state(self, model, state):
        raise NotImplementedError

    def aggregate(self, client_states, prev_state):
        raise NotImplementedError

    def communication(self, state, num_clients):
        raise NotImplementedError


class UnsupportedMethod(MethodBase):
    def inject(self, model):
        raise NotImplementedError(
            f"{self.name} is intentionally not implemented because the guide notes missing public details."
        )


def build_method(method_name, cfg):
    name = method_name.lower()
    if name in {"florg", "florg_no_procrustes"}:
        return FLoRGMethod(cfg, use_procrustes=name == "florg")
    if name == "fedit":
        return FedITMethod(cfg)
    if name == "federa":
        return FeDeRAMethod(cfg)
    if name == "ffa_lora":
        return FFALoRAMethod(cfg)
    if name in {"fedsa_lora", "fedex_lora"}:
        method = UnsupportedMethod(cfg)
        method.name = name
        return method
    raise ValueError(f"Unknown method: {method_name}")


def inject_lora_for_cfg(model, cfg):
    return inject_lora(
        model,
        cfg.model.target_modules,
        cfg.adapter.rank,
        cfg.adapter.alpha,
        cfg.adapter.dropout,
    )


def inject_florg_for_cfg(model, cfg):
    return inject_florg(
        model,
        cfg.model.target_modules,
        cfg.adapter.rank,
        cfg.adapter.alpha,
        cfg.adapter.dropout,
        cfg.adapter.init_scale,
    )

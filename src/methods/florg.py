from __future__ import annotations

import torch

from src.models.florg_linear import FLoRGLinear
from src.models.inject import inject_florg
from src.models.model_utils import (
    average_classifier_states,
    freeze_non_adapter_params,
    get_classifier_state,
    set_classifier_state,
    tensor_tree_numel,
)


@torch.no_grad()
def florg_aggregate_one_layer(client_As, prev_A, use_procrustes=True, eps=1e-8):
    device = prev_A.device
    dtype = prev_A.dtype
    client_As = [A.to(device=device, dtype=torch.float32) for A in client_As]
    prev_A_f = prev_A.to(device=device, dtype=torch.float32)
    r, k = prev_A_f.shape

    q = torch.zeros(k, k, device=device, dtype=torch.float32)
    for A in client_As:
        q += A.T @ A
    q /= len(client_As)
    q = 0.5 * (q + q.T)

    eigvals, eigvecs = torch.linalg.eigh(q)
    idx = torch.argsort(eigvals, descending=True)[:r]
    vals = eigvals[idx].clamp_min(eps)
    vecs = eigvecs[:, idx]
    a_tilde = torch.diag(torch.sqrt(vals)) @ vecs.T

    if use_procrustes:
        m = prev_A_f @ a_tilde.T
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        a_next = (u @ vh) @ a_tilde
    else:
        a_next = a_tilde
    return a_next.to(dtype=dtype)


@torch.no_grad()
def aggregate_florg_states(client_states, prev_global_state, use_procrustes=True):
    next_state = {}
    for layer_name, prev_A in prev_global_state.items():
        next_state[layer_name] = florg_aggregate_one_layer(
            [cs[layer_name] for cs in client_states],
            prev_A,
            use_procrustes=use_procrustes,
        ).cpu()
    return next_state


def florg_diagnostics(A_prev, A_next, client_As):
    with torch.no_grad():
        q = sum(A.float().T @ A.float() for A in client_As) / len(client_As)
        q = 0.5 * (q + q.T)
        eigvals = torch.linalg.eigvalsh(q)
        return {
            "q_min_eig": eigvals.min().item(),
            "q_max_eig": eigvals.max().item(),
            "q_rank_eps1e_6": int((eigvals > 1e-6).sum().item()),
            "a_drift": torch.norm(A_next.float() - A_prev.float()).item(),
            "a_norm": torch.norm(A_next.float()).item(),
        }


class FLoRGMethod:
    name = "florg"

    def __init__(self, cfg, use_procrustes=True):
        self.cfg = cfg
        self.use_procrustes = use_procrustes
        self.name = "florg" if use_procrustes else "florg_no_procrustes"

    def inject(self, model):
        return inject_florg(
            model,
            self.cfg.model.target_modules,
            self.cfg.adapter.rank,
            self.cfg.adapter.alpha,
            self.cfg.adapter.dropout,
            self.cfg.adapter.init_scale,
        )

    def freeze(self, model):
        freeze_non_adapter_params(model, self.cfg.model.train_classifier_head)

    def get_state(self, model):
        state = {name: module.get_A().cpu() for name, module in model.named_modules() if isinstance(module, FLoRGLinear)}
        if self.cfg.model.train_classifier_head:
            state["__classifier_head__"] = get_classifier_state(model)
        return state

    @torch.no_grad()
    def set_state(self, model, state):
        for name, module in model.named_modules():
            if isinstance(module, FLoRGLinear):
                module.set_A(state[name].to(module.A.device))
        if self.cfg.model.train_classifier_head:
            set_classifier_state(model, state.get("__classifier_head__", {}))

    def aggregate(self, client_states, prev_state):
        adapter_prev = {k: v for k, v in prev_state.items() if k != "__classifier_head__"}
        next_state = aggregate_florg_states(client_states, adapter_prev, self.use_procrustes)
        if self.cfg.model.train_classifier_head:
            next_state["__classifier_head__"] = average_classifier_states([cs["__classifier_head__"] for cs in client_states])
        return next_state

    def communication(self, state, num_clients):
        one_direction = tensor_tree_numel(state)
        upload = num_clients * one_direction
        download = num_clients * one_direction
        return {"upload": upload, "download": download, "total": upload + download}

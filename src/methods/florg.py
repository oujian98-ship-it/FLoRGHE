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
    """
    Aggregate one FLoRG layer using Gram averaging and a fixed-rank top-r projection.
    """
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
    vals = eigvals[idx].clamp_min(0.0)
    vecs = eigvecs[:, idx]

    a_tilde = torch.diag(torch.sqrt(vals + eps)) @ vecs.T

    if use_procrustes:
        m = prev_A_f @ a_tilde.T
        u, _, vh = torch.linalg.svd(m, full_matrices=False)
        a_next = (u @ vh) @ a_tilde
    else:
        a_next = a_tilde

    return a_next.to(dtype=dtype)


@torch.no_grad()
def aggregate_florg_A_states(client_A_states, prev_A_state, use_procrustes=True):
    next_A_state = {}

    for layer_name, prev_A in prev_A_state.items():
        next_A_state[layer_name] = florg_aggregate_one_layer(
            [cs[layer_name] for cs in client_A_states],
            prev_A,
            use_procrustes=use_procrustes,
        ).cpu()

    return next_A_state


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
        self.last_diagnostics = {}

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
        """
        Store A plus the shared frozen L/R bases.

        FLoRG aggregation is only meaningful when every client trains A in the
        same coordinate system, so L/R must be synchronized through global state.
        """
        state = {}

        for name, module in model.named_modules():
            if isinstance(module, FLoRGLinear):
                state[name] = {
                    "A": module.get_A().cpu(),
                    "L": module.L.detach().cpu().clone(),
                    "R": module.R.detach().cpu().clone(),
                }

        state["__classifier_head__"] = get_classifier_state(model)
        return state

    @torch.no_grad()
    def set_state(self, model, state):
        for name, module in model.named_modules():
            if isinstance(module, FLoRGLinear):
                layer_state = state[name]
                module.set_A(layer_state["A"].to(module.A.device))
                module.L.copy_(layer_state["L"].to(device=module.L.device, dtype=module.L.dtype))
                module.R.copy_(layer_state["R"].to(device=module.R.device, dtype=module.R.dtype))

        set_classifier_state(model, state.get("__classifier_head__", {}))

    def aggregate(self, client_states, prev_state):
        prev_A_state = {
            layer_name: layer_state["A"]
            for layer_name, layer_state in prev_state.items()
            if layer_name != "__classifier_head__"
        }

        client_A_states = [
            {
                layer_name: layer_state["A"]
                for layer_name, layer_state in cs.items()
                if layer_name != "__classifier_head__"
            }
            for cs in client_states
        ]

        next_A_state = aggregate_florg_A_states(
            client_A_states,
            prev_A_state,
            use_procrustes=self.use_procrustes,
        )

        self.last_diagnostics = {
            layer_name: florg_diagnostics(
                A_prev=prev_A_state[layer_name],
                A_next=A_next,
                client_As=[cs[layer_name] for cs in client_A_states],
            )
            for layer_name, A_next in next_A_state.items()
        }

        next_state = {}
        for layer_name, A_next in next_A_state.items():
            next_state[layer_name] = {
                "A": A_next,
                "L": prev_state[layer_name]["L"],
                "R": prev_state[layer_name]["R"],
            }

        if self.cfg.model.train_classifier_head:
            next_state["__classifier_head__"] = average_classifier_states(
                [cs["__classifier_head__"] for cs in client_states]
            )
        else:
            next_state["__classifier_head__"] = prev_state.get("__classifier_head__", {})

        return next_state

    def communication(self, state, num_clients):
        one_direction = 0

        for layer_name, layer_state in state.items():
            if layer_name == "__classifier_head__":
                if self.cfg.model.train_classifier_head:
                    one_direction += tensor_tree_numel(layer_state)
            else:
                one_direction += layer_state["A"].numel()

        upload = num_clients * one_direction
        download = num_clients * one_direction
        return {"upload": upload, "download": download, "total": upload + download}

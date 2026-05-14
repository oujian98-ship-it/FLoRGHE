from __future__ import annotations

import time

import torch

from src.models.hetero_lora_linear import HeteroLoRALinear
from src.models.inject_hetero_lora import inject_hetero_lora
from src.models.model_utils import (
    get_classifier_state,
    set_classifier_state,
)


@torch.no_grad()
def randomized_svd(matrix, rank, n_oversamples=8, n_iter=1):
    m, n = matrix.shape
    q = min(min(m, n), int(rank) + int(n_oversamples))
    omega = torch.randn(n, q, device=matrix.device, dtype=matrix.dtype)
    Y = matrix @ omega
    for _ in range(int(n_iter)):
        Y = matrix @ (matrix.T @ Y)
    Q, _ = torch.linalg.qr(Y, mode="reduced")
    B = Q.T @ matrix
    Uh, S, Vh = torch.linalg.svd(B, full_matrices=False)
    U = Q @ Uh
    return U, S, Vh


@torch.no_grad()
def compute_svd(matrix, rank, svd_type="exact", n_oversamples=8, n_iter=1):
    if svd_type == "exact":
        return torch.linalg.svd(matrix, full_matrices=False)
    if svd_type == "randomized":
        return randomized_svd(matrix, rank=rank, n_oversamples=n_oversamples, n_iter=n_iter)
    raise ValueError(f"Unknown SVD type: {svd_type}")


@torch.no_grad()
def aggregate_idea_one_layer(
    client_layer_states,
    prev_global,
    server_rank,
    alpha=16.0,
    gamma=0.5,
    procrustes="full",
    svd_type="exact",
    n_oversamples=8,
    n_iter=1,
    eps=1e-8,
):
    valid = [s for s in client_layer_states if int(s["rank"]) > 0]
    if len(valid) == 0:
        out = dict(prev_global)
        out["trunc_error"] = None
        out["update_norm"] = 0.0
        out["drift_before"] = None
        out["drift_after"] = None
        out["server_svd_time_sec"] = 0.0
        out["procrustes_time_sec"] = 0.0
        return out

    device = prev_global["B_eff"].device
    dtype = prev_global["B_eff"].dtype
    dout = valid[0]["B"].shape[0]
    din = valid[0]["A"].shape[1]

    weights_raw = []
    deltas = []
    for s in valid:
        r = int(s["rank"])
        n = float(s.get("num_samples", 1.0))
        scale = float(s.get("alpha", alpha)) / r
        B = s["B"].to(device=device, dtype=torch.float32)
        A = s["A"].to(device=device, dtype=torch.float32)
        deltas.append(scale * (B @ A))
        weights_raw.append(n * (r ** gamma))

    weights_raw = torch.tensor(weights_raw, device=device, dtype=torch.float32)
    weights = weights_raw / weights_raw.sum().clamp_min(eps)

    delta_bar = torch.zeros(dout, din, device=device, dtype=torch.float32)
    for w, delta in zip(weights, deltas):
        delta_bar += w * delta

    svd_start = time.time()
    U, S, Vh = compute_svd(
        delta_bar,
        rank=server_rank,
        svd_type=svd_type,
        n_oversamples=n_oversamples,
        n_iter=n_iter,
    )
    svd_time = time.time() - svd_start

    R = min(int(server_rank), S.numel())
    U_r = U[:, :R]
    S_r = S[:R].clamp_min(eps)
    Vh_r = Vh[:R, :]
    sqrtS = torch.sqrt(S_r)
    B_tilde = U_r * sqrtS.unsqueeze(0)
    A_tilde = sqrtS.unsqueeze(1) * Vh_r

    B_prev = prev_global["B_eff"].to(device=device, dtype=torch.float32)
    A_prev = prev_global["A_eff"].to(device=device, dtype=torch.float32)
    if B_prev.shape[1] != R:
        B_prev = B_prev[:, :R]
        A_prev = A_prev[:R, :]

    proc_start = time.time()
    if procrustes == "full":
        K = B_tilde.T @ B_prev + A_tilde @ A_prev.T
        Uq, _, Vhq = torch.linalg.svd(K, full_matrices=False)
        Q = Uq @ Vhq
        B_final = B_tilde @ Q
        A_final = Q.T @ A_tilde
    elif procrustes == "prefix_safe":
        B_final = B_tilde.clone()
        A_final = A_tilde.clone()
        score = (B_tilde * B_prev).sum(dim=0) + (A_tilde * A_prev).sum(dim=1)
        signs = torch.sign(score)
        signs[signs == 0] = 1.0
        B_final = B_final * signs.unsqueeze(0)
        A_final = A_final * signs.unsqueeze(1)
    elif procrustes == "none":
        B_final = B_tilde
        A_final = A_tilde
    else:
        raise ValueError(f"Unknown procrustes mode: {procrustes}")
    proc_time = time.time() - proc_start

    approx = B_final @ A_final
    denom = torch.norm(delta_bar).clamp_min(eps)
    trunc_error = torch.norm(delta_bar - approx) / denom
    drift_before = torch.norm(B_tilde - B_prev) + torch.norm(A_tilde - A_prev)
    drift_after = torch.norm(B_final - B_prev) + torch.norm(A_final - A_prev)

    return {
        "B_eff": B_final.to(dtype=dtype).cpu(),
        "A_eff": A_final.to(dtype=dtype).cpu(),
        "singular_values": S_r.detach().cpu(),
        "trunc_error": float(trunc_error.detach().cpu()),
        "update_norm": float(torch.norm(delta_bar).detach().cpu()),
        "drift_before": float(drift_before.detach().cpu()),
        "drift_after": float(drift_after.detach().cpu()),
        "server_svd_time_sec": svd_time,
        "procrustes_time_sec": proc_time,
        "svd_type": svd_type,
    }


@torch.no_grad()
def aggregate_idea_states(
    client_states,
    prev_global_state,
    server_rank_by_layer,
    alpha=16.0,
    gamma=0.5,
    procrustes="full",
    svd_type="exact",
    n_oversamples=8,
    n_iter=1,
):
    next_state = {}
    for layer_name, prev_global in prev_global_state.items():
        if layer_name.startswith("__"):
            continue
        layer_client_states = [cs[layer_name] for cs in client_states]
        next_state[layer_name] = aggregate_idea_one_layer(
            client_layer_states=layer_client_states,
            prev_global=prev_global,
            server_rank=server_rank_by_layer[layer_name],
            alpha=alpha,
            gamma=gamma,
            procrustes=procrustes,
            svd_type=svd_type,
            n_oversamples=n_oversamples,
            n_iter=n_iter,
        )
    return next_state


def weighted_average_tensor_tree(states, weights, eps=1e-8):
    if not states:
        return {}
    weights = weights.float()
    weights = weights / weights.sum().clamp_min(eps)
    out = {}
    first = states[0]
    for name in first:
        acc = None
        for w, state in zip(weights, states):
            value = state[name].float()
            acc = w * value if acc is None else acc + w * value
        out[name] = acc.to(dtype=first[name].dtype).cpu()
    return out


class IdeaUpdateSpaceMethod:
    name = "idea_update_space"

    def __init__(self, cfg, rank_map=None, server_rank_by_layer=None, procrustes=None):
        self.cfg = cfg
        self.rank_map = rank_map or {}
        self.server_rank_by_layer = server_rank_by_layer or {}
        aggregation_cfg = getattr(cfg, "aggregation", None)
        self.gamma = float(getattr(aggregation_cfg, "rank_weight_gamma", 0.5))
        self.procrustes = procrustes or getattr(aggregation_cfg, "procrustes", "full")
        svd_cfg = getattr(aggregation_cfg, "svd", None)
        self.svd_type = getattr(svd_cfg, "type", "exact")
        self.svd_n_oversamples = int(getattr(svd_cfg, "n_oversamples", 8))
        self.svd_n_iter = int(getattr(svd_cfg, "n_iter", 1))
        if self.procrustes == "none":
            self.name = "idea_no_procrustes"
        elif self.procrustes == "prefix_safe":
            self.name = "idea_prefix_safe"
        elif self.procrustes == "full":
            self.name = "idea_update_space"
        self.last_diagnostics = {}

    @property
    def alpha(self):
        return float(getattr(self.cfg.adapter, "alpha", 16.0))

    def inject(self, model, rank_by_layer=None, default_rank=None):
        rank = default_rank if default_rank is not None else int(getattr(self.cfg.adapter, "server_rank", 4))
        return inject_hetero_lora(
            model,
            self.cfg.model.target_modules,
            rank=rank,
            alpha=self.alpha,
            dropout=getattr(self.cfg.adapter, "dropout", 0.0),
            init_scale=getattr(self.cfg.adapter, "init_scale", 0.01),
            rank_by_layer=rank_by_layer,
        )

    def freeze(self, model):
        for _, p in model.named_parameters():
            p.requires_grad = False
        for _, module in model.named_modules():
            if isinstance(module, HeteroLoRALinear) and module.rank > 0:
                module.lora_A.requires_grad = True
                module.lora_B.requires_grad = True
        if self.cfg.model.train_classifier_head:
            for name, p in model.named_parameters():
                if "classifier" in name or "score" in name:
                    p.requires_grad = True

    def init_global_state(self, model):
        state = {}
        for name, module in model.named_modules():
            if isinstance(module, HeteroLoRALinear):
                R = int(self.server_rank_by_layer.get(name, module.rank))
                state[name] = {
                    "B_eff": torch.zeros(module.out_features, R, dtype=module.base.weight.dtype),
                    "A_eff": torch.randn(R, module.in_features, dtype=module.base.weight.dtype)
                    * getattr(self.cfg.adapter, "init_scale", 0.01),
                    "singular_values": None,
                    "trunc_error": None,
                    "update_norm": None,
                    "drift_before": None,
                    "drift_after": None,
                    "server_svd_time_sec": 0.0,
                    "procrustes_time_sec": 0.0,
                }
        state["__classifier_head__"] = get_classifier_state(model)
        return state

    @torch.no_grad()
    def set_client_state(self, model, global_state, client_id):
        rank_for_client = self.rank_map[int(client_id)]
        for name, module in model.named_modules():
            if isinstance(module, HeteroLoRALinear):
                r = int(rank_for_client[name])
                if r != module.rank:
                    raise ValueError(f"Client {client_id} layer {name} has module rank {module.rank}, expected {r}")
                if r > 0:
                    module.set_from_effective_prefix(global_state[name]["B_eff"], global_state[name]["A_eff"])
        set_classifier_state(model, global_state.get("__classifier_head__", {}))

    @torch.no_grad()
    def set_global_state(self, model, global_state):
        for name, module in model.named_modules():
            if isinstance(module, HeteroLoRALinear) and module.rank > 0:
                module.set_from_effective_prefix(global_state[name]["B_eff"], global_state[name]["A_eff"])
        set_classifier_state(model, global_state.get("__classifier_head__", {}))

    def get_client_state(self, model, client_id, num_samples):
        state = {}
        for name, module in model.named_modules():
            if isinstance(module, HeteroLoRALinear):
                layer_state = module.get_raw_state()
                layer_state["num_samples"] = int(num_samples)
                state[name] = layer_state
        state["__classifier_head__"] = get_classifier_state(model)
        state["__num_samples__"] = int(num_samples)
        state["__client_id__"] = int(client_id)
        return state

    def aggregate(self, client_states, prev_global_state):
        next_state = aggregate_idea_states(
            client_states=client_states,
            prev_global_state=prev_global_state,
            server_rank_by_layer=self.server_rank_by_layer,
            alpha=self.alpha,
            gamma=self.gamma,
            procrustes=self.procrustes,
            svd_type=self.svd_type,
            n_oversamples=self.svd_n_oversamples,
            n_iter=self.svd_n_iter,
        )

        if self.cfg.model.train_classifier_head:
            weights = torch.tensor([float(cs.get("__num_samples__", 1.0)) for cs in client_states], dtype=torch.float32)
            next_state["__classifier_head__"] = weighted_average_tensor_tree(
                [cs["__classifier_head__"] for cs in client_states],
                weights,
            )
        else:
            next_state["__classifier_head__"] = prev_global_state.get("__classifier_head__", {})

        diagnostics = {}
        for name, layer_state in next_state.items():
            if name.startswith("__"):
                continue
            diagnostics[name] = {
                "trunc_error": layer_state.get("trunc_error"),
                "update_norm": layer_state.get("update_norm"),
                "drift_before": layer_state.get("drift_before"),
                "drift_after": layer_state.get("drift_after"),
                "server_svd_time_sec": layer_state.get("server_svd_time_sec", 0.0),
                "procrustes_time_sec": layer_state.get("procrustes_time_sec", 0.0),
                "svd_type": layer_state.get("svd_type", self.svd_type),
            }
        self.last_diagnostics = diagnostics
        return next_state

    def communication(self, selected_clients):
        upload = 0
        download = 0
        for cid in selected_clients:
            for layer_name, r in self.rank_map[int(cid)].items():
                r = int(r)
                if r <= 0:
                    continue
                B_eff = self._shape_state[layer_name]["B_eff"]
                A_eff = self._shape_state[layer_name]["A_eff"]
                params = r * (B_eff.shape[0] + A_eff.shape[1])
                upload += params
                download += params
        return {"upload": upload, "download": download, "total": upload + download}

    def bind_shape_state_for_comm(self, global_state):
        self._shape_state = global_state

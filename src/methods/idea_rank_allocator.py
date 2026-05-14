from __future__ import annotations

import math
import random


def _cfg_get(obj, name, default=None):
    return getattr(obj, name, default) if obj is not None else default


def _discretize_rank(value, candidates):
    candidates = sorted({int(x) for x in candidates})
    if value <= 0:
        return 0 if 0 in candidates else candidates[0]
    return min(candidates, key=lambda r: abs(r - value))


def _matches_any(name, patterns):
    return any(p in name for p in patterns)


def layer_sensitivity(layer_name, cfg):
    hetero_cfg = getattr(cfg, "hetero_rank", None)
    sensitivities = _cfg_get(hetero_cfg, "layer_sensitivity", {}) or {}
    betas = _cfg_get(hetero_cfg, "layer_beta", {}) or {}
    sensitivity = float(sensitivities.get(layer_name, 1.0)) if isinstance(sensitivities, dict) else 1.0
    beta = float(betas.get(layer_name, 1.0)) if isinstance(betas, dict) else 1.0
    return sensitivity * beta


def semantic_gate_rank(layer_name, rank, cfg):
    hetero_cfg = getattr(cfg, "hetero_rank", None)
    semantic_patterns = _cfg_get(hetero_cfg, "semantic_layer_patterns", []) or []
    min_semantic = int(_cfg_get(hetero_cfg, "r_min_semantic", 0) or 0)
    if semantic_patterns and min_semantic > 0 and _matches_any(layer_name, semantic_patterns) and rank < min_semantic:
        return 0
    return rank


def manual_client_rank(client_position, num_clients, cfg):
    hetero_cfg = cfg.hetero_rank
    weak_rank = int(_cfg_get(hetero_cfg, "weak_rank", 2))
    medium_rank = int(_cfg_get(hetero_cfg, "medium_rank", 4))
    strong_rank = int(_cfg_get(hetero_cfg, "strong_rank", 8))
    weak_ratio = float(_cfg_get(hetero_cfg, "weak_ratio", 0.3))
    medium_ratio = float(_cfg_get(hetero_cfg, "medium_ratio", 0.4))

    weak_cut = int(round(num_clients * weak_ratio))
    medium_cut = weak_cut + int(round(num_clients * medium_ratio))
    if client_position < weak_cut:
        return weak_rank
    if client_position < medium_cut:
        return medium_rank
    return strong_rank


def homogeneous_client_rank(cfg):
    return int(getattr(cfg.adapter, "server_rank", getattr(cfg.adapter, "rank", 4)))


def client_budget(client_position, num_clients, cfg):
    hetero_cfg = getattr(cfg, "hetero_rank", None)
    weak_budget = float(_cfg_get(hetero_cfg, "weak_budget", 2.0))
    medium_budget = float(_cfg_get(hetero_cfg, "medium_budget", 4.0))
    strong_budget = float(_cfg_get(hetero_cfg, "strong_budget", 8.0))
    weak_ratio = float(_cfg_get(hetero_cfg, "weak_ratio", 0.3))
    medium_ratio = float(_cfg_get(hetero_cfg, "medium_ratio", 0.4))
    weak_cut = int(round(num_clients * weak_ratio))
    medium_cut = weak_cut + int(round(num_clients * medium_ratio))
    if client_position < weak_cut:
        return weak_budget
    if client_position < medium_cut:
        return medium_budget
    return strong_budget


def build_client_rank_maps(layer_names, cfg, seed=0, layer_shapes=None):
    hetero_cfg = getattr(cfg, "hetero_rank", None)
    mode = _cfg_get(hetero_cfg, "mode", "homogeneous").lower()
    num_clients = int(cfg.federated.num_clients)
    rank_maps = {}

    order = list(range(num_clients))
    random.Random(seed).shuffle(order)
    inverse_order = {cid: pos for pos, cid in enumerate(order)}

    if mode == "sensitivity":
        candidates = _cfg_get(hetero_cfg, "rank_candidates", [0, 2, 4, 8]) or [0, 2, 4, 8]
        layer_scores = {name: layer_sensitivity(name, cfg) for name in layer_names}
        score_sum = sum(layer_scores.values()) or 1.0

    for cid in range(num_clients):
        pos = inverse_order[cid]
        if mode == "manual":
            rank = manual_client_rank(pos, num_clients, cfg)
            rank_maps[cid] = {layer_name: semantic_gate_rank(layer_name, int(rank), cfg) for layer_name in layer_names}
        elif mode == "strong_hetero":
            ranks = list(_cfg_get(hetero_cfg, "rank_candidates", [0, 2, 4, 8]) or [0, 2, 4, 8])
            rank = int(ranks[pos % len(ranks)])
            rank_maps[cid] = {layer_name: semantic_gate_rank(layer_name, rank, cfg) for layer_name in layer_names}
        elif mode == "homogeneous":
            rank = homogeneous_client_rank(cfg)
            rank_maps[cid] = {layer_name: semantic_gate_rank(layer_name, int(rank), cfg) for layer_name in layer_names}
        elif mode == "sensitivity":
            budget = client_budget(pos, num_clients, cfg)
            rank_maps[cid] = {}
            for layer_name in layer_names:
                shape_penalty = 1.0
                if layer_shapes and layer_name in layer_shapes:
                    dout, din = layer_shapes[layer_name]
                    shape_penalty = max(1.0, math.sqrt(float(dout + din)))
                raw = (layer_scores[layer_name] / score_sum) * budget * len(layer_names) / shape_penalty
                rank = _discretize_rank(raw, candidates)
                rank_maps[cid][layer_name] = semantic_gate_rank(layer_name, int(rank), cfg)
        else:
            raise ValueError(f"Unknown hetero_rank.mode={mode!r}")

    return rank_maps


def rank_distribution(rank_map, selected_clients=None):
    selected = selected_clients if selected_clients is not None else sorted(rank_map)
    counts = {}
    for cid in selected:
        for rank in rank_map[cid].values():
            counts[str(int(rank))] = counts.get(str(int(rank)), 0) + 1
    return counts

from __future__ import annotations

from pathlib import Path

import yaml


BASE = {
    "seed": 42,
    "method": "idea_update_space",
    "data": {"root": "D:/data"},
    "task": {"name": "rte", "type": "glue"},
    "model": {
        "name_or_path": "roberta-large",
        "task_type": "sequence_classification",
        "target_modules": ["query", "value"],
        "num_labels": 2,
        "train_classifier_head": True,
        "use_fast_tokenizer": True,
    },
    "adapter": {
        "type": "hetero_lora",
        "alpha": 16,
        "dropout": 0.0,
        "init_scale": 0.01,
        "server_rank": 8,
    },
    "aggregation": {
        "space": "delta_w",
        "rank_weight_gamma": 0.5,
        "procrustes": "full",
        "svd": {"type": "exact", "eps": 1.0e-8, "n_oversamples": 8, "n_iter": 1},
    },
    "hetero_rank": {
        "mode": "manual",
        "weak_rank": 2,
        "medium_rank": 4,
        "strong_rank": 8,
        "weak_ratio": 0.3,
        "medium_ratio": 0.4,
        "strong_ratio": 0.3,
        "rank_candidates": [0, 2, 4, 8],
        "r_min_semantic": 4,
        "semantic_layer_patterns": [],
    },
    "federated": {
        "num_clients": 20,
        "participation_ratio": 1.0,
        "rounds": 50,
        "local_epochs": 1,
        "non_iid": {"type": "dirichlet", "alpha": 0.5, "min_size": 1},
    },
    "training": {
        "lr": 5.0e-5,
        "weight_decay": 0.0,
        "batch_size": 4,
        "max_seq_length": 128,
        "optimizer": "adamw",
        "max_grad_norm": 1.0,
    },
    "eval": {"every_round": 1, "batch_size": 64, "metric": "accuracy", "debug_predictions": True},
    "output_dir": "outputs/idea_update_space_roberta_large_rte",
}


def deep_copy(obj):
    return yaml.safe_load(yaml.safe_dump(obj, sort_keys=False))


def write_config(name, updates):
    cfg = deep_copy(BASE)
    for path, value in updates.items():
        target = cfg
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    cfg["output_dir"] = f"outputs/{name}"
    path = Path("configs") / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return path


def main():
    configs = {
        "idea_update_space_roberta_large_rte": {},
        "idea_no_procrustes_roberta_large_rte": {
            "method": "idea_no_procrustes",
            "aggregation.procrustes": "none",
        },
        "idea_prefix_safe_roberta_large_rte": {
            "method": "idea_prefix_safe",
            "aggregation.procrustes": "prefix_safe",
        },
        "idea_gamma0_roberta_large_rte": {"aggregation.rank_weight_gamma": 0.0},
        "idea_gamma05_roberta_large_rte": {"aggregation.rank_weight_gamma": 0.5},
        "idea_gamma1_roberta_large_rte": {"aggregation.rank_weight_gamma": 1.0},
        "idea_homogeneous_r4_roberta_large_rte": {
            "adapter.server_rank": 4,
            "hetero_rank.mode": "homogeneous",
        },
        "idea_mild_hetero_roberta_large_rte": {
            "adapter.server_rank": 8,
            "hetero_rank.mode": "manual",
            "hetero_rank.weak_rank": 2,
            "hetero_rank.medium_rank": 4,
            "hetero_rank.strong_rank": 8,
        },
        "idea_strong_hetero_roberta_large_rte": {
            "adapter.server_rank": 8,
            "hetero_rank.mode": "strong_hetero",
            "hetero_rank.rank_candidates": [0, 2, 4, 8],
        },
        "idea_sensitivity_roberta_large_rte": {
            "adapter.server_rank": 8,
            "hetero_rank.mode": "sensitivity",
            "hetero_rank.weak_budget": 2.0,
            "hetero_rank.medium_budget": 4.0,
            "hetero_rank.strong_budget": 8.0,
        },
        "idea_rho1_roberta_large_rte": {"federated.non_iid.alpha": 1.0},
        "idea_rho05_roberta_large_rte": {"federated.non_iid.alpha": 0.5},
        "idea_rho01_roberta_large_rte": {"federated.non_iid.alpha": 0.1},
        "idea_participation1_roberta_large_rte": {"federated.participation_ratio": 1.0},
        "idea_participation05_roberta_large_rte": {"federated.participation_ratio": 0.5},
        "idea_participation02_roberta_large_rte": {"federated.participation_ratio": 0.2},
        "idea_randomized_svd_roberta_large_rte": {"aggregation.svd.type": "randomized"},
    }

    task_labels = {
        "mrpc": 2,
        "qnli": 2,
        "wnli": 2,
        "rte": 2,
    }
    for task, num_labels in task_labels.items():
        suffix = f"roberta_large_{task}"
        configs[f"idea_update_space_{suffix}"] = {
            "task.name": task,
            "model.num_labels": num_labels,
        }
        configs[f"idea_prefix_safe_{suffix}"] = {
            "method": "idea_prefix_safe",
            "task.name": task,
            "model.num_labels": num_labels,
            "aggregation.procrustes": "prefix_safe",
        }
        configs[f"idea_no_procrustes_{suffix}"] = {
            "method": "idea_no_procrustes",
            "task.name": task,
            "model.num_labels": num_labels,
            "aggregation.procrustes": "none",
        }

    for name, updates in configs.items():
        path = write_config(name, updates)
        print(path)


if __name__ == "__main__":
    main()

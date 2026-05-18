from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.utils import logging as hf_logging
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}. Install all runtime dependencies with:\n"
        "  python -m pip install -r requirements.txt"
    ) from exc

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arguments import load_config, parse_run_args
from src.datasets.glue import load_glue_dataset, validate_task_num_labels
from src.datasets.partition import client_label_stats, dirichlet_partition, iid_partition
from src.eval.glue_eval import evaluate_glue
from src.federated.idea_trainer import IdeaFederatedTrainer
from src.logging_utils import write_json
from src.methods.idea_rank_allocator import build_client_rank_maps
from src.methods.idea_update_space import IdeaUpdateSpaceMethod
from src.models.inject_hetero_lora import discover_lora_targets
from src.seed import set_seed


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrained_kwargs(cfg):
    return {"local_files_only": bool(getattr(cfg.model, "local_files_only", False))}


def build_partition(labels, cfg):
    partition_cfg = cfg.federated.non_iid
    partition_type = getattr(partition_cfg, "type", "dirichlet").lower()
    if partition_type == "iid":
        return iid_partition(
            labels=labels,
            num_clients=cfg.federated.num_clients,
            seed=cfg.seed,
            min_size=getattr(partition_cfg, "min_size", 1),
        )
    if partition_type == "dirichlet":
        return dirichlet_partition(
            labels=labels,
            num_clients=cfg.federated.num_clients,
            alpha=partition_cfg.alpha,
            seed=cfg.seed,
            min_size=getattr(partition_cfg, "min_size", 1),
            min_classes_per_client=getattr(partition_cfg, "min_classes_per_client", 1),
            max_size_ratio=getattr(partition_cfg, "max_size_ratio", None),
            max_tries=getattr(partition_cfg, "max_tries", 5000),
        )
    raise ValueError(f"Unknown partition type: {partition_type}")


def main():
    args = parse_run_args()
    overrides = {}
    if args.method is not None:
        overrides["method"] = args.method
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    cfg = load_config(args.config, overrides)
    validate_task_num_labels(cfg.task.name, cfg.model.num_labels)
    set_seed(cfg.seed)

    device = get_device()
    if device.type == "cuda":
        print(f"[runtime] using cuda: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[runtime] using cpu: torch.cuda.is_available() is False. Training will be very slow.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        use_fast=getattr(cfg.model, "use_fast_tokenizer", True),
        **pretrained_kwargs(cfg),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoded = load_glue_dataset(
        cfg.task.name,
        tokenizer,
        cfg.training.max_seq_length,
        data_root=getattr(getattr(cfg, "data", None), "root", None),
    )
    train_dataset = encoded["train"]
    eval_dataset = encoded["validation_matched"] if cfg.task.name.lower() == "mnli" else encoded["validation"]
    labels = [int(x["labels"]) for x in train_dataset]
    client_indices = build_partition(labels, cfg)
    write_json(Path(cfg.output_dir) / "client_stats.json", client_label_stats(labels, client_indices))

    probe_model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=cfg.model.num_labels,
        **pretrained_kwargs(cfg),
    )
    if tokenizer.pad_token is not None:
        probe_model.config.pad_token_id = tokenizer.pad_token_id
    targets = discover_lora_targets(probe_model, cfg.model.target_modules)
    layer_names = [name for name, _, _ in targets]
    layer_shapes = {name: (out_features, in_features) for name, out_features, in_features in targets}
    server_rank = int(getattr(cfg.adapter, "server_rank", getattr(cfg.adapter, "rank", 4)))
    server_rank_by_layer = {name: server_rank for name in layer_names}
    rank_map = build_client_rank_maps(layer_names, cfg, seed=cfg.seed, layer_shapes=layer_shapes)
    del probe_model

    procrustes = getattr(getattr(cfg, "aggregation", None), "procrustes", "full")
    if cfg.method == "idea_no_procrustes":
        procrustes = "none"
    elif cfg.method == "idea_prefix_safe":
        procrustes = "prefix_safe"

    method = IdeaUpdateSpaceMethod(
        cfg,
        rank_map=rank_map,
        server_rank_by_layer=server_rank_by_layer,
        procrustes=procrustes,
    )

    def make_base_model():
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name_or_path,
            num_labels=cfg.model.num_labels,
            **pretrained_kwargs(cfg),
        )
        if tokenizer.pad_token is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        return model

    def global_model_fn():
        model = make_base_model()
        method.inject(model, rank_by_layer=server_rank_by_layer, default_rank=server_rank)
        method.freeze(model)
        return model

    def client_model_fn(client_id):
        model = make_base_model()
        method.inject(model, rank_by_layer=rank_map[int(client_id)], default_rank=server_rank)
        method.freeze(model)
        return model

    trainer = IdeaFederatedTrainer(
        global_model_fn=global_model_fn,
        client_model_fn=client_model_fn,
        method=method,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        client_indices=client_indices,
        cfg=cfg,
        device=device,
        evaluator=evaluate_glue,
        resume_checkpoint=args.resume_checkpoint,
    )
    trainer.run()


if __name__ == "__main__":
    main()

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
LOCAL_DEPS = ROOT / ".python_deps"
if LOCAL_DEPS.exists() and str(LOCAL_DEPS) not in sys.path:
    sys.path.insert(0, str(LOCAL_DEPS))

from src.arguments import load_config, parse_run_args
from src.datasets.glue import load_glue_dataset, validate_task_num_labels
from src.datasets.partition import client_label_stats, dirichlet_partition
from src.eval.glue_eval import evaluate_glue
from src.federated.trainer import FederatedTrainer
from src.logging_utils import write_json
from src.methods import build_method
from src.seed import set_seed


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrained_kwargs(cfg):
    return {"local_files_only": bool(getattr(cfg.model, "local_files_only", False))}


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
    if cfg.task.name.lower() == "mnli":
        eval_dataset = {
            "matched": encoded["validation_matched"],
            "mismatched": encoded["validation_mismatched"],
        }
    else:
        eval_dataset = encoded["validation"]

    labels = [int(x["labels"]) for x in train_dataset]
    client_indices = dirichlet_partition(
        labels=labels,
        num_clients=cfg.federated.num_clients,
        alpha=cfg.federated.non_iid.alpha,
        seed=cfg.seed,
        min_size=getattr(cfg.federated.non_iid, "min_size", 1),
    )
    write_json(Path(cfg.output_dir) / "client_stats.json", client_label_stats(labels, client_indices))

    method = build_method(cfg.method, cfg)

    def model_fn():
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name_or_path,
            num_labels=cfg.model.num_labels,
            **pretrained_kwargs(cfg),
        )
        if tokenizer.pad_token is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        method.inject(model)
        method.freeze(model)
        return model

    trainer = FederatedTrainer(
        model_fn=model_fn,
        method=method,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        client_indices=client_indices,
        cfg=cfg,
        device=device,
        evaluator=evaluate_glue,
    )
    trainer.run()


if __name__ == "__main__":
    main()

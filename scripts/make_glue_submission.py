from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arguments import load_config
from src.datasets.glue import load_glue_dataset
from src.methods import build_method
from src.methods.idea_rank_allocator import build_client_rank_maps
from src.methods.idea_update_space import IdeaUpdateSpaceMethod
from src.models.inject_hetero_lora import discover_lora_targets


SUBMISSION_TASKS = [
    "CoLA",
    "SST-2",
    "MRPC",
    "STS-B",
    "QQP",
    "MNLI-m",
    "MNLI-mm",
    "QNLI",
    "RTE",
    "WNLI",
    "AX",
]

TEST_FILES = {
    "CoLA": ("CoLA", "test.tsv"),
    "SST-2": ("SST-2", "test.tsv"),
    "MRPC": ("MRPC", "test.tsv"),
    "STS-B": ("STS-B", "test.tsv"),
    "QQP": ("QQP", "test.tsv"),
    "MNLI-m": ("MNLI", "test_matched.tsv"),
    "MNLI-mm": ("MNLI", "test_mismatched.tsv"),
    "QNLI": ("QNLI", "test.tsv"),
    "RTE": ("RTE", "test.tsv"),
    "WNLI": ("WNLI", "test.tsv"),
    "AX": ("diagnostic", "diagnostic.tsv"),
}

OFFICIAL_TEST_ROWS = {
    "CoLA": 1063,
    "SST-2": 1821,
    "MRPC": 1725,
    "STS-B": 1379,
    "QQP": 390965,
    "MNLI-m": 9796,
    "MNLI-mm": 9847,
    "QNLI": 5463,
    "RTE": 3000,
    "WNLI": 146,
    "AX": 1104,
}

PLACEHOLDER_PREDICTIONS = {
    "CoLA": "0",
    "SST-2": "1",
    "MRPC": "0",
    "STS-B": "0.0",
    "QQP": "0",
    "MNLI-m": "entailment",
    "MNLI-mm": "entailment",
    "QNLI": "entailment",
    "RTE": "entailment",
    "WNLI": "0",
    "AX": "entailment",
}

LABEL_TEXT = {
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "wnli": {0: "0", 1: "1"},
}


def parse_model_spec(spec: str) -> tuple[str, Path, Path]:
    parts = spec.split("=", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--model must look like TASK=config.yaml|checkpoint.pt")
    task = parts[0]
    rhs = parts[1]
    paths = rhs.split("|", 1) if "|" in rhs else rhs.split(":", 1)
    if len(paths) != 2:
        raise argparse.ArgumentTypeError("--model must look like TASK=config.yaml|checkpoint.pt")
    return task, Path(paths[0]), Path(paths[1])


def pretrained_kwargs(cfg):
    return {"local_files_only": bool(getattr(cfg.model, "local_files_only", False))}


def get_test_row_count(data_root: Path, task: str, allow_official_counts: bool) -> int:
    task_dir, file_name = TEST_FILES[task]
    path = data_root / task_dir / file_name
    if path.exists():
        return int(len(pd.read_csv(path, sep="\t", quoting=3)))
    if allow_official_counts:
        return OFFICIAL_TEST_ROWS[task]
    raise FileNotFoundError(
        f"Missing local test file for {task}: {path}. "
        "Pass --allow-official-counts to create a placeholder file with the official row count."
    )


def write_submission_file(path: Path, predictions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "prediction"])
        for idx, pred in enumerate(predictions):
            writer.writerow([idx, pred])


def build_idea_model(cfg, checkpoint_path: Path, tokenizer):
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
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=cfg.model.num_labels,
        **pretrained_kwargs(cfg),
    )
    if tokenizer.pad_token is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    method.inject(model, rank_by_layer=server_rank_by_layer, default_rank=server_rank)
    method.freeze(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    method.set_global_state(model, checkpoint["state"])
    return model


def build_standard_model(cfg, checkpoint_path: Path, tokenizer):
    method = build_method(cfg.method, cfg)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.name_or_path,
        num_labels=cfg.model.num_labels,
        **pretrained_kwargs(cfg),
    )
    if tokenizer.pad_token is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    method.inject(model)
    method.freeze(model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    method.set_state(model, checkpoint["state"])
    return model


def predict_from_checkpoint(config_path: Path, checkpoint_path: Path, split_name: str, device: torch.device) -> list[str]:
    cfg = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.name_or_path,
        use_fast=getattr(cfg.model, "use_fast_tokenizer", True),
        **pretrained_kwargs(cfg),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if str(cfg.method).startswith("idea"):
        model = build_idea_model(cfg, checkpoint_path, tokenizer)
    else:
        model = build_standard_model(cfg, checkpoint_path, tokenizer)
    model.to(device)
    model.eval()

    encoded = load_glue_dataset(
        cfg.task.name,
        tokenizer,
        cfg.training.max_seq_length,
        data_root=getattr(getattr(cfg, "data", None), "root", None),
    )
    dataset = encoded[split_name]
    loader = DataLoader(
        dataset,
        batch_size=getattr(cfg.eval, "batch_size", 32),
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    task_name = cfg.task.name.lower()
    pred_text = LABEL_TEXT[task_name]
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels", None)
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            predictions.extend(pred_text[int(pred)] for pred in preds)
    return predictions


def make_placeholder(task: str, data_root: Path, allow_official_counts: bool) -> list[str]:
    n_rows = get_test_row_count(data_root, task, allow_official_counts=allow_official_counts)
    return [PLACEHOLDER_PREDICTIONS[task]] * n_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="D:/data")
    parser.add_argument("--out-dir", default="glue_submit/submission")
    parser.add_argument("--zip", default="glue_submit/glue_submission.zip")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        type=parse_model_spec,
        help="Use a trained checkpoint for one task, e.g. WNLI=configs/x.yaml|outputs/x/checkpoints/round_014/global_adapter.pt",
    )
    parser.add_argument(
        "--allow-official-counts",
        action="store_true",
        help="For missing local test files, create placeholder files using known official GLUE test row counts.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    zip_path = Path(args.zip)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_specs = {task: (config, checkpoint) for task, config, checkpoint in args.model}
    split_for_task = {
        "MNLI-m": "test_matched",
        "MNLI-mm": "test_mismatched",
        "QNLI": "test",
        "RTE": "test",
        "WNLI": "test",
    }

    written = []
    for task in SUBMISSION_TASKS:
        if task in model_specs:
            config_path, checkpoint_path = model_specs[task]
            if task not in split_for_task:
                raise ValueError(f"Model prediction is not implemented for {task}; use placeholders for this task.")
            predictions = predict_from_checkpoint(config_path, checkpoint_path, split_for_task[task], device)
            source = f"model:{checkpoint_path}"
        else:
            predictions = make_placeholder(task, data_root, args.allow_official_counts)
            source = "placeholder"

        out_path = out_dir / f"{task}.tsv"
        write_submission_file(out_path, predictions)
        written.append((task, len(predictions), source, out_path))

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for task in SUBMISSION_TASKS:
            path = out_dir / f"{task}.tsv"
            zf.write(path, arcname=path.name)

    print(f"Wrote {zip_path}")
    for task, n_rows, source, path in written:
        print(f"{task}: {n_rows} rows, {source}, {path}")


if __name__ == "__main__":
    main()

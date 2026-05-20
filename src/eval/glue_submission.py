from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


TASK_OUTPUTS = {
    "cola": [("test", "CoLA.tsv")],
    "sst2": [("test", "SST-2.tsv")],
    "mrpc": [("test", "MRPC.tsv")],
    "stsb": [("test", "STS-B.tsv")],
    "qqp": [("test", "QQP.tsv")],
    "mnli": [("test_matched", "MNLI-m.tsv"), ("test_mismatched", "MNLI-mm.tsv")],
    "qnli": [("test", "QNLI.tsv")],
    "rte": [("test", "RTE.tsv")],
    "wnli": [("test", "WNLI.tsv")],
}

CLASSIFICATION_LABELS = {
    "mnli": {0: "entailment", 1: "neutral", 2: "contradiction"},
    "qnli": {0: "entailment", 1: "not_entailment"},
    "rte": {0: "entailment", 1: "not_entailment"},
    "wnli": {0: "0", 1: "1"},
    "cola": {0: "0", 1: "1"},
    "sst2": {0: "0", 1: "1"},
    "mrpc": {0: "0", 1: "1"},
    "qqp": {0: "0", 1: "1"},
}


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("\\", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _prediction_text(task_name: str, logits: torch.Tensor) -> list[str]:
    task_name = task_name.lower()
    if task_name == "stsb":
        return [str(float(x)) for x in logits.squeeze(-1).detach().cpu().tolist()]

    labels = CLASSIFICATION_LABELS.get(task_name)
    if labels is None:
        raise ValueError(f"Unsupported GLUE submission task: {task_name}")
    preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    return [labels[int(pred)] for pred in preds]


def _write_tsv(path: Path, predictions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["index", "prediction"])
        for idx, pred in enumerate(predictions):
            writer.writerow([idx, pred])


def predict_split(model, dataset, tokenizer, device, cfg) -> list[str]:
    loader = DataLoader(
        dataset,
        batch_size=getattr(cfg.eval, "batch_size", 32),
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch.pop("labels", None)
            logits = model(**batch).logits
            predictions.extend(_prediction_text(cfg.task.name, logits))
    return predictions


def write_glue_test_predictions(
    model,
    encoded_dataset,
    tokenizer,
    device,
    cfg,
    output_root: str | Path = "glue_submit",
) -> list[Path]:
    task_name = cfg.task.name.lower()
    outputs = TASK_OUTPUTS.get(task_name)
    if outputs is None:
        raise ValueError(f"Unsupported GLUE submission task: {task_name}")

    experiment_name = _safe_name(Path(cfg.output_dir).name)
    out_dir = Path(output_root) / experiment_name
    written = []

    for split_name, file_name in outputs:
        if split_name not in encoded_dataset:
            print(
                f"[submission] skip {file_name}: split '{split_name}' not found. "
                "Make sure the local GLUE test file exists.",
                flush=True,
            )
            continue
        predictions = predict_split(model, encoded_dataset[split_name], tokenizer, device, cfg)
        path = out_dir / file_name
        _write_tsv(path, predictions)
        written.append(path)
        print(f"[submission] wrote {path} ({len(predictions)} rows)", flush=True)

    return written

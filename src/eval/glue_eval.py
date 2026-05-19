from collections import Counter

import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def _evaluate_one_split(model, dataset, tokenizer, device, cfg):
    task_name = cfg.task.name.lower()
    metric = evaluate.load("glue", task_name)
    loader = DataLoader(
        dataset,
        batch_size=getattr(cfg.eval, "batch_size", 32),
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    model.eval()

    debug_predictions = getattr(cfg.eval, "debug_predictions", False)
    pred_counter = Counter()
    label_counter = Counter()
    pair_counter = Counter()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        with torch.no_grad():
            logits = model(**batch).logits

        preds = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=preds.cpu(), references=labels.cpu())

        if debug_predictions:
            pred_list = [int(x) for x in preds.cpu().tolist()]
            label_list = [int(x) for x in labels.cpu().tolist()]
            pred_counter.update(pred_list)
            label_counter.update(label_list)
            pair_counter.update((label, pred) for label, pred in zip(label_list, pred_list))

    result = metric.compute()

    if debug_predictions:
        result["pred_counts"] = {str(k): int(v) for k, v in sorted(pred_counter.items())}
        result["label_counts"] = {str(k): int(v) for k, v in sorted(label_counter.items())}
        result["confusion"] = {
            f"{label}->{pred}": int(v)
            for (label, pred), v in sorted(pair_counter.items())
        }
        result["per_class_accuracy"] = {}
        for label, total in sorted(label_counter.items()):
            correct = pair_counter.get((label, label), 0)
            result["per_class_accuracy"][str(label)] = float(correct / total) if total else 0.0

    return result


def evaluate_glue(model, dataset, tokenizer, device, cfg):
    if isinstance(dataset, dict):
        out = {}
        for split_name, split_dataset in dataset.items():
            split_metrics = _evaluate_one_split(model, split_dataset, tokenizer, device, cfg)
            for key, value in split_metrics.items():
                out[f"{split_name}_{key}"] = value
        if "matched_accuracy" in out and "mismatched_accuracy" in out:
            out = {
                "accuracy": (out["matched_accuracy"] + out["mismatched_accuracy"]) / 2.0,
                **out,
            }
        return out
    return _evaluate_one_split(model, dataset, tokenizer, device, cfg)

from collections import Counter

import evaluate
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def evaluate_glue(model, dataset, tokenizer, device, cfg):
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

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")

        with torch.no_grad():
            logits = model(**batch).logits

        preds = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=preds.cpu(), references=labels.cpu())

        if debug_predictions:
            pred_counter.update(int(x) for x in preds.cpu().tolist())
            label_counter.update(int(x) for x in labels.cpu().tolist())

    result = metric.compute()

    if debug_predictions:
        result["pred_counts"] = {str(k): int(v) for k, v in sorted(pred_counter.items())}
        result["label_counts"] = {str(k): int(v) for k, v in sorted(label_counter.items())}

    return result

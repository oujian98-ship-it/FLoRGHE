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
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.no_grad():
            logits = model(**batch).logits
        preds = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=preds.cpu(), references=labels.cpu())
    return metric.compute()

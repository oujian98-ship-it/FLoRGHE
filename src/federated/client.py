from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding


class ClientTrainer:
    def __init__(self, client_id, dataset, indices, tokenizer, cfg, device):
        self.client_id = client_id
        self.dataset = dataset
        self.indices = indices
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device

    def make_loader(self):
        return DataLoader(
            Subset(self.dataset, self.indices),
            batch_size=self.cfg.training.batch_size,
            shuffle=True,
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )

    def train(self, model, round_id=None, show_progress=True):
        model.to(self.device)
        model.train()

        adapter_params = []
        head_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "classifier" in name or "score" in name:
                head_params.append(p)
            else:
                adapter_params.append(p)

        lr = float(self.cfg.training.lr)
        head_lr = float(getattr(self.cfg.training, "head_lr", lr))
        weight_decay = float(getattr(self.cfg.training, "weight_decay", 0.0))
        param_groups = []
        if adapter_params:
            param_groups.append(
                {
                    "params": adapter_params,
                    "lr": lr,
                    "weight_decay": weight_decay,
                }
            )
        if head_params:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": head_lr,
                    "weight_decay": weight_decay,
                }
            )

        trainable_params = adapter_params + head_params
        optimizer = torch.optim.AdamW(param_groups)
        loader = self.make_loader()
        total_loss = 0.0
        steps = 0
        total_batches = len(loader) * self.cfg.federated.local_epochs
        desc = f"round {round_id} client {self.client_id}" if round_id is not None else f"client {self.client_id}"
        progress = tqdm(
            total=total_batches,
            desc=desc,
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress,
        )

        for _ in range(self.cfg.federated.local_epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.cfg.training.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.cfg.training.max_grad_norm)
                optimizer.step()
                total_loss += loss.item()
                steps += 1
                progress.set_postfix(loss=f"{total_loss / max(1, steps):.4f}")
                progress.update(1)

        progress.close()

        return {"loss": total_loss / max(1, steps), "steps": steps}

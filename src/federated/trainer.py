from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.arguments import namespace_to_dict
from src.datasets.partition import client_label_stats
from src.federated.client import ClientTrainer
from src.logging_utils import append_jsonl, ensure_dir, write_json


class FederatedTrainer:
    def __init__(
        self,
        model_fn,
        method,
        tokenizer,
        train_dataset,
        eval_dataset,
        client_indices,
        cfg,
        device,
        evaluator,
    ):
        self.model_fn = model_fn
        self.method = method
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.client_indices = client_indices
        self.train_labels = [int(row["labels"]) for row in train_dataset]
        self.cfg = cfg
        self.device = device
        self.evaluator = evaluator
        self.global_model = model_fn().to(device)
        self.global_state = method.get_state(self.global_model)
        self.output_dir = ensure_dir(cfg.output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = timestamp
        self.log_path = Path(self.output_dir) / "logs" / f"{timestamp}.log"
        runtime_cfg = getattr(cfg, "runtime", None)
        self.reuse_client_model = getattr(runtime_cfg, "reuse_client_model", True)

    def select_clients(self, round_id):
        n = self.cfg.federated.num_clients
        m = max(1, int(n * self.cfg.federated.participation_ratio))
        rng = random.Random(self.cfg.seed + round_id)
        return rng.sample(list(range(n)), m)

    def save_checkpoint(self, round_id, metrics, comm):
        ckpt_dir = ensure_dir(Path(self.output_dir) / "checkpoints" / f"round_{round_id:03d}")
        torch.save(
            {
                "round": round_id,
                "method": self.method.name,
                "state": self.global_state,
                "config": namespace_to_dict(self.cfg),
            },
            ckpt_dir / "global_adapter.pt",
        )
        write_json(ckpt_dir / "metrics.json", metrics)
        write_json(ckpt_dir / "comm.json", comm)
        diagnostics = getattr(self.method, "last_diagnostics", None)
        if diagnostics:
            write_json(ckpt_dir / "diagnostics.json", diagnostics)

    def run(self):
        logs = []
        round_iter = tqdm(
            range(1, self.cfg.federated.rounds + 1),
            desc="federated rounds",
            dynamic_ncols=True,
        )
        for round_id in round_iter:
            selected = self.select_clients(round_id)
            client_states = []
            client_logs = []
            reusable_local_model = None
            if self.reuse_client_model:
                reusable_local_model = self.model_fn().to(self.device)

            train_start = time.time()
            for cid in tqdm(selected, desc=f"round {round_id} clients", leave=False, dynamic_ncols=True):
                local_model = reusable_local_model if reusable_local_model is not None else self.model_fn().to(self.device)
                self.method.set_state(local_model, self.global_state)
                client = ClientTrainer(
                    client_id=cid,
                    dataset=self.train_dataset,
                    indices=self.client_indices[cid],
                    tokenizer=self.tokenizer,
                    cfg=self.cfg,
                    device=self.device,
                )
                client_log = client.train(local_model, round_id=round_id)
                client_logs.append(client_log)
                client_states.append(self.method.get_state(local_model))
                tqdm.write(
                    f"[round {round_id}/{self.cfg.federated.rounds}] "
                    f"client {cid} finished: loss={client_log['loss']:.4f}, steps={client_log['steps']}"
                )
                if reusable_local_model is None:
                    del local_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if reusable_local_model is not None:
                del reusable_local_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            train_time = time.time() - train_start

            agg_start = time.time()
            self.global_state = self.method.aggregate(client_states, self.global_state)
            agg_time = time.time() - agg_start
            self.method.set_state(self.global_model, self.global_state)

            eval_metrics = {}
            should_eval = (
                round_id % self.cfg.eval.every_round == 0
                or round_id == self.cfg.federated.rounds
            )
            if should_eval:
                eval_metrics = self.evaluator(
                    self.global_model,
                    self.eval_dataset,
                    self.tokenizer,
                    self.device,
                    self.cfg,
                )

            comm = self.method.communication(self.global_state, len(selected))
            eval_log = {f"eval_{k}": v for k, v in eval_metrics.items()}
            round_log = {
                "round": round_id,
                **eval_log,
                "method": self.method.name,
                "model": self.cfg.model.name_or_path,
                "task": self.cfg.task.name,
                "seed": self.cfg.seed,
                "run_id": self.run_id,
                "selected_clients": selected,
                "selected_client_label_stats": [
                    stat for stat in client_label_stats(self.train_labels, self.client_indices)
                    if stat["client_id"] in selected
                ],
                "client_loss": sum(x["loss"] for x in client_logs) / max(1, len(client_logs)),
                "upload_params": comm["upload"],
                "download_params": comm["download"],
                "total_comm_params": comm["total"],
                "server_agg_time_sec": agg_time,
                "train_time_sec": train_time,
            }
            logs.append(round_log)
            append_jsonl(self.log_path, round_log, blank_line=True)
            self.save_checkpoint(round_id, eval_metrics, comm)
            round_iter.set_postfix(loss=f"{round_log['client_loss']:.4f}", **{k: f"{v:.4f}" for k, v in round_log.items() if k.startswith("eval_") and isinstance(v, float)})
            metric_text = ", ".join(
                f"{k}={v:.4f}" for k, v in round_log.items() if k.startswith("eval_") and isinstance(v, float)
            )
            if not metric_text:
                metric_text = "no eval this round"
            tqdm.write(
                f"[round {round_id}/{self.cfg.federated.rounds}] finished: "
                f"client_loss={round_log['client_loss']:.4f}, {metric_text}, "
                f"comm={round_log['total_comm_params']}"
            )
        return logs

from __future__ import annotations

import random
import time
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.arguments import namespace_to_dict
from src.datasets.partition import client_label_stats
from src.eval.glue_submission import write_glue_test_predictions
from src.federated.client import ClientTrainer
from src.logging_utils import append_jsonl, ensure_dir, write_json
from src.methods.idea_rank_allocator import rank_distribution


class IdeaFederatedTrainer:
    """
    Standalone trainer for the idea_update_space prototype.

    This file is intentionally separate from src/federated/trainer.py so the
    original FLoRG training path remains untouched.
    """

    def __init__(
        self,
        global_model_fn,
        client_model_fn,
        method,
        tokenizer,
        train_dataset,
        eval_dataset,
        client_indices,
        cfg,
        device,
        evaluator,
        resume_checkpoint=None,
        submission_dataset=None,
    ):
        self.global_model_fn = global_model_fn
        self.client_model_fn = client_model_fn
        self.method = method
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.client_indices = client_indices
        self.train_labels = [int(row["labels"]) for row in train_dataset]
        self.cfg = cfg
        self.device = device
        self.evaluator = evaluator
        self.submission_dataset = submission_dataset
        self.best_submission_accuracy = None
        self.best_submission_round = None

        self.global_model = global_model_fn().to(device)
        self.global_state = method.init_global_state(self.global_model)
        self.start_round = 1
        if resume_checkpoint is not None:
            checkpoint = torch.load(resume_checkpoint, map_location="cpu")
            self.global_state = checkpoint["state"]
            self.start_round = int(checkpoint["round"]) + 1
        self.method.bind_shape_state_for_comm(self.global_state)
        self.method.set_global_state(self.global_model, self.global_state)

        self.output_dir = ensure_dir(cfg.output_dir)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(self.output_dir) / "logs" / f"{self.run_id}.log"

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
            range(self.start_round, self.cfg.federated.rounds + 1),
            desc="idea federated rounds",
            dynamic_ncols=True,
        )
        for round_id in round_iter:
            selected = self.select_clients(round_id)
            client_states = []
            client_logs = []

            train_start = time.time()
            for cid in tqdm(selected, desc=f"round {round_id} clients", leave=False, dynamic_ncols=True):
                local_model = self.client_model_fn(cid).to(self.device)
                self.method.set_client_state(local_model, self.global_state, client_id=cid)
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
                client_states.append(
                    self.method.get_client_state(
                        local_model,
                        client_id=cid,
                        num_samples=len(self.client_indices[cid]),
                    )
                )
                tqdm.write(
                    f"[round {round_id}/{self.cfg.federated.rounds}] "
                    f"client {cid} finished: loss={client_log['loss']:.4f}, steps={client_log['steps']}"
                )
                del local_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            train_time = time.time() - train_start

            agg_start = time.time()
            self.global_state = self.method.aggregate(client_states, self.global_state)
            self.method.bind_shape_state_for_comm(self.global_state)
            agg_time = time.time() - agg_start
            self.method.set_global_state(self.global_model, self.global_state)

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
                eval_accuracy = eval_metrics.get("accuracy")
                if (
                    self.submission_dataset is not None
                    and eval_accuracy is not None
                    and (
                        self.best_submission_accuracy is None
                        or float(eval_accuracy) > self.best_submission_accuracy
                    )
                ):
                    self.best_submission_accuracy = float(eval_accuracy)
                    self.best_submission_round = int(round_id)
                    written = write_glue_test_predictions(
                        model=self.global_model,
                        encoded_dataset=self.submission_dataset,
                        tokenizer=self.tokenizer,
                        device=self.device,
                        cfg=self.cfg,
                        output_root="glue_submit",
                    )
                    write_json(
                        Path("glue_submit") / Path(self.cfg.output_dir).name / "best_submission.json",
                        {
                            "round": self.best_submission_round,
                            "eval_accuracy": self.best_submission_accuracy,
                            "files": [str(path) for path in written],
                        },
                    )

            comm = self.method.communication(selected)
            diagnostics = getattr(self.method, "last_diagnostics", {})
            trunc_errors = [
                v["trunc_error"] for v in diagnostics.values() if v.get("trunc_error") is not None
            ]
            svd_time = sum(float(v.get("server_svd_time_sec", 0.0)) for v in diagnostics.values())
            proc_time = sum(float(v.get("procrustes_time_sec", 0.0)) for v in diagnostics.values())

            eval_log = {f"eval_{k}": v for k, v in eval_metrics.items()}
            all_client_label_stats = client_label_stats(self.train_labels, self.client_indices)
            selected_client_label_stats = [
                stat for stat in all_client_label_stats if stat["client_id"] in selected
            ]
            round_log = {
                "round": round_id,
                **eval_log,
                "method": self.method.name,
                "model": self.cfg.model.name_or_path,
                "task": self.cfg.task.name,
                "seed": self.cfg.seed,
                "run_id": self.run_id,
                "selected_clients": selected,
                "selected_client_label_stats": selected_client_label_stats,
                "rank_distribution": rank_distribution(self.method.rank_map, selected),
                "client_loss": sum(x["loss"] for x in client_logs) / max(1, len(client_logs)),
                "client_loss_std": float(torch.tensor([x["loss"] for x in client_logs]).std(unbiased=False).item()),
                "upload_params": comm["upload"],
                "download_params": comm["download"],
                "total_comm_params": comm["total"],
                "server_agg_time_sec": agg_time,
                "server_svd_time_sec": svd_time,
                "procrustes_time_sec": proc_time,
                "trunc_error_mean": sum(trunc_errors) / len(trunc_errors) if trunc_errors else None,
                "trunc_error_max": max(trunc_errors) if trunc_errors else None,
                "train_time_sec": train_time,
            }
            logs.append(round_log)
            append_jsonl(self.log_path, round_log, blank_line=True)
            self.save_checkpoint(round_id, eval_metrics, comm)

            round_iter.set_postfix(
                loss=f"{round_log['client_loss']:.4f}",
                **{k: f"{v:.4f}" for k, v in round_log.items() if k.startswith("eval_") and isinstance(v, float)},
            )
            metric_text = ", ".join(
                f"{k}={v:.4f}" for k, v in round_log.items() if k.startswith("eval_") and isinstance(v, float)
            ) or "no eval this round"
            tqdm.write(
                f"[round {round_id}/{self.cfg.federated.rounds}] finished: "
                f"client_loss={round_log['client_loss']:.4f}, {metric_text}, "
                f"comm={round_log['total_comm_params']}, trunc_mean={round_log['trunc_error_mean']}"
            )
        return logs

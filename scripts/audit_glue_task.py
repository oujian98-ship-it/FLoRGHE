from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arguments import load_config
from src.datasets.partition import (
    client_label_stats,
    dirichlet_partition,
    iid_partition,
)


LABEL_MAPS = {
    "rte": {"entailment": 0, "not_entailment": 1},
    "qnli": {"entailment": 0, "not_entailment": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction": 2},
}


def read_split(task_dir: Path, split_name: str):
    path = task_dir / split_name
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t", quoting=3)


def label_counts(df):
    if df is None:
        return {}
    label_col = "gold_label" if "gold_label" in df.columns else "label" if "label" in df.columns else None
    if label_col is None:
        return {}
    return {str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).sort_index().items()}


def majority_baseline(df):
    counts = label_counts(df)
    total = sum(counts.values())
    if total == 0:
        return None
    return max(counts.values()) / total


def pair_overlap(train, dev):
    keys = [c for c in ("sentence1", "sentence2") if c in train.columns and c in dev.columns]
    if len(keys) != 2:
        return None
    train_pairs = set(map(tuple, train[keys].astype(str).values.tolist()))
    dev_pairs = set(map(tuple, dev[keys].astype(str).values.tolist()))
    rev_train_pairs = {(b, a) for a, b in train_pairs}
    return {
        "exact": len(train_pairs & dev_pairs),
        "reversed": len(rev_train_pairs & dev_pairs),
        "dev_pairs": len(dev_pairs),
    }


def sentence_overlap(train, dev):
    keys = [c for c in ("sentence1", "sentence2") if c in train.columns and c in dev.columns]
    if len(keys) != 2:
        return None
    train_sentences = set(train[keys[0]].astype(str)) | set(train[keys[1]].astype(str))
    dev_sentences = set(dev[keys[0]].astype(str)) | set(dev[keys[1]].astype(str))
    return {
        "overlap": len(train_sentences & dev_sentences),
        "dev_unique_sentences": len(dev_sentences),
    }


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    task = cfg.task.name.lower()
    task_dir = Path(cfg.data.root) / task.upper()
    train = read_split(task_dir, "train.tsv")
    dev_name = "dev_matched.tsv" if task == "mnli" else "dev.tsv"
    dev = read_split(task_dir, dev_name)

    if train is None or dev is None:
        raise SystemExit(f"Missing train/dev split under {task_dir}")

    train_label_col = "gold_label" if "gold_label" in train.columns else "label"
    train_labels = train[train_label_col].to_numpy()
    if train_labels.dtype.kind not in {"i", "u"}:
        label_map = LABEL_MAPS.get(task)
        if label_map is None:
            unique = {v: i for i, v in enumerate(sorted(pd.unique(train_labels)))}
            label_map = unique
        train_labels = np.asarray([label_map[v] for v in train_labels])

    partition = build_partition(train_labels, cfg)
    stats = client_label_stats(train_labels, partition)
    sizes = [x["num_samples"] for x in stats]
    single_class = sum(1 for x in stats if len(x["label_counts"]) < 2)

    report = {
        "config": args.config,
        "task": task,
        "task_dir": str(task_dir),
        "train_rows": int(len(train)),
        "dev_rows": int(len(dev)),
        "train_label_counts": label_counts(train),
        "dev_label_counts": label_counts(dev),
        "dev_majority_baseline": majority_baseline(dev),
        "pair_overlap": pair_overlap(train, dev),
        "sentence_overlap": sentence_overlap(train, dev),
        "num_clients": int(cfg.federated.num_clients),
        "partition_size_min": int(min(sizes)),
        "partition_size_max": int(max(sizes)),
        "partition_size_mean": float(np.mean(sizes)),
        "single_class_clients": int(single_class),
        "client_stats": stats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

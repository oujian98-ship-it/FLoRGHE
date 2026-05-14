from __future__ import annotations

from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset


GLUE_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

LOCAL_SPLITS = {
    "rte": {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"},
    "wnli": {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"},
    "qnli": {"train": "train.tsv", "validation": "dev.tsv", "test": "test.tsv"},
    "mnli": {
        "train": "train.tsv",
        "validation_matched": "dev_matched.tsv",
        "validation_mismatched": "dev_mismatched.tsv",
        "test_matched": "test_matched.tsv",
        "test_mismatched": "test_mismatched.tsv",
    },
}

LABEL_MAPS = {
    "rte": {"entailment": 0, "not_entailment": 1},
    "qnli": {"entailment": 0, "not_entailment": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction": 2},
}


def _task_dir(task_name, data_root):
    if data_root is None:
        candidates = [Path("D:/data") / task_name.upper(), Path("data") / task_name.upper()]
    else:
        root = Path(data_root)
        candidates = [root / task_name.upper(), root / task_name.lower(), root / task_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _read_local_split(path, task_name):
    df = pd.read_csv(path, sep="\t", quoting=3)
    task_name = task_name.lower()
    if "gold_label" in df.columns:
        df = df.rename(columns={"gold_label": "label"})
    if "label" in df.columns:
        if task_name in LABEL_MAPS and df["label"].dtype == object:
            df["labels"] = df["label"].map(LABEL_MAPS[task_name])
        else:
            df["labels"] = pd.to_numeric(df["label"], errors="coerce")
        df = df.dropna(subset=["labels"])
        df["labels"] = df["labels"].astype("int64")
    return Dataset.from_pandas(df, preserve_index=False)


def load_local_glue_dataset(task_name, tokenizer, max_seq_length, data_root=None):
    task_name = task_name.lower()
    task_dir = _task_dir(task_name, data_root)
    if task_dir is None or task_name not in LOCAL_SPLITS:
        return None

    splits = {}
    for split_name, file_name in LOCAL_SPLITS[task_name].items():
        path = task_dir / file_name
        if path.exists():
            splits[split_name] = _read_local_split(path, task_name)
    if "train" not in splits:
        return None

    raw = DatasetDict(splits)
    return _tokenize_glue_dataset(raw, task_name, tokenizer, max_seq_length)


def _tokenize_glue_dataset(raw, task_name, tokenizer, max_seq_length):
    task_name = task_name.lower()
    task_name = task_name.lower()
    key1, key2 = GLUE_KEYS[task_name]

    def preprocess(examples):
        args = (examples[key1],) if key2 is None else (examples[key1], examples[key2])
        return tokenizer(
            *args,
            truncation=True,
            padding=False,
            max_length=max_seq_length,
        )

    encoded = raw.map(preprocess, batched=True)
    if "label" in encoded["train"].column_names and "labels" not in encoded["train"].column_names:
        encoded = encoded.rename_column("label", "labels")
    keep = {"input_ids", "attention_mask", "token_type_ids", "labels"}
    for split_name in list(encoded.keys()):
        drop = [c for c in encoded[split_name].column_names if c not in keep]
        if drop:
            encoded[split_name] = encoded[split_name].remove_columns(drop)
    encoded.set_format(type="torch")
    return encoded


def load_glue_dataset(task_name, tokenizer, max_seq_length, data_root=None):
    local = load_local_glue_dataset(task_name, tokenizer, max_seq_length, data_root)
    if local is not None:
        return local
    raw = load_dataset("glue", task_name.lower())
    return _tokenize_glue_dataset(raw, task_name, tokenizer, max_seq_length)

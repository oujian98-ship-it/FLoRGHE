from __future__ import annotations

from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def namespace_to_dict(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [namespace_to_dict(v) for v in value]
    return value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> SimpleNamespace:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _merge_dict(data, overrides)
    return _to_namespace(data)


def parse_run_args():
    parser = ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()

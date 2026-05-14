from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--out", default="outputs/results/all_rounds.csv")
    args = parser.parse_args()

    frames = []
    log_paths = list(Path(args.root).glob("**/*.log")) + list(Path(args.root).glob("**/rounds.jsonl"))
    log_paths = [p for p in log_paths if p.parent.name == "logs"]
    for path in log_paths:
        frames.append(pd.read_json(path, lines=True))
    if not frames:
        raise SystemExit(f"No rounds.log or rounds.jsonl files found under {args.root}")
    df = pd.concat(frames, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

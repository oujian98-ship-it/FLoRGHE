import pandas as pd


def summarize_accuracy(path):
    df = pd.read_json(path, lines=True)
    last = df.sort_values("round").groupby(["model", "task", "method", "seed"]).tail(1)
    metric_cols = [c for c in last.columns if c.startswith("eval_")]
    rows = []
    for metric in metric_cols:
        summary = (
            last.groupby(["model", "task", "method"])[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        summary["metric"] = metric
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

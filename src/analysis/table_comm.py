import pandas as pd


def summarize_comm_to_target(path, metric="eval_accuracy", target=0.8):
    df = pd.read_json(path, lines=True)
    df = df.sort_values("round")
    df["cumulative_comm_params"] = df.groupby(["model", "task", "method", "seed"])["total_comm_params"].cumsum()
    hits = df[df[metric] >= target]
    if hits.empty:
        return pd.DataFrame(columns=["model", "task", "method", "seed", "round", "cumulative_comm_params"])
    return hits.groupby(["model", "task", "method", "seed"]).head(1)[
        ["model", "task", "method", "seed", "round", "cumulative_comm_params"]
    ]

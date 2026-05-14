import pandas as pd


def load_curve_frame(path):
    df = pd.read_json(path, lines=True)
    df["cumulative_comm_params"] = df.groupby(["model", "task", "method", "seed"])["total_comm_params"].cumsum()
    return df

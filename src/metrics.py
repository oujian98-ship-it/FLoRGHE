def primary_metric(task_name):
    task_name = task_name.lower()
    if task_name in {"mrpc", "qqp"}:
        return "f1"
    if task_name == "cola":
        return "matthews_correlation"
    return "accuracy"

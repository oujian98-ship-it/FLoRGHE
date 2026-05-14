def cumulative_comm(logs):
    total = 0
    out = []
    for row in logs:
        total += int(row.get("total_comm_params", 0))
        row = dict(row)
        row["cumulative_comm_params"] = total
        out.append(row)
    return out

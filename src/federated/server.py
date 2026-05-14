from dataclasses import dataclass


@dataclass
class RoundResult:
    round: int
    selected_clients: list[int]
    metrics: dict
    comm: dict

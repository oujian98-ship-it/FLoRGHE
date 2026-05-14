from __future__ import annotations

import torch
import torch.nn as nn


class HeteroLoRALinear(nn.Module):
    """
    Standard LoRA layer with a fixed per-layer rank.

    This module intentionally does not resize parameters dynamically. For
    heterogeneous-rank experiments, build each client model with its own rank map.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        layer_name: str = "",
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("HeteroLoRALinear only wraps nn.Linear")
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        out_features, in_features = base_linear.weight.shape
        self.out_features = out_features
        self.in_features = in_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0
        self.layer_name = layer_name
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if self.rank > 0:
            self.lora_B = nn.Parameter(
                torch.zeros(
                    out_features,
                    self.rank,
                    device=base_linear.weight.device,
                    dtype=base_linear.weight.dtype,
                )
            )
            self.lora_A = nn.Parameter(
                torch.randn(
                    self.rank,
                    in_features,
                    device=base_linear.weight.device,
                    dtype=base_linear.weight.dtype,
                )
                * init_scale
            )
        else:
            self.register_parameter("lora_B", None)
            self.register_parameter("lora_A", None)

    def forward(self, x):
        y = self.base(x)
        if self.rank == 0:
            return y
        z = self.dropout(x)
        z = z @ self.lora_A.T
        z = z @ self.lora_B.T
        return y + self.scaling * z

    @torch.no_grad()
    def effective_delta(self):
        if self.rank == 0:
            return None
        return self.scaling * (self.lora_B @ self.lora_A)

    @torch.no_grad()
    def get_raw_state(self):
        if self.rank == 0:
            return {"B": None, "A": None, "rank": 0, "alpha": self.alpha}
        return {
            "B": self.lora_B.detach().cpu().clone(),
            "A": self.lora_A.detach().cpu().clone(),
            "rank": self.rank,
            "alpha": self.alpha,
        }

    @torch.no_grad()
    def set_raw_state(self, B, A):
        if self.rank == 0:
            return
        if tuple(B.shape) != tuple(self.lora_B.shape):
            raise ValueError(f"B shape mismatch: got {tuple(B.shape)}, expected {tuple(self.lora_B.shape)}")
        if tuple(A.shape) != tuple(self.lora_A.shape):
            raise ValueError(f"A shape mismatch: got {tuple(A.shape)}, expected {tuple(self.lora_A.shape)}")
        self.lora_B.copy_(B.to(device=self.lora_B.device, dtype=self.lora_B.dtype))
        self.lora_A.copy_(A.to(device=self.lora_A.device, dtype=self.lora_A.dtype))

    @torch.no_grad()
    def set_from_effective_prefix(self, B_eff, A_eff):
        if self.rank == 0:
            return
        raw_scale = (self.rank / self.alpha) ** 0.5
        B_raw = raw_scale * B_eff[:, : self.rank]
        A_raw = raw_scale * A_eff[: self.rank, :]
        self.set_raw_state(B_raw, A_raw)

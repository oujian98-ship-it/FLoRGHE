import torch
import torch.nn as nn


def semi_orthogonal_L(out_features, k, device=None, dtype=None):
    x = torch.randn(out_features, k, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q


def semi_orthogonal_R(k, in_features, device=None, dtype=None):
    x = torch.randn(in_features, k, device=device, dtype=dtype)
    q, _ = torch.linalg.qr(x, mode="reduced")
    return q.T


class FLoRGLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.0,
        init_scale: float = 0.01,
        layer_name: str = "",
    ):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("FLoRGLinear only wraps nn.Linear")

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.out_features, self.in_features = self.base.weight.shape
        self.k = min(self.out_features, self.in_features)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.layer_name = layer_name
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.register_buffer("L", semi_orthogonal_L(self.out_features, self.k, self.base.weight.device, self.base.weight.dtype))
        self.register_buffer("R", semi_orthogonal_R(self.k, self.in_features, self.base.weight.device, self.base.weight.dtype))
        self.A = nn.Parameter(
            torch.randn(rank, self.k, device=self.base.weight.device, dtype=self.base.weight.dtype) * init_scale
        )

    def adapter_weight(self):
        q = self.A.transpose(0, 1) @ self.A
        return self.L @ q @ self.R

    def forward(self, x):
        y = self.base(x)
        q = self.A.transpose(0, 1) @ self.A
        z = self.dropout(x) @ self.R.transpose(0, 1)
        z = z @ q
        z = z @ self.L.transpose(0, 1)
        return y + self.scaling * z

    @torch.no_grad()
    def get_A(self):
        return self.A.detach().clone()

    @torch.no_grad()
    def set_A(self, new_A):
        if tuple(new_A.shape) != tuple(self.A.shape):
            raise ValueError(f"A shape mismatch: got {tuple(new_A.shape)}, expected {tuple(self.A.shape)}")
        self.A.copy_(new_A.to(device=self.A.device, dtype=self.A.dtype))

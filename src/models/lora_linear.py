import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank=4, alpha=16.0, dropout=0.0, layer_name=""):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear only wraps nn.Linear")
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        out_features, in_features = base_linear.weight.shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.layer_name = layer_name
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=base_linear.weight.device, dtype=base_linear.weight.dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=base_linear.weight.device, dtype=base_linear.weight.dtype))

    def forward(self, x):
        y = self.base(x)
        z = self.dropout(x) @ self.lora_A.T
        z = z @ self.lora_B.T
        return y + self.scaling * z

    def delta_weight(self):
        return self.lora_B @ self.lora_A

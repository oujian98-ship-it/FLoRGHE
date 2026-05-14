from __future__ import annotations

import torch

from src.methods.fedit import FedITMethod


@torch.no_grad()
def aggregate_federa_one_layer(client_pairs, rank, eps=1e-8):
    d = sum(B.float() @ A.float() for B, A in client_pairs) / len(client_pairs)
    u, s, vh = torch.linalg.svd(d, full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank].clamp_min(eps)
    vh_r = vh[:rank, :]
    b_new = u_r @ torch.diag(torch.sqrt(s_r))
    a_new = torch.diag(torch.sqrt(s_r)) @ vh_r
    return b_new, a_new


class FeDeRAMethod(FedITMethod):
    name = "federa"

    def aggregate(self, client_states, prev_state):
        out = {}
        rank = self.cfg.adapter.rank
        for layer in [k for k in prev_state if k != "__classifier_head__"]:
            pairs = [(cs[layer]["lora_B"], cs[layer]["lora_A"]) for cs in client_states]
            b_new, a_new = aggregate_federa_one_layer(pairs, rank)
            out[layer] = {
                "lora_A": a_new.to(prev_state[layer]["lora_A"].dtype).cpu(),
                "lora_B": b_new.to(prev_state[layer]["lora_B"].dtype).cpu(),
            }
        if self.cfg.model.train_classifier_head:
            from src.models.model_utils import average_classifier_states

            out["__classifier_head__"] = average_classifier_states([cs["__classifier_head__"] for cs in client_states])
        return out

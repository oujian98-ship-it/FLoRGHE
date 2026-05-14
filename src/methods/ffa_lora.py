from __future__ import annotations

from src.methods.fedit import FedITMethod
from src.models.model_utils import freeze_non_adapter_params


class FFALoRAMethod(FedITMethod):
    name = "ffa_lora"

    def freeze(self, model):
        freeze_non_adapter_params(model, self.cfg.model.train_classifier_head, ffa_lora=True)

    def aggregate(self, client_states, prev_state):
        out = {}
        for layer in [k for k in prev_state if k != "__classifier_head__"]:
            out[layer] = {
                "lora_A": prev_state[layer]["lora_A"],
                "lora_B": sum(cs[layer]["lora_B"] for cs in client_states) / len(client_states),
            }
        if self.cfg.model.train_classifier_head:
            from src.models.model_utils import average_classifier_states

            out["__classifier_head__"] = average_classifier_states([cs["__classifier_head__"] for cs in client_states])
        return out

from src.models.florg_linear import FLoRGLinear
from src.models.lora_linear import LoRALinear


def freeze_non_adapter_params(model, train_classifier_head=False, ffa_lora=False):
    for _, p in model.named_parameters():
        p.requires_grad = False

    for _, module in model.named_modules():
        if isinstance(module, FLoRGLinear):
            module.A.requires_grad = True
        elif isinstance(module, LoRALinear):
            module.lora_A.requires_grad = not ffa_lora
            module.lora_B.requires_grad = True

    if train_classifier_head:
        for name, p in model.named_parameters():
            if "classifier" in name or "score" in name:
                p.requires_grad = True


def trainable_parameter_names(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


def is_classifier_param(name):
    return "classifier" in name or "score" in name


def get_classifier_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if is_classifier_param(name)}


def set_classifier_state(model, state):
    if not state:
        return
    named = dict(model.named_parameters())
    for name, value in state.items():
        named[name].data.copy_(value.to(named[name].device, named[name].dtype))


def average_classifier_states(client_states):
    first = client_states[0]
    return {name: sum(cs[name] for cs in client_states) / len(client_states) for name in first}


def tensor_tree_numel(state):
    if isinstance(state, dict):
        return sum(tensor_tree_numel(v) for v in state.values())
    return state.numel()

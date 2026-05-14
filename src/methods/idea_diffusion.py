from __future__ import annotations


DEFAULT_CROSS_ATTENTION_PATTERNS = [
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out",
    "cross_attn",
    "cross_attention",
]

DEFAULT_SELF_ATTENTION_PATTERNS = [
    "attn1.to_q",
    "attn1.to_v",
    "self_attn",
    "self_attention",
]

DEFAULT_FFN_PATTERNS = [
    "proj_in",
    "proj_out",
    "ff",
    "feed_forward",
]


def is_semantic_diffusion_layer(layer_name, patterns=None):
    patterns = patterns or DEFAULT_CROSS_ATTENTION_PATTERNS
    return any(pattern in layer_name for pattern in patterns)


def diffusion_semantic_gate(layer_name, rank, r_min_semantic=4, patterns=None):
    if is_semantic_diffusion_layer(layer_name, patterns) and rank < r_min_semantic:
        return 0
    return rank


def default_diffusion_target_modules(include_ffn=True):
    targets = ["to_q", "to_k", "to_v", "to_out.0", "to_out"]
    if include_ffn:
        targets.extend(["proj_in", "proj_out"])
    return targets

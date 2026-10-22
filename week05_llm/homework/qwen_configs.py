"""Given config for the Qwen3.5 hybrid architecture -- not a blank."""

import dataclasses


@dataclasses.dataclass
class Qwen35Config:
    vocab_size: int = 300
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 8

    # Required full attention uses one key/value head per query head (MHA).
    # Fewer key/value heads select the optional GQA extension.
    num_attention_heads: int = 4
    num_key_value_heads: int = 4
    head_dim: int = 32
    attention_bias: bool = False

    # linear attention (Gated DeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 16
    linear_value_head_dim: int = 16
    linear_num_key_heads: int = 4
    linear_num_value_heads: int = 4

    # rotary embedding (partial RoPE)
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    max_position_embeddings: int = 512

    # norms / misc
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True

    # every `full_attention_interval`-th layer is full attention, the rest are linear attention
    full_attention_interval: int = 4
    layer_types: list = None

    # bonus: multi-token prediction head
    mtp_num_layers: int = 1

    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = [
                "full_attention" if (i + 1) % self.full_attention_interval == 0 else "linear_attention"
                for i in range(self.num_hidden_layers)
            ]


def make_small_config(**overrides):
    """Small CPU-friendly config; regular attention unless KV heads are explicit."""
    config = Qwen35Config()
    for key, value in overrides.items():
        setattr(config, key, value)
    if "num_key_value_heads" not in overrides:
        config.num_key_value_heads = config.num_attention_heads
    if "layer_types" not in overrides:
        config.layer_types = [
            "full_attention" if (i + 1) % config.full_attention_interval == 0 else "linear_attention"
            for i in range(config.num_hidden_layers)
        ]
    return config

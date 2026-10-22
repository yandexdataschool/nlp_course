"""
Public correctness tests for qwen.py.

Each test builds a same-shaped real `transformers` Qwen3.5 module, copies a
student module's (randomized) weights into it, and checks the outputs match
-- the same "steal the weights" trick seminar.ipynb uses to check its
from-scratch GPT-2 against a real one. Tests are grouped into classes, one
per architectural block; besides the reference match, each class also
asserts the basic shape/dtype/finiteness contract for that block -- this is
the place to look for exact shapes if the notebook markdown doesn't spell
them out.

Run all of these: `pytest tests/test_qwen_public.py -v`
Run just one block: `pytest tests/test_qwen_public.py -k rope -v`
"""

import pytest
import torch

from transformers.models.qwen3_5 import modeling_qwen3_5 as ref

import qwen
from conftest import randomize, randomize_gdn, assert_close, real_text_config


pytestmark = [pytest.mark.main, pytest.mark.public]


# ---------------------------------------------------------------------------
# RMSNorm / RMSNormGated
# ---------------------------------------------------------------------------

class TestRMSNorm:
    def test_rmsnorm_matches_reference(self, small_config, device):
        mine = randomize(qwen.RMSNorm(small_config.hidden_size, eps=small_config.rms_norm_eps)).to(device)
        real = ref.Qwen3_5RMSNorm(small_config.hidden_size, eps=small_config.rms_norm_eps).to(device)
        real.load_state_dict(mine.state_dict())

        x = torch.randn(2, 9, small_config.hidden_size, device=device)
        out_mine = mine(x)

        assert out_mine.shape == x.shape
        assert out_mine.dtype == x.dtype
        assert torch.isfinite(out_mine).all()
        assert_close(out_mine, real(x), msg="RMSNorm")

    def test_rmsnorm_zero_weight_is_identity_normalization(self, device):
        """Sanity check on the *specific* Qwen RMSNorm convention: with weight==0
        (its init value), output should just be the plain normalized input."""
        mine = qwen.RMSNorm(8, eps=1e-6).to(device)
        x = torch.randn(3, 8, device=device) * 5
        out = mine(x)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert_close(out, expected, msg="RMSNorm@weight=0")


class TestRMSNormGated:
    def test_rmsnorm_gated_matches_reference(self, small_config, device):
        dim = small_config.linear_value_head_dim
        mine = randomize(qwen.RMSNormGated(dim, eps=small_config.rms_norm_eps)).to(device)
        real = ref.Qwen3_5RMSNormGated(dim, eps=small_config.rms_norm_eps).to(device)
        real.load_state_dict(mine.state_dict())

        x = torch.randn(2, 9, dim, device=device)
        gate = torch.randn(2, 9, dim, device=device)
        out_mine = mine(x, gate)

        assert out_mine.shape == x.shape
        assert torch.isfinite(out_mine).all()
        assert_close(out_mine, real(x, gate), msg="RMSNormGated")


# ---------------------------------------------------------------------------
# Rotary embedding
# ---------------------------------------------------------------------------

class TestRotaryEmbedding:
    def test_rotate_half_matches_reference(self, device):
        x = torch.randn(2, 3, 5, 8, device=device)
        out = qwen.rotate_half(x)
        assert out.shape == x.shape
        assert_close(out, ref.rotate_half(x), msg="rotate_half")

    def test_rope_cos_sin_matches_reference(self, small_config, real_config, device):
        mine_rope = qwen.Qwen35RotaryEmbedding(small_config).to(device)
        real_rope = ref.Qwen3_5TextRotaryEmbedding(real_config).to(device)

        batch, seq_len = 2, 13
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        dummy_hidden = torch.randn(batch, seq_len, small_config.hidden_size, device=device)
        mcos, msin = mine_rope(dummy_hidden, position_ids)
        rcos, rsin = real_rope(dummy_hidden, position_ids)

        rotary_dim = int(small_config.head_dim * small_config.partial_rotary_factor)
        assert mcos.shape == (batch, seq_len, rotary_dim)
        assert msin.shape == (batch, seq_len, rotary_dim)
        assert_close(mcos, rcos, msg="rope cos")
        assert_close(msin, rsin, msg="rope sin")

    def test_rope_cos_sin_cast_to_hidden_states_dtype(self, small_config, real_config, device):
        """A model running in bf16/fp16 needs cos/sin cast to match, or q/k
        silently upcast to float32 and later matmuls against bf16 tensors crash."""
        mine_rope = qwen.Qwen35RotaryEmbedding(small_config).to(device)
        real_rope = ref.Qwen3_5TextRotaryEmbedding(real_config).to(device)

        batch, seq_len = 2, 7
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        dummy_hidden = torch.randn(batch, seq_len, small_config.hidden_size, dtype=torch.bfloat16, device=device)
        mcos, msin = mine_rope(dummy_hidden, position_ids)
        rcos, rsin = real_rope(dummy_hidden, position_ids)

        assert mcos.dtype == torch.bfloat16, f"cos should be cast to hidden_states.dtype, got {mcos.dtype}"
        assert msin.dtype == torch.bfloat16, f"sin should be cast to hidden_states.dtype, got {msin.dtype}"
        assert_close(mcos.float(), rcos.float(), msg="rope cos (bf16)")
        assert_close(msin.float(), rsin.float(), msg="rope sin (bf16)")

    def test_apply_rotary_pos_emb_matches_reference(self, small_config, device):
        rotary_dim = int(small_config.head_dim * small_config.partial_rotary_factor)
        batch, heads, seq_len = 2, 4, 11
        q = torch.randn(batch, heads, seq_len, small_config.head_dim, device=device)
        k = torch.randn(batch, heads, seq_len, small_config.head_dim, device=device)
        cos = torch.randn(batch, seq_len, rotary_dim, device=device)
        sin = torch.randn(batch, seq_len, rotary_dim, device=device)

        q_mine, k_mine = qwen.apply_rotary_pos_emb(q, k, cos, sin)
        q_real, k_real = ref.apply_rotary_pos_emb(q, k, cos, sin)

        assert q_mine.shape == q.shape
        assert k_mine.shape == k.shape
        assert_close(q_mine, q_real, msg="apply_rotary_pos_emb (q)")
        assert_close(k_mine, k_real, msg="apply_rotary_pos_emb (k)")

        # only the first `rotary_dim` channels should have changed
        assert torch.equal(q_mine[..., rotary_dim:], q[..., rotary_dim:]), "pass-through channels must be untouched"


# ---------------------------------------------------------------------------
# Full attention (GatedAttention)
# ---------------------------------------------------------------------------

class TestGatedAttention:
    def test_required_attention_does_not_need_gqa(self, small_config, monkeypatch, device):
        """Equal Q/K/V head counts are sufficient for every required task."""
        assert small_config.num_attention_heads == small_config.num_key_value_heads

        def unfinished_bonus(*args, **kwargs):
            raise NotImplementedError("GQA is an optional bonus")

        monkeypatch.setattr(qwen.GatedAttention, "forward_gqa", unfinished_bonus)
        mine = randomize(qwen.GatedAttention(small_config, 3)).to(device)
        hidden = torch.randn(2, 7, small_config.hidden_size, device=device)
        positions = qwen.Qwen35RotaryEmbedding(small_config).to(device)(
            hidden, torch.arange(7, device=device)[None].expand(2, -1),
        )
        with torch.no_grad():
            actual = mine(hidden, positions)
            expected = qwen.naive_looped_gated_attention(mine, hidden, positions)
        assert_close(actual, expected, atol=1e-4, msg="required regular multi-head attention")

    def test_gated_attention_matches_reference(self, small_config, real_config, device):
        layer_idx = small_config.layer_types.index("full_attention")
        mine = randomize(qwen.GatedAttention(small_config, layer_idx)).to(device)
        real = ref.Qwen3_5Attention(real_config, layer_idx).to(device)
        real.load_state_dict(mine.state_dict())
        mine.eval()
        real.eval()

        batch, seq_len = 2, 15
        hidden_states = torch.randn(batch, seq_len, small_config.hidden_size, device=device)
        mask = qwen.build_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
        position_embeddings = qwen.Qwen35RotaryEmbedding(small_config).to(device)(
            hidden_states, torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        )

        with torch.no_grad():
            out_mine = mine(hidden_states, position_embeddings, mask)
            out_real = real(hidden_states, position_embeddings=position_embeddings, attention_mask=mask)
            if isinstance(out_real, tuple):
                out_real = out_real[0]

        assert out_mine.shape == hidden_states.shape
        assert torch.isfinite(out_mine).all()
        assert_close(out_mine, out_real, atol=1e-3, msg="GatedAttention vs real transformers")

    def test_gated_attention_matches_naive_loop(self, small_config, device):
        """Your vectorized GatedAttention.forward must agree with the given,
        slow, per-head for-loop reference `naive_looped_gated_attention`."""
        layer_idx = small_config.layer_types.index("full_attention")
        mine = randomize(qwen.GatedAttention(small_config, layer_idx)).to(device)
        mine.eval()

        batch, seq_len = 2, 12
        hidden_states = torch.randn(batch, seq_len, small_config.hidden_size, device=device)
        mask = qwen.build_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
        position_embeddings = qwen.Qwen35RotaryEmbedding(small_config).to(device)(
            hidden_states, torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        )

        with torch.no_grad():
            out_fast = mine(hidden_states, position_embeddings, mask)
            out_naive = qwen.naive_looped_gated_attention(mine, hidden_states, position_embeddings, mask)

        assert out_fast.shape == hidden_states.shape
        assert_close(out_fast, out_naive, atol=1e-4, msg="GatedAttention vs naive_looped_gated_attention")

    def test_gated_attention_is_causal(self, small_config, device):
        """Changing a future token must not change an earlier position's output."""
        layer_idx = small_config.layer_types.index("full_attention")
        mine = randomize(qwen.GatedAttention(small_config, layer_idx)).to(device)
        mine.eval()

        batch, seq_len = 1, 10
        hidden_states = torch.randn(batch, seq_len, small_config.hidden_size, device=device)
        mask = qwen.build_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
        position_embeddings = qwen.Qwen35RotaryEmbedding(small_config).to(device)(
            hidden_states, torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        )

        perturbed = hidden_states.clone()
        perturbed[:, -1] += 100.0

        with torch.no_grad():
            out_a = mine(hidden_states, position_embeddings, mask)
            out_b = mine(perturbed, position_embeddings, mask)

        assert out_a.shape == hidden_states.shape
        assert_close(out_a[:, :-1], out_b[:, :-1], msg="GatedAttention must be causal")
        assert not torch.allclose(out_a[:, -1], out_b[:, -1]), "last position should still change"


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class TestMLP:
    def test_mlp_matches_reference(self, small_config, real_config, device):
        mine = randomize(qwen.MLP(small_config)).to(device)
        real = ref.Qwen3_5MLP(real_config, intermediate_size=small_config.intermediate_size).to(device)
        real.load_state_dict(mine.state_dict())

        x = torch.randn(2, 9, small_config.hidden_size, device=device)
        out_mine = mine(x)

        assert out_mine.shape == x.shape
        assert torch.isfinite(out_mine).all()
        assert_close(out_mine, real(x), msg="MLP")


# ---------------------------------------------------------------------------
# Gated DeltaNet
# ---------------------------------------------------------------------------

class TestGatedDeltaNet:
    def test_gated_delta_net_matches_reference(self, small_config, real_config, device):
        layer_idx = small_config.layer_types.index("linear_attention")
        mine = randomize_gdn(qwen.GatedDeltaNet(small_config, layer_idx)).to(device)
        real = ref.Qwen3_5GatedDeltaNet(real_config, layer_idx).to(device)
        real.load_state_dict(mine.state_dict())
        mine.eval()
        real.eval()

        hidden_states = torch.randn(2, 19, small_config.hidden_size, device=device)
        with torch.no_grad():
            out_mine = mine(hidden_states)
            out_real = real(hidden_states)
            if isinstance(out_real, tuple):
                out_real = out_real[0]

        assert out_mine.shape == hidden_states.shape
        assert torch.isfinite(out_mine).all()
        assert_close(out_mine, out_real, atol=2e-5, rtol=2e-4, msg="GatedDeltaNet vs real transformers")

    @pytest.mark.parametrize("decay_parameter", ["A_log", "dt_bias"])
    def test_each_decay_parameter_changes_the_output(self, small_config, real_config, device, decay_parameter):
        """Both decay parameters must affect history, with a visible signal.

        Comparing before/after separately prevents a missing A_log or dt_bias
        from hiding behind near-zero weights or a loose output tolerance.
        """
        layer_idx = small_config.layer_types.index("linear_attention")
        mine = randomize_gdn(qwen.GatedDeltaNet(small_config, layer_idx)).to(device).eval()
        real = ref.Qwen3_5GatedDeltaNet(real_config, layer_idx).to(device).eval()
        hidden_states = torch.randn(2, 29, small_config.hidden_size, device=device)

        with torch.no_grad():
            real.load_state_dict(mine.state_dict())
            baseline_mine = mine(hidden_states)
            baseline_real = real(hidden_states)
            if isinstance(baseline_real, tuple):
                baseline_real = baseline_real[0]
            getattr(mine, decay_parameter).add_(1.3)
            real.load_state_dict(mine.state_dict())
            changed_mine = mine(hidden_states)
            changed_real = real(hidden_states)
            if isinstance(changed_real, tuple):
                changed_real = changed_real[0]

        expected_change = changed_real - baseline_real
        assert expected_change.abs().max() > 1e-3, "test setup must visibly exercise decay"
        assert_close(baseline_mine, baseline_real, atol=2e-5, rtol=2e-4, msg=f"{decay_parameter} baseline")
        assert_close(changed_mine, changed_real, atol=2e-5, rtol=2e-4, msg=f"{decay_parameter} changed")
        assert_close(changed_mine - baseline_mine, expected_change, atol=2e-5, rtol=2e-4, msg=f"{decay_parameter} effect")


class TestGatedDeltaRuleRecurrent:
    def test_recurrent_keeps_gradients_for_every_input(self, device):
        """The supplied recurrence is trainable, including decay and write gates."""
        query = torch.randn(2, 7, 3, 4, device=device, requires_grad=True)
        key = torch.randn(2, 7, 3, 4, device=device, requires_grad=True)
        value = torch.randn(2, 7, 3, 5, device=device, requires_grad=True)
        g = (-torch.rand(2, 7, 3, device=device)).requires_grad_()
        beta = torch.rand(2, 7, 3, device=device, requires_grad=True)
        inputs = (query, key, value, g, beta)
        output = qwen.gated_delta_rule_recurrent(*inputs)
        assert output.requires_grad, "recurrence must preserve autograd (no @torch.no_grad)"
        gradients = torch.autograd.grad(output.square().sum(), inputs)
        for name, gradient in zip(("query", "key", "value", "g", "beta"), gradients):
            assert torch.isfinite(gradient).all(), f"{name} gradient must be finite"
            assert gradient.abs().sum() > 0, f"{name} must receive a nonzero gradient"

    def test_recurrent_output_cast_to_input_dtype(self, small_config, device):
        """A model running in bf16/fp16 needs the recurrence's internal float32
        upcast undone before returning, or the output silently stays float32
        and later ops against bf16 tensors crash."""
        batch, seq_len = 2, 11
        heads = small_config.linear_num_value_heads
        k_dim, v_dim = small_config.linear_key_head_dim, small_config.linear_value_head_dim

        query = torch.randn(batch, seq_len, heads, k_dim, device=device)
        key = torch.randn(batch, seq_len, heads, k_dim, device=device)
        value = torch.randn(batch, seq_len, heads, v_dim, device=device)
        g = -torch.rand(batch, seq_len, heads, device=device)
        beta = torch.rand(batch, seq_len, heads, device=device)

        out_fp32 = qwen.gated_delta_rule_recurrent(query, key, value, g, beta)

        bf16 = lambda x: x.to(torch.bfloat16)
        out_bf16 = qwen.gated_delta_rule_recurrent(bf16(query), bf16(key), bf16(value), bf16(g), bf16(beta))

        assert out_bf16.dtype == torch.bfloat16, (
            f"gated_delta_rule_recurrent should return output cast to the input dtype, got {out_bf16.dtype}"
        )
        assert_close(out_bf16.float(), out_fp32, atol=1e-1, msg="gated_delta_rule_recurrent bf16 vs fp32")


# ---------------------------------------------------------------------------
# Decoder layer / full model
# ---------------------------------------------------------------------------

class TestDecoderLayer:
    def test_decoder_layer_matches_reference(self, small_config, real_config, device):
        for layer_idx, block_type in enumerate(small_config.layer_types):
            mine = randomize(qwen.Qwen35DecoderLayer(small_config, layer_idx)).to(device)
            real = ref.Qwen3_5DecoderLayer(real_config, layer_idx).to(device)
            real.load_state_dict(mine.state_dict())
            mine.eval()
            real.eval()

            batch, seq_len = 2, 14
            hidden_states = torch.randn(batch, seq_len, small_config.hidden_size, device=device)
            mask = qwen.build_causal_mask(seq_len, hidden_states.dtype, hidden_states.device)
            position_embeddings = qwen.Qwen35RotaryEmbedding(small_config).to(device)(
                hidden_states, torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
            )

            # the 4D additive causal mask is only meaningful for full_attention layers;
            # passing it to a linear_attention layer's real reference module makes newer
            # transformers versions try to use it as a 2D padding mask and crash -- this
            # assignment is out of scope for padding masks, so pass None there instead,
            # matching what qwen.Qwen35DecoderLayer.forward does internally.
            real_mask = mask if block_type == "full_attention" else None

            with torch.no_grad():
                out_mine = mine(hidden_states, position_embeddings, mask)
                out_real = real(hidden_states, position_embeddings=position_embeddings, attention_mask=real_mask)
                if isinstance(out_real, tuple):
                    out_real = out_real[0]

            assert out_mine.shape == hidden_states.shape
            assert_close(out_mine, out_real, atol=1e-3, msg=f"DecoderLayer[{block_type}] (layer {layer_idx})")


class TestFullModel:
    def test_full_model_matches_reference(self, small_config, real_config, device):
        mine = randomize(qwen.Qwen35Model(small_config), std=0.02).to(device)
        real = ref.Qwen3_5TextModel(real_config).to(device)
        real.load_state_dict(mine.state_dict())
        mine.eval()
        real.eval()

        batch, seq_len = 2, 21
        input_ids = torch.randint(0, small_config.vocab_size, (batch, seq_len), device=device)
        with torch.no_grad():
            out_mine = mine(input_ids)
            out_real = real(input_ids, use_cache=False).last_hidden_state

        assert out_mine.shape == (batch, seq_len, small_config.hidden_size)
        assert torch.isfinite(out_mine).all()
        assert_close(out_mine, out_real, atol=1e-2, msg="Full Qwen35Model")


class TestCausalLM:
    def test_causal_lm_logits_are_tied_embedding_matmul(self, small_config, device):
        model = randomize(qwen.Qwen35ForCausalLM(small_config), std=0.02).to(device)
        model.eval()
        batch, seq_len = 2, 10
        input_ids = torch.randint(0, small_config.vocab_size, (batch, seq_len), device=device)
        with torch.no_grad():
            hidden = model.model(input_ids)
            logits, loss = model(input_ids)
            expected_logits = hidden @ model.model.embed_tokens.weight.T

        assert loss is None, "loss must be None when labels is not given"
        assert logits.shape == (batch, seq_len, small_config.vocab_size)
        assert_close(logits, expected_logits, msg="ForCausalLM logits must be the tied-embedding matmul")

    def test_causal_lm_loss_matches_manual_cross_entropy(self, small_config, device):
        import torch.nn.functional as F

        model = randomize(qwen.Qwen35ForCausalLM(small_config), std=0.02).to(device)
        model.eval()
        input_ids = torch.randint(0, small_config.vocab_size, (2, 10), device=device)
        labels = torch.randint(0, small_config.vocab_size, (2, 10), device=device)
        with torch.no_grad():
            logits, loss = model(input_ids, labels=labels)
            expected_loss = F.cross_entropy(logits.view(-1, small_config.vocab_size), labels.view(-1))

        assert loss is not None, "loss must be returned when labels is given"
        assert loss.dim() == 0, "loss must be a scalar"
        assert_close(loss.unsqueeze(0), expected_loss.unsqueeze(0), msg="ForCausalLM loss")

    def test_causal_lm_tied_lm_head_is_same_tensor_as_embeddings(self, small_config, device):
        model = randomize(qwen.Qwen35ForCausalLM(small_config), std=0.02).to(device)
        model.eval()
        assert model.lm_head.weight is model.model.embed_tokens.weight

        input_ids = torch.randint(0, small_config.vocab_size, (2, 10), device=device)
        with torch.no_grad():
            logits, _ = model(input_ids)
        assert logits.shape == (2, 10, small_config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_causal_lm_untied_lm_head_differs_from_embeddings(self, device):
        config = qwen.make_small_config(tie_word_embeddings=False)
        model = randomize(qwen.Qwen35ForCausalLM(config), std=0.02).to(device)
        model.eval()

        assert model.lm_head.weight is not model.model.embed_tokens.weight

        input_ids = torch.randint(0, config.vocab_size, (2, 10), device=device)
        with torch.no_grad():
            logits, _ = model(input_ids)
        assert logits.shape == (2, 10, config.vocab_size)
        assert torch.isfinite(logits).all()


# ---------------------------------------------------------------------------
# Scaling to other Qwen3.5 sizes
# ---------------------------------------------------------------------------
#
# Hardcoded (not downloaded) `text_config`/`config` values, copied from the
# real config.json's linked in qwen.qwen35_config_from_hf's docstring --
# offline and deterministic on purpose. Notice 9B/27B's `text_config` has no
# "tie_word_embeddings" key at all (only the top-level one), unlike 0.8B/4B.

_ROPE_PARAMETERS = {"rope_theta": 10000000, "partial_rotary_factor": 0.25}

_HF_CONFIGS = {
    "0.8B": {
        "tie_word_embeddings": True,
        "text_config": {
            "vocab_size": 248320, "hidden_size": 1024, "intermediate_size": 3584,
            "num_hidden_layers": 24, "num_attention_heads": 8, "num_key_value_heads": 2,
            "head_dim": 256, "attention_bias": False,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 16,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-06, "hidden_act": "silu",
            "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": True, "rope_parameters": _ROPE_PARAMETERS,
        },
    },
    "4B": {
        "tie_word_embeddings": True,
        "text_config": {
            "vocab_size": 248320, "hidden_size": 2560, "intermediate_size": 9216,
            "num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4,
            "head_dim": 256, "attention_bias": False,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-06, "hidden_act": "silu",
            "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
            "tie_word_embeddings": True, "rope_parameters": _ROPE_PARAMETERS,
        },
    },
    "9B": {
        "tie_word_embeddings": False,
        "text_config": {
            "vocab_size": 248320, "hidden_size": 4096, "intermediate_size": 12288,
            "num_hidden_layers": 32, "num_attention_heads": 16, "num_key_value_heads": 4,
            "head_dim": 256, "attention_bias": False,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-06, "hidden_act": "silu",
            "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
            # no "tie_word_embeddings" here on purpose -- only at the top level
            "rope_parameters": _ROPE_PARAMETERS,
        },
    },
    "27B": {
        "tie_word_embeddings": False,
        "text_config": {
            "vocab_size": 248320, "hidden_size": 5120, "intermediate_size": 17408,
            "num_hidden_layers": 64, "num_attention_heads": 24, "num_key_value_heads": 4,
            "head_dim": 256, "attention_bias": False,
            "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-06, "hidden_act": "silu",
            "full_attention_interval": 4, "mtp_num_hidden_layers": 1,
            "rope_parameters": _ROPE_PARAMETERS,
        },
    },
}


class TestScalingToOtherSizes:
    @pytest.mark.parametrize("size", ["0.8B", "4B", "9B", "27B"])
    @pytest.mark.parametrize("nested", [True, False], ids=["multimodal-config", "text-config"])
    def test_qwen35_config_from_hf_matches_known_shapes(self, size, nested):
        hf_config = _HF_CONFIGS[size]
        text_config = hf_config["text_config"]
        source_config = hf_config if nested else {**text_config, "tie_word_embeddings": hf_config["tie_word_embeddings"]}
        config = qwen.qwen35_config_from_hf(source_config)

        for field in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "attention_bias",
            "linear_conv_kernel_dim", "linear_key_head_dim", "linear_value_head_dim",
            "linear_num_key_heads", "linear_num_value_heads", "full_attention_interval",
            "max_position_embeddings", "rms_norm_eps", "hidden_act",
        ):
            assert getattr(config, field) == text_config[field], f"{size}: incorrect {field}"
        assert config.rope_theta == text_config["rope_parameters"]["rope_theta"]
        assert config.partial_rotary_factor == text_config["rope_parameters"]["partial_rotary_factor"]
        assert config.mtp_num_layers == text_config["mtp_num_hidden_layers"]
        assert config.tie_word_embeddings == hf_config["tie_word_embeddings"], (
            f"{size}: tie_word_embeddings must come from the top-level config when "
            f"text_config doesn't repeat it"
        )
        assert config.layer_types == [
            "full_attention" if (index + 1) % text_config["full_attention_interval"] == 0 else "linear_attention"
            for index in range(text_config["num_hidden_layers"])
        ]


# ---------------------------------------------------------------------------
# End-to-end: logits stay matched across several forward passes
# ---------------------------------------------------------------------------
#
# Every class above checks a single forward pass. This one greedily extends
# the sequence a few tokens at a time (each step re-running the *full*
# forward pass from scratch, no KV cache) and checks logits still agree with
# real `transformers` at every step. Any divergence should stay at
# float32-rounding scale (~1e-6) -- it should not grow with sequence length,
# and it should never approach the scale of a real implementation bug.

class TestEndToEndForwardPasses:
    @pytest.mark.parametrize("tie_word_embeddings", [True, False])
    def test_causal_lm_matches_reference_over_several_forward_passes(self, tie_word_embeddings, device):
        config = qwen.make_small_config(tie_word_embeddings=tie_word_embeddings)
        real_config = real_text_config(config)

        mine = randomize(qwen.Qwen35ForCausalLM(config), std=0.02).to(device)
        real = ref.Qwen3_5ForCausalLM(real_config).to(device)
        real.load_state_dict(mine.state_dict(), strict=True)
        mine.eval()
        real.eval()

        batch, seq_len = 2, 5
        input_ids = torch.randint(0, config.vocab_size, (batch, seq_len), device=device)

        for step in range(5):
            with torch.no_grad():
                logits_mine, _ = mine(input_ids)
                logits_real = real(input_ids, use_cache=False).logits

            assert logits_mine.shape == (batch, input_ids.shape[1], config.vocab_size)
            assert_close(
                logits_mine, logits_real, atol=1e-3, rtol=1e-3,
                msg=f"CausalLM logits vs real transformers (step {step}, seq_len={input_ids.shape[1]})",
            )

            next_token = logits_mine[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

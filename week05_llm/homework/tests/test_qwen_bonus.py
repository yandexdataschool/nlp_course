"""PUBLIC examples for optional GQA, chunked delta rule, and MTP bonuses.

Run this file explicitly, or pass --run-bonus. Numerical agreement alone does
not demonstrate chunking: batched within-chunk computation is also required.
"""

import pytest
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as ref

import qwen
from conftest import assert_close, randomize, real_text_config

pytestmark = [pytest.mark.bonus, pytest.mark.public]


@pytest.mark.gqa
class TestGroupedQueryAttention:
    @pytest.mark.parametrize("query_heads,kv_heads", [(8, 4), (6, 2), (4, 1)])
    def test_group_ratios_match_transformers(self, query_heads, kv_heads, device):
        """The bonus generalizes regular attention to fewer shared K/V heads."""
        config = qwen.make_small_config(num_attention_heads=query_heads, num_key_value_heads=kv_heads)
        mine = randomize(qwen.GatedAttention(config, 3), std=0.12).to(device).eval()
        real = ref.Qwen3_5Attention(real_text_config(config), 3).to(device).eval()
        real.load_state_dict(mine.state_dict())
        hidden = torch.randn(2, 11, config.hidden_size, device=device)
        positions = qwen.Qwen35RotaryEmbedding(config).to(device)(
            hidden, torch.arange(11, device=device)[None].expand(2, -1),
        )
        mask = qwen.build_causal_mask(11, hidden.dtype, device)
        with torch.no_grad():
            actual = mine.forward_gqa(hidden, positions, mask)
            dispatched = mine(hidden, positions, mask)
            expected = real(hidden, position_embeddings=positions, attention_mask=mask)[0]
        assert_close(actual, expected, atol=2e-5, rtol=2e-4, msg="GQA head grouping")
        assert_close(dispatched, actual, atol=2e-5, msg="forward dispatch to GQA")

    def test_default_mask_is_causal(self, device):
        config = qwen.make_small_config(num_attention_heads=6, num_key_value_heads=2)
        mine = randomize(qwen.GatedAttention(config, 3), std=0.12).to(device).eval()
        hidden = torch.randn(2, 9, config.hidden_size, device=device)
        changed = hidden.clone()
        changed[:, 5:] += torch.randn_like(changed[:, 5:]) * 3
        positions = qwen.Qwen35RotaryEmbedding(config).to(device)(
            hidden, torch.arange(9, device=device)[None].expand(2, -1),
        )
        with torch.no_grad():
            actual = mine(hidden, positions)
            altered = mine(changed, positions)
            expected = qwen.naive_looped_gated_attention(mine, hidden, positions)
        assert_close(actual, expected, atol=2e-5, rtol=2e-4, msg="GQA without explicit mask")
        assert_close(actual[:, :5], altered[:, :5], msg="GQA must not read future tokens")
        assert not torch.allclose(actual[:, 5:], altered[:, 5:])

    def test_full_model_with_shared_kv_heads(self, device):
        config = qwen.make_small_config(num_attention_heads=6, num_key_value_heads=2)
        mine = randomize(qwen.Qwen35ForCausalLM(config), std=0.08).to(device).eval()
        real = ref.Qwen3_5ForCausalLM(real_text_config(config)).to(device).eval()
        real.load_state_dict(mine.state_dict(), strict=True)
        input_ids = torch.randint(0, config.vocab_size, (2, 13), device=device)
        with torch.no_grad():
            actual, loss = mine(input_ids)
            expected = real(input_ids, use_cache=False).logits
        assert loss is None
        assert_close(actual, expected, atol=3e-5, rtol=3e-4, msg="GQA decoder/model integration")


class TestChunkedDeltaRuleCorrectness:
    @pytest.mark.parametrize("length,chunk_size", [(1, 8), (23, 8), (37, 16)])
    def test_matches_recurrent_and_transformers(self, small_config, length, chunk_size, device):
        batch, heads = 2, small_config.linear_num_value_heads
        k_dim, v_dim = small_config.linear_key_head_dim, small_config.linear_value_head_dim
        query = torch.randn(batch, length, heads, k_dim, device=device)
        key = torch.randn(batch, length, heads, k_dim, device=device)
        value = torch.randn(batch, length, heads, v_dim, device=device)
        g = -torch.rand(batch, length, heads, device=device)
        beta = torch.rand(batch, length, heads, device=device)
        out = qwen.gated_delta_rule_chunked(query, key, value, g, beta, chunk_size=chunk_size)
        expected = qwen.gated_delta_rule_recurrent(query, key, value, g, beta)
        real, _ = ref.torch_chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, chunk_size=chunk_size, use_qk_l2norm_in_kernel=True,
        )
        assert out.shape == (batch, length, heads, v_dim)
        assert out.device == query.device
        assert_close(out, expected, atol=1e-4, msg="chunked vs given recurrence")
        assert_close(out, real, atol=1e-4, msg="chunked vs transformers")


class TestMTPHead:
    def test_output_shape(self, small_config, device):
        """The head combines h[t] and the embedding of token[t+1]."""
        mtp = randomize(qwen.Qwen35MTPHead(small_config), std=0.02).to(device).eval()
        embedding = torch.nn.Embedding(small_config.vocab_size, small_config.hidden_size).to(device)
        hidden = torch.randn(2, 16, small_config.hidden_size, device=device)
        input_ids = torch.randint(0, small_config.vocab_size, (2, 16), device=device)
        with torch.no_grad():
            logits = mtp(hidden, input_ids, embedding)
        assert logits.shape == (2, 14, small_config.vocab_size)
        assert logits.device == hidden.device
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("length", [1, 2])
    def test_needs_at_least_three_tokens(self, small_config, length, device):
        mtp = qwen.Qwen35MTPHead(small_config).to(device)
        embedding = torch.nn.Embedding(small_config.vocab_size, small_config.hidden_size).to(device)
        hidden = torch.randn(1, length, small_config.hidden_size, device=device)
        input_ids = torch.zeros(1, length, dtype=torch.long, device=device)
        with pytest.raises(ValueError):
            mtp(hidden, input_ids, embedding)

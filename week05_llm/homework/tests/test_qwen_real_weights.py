"""
Intermediate tests using real Qwen3.5-0.8B checkpoint weights and real
generation, plus an opt-in GPU flag.

Off by default -- these need the relevant flag. Real checkpoints use GQA,
so logits/generation require the optional GQA implementation and opt-in:
    pytest tests/test_qwen_real_weights.py --run-real-weights              # main: config metadata only
    pytest tests/test_qwen_real_weights.py --run-real-weights --run-bonus  # CPU, real GQA checkpoint
    pytest tests/test_qwen_real_weights.py --run-real-weights --run-bonus --gpu
    pytest tests/test_qwen_real_weights.py --gpu                           # tiny main GPU check, no download

`--run-real-weights` downloads (or reuses a cached) ~1.8GB checkpoint from
the HF Hub -- needs internet the first time. `--gpu` requires an available
CUDA device.
"""

import pytest
import torch

from transformers.models.qwen3_5 import modeling_qwen3_5 as ref

import qwen
from conftest import randomize, assert_close


pytestmark = pytest.mark.public


def _load_student_and_real_models(real_hf_config, real_full_config, real_checkpoint_state_dict, device):
    # load_state_dict transfers each tensor; moving the whole checkpoint first
    # would hold an unnecessary third model-sized allocation on the GPU.
    state_dict = real_checkpoint_state_dict

    student_model = qwen.Qwen35ForCausalLM(real_full_config).to(device).eval()
    missing, unexpected = student_model.model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, f"student checkpoint load: missing={missing} unexpected={unexpected}"

    from copy import deepcopy

    text_config = deepcopy(real_hf_config.get_text_config())
    text_config._attn_implementation = "eager"
    real_model = ref.Qwen3_5TextModel(text_config).to(device).eval()
    missing, unexpected = real_model.load_state_dict(state_dict, strict=False)
    assert not missing and not unexpected, f"real checkpoint load: missing={missing} unexpected={unexpected}"

    return student_model, real_model


@pytest.fixture(scope="session")
def checkpoint_models(real_hf_config, real_full_config, real_checkpoint_state_dict, device):
    return _load_student_and_real_models(real_hf_config, real_full_config, real_checkpoint_state_dict, device)


def _real_logits(real_model, input_ids):
    hidden = real_model(input_ids, use_cache=False).last_hidden_state
    return hidden @ real_model.embed_tokens.weight.T


# ---------------------------------------------------------------------------
# GPU sanity check -- no download, just device consistency on the tiny config
# ---------------------------------------------------------------------------

@pytest.mark.gpu
@pytest.mark.main
class TestGPUDeviceConsistency:
    def test_full_model_cpu_matches_gpu(self, small_config, device):
        model_cpu = randomize(qwen.Qwen35ForCausalLM(small_config), std=0.02)
        model_cpu.eval()
        model_gpu = qwen.Qwen35ForCausalLM(small_config)
        model_gpu.load_state_dict(model_cpu.state_dict())
        model_gpu = model_gpu.to(device).eval()

        input_ids = torch.randint(0, small_config.vocab_size, (2, 17))
        with torch.no_grad():
            logits_cpu, _ = model_cpu(input_ids)
            logits_gpu, _ = model_gpu(input_ids.to(device))

        assert logits_cpu.device.type == "cpu"
        assert logits_gpu.device.type == "cuda"
        assert all(parameter.device.type == "cuda" for parameter in model_gpu.parameters())
        assert all(buffer.device.type == "cuda" for buffer in model_gpu.buffers())
        assert_close(logits_cpu, logits_gpu.cpu(), atol=1e-3, rtol=1e-3, msg="CPU vs GPU logits")


# ---------------------------------------------------------------------------
# Real checkpoint: single forward pass vs. real transformers
# ---------------------------------------------------------------------------

@pytest.mark.real_weights
@pytest.mark.main
class TestRealWeightsConfig:
    def test_converted_config_matches_checkpoint_metadata(self, real_hf_config, real_full_config):
        text = real_hf_config.get_text_config()
        for field in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "head_dim", "attention_bias",
            "linear_conv_kernel_dim", "linear_key_head_dim", "linear_value_head_dim",
            "linear_num_key_heads", "linear_num_value_heads", "max_position_embeddings",
            "rms_norm_eps", "hidden_act", "layer_types",
        ):
            assert getattr(real_full_config, field) == getattr(text, field), field


@pytest.mark.real_weights
@pytest.mark.bonus
@pytest.mark.gqa
class TestRealWeightsForwardPass:
    def test_logits_match_real_transformers(
        self, checkpoint_models, real_tokenizer, device
    ):
        student_model, real_model = checkpoint_models

        input_ids = torch.as_tensor([real_tokenizer.encode("The Fermi paradox ")], device=device)
        with torch.no_grad():
            student_logits, _ = student_model(input_ids)
            real_logits = _real_logits(real_model, input_ids)

        assert student_logits.shape == real_logits.shape
        assert student_logits.device == input_ids.device
        assert_close(student_logits, real_logits, atol=1e-3, rtol=1e-3, msg="real-checkpoint logits")


# ---------------------------------------------------------------------------
# Real checkpoint: multi-step greedy generation vs. real transformers
# ---------------------------------------------------------------------------

@pytest.mark.real_weights
@pytest.mark.bonus
@pytest.mark.gqa
class TestRealWeightsGeneration:
    def test_greedy_generation_matches_real_transformers(
        self, checkpoint_models, real_tokenizer, device
    ):
        student_model, real_model = checkpoint_models

        student_tokens = real_tokenizer.encode("The Fermi paradox ")
        real_tokens = list(student_tokens)

        for _ in range(5):
            with torch.no_grad():
                student_logits, _ = student_model(torch.as_tensor([student_tokens], device=device))
                real_logits = _real_logits(real_model, torch.as_tensor([real_tokens], device=device))
            student_tokens.append(int(student_logits[0, -1].argmax()))
            real_tokens.append(int(real_logits[0, -1].argmax()))

        assert student_tokens == real_tokens, (
            f"greedy generation diverged from real transformers:\n"
            f"student: {student_tokens}\nreal:    {real_tokens}"
        )

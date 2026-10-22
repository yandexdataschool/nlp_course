import os
import sys
from pathlib import Path

# Default to local caches; an explicit cache path (e.g. the GPU server's
# shared HF_HOME) takes precedence so existing checkpoints can be reused.
HOMEWORK_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = HOMEWORK_DIR / ".cache"
os.environ.setdefault("HF_HOME", str(CACHE_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HF_XET_CACHE", str(Path(os.environ["HF_HOME"]) / "xet"))
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import qwen  # noqa: E402  (import after sys.path fixup)


def pytest_addoption(parser):
    parser.addoption(
        "--run-real-weights", action="store_true", default=False,
        help="Run tests that download and use the real Qwen/Qwen3.5-0.8B checkpoint from the HF Hub (~1.8GB).",
    )
    parser.addoption(
        "--gpu", action="store_true", default=False,
        help="Run tests on CUDA instead of CPU, and enable GPU-only tests.",
    )
    parser.addoption(
        "--run-bonus", action="store_true", default=False,
        help="Include optional bonus tests in directory-wide runs (explicit bonus files also run).",
    )


def pytest_configure(config):
    for marker, description in {
        "main": "required homework tasks",
        "bonus": "optional GQA, chunked delta rule, and MTP tasks",
        "gqa": "optional grouped-query attention (select with -m gqa or --run-bonus)",
        "public": "student-facing checks",
    }.items():
        config.addinivalue_line("markers", f"{marker}: {description}")
    config.addinivalue_line("markers", "real_weights: needs --run-real-weights (downloads the real checkpoint)")
    config.addinivalue_line("markers", "gpu: needs --gpu and an available CUDA device")
    if config.getoption("--gpu") and not torch.cuda.is_available():
        raise pytest.UsageError("--gpu was passed but torch.cuda.is_available() is False")


def pytest_collection_modifyitems(config, items):
    # A bare directory run grades only the required tasks. Selecting a bonus
    # file/node or explicitly mentioning bonus in -m is also an opt-in.
    include_bonus = (
        config.getoption("--run-bonus")
        or "bonus" in config.getoption("markexpr")
        or "gqa" in config.getoption("markexpr")
        or any("bonus" in Path(str(arg).split("::", 1)[0]).name for arg in config.args)
    )
    if not include_bonus:
        deselected = [item for item in items if "bonus" in item.keywords]
        items[:] = [item for item in items if "bonus" not in item.keywords]
        config.hook.pytest_deselected(items=deselected)
    skip_real_weights = pytest.mark.skip(reason="need --run-real-weights to run")
    skip_gpu = pytest.mark.skip(reason="need --gpu to run")
    for item in items:
        if "real_weights" in item.keywords and not config.getoption("--run-real-weights"):
            item.add_marker(skip_real_weights)
        if "gpu" in item.keywords and not config.getoption("--gpu"):
            item.add_marker(skip_gpu)


@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def device(request):
    if request.config.getoption("--gpu"):
        if not torch.cuda.is_available():
            pytest.fail("--gpu was passed but torch.cuda.is_available() is False")
        return "cuda"
    return "cpu"


@pytest.fixture
def small_config():
    return qwen.make_small_config()


def real_text_config(small: "qwen.Qwen35Config"):
    """Build a real `transformers` Qwen3_5TextConfig with the same shapes as
    our small student config, so a real module and a student module are
    directly weight-compatible."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        vocab_size=small.vocab_size,
        hidden_size=small.hidden_size,
        intermediate_size=small.intermediate_size,
        num_hidden_layers=small.num_hidden_layers,
        num_attention_heads=small.num_attention_heads,
        num_key_value_heads=small.num_key_value_heads,
        head_dim=small.head_dim,
        linear_conv_kernel_dim=small.linear_conv_kernel_dim,
        linear_key_head_dim=small.linear_key_head_dim,
        linear_value_head_dim=small.linear_value_head_dim,
        linear_num_key_heads=small.linear_num_key_heads,
        linear_num_value_heads=small.linear_num_value_heads,
        full_attention_interval=small.full_attention_interval,
        rope_theta=small.rope_theta,
        partial_rotary_factor=small.partial_rotary_factor,
        rms_norm_eps=small.rms_norm_eps,
        max_position_embeddings=small.max_position_embeddings,
        tie_word_embeddings=small.tie_word_embeddings,
        attn_implementation="eager",
    )


@pytest.fixture
def real_config(small_config):
    return real_text_config(small_config)


def randomize(module: torch.nn.Module, std: float = 0.05):
    """In-place randomize every parameter of a module (default nn.Linear
    init is too close to identity/zero for some of these blocks to be a
    meaningful correctness check)."""
    with torch.no_grad():
        for p in module.parameters():
            torch.nn.init.normal_(p, std=std)
    return module


def randomize_gdn(module: torch.nn.Module, std: float = 0.12):
    """Use a visible signal and separated decay parameters for GDN checks.

    Tiny normally distributed A_log/dt_bias and norm weights hide a missing
    decay factor under output tolerances. These values exercise different
    memory timescales while keeping all outputs numerically well behaved.
    """
    randomize(module, std=std)
    with torch.no_grad():
        for child in module.modules():
            if isinstance(child, qwen.GatedDeltaNet):
                n_heads = child.num_v_heads
                child.A_log.copy_(torch.linspace(-1.4, 1.2, n_heads, device=child.A_log.device))
                child.dt_bias.copy_(torch.linspace(-1.1, 1.4, n_heads, device=child.dt_bias.device))
                child.norm.weight.copy_(torch.linspace(0.7, 1.3, child.head_v_dim, device=child.norm.weight.device))
    return module


def assert_close(actual, expected, atol=1e-4, rtol=1e-4, msg=""):
    assert actual.shape == expected.shape, f"{msg} shape: {actual.shape} != {expected.shape}"
    assert actual.device == expected.device, f"{msg} device: {actual.device} != {expected.device}"
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol), (
        f"{msg} max abs diff = {(actual - expected).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# Real Qwen3.5-0.8B checkpoint fixtures -- only touched by tests marked
# `real_weights` (gated behind --run-real-weights). Checkpoints and metadata
# default to homework/.cache, or reuse explicitly configured cache paths.
# ---------------------------------------------------------------------------

REAL_MODEL_ID = "Qwen/Qwen3.5-0.8B"


@pytest.fixture(scope="session")
def real_hf_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(REAL_MODEL_ID, cache_dir=os.environ["HF_HUB_CACHE"])


@pytest.fixture(scope="session")
def real_full_config(real_hf_config):
    return qwen.qwen35_config_from_hf(real_hf_config.to_dict())


@pytest.fixture(scope="session")
def real_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REAL_MODEL_ID, cache_dir=os.environ["HF_HUB_CACHE"])


@pytest.fixture(scope="session")
def real_checkpoint_state_dict():
    """Downloads (or reuses the cached) checkpoint, merges its safetensors
    shards, and strips them down to a state dict loadable into `Qwen35Model`
    (backbone only -- drops the MTP head, which our model doesn't have)."""
    import glob
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    ckpt_dir = snapshot_download(
        REAL_MODEL_ID, cache_dir=os.environ["HF_HUB_CACHE"],
        allow_patterns=["*.safetensors"],
    )
    state_dict = {}
    for path in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        state_dict.update(load_file(path))

    prefix = "model.language_model."
    return {
        key[len(prefix):]: value for key, value in state_dict.items()
        if key.startswith(prefix) and not key.startswith(prefix + "mtp.")
    }

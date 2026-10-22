# Week 05: Qwen3.5 homework

Open `homework.ipynb` for the assignment and implement the TODOs in `qwen.py`.
`qwen_configs.py` supplies the model configuration. The notebook contains the
explanations and demonstrations; run the public tests separately.

Required tasks cover RMSNorm, gated normalization, partial RoPE, vectorized
regular multi-head attention (equal query/key/value head counts), SwiGLU,
Gated DeltaNet, the decoder/backbone, causal LM loss, and config conversion.
GQA, chunked recurrence, and MTP are three independent optional bonuses.
The `week07_efficiency` lecture will explain GQA and many more efficiency
techniques.

## Public tests

Run these commands in a terminal from this directory:

```sh
# Required tasks; CPU, no checkpoint download:
python -m pytest tests/test_qwen_public.py -q

# Optional bonuses:
python -m pytest tests/test_qwen_bonus.py -q

# Required tasks and bonuses together:
python -m pytest --run-bonus -q
```

A default run excludes bonuses. Unfinished TODOs fail until implemented.
Real-weight tests are opt-in and download approximately 1.8 GB.
The actual checkpoint uses GQA: its logits and generation checks require the
GQA bonus, enabled below with `--run-bonus`. Omit that flag to run only the
required-task config metadata checks:

```sh
python -m pytest tests/test_qwen_real_weights.py --run-real-weights --run-bonus -q
```

Add `--gpu` on a CUDA machine to move test inputs and modules to CUDA.
Ordinary small-model tests run on CPU without downloading a checkpoint.

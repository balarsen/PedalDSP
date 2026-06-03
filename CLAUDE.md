# PedalDSP — Claude Code Guidelines

## Project Overview

Guitar pedal modeling project: capture dry/wet audio pairs, fit classical DSP models (FIR, IIR, Hammerstein, Volterra), train neural models (LSTM, TCN, WaveNet, DDSP), and compare them on a metrics suite (ESR, STFT loss, THD, MCD).

Stack: Python 3.11+, PyTorch, torchaudio, NumPy, SciPy, soundfile, seaborn. Runs on WSL2 with CUDA.

---

## Code Style

- Follow PEP 8. Use `ruff` for linting (`ruff check .`).
- Type-annotate all function signatures. No `Any` unless unavoidable.
- Prefer `np.ndarray` and `torch.Tensor` over bare arrays in signatures — be explicit about shapes in docstrings.
- Use `pathlib.Path` for all file paths, never raw strings.
- Constants in `UPPER_SNAKE_CASE` at module level.

---

## Documentation Rules

**Functions**: every public function gets a one-line summary and a `Args` / `Returns` block. Skip obvious parameters (no `x: float  # the float x`). Document units (`sr: int  # sample rate in Hz`), valid ranges, and non-obvious defaults.

```python
def compute_esr(target: np.ndarray, predicted: np.ndarray) -> float:
    """Error-to-Signal Ratio: lower is better, 0 = perfect.

    Args:
        target: Reference audio, shape (N,), float32, range [-1, 1].
        predicted: Model output, same shape as target.

    Returns:
        ESR in [0, ∞). Values > 1 indicate the model is worse than silence.
    """
```

**Inline comments**: only when the WHY is non-obvious — a hidden constraint, a workaround for a known numerical issue, a DSP identity that would surprise a reader. No comments that restate what the code already says.

**Modules**: each module gets a one-line module docstring naming its responsibility. No paragraph essays.

**No comments for**:
- What a loop does when the variable names are clear
- Closing braces or blocks
- Removed or future code

---

## Testing Rules

Every module in `capture/`, `models/`, `metrics/`, and `train/` must have a corresponding test file at `tests/test_<module>.py`.

### What to test

**Pure functions** (metrics, alignment, signal generation):
- Test on synthetic inputs where the expected output is known analytically.
- Test edge cases: empty array, single sample, all-zeros signal, perfect prediction.
- Assert numerical precision with `np.testing.assert_allclose(rtol=1e-5)` — not `==`.

**Model forward passes**:
- Test that output shape matches expected shape for a batch of random inputs.
- Test that forward pass runs on CPU. If CUDA is available, add a GPU variant with `pytest.mark.skipif`.
- Test that loss is finite and > 0 on random inputs.

**Data pipeline** (`data/dataset.py`):
- Test with a short synthetic WAV (generated in the test, not from disk).
- Test that windows don't exceed audio length.
- Test that dry/wet alignment is preserved through slicing.

### Structure

```python
import pytest
import numpy as np
from metrics.time_domain import compute_esr

def test_esr_perfect_prediction():
    signal = np.random.randn(1000).astype(np.float32)
    assert compute_esr(signal, signal) == pytest.approx(0.0, abs=1e-7)

def test_esr_zero_prediction():
    signal = np.ones(100, dtype=np.float32)
    assert compute_esr(signal, np.zeros(100, dtype=np.float32)) == pytest.approx(1.0)
```

- Use `pytest`. No `unittest.TestCase`.
- No mocking of NumPy, SciPy, or PyTorch internals.
- No file I/O in tests — generate synthetic data inline or use `tmp_path`.
- Tests must be deterministic: set `np.random.seed` and `torch.manual_seed` at the top of any test that uses random data.
- Tests must pass on CPU-only. GPU is optional / skipped.

### Running tests

```bash
pytest tests/ -v
pytest tests/ -v --tb=short   # shorter tracebacks
```

---

## Model Implementation Rules

- All models inherit from `models/base.py`. Implement `forward()` and `receptive_field` property.
- Classical models (FIR, IIR, etc.) must be runnable without PyTorch — NumPy/SciPy only.
- Neural models must support `torch.no_grad()` inference.
- Never hard-code sample rate; always pass `sr: int` as a parameter.
- No global mutable state. Models are stateless between calls unless they explicitly carry hidden state (LSTM, GRU).

---

## Training Rules

- Checkpoints go in `checkpoints/<model_name>/<timestamp>/`.
- Always log: epoch, train loss, val loss, ESR on val set.
- Never overwrite a checkpoint without a new timestamp.
- Training scripts must be runnable from the CLI with a config YAML: `python train/trainer.py --config configs/experiment.yaml`.

---

## Metrics Rules

- All metrics return a single `float` (not a tensor).
- All metrics are detached from the compute graph before return.
- `metrics/suite.py` is the only place that calls all metrics together. Individual metric modules have no cross-imports.

---

## Environment

- Python 3.11+, CUDA 12.1, PyTorch ≥ 2.2.
- All dependencies in `requirements.txt`. Pin major versions.
- WSL2 Ubuntu. Use `pathlib.Path` — never Windows-style backslash paths.
- Audio files: 48kHz, 24-bit, float32 in memory after load.

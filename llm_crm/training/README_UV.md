# UV Environment Setup

This directory uses [uv](https://github.com/astral-sh/uv) for Python dependency management.

## Setup

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Create and activate the virtual environment**:
   ```bash
   cd train
   uv venv
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -e .
   ```

   Or install directly from pyproject.toml:
   ```bash
   uv pip install -r pyproject.toml
   ```

## Dependencies

The project requires:
- `numpy>=1.21.0` - Numerical computing
- `torch>=2.0.0` - PyTorch deep learning framework
- `transformers>=4.30.0` - Hugging Face Transformers library
- `peft>=0.4.0` - Parameter-Efficient Fine-Tuning

## Checkpoints

Model checkpoints are saved to:
```
train/checkpoints/llama_il_lora_webarena/
```

This directory is created automatically when training starts. Consider adding it to `.gitignore` if you don't want to commit checkpoints.

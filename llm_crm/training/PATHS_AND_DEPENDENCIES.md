# Train Directory: Paths and Dependencies Analysis

## Overview
This document lists all paths and dependencies for each file in the `train/` directory.

---

## 1. train_llama_il.py

### 1.1 Library Dependencies (Python Packages)

#### Standard Library
- `json` - JSON file handling
- `os` - Operating system interface (path operations)
- `re` - Regular expressions
- `csv` - CSV file handling
- `gc` - Garbage collection
- `math` - Mathematical operations
- `time` - Time-related functions
- `fnmatch` - Filename matching
- `typing` - Type hints (Any, Dict, List, Optional, Tuple)
- `collections` - Counter, defaultdict
- `traceback` - Exception tracebacks
- `argparse` - Command-line argument parsing

#### Third-Party Libraries
- `numpy` (as `np`) - Numerical computing
- `torch` - PyTorch deep learning framework
- `torch.nn` (as `nn`) - Neural network modules
- `torch.utils.data` - Dataset and DataLoader
- `transformers` - Hugging Face Transformers library
  - `AutoTokenizer`
  - `AutoModelForCausalLM`
  - `default_data_collator`
  - `get_linear_schedule_with_warmup`
  - `StoppingCriteria`
  - `StoppingCriteriaList`
- `peft` - Parameter-Efficient Fine-Tuning
  - `LoraConfig`
  - `get_peft_model`
  - `TaskType`
  - `PeftModel`

### 1.2 File Dependencies (Local Files)

#### Required Input Files
1. **Training Data Files:**
   - `training_data/vanilla_trajectories_common.json` (via `VANILLA_DATA_PATH`)
   - `training_data/safe_trajectories_common.json` (via `SAFE_DATA_PATH`)

2. **Task Configuration Files (optional, used for fallback):**
   - `stwebagentbench/test.augmented.json` (relative to script directory)
   - `stwebagentbench/test.raw.json` (relative to script directory)

#### Generated Output Files
1. **Model Checkpoints:**
   - `train/checkpoints/llama_il_lora_webarena/{save_dir}/adapter_model.safetensors`
   - `train/checkpoints/llama_il_lora_webarena/{save_dir}/adapter_config.json`
   - `train/checkpoints/llama_il_lora_webarena/{save_dir}/best_checkpoint.json`
   - `train/checkpoints/llama_il_lora_webarena/{save_dir}/best_intra_checkpoint.json` (if intra_evaluate)

2. **Debug/Evaluation Files:**
   - `{save_dir}/debug_prompts.json`
   - `{save_dir}/metrics_{safe_data_str}_seed{seed}.csv`
   - `{save_dir}/final_comparison.json`

### 1.3 Paths Used

#### Absolute Paths (Hardcoded)
1. **Model Paths:**
   - `/home/fodl/tomerslor/safe-control/Llama-3.2-1B-Instruct` (MODEL_PATH_1B)
   - `/home/fodl/tomerslor/safe-control/Llama-3.1-8B-Instruct` (MODEL_PATH_8B)

2. **Training Data Directory:**
   - `/home/fodl/tomerslor/safe-control/WebArena/ST-WebAgentBench/training_data` (TRAINING_DATA_DIR)

#### Relative Paths
1. **Output Directory:**
   - `train/checkpoints/llama_il_lora_webarena` (OUTPUT_DIR - relative to train directory)

2. **Script-Relative Paths:**
   - `stwebagentbench/test.augmented.json` (relative to `__file__` directory)
   - `stwebagentbench/test.raw.json` (relative to `__file__` directory)

#### Dynamic Paths (Constructed at Runtime)
- `{save_dir}/adapter_config.json`
- `{save_dir}/best_checkpoint.json`
- `{save_dir}/best_intra_checkpoint.json`
- `{save_dir}/debug_prompts.json`
- `{save_dir}/metrics_{safe_data_str}_seed{seed}.csv`
- `{save_dir}/final_comparison.json`

Where `save_dir` is constructed as:
```
{OUTPUT_DIR}/{model_size}_{data}_seed{seed}_lora{r}_lr{lr}_bs{batch_size}_.../
```

### 1.4 Environment Variables
- `LLAMA_PATH` - Fallback model path (if set, overrides MODEL_PATH_1B default)

---

## 2. train_script.sh

### 2.1 Dependencies (Shell Commands/Executables)
- `bash` - Bash shell
- `screen` - Terminal multiplexer
- `conda` - Conda package manager
- `python` - Python interpreter
- `source` - Shell builtin for sourcing files
- `mkdir` - Create directories
- `grep` - Pattern matching
- `sed` - Stream editor
- `date` - Date/time formatting
- `printf` - Formatted output

### 2.2 File Dependencies
- `train_llama_il.py` - The Python training script (called from WORKDIR)

### 2.3 Paths Used

#### Absolute Paths (Hardcoded)
1. **Working Directory:**
   - `/home/fodl/yoavyosefn/WebArena/ST-WebAgentBench` (WORKDIR - default)

2. **Log Directory:**
   - `/home/fodl/yoavyosefn/WebArena/logs` (LOG_DIR - default)

3. **Conda/Bash Configuration:**
   - `/home/fodl/yoavyosefn/.bashrc`
   - `/home/fodl/yoavyosefn/miniconda3/etc/profile.d/conda.sh`

#### Dynamic Paths (Constructed at Runtime)
- `{LOG_DIR}/{SESSION_PREFIX}_{TIME_STAMP}.log` - Log file path
  - Example: `/home/fodl/yoavyosefn/WebArena/logs/train__20260129_104520.log`

### 2.4 Environment Variables
- `ENV_NAME` - Conda environment name (default: "webarena")
- `SESSION_PREFIX` - Screen session prefix (default: "train_")

### 2.5 Command-Line Arguments
- `--env NAME` - Override conda environment name
- `--workdir DIR` - Override working directory
- `--logdir DIR` - Override log directory
- `--prefix STR` - Override session prefix
- `--dry-run` - Print command without executing
- All remaining arguments passed to `train_llama_il.py`

---

## 3. multiple_trains.sh

### 3.1 Dependencies (Shell Commands/Executables)
- `bash` - Bash shell
- `./train_script.sh` - Calls the train_script.sh script

### 3.2 File Dependencies
- `train_script.sh` - The training script wrapper (in same directory)

### 3.3 Paths Used
- None (all paths are handled by `train_script.sh`)

### 3.4 Configuration Variables
- `SAMPLING_PAIRS` - Boolean flag for sampling pairs
- `SAMPLING_PAIRS_K` - Number of pairs to sample (default: 15)
- `MAX_HISTORY_ACTIONS` - Max history actions (default: 0)
- `SEEDS` - Array of random seeds (default: [45, 121])
- `LORA_R` - LoRA rank (default: 4)
- `BATCH_SIZE` - Batch size (default: 2)
- `GPU_ID` - Starting GPU ID (default: 4)

---

## 4. SUGGESTED_CHANGES.md

### 4.1 Dependencies
- None (markdown documentation file)

### 4.2 Paths Used
- None (documentation only)

### 4.3 File References (Mentioned in Documentation)
- `ST-WebAgentBench/train_llama_il.py` - Main training script (mentioned as file to modify)
- `logs/train__20260129_104520.log` - Example log file (mentioned as reference)
- `baseline_models/..._best_test_acc/adapter_model.safetensors` - Model checkpoint (mentioned as example)

---

## Summary: All Unique Paths Across All Files

### Absolute Paths
1. `/home/fodl/tomerslor/safe-control/Llama-3.2-1B-Instruct`
2. `/home/fodl/tomerslor/safe-control/Llama-3.1-8B-Instruct`
3. `/home/fodl/tomerslor/safe-control/WebArena/ST-WebAgentBench/training_data`
4. `/home/fodl/yoavyosefn/WebArena/ST-WebAgentBench`
5. `/home/fodl/yoavyosefn/WebArena/logs`
6. `/home/fodl/yoavyosefn/.bashrc`
7. `/home/fodl/yoavyosefn/miniconda3/etc/profile.d/conda.sh`

### Relative Paths
1. `train/checkpoints/llama_il_lora_webarena`
2. `stwebagentbench/test.augmented.json`
3. `stwebagentbench/test.raw.json`
4. `train_llama_il.py`

### File Patterns (Dynamic)
1. `{save_dir}/adapter_config.json`
2. `{save_dir}/adapter_model.safetensors`
3. `{save_dir}/best_checkpoint.json`
4. `{save_dir}/best_intra_checkpoint.json`
5. `{save_dir}/debug_prompts.json`
6. `{save_dir}/final_comparison.json`
7. `{save_dir}/metrics_{safe_data_str}_seed{seed}.csv`
8. `{LOG_DIR}/{SESSION_PREFIX}_{TIME_STAMP}.log`
9. `training_data/vanilla_trajectories_common.json`
10. `training_data/safe_trajectories_common.json`

---

## Summary: All Dependencies

### Python Libraries (train_llama_il.py)
- Standard Library: json, os, re, csv, gc, math, time, fnmatch, typing, collections, traceback, argparse
- Third-Party: numpy, torch, transformers, peft

### Shell Commands (train_script.sh, multiple_trains.sh)
- bash, screen, conda, python, source, mkdir, grep, sed, date, printf

### Local Files (Required)
- `training_data/vanilla_trajectories_common.json`
- `training_data/safe_trajectories_common.json`
- `stwebagentbench/test.augmented.json` (optional)
- `stwebagentbench/test.raw.json` (optional)
- `train_llama_il.py`

---

## Notes

1. **Path Inconsistencies:**
   - `train_script.sh` uses `/home/fodl/yoavyosefn/` paths
   - `train_llama_il.py` uses `/home/fodl/tomerslor/` paths
   - These should be aligned to the current workspace structure

2. **Relative vs Absolute:**
   - `train_llama_il.py` uses `__file__` to resolve relative paths for `stwebagentbench/` files
   - Output directory is relative to train directory (`train/checkpoints/...`)

3. **Dynamic Path Construction:**
   - Most output paths are constructed at runtime based on model size, data type, seed, and hyperparameters
   - Log paths include timestamps for uniqueness

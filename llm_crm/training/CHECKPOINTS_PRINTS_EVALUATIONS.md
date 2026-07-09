# Checkpoints, Prints, and Evaluations in train_llama_il.py

## 1. CHECKPOINTS SAVED

The script saves **4 types of checkpoints**:

### Directory Structure

```
train/
├── checkpoints/{experiment_name}/{seed}/{vanilla|safe}/
│   ├── epoch0_state/           # Per-epoch checkpoint
│   ├── epoch1_state/
│   ├── ...
│   ├── best_test_acc/          # Best test accuracy checkpoint
│   ├── best_intra_acc/         # Best validation accuracy (if --intra_evaluate)
│   └── final_comparison.json   # Comparison results
└── results/{seed}/
    └── metrics_{experiment_name}_{data}.csv  # Training metrics
```

### 1.1 Per-Epoch Checkpoints (NEW)
**Location**: `train/checkpoints/{experiment_name}/{seed}/{vanilla|safe}/epoch{N}_state/`
- **When**: Saved at the end of each epoch
- **Contains**:
  - `adapter_model.safetensors` - LoRA adapter weights at this epoch
  - `adapter_config.json` - LoRA configuration
  - `tokenizer_config.json` - Tokenizer configuration
  - `tokenizer.json` - Tokenizer files
  - `special_tokens_map.json` - Special tokens mapping

### 1.2 Final Model Checkpoint
**Location**: `train/checkpoints/{experiment_name}/{seed}/{vanilla|safe}/`
- **When**: After all training epochs complete
- **Contains**:
  - `adapter_model.safetensors` - Final LoRA adapter weights
  - `adapter_config.json` - LoRA configuration
  - `tokenizer_config.json` - Tokenizer configuration
  - `tokenizer.json` - Tokenizer files
  - `special_tokens_map.json` - Special tokens mapping
  - `final_comparison.json` - Comparison between FINAL and BEST models
  - `debug_prompts.json` - Debug output

### 1.3 Best Test Accuracy Checkpoint
**Location**: `train/checkpoints/{experiment_name}/{seed}/{vanilla|safe}/best_test_acc/`
- **When**: Saved during training whenever test accuracy improves
- **Trigger**: New highest `test_overall_accuracy` achieved
- **Contains**:
  - `adapter_model.safetensors` - Best LoRA adapter weights (by test accuracy)
  - `adapter_config.json` - LoRA configuration
  - `tokenizer_config.json` - Tokenizer configuration
  - `best_checkpoint.json` - Metadata:
    ```json
    {
      "best_epoch": 8,
      "best_test_overall_accuracy": 0.595
    }
    ```

### 1.4 Best Intra/Validation Accuracy Checkpoint (Optional)
**Location**: `train/checkpoints/{experiment_name}/{seed}/{vanilla|safe}/best_intra_acc/`
- **When**: Only if `--intra_evaluate` is enabled
- **Trigger**: Saved during training whenever validation accuracy improves
- **Contains**:
  - `adapter_model.safetensors` - Best LoRA adapter weights (by validation accuracy)
  - `adapter_config.json` - LoRA configuration
  - `tokenizer_config.json` - Tokenizer configuration
  - `best_checkpoint.json` - Metadata:
    ```json
    {
      "best_epoch": 6,
      "best_val_overall_accuracy": 0.612
    }
    ```

### 1.5 Results CSV (NEW)
**Location**: `train/results/{seed}/metrics_{experiment_name}_{data}.csv`
- **When**: After training completes
- **Contains**: Per-epoch metrics including train loss, test loss, train accuracy, test accuracy, and all granular metrics

---

## 2. PRINT STATEMENTS

### 2.1 Initialization & Configuration

```
============================================================
Configuration
============================================================
Model: /home/fodl/tomerslor/safe-control/Llama-3.2-1B-Instruct
Data: /home/fodl/tomerslor/safe-control/WebArena/ST-WebAgentBench/training_data/vanilla_trajectories_common.json
Seed: 42
Batch size: 4
Learning rate: 0.0001
LoRA r: 8, alpha: 8, dropout: 0.005
GPU ID: 2
Max history actions: 15
Error on truncation: False
============================================================
Checkpoints will be saved to: train/checkpoints/my_experiment/42/vanilla
Results will be saved to: train/results/42
```

**Policy-Aware Mode** (if `--policy_aware`):
```
[POLICY_AWARE] Enabled: injecting dataset safety policy into the system prompt.
[POLICY_AWARE] Loaded policy with 3 rules. Preview: P1: ... | P2: ... | P3: ... (+0 more)
```

### 2.2 Dataset Loading

```
[Dataset] Loaded 1234 samples from {data_path}
```

**Warnings**:
```
[WARNING] 5 samples missing meta_data.task_template_id (or intent_template_id fallback).
```

### 2.3 Data Splitting

**Template-based split**:
```
[Split Mode: template] OOD split by task_template_id
[WARNING] 3 samples still missing task_template_id after fallbacks; they will be excluded from the template-based split.
Total templates: 15
Train templates: 12, Test templates: 3
Split: train=850, test=200
```

**Random split**:
```
[Split Mode: random] In-distribution 80/20 split by sample
Total samples: 1050
Split: train=840, test=210
```

**Sampling** (if enabled):
```
[Sampling][Traj] template_id=2001: trajectories 5/10 (~50.0%)
[Sampling][Traj] template_id=2002: trajectories 5/8 (~62.5%)
...
[Sampling] Summary: kept 50/100 trajectories (50.0%)
```

**Intra-validation** (if `--intra_evaluate`):
```
[Intra-Val] Enabled: built VAL from unsampled trajectories within train templates (seed=42).
[Intra-Val] Cap: val_max_trajectories=3 per task_template_id
[Intra-Val] template_id=2001: total_traj=10 train_traj=5 val_traj=3
[Intra-Val] Resulting val samples (steps): 150
Split (with intra): train=850, val=150, test=200
```

### 2.4 Action Distribution

```
============================================================
Action Distribution: Train Set
============================================================
Total samples: 850
  click: 450 (52.9%)
  fill: 300 (35.3%)
  finish: 50 (5.9%)
  select_option: 50 (5.9%)
============================================================
```

### 2.5 Pre-Training Evaluation

```
======================================================================
PRE-TRAIN EVALUATION (before any training updates)
======================================================================
[Pre-Train] Train Loss: 3.2456, Test Loss: 3.3124
[Pre-Train] Val Loss (intra): 3.2891
[Pre-Train][TEST] Overall: 12.50%
[Pre-Train][TEST] Type correct: 25.00%
[Pre-Train][TEST] BID correct: 10.00%
[Pre-Train][VAL] Overall: 15.00%
[Pre-Train][TRAIN] Overall: 18.00%
```

### 2.6 Training Loop (Per Epoch)

**Epoch Start**:
```
[Epoch 0] Learning rate: 1.00e-04
```

**Training Progress** (every N steps):
```
Step 10/250 | Loss: 2.345 | 1.23s/microbatch | 2.45s/update_step | 0.41 update_steps/s | ETA: ~15.2 min
```

**Epoch End - Test Evaluation**:
```
[Epoch 0] Evaluating on TEST set...
[TEST] Click (full): 45.23% (95/210)
[TEST] Fill (full): 38.50% (77/200)
[TEST] Other (full): 60.00% (12/20)
[TEST] Overall: 42.55% (184/430)
[TEST] --- Granular Breakdown ---
[TEST] Type correct: 65.12% (280/430)
[TEST] BID correct (click/fill/select): 45.23% (172/380)
[TEST] Fill BID correct: 38.50% (77/200)
[TEST] Fill TEXT correct (given BID): 85.71% (66/77)
```

**Best Checkpoint Saved**:
```
[BEST_TEST] New best test overall accuracy 42.55% at epoch 0. Saving to {best_save_dir}
[BEST_INTRA] New best intra/val overall accuracy 48.20% at epoch 2. Saving to {best_intra_save_dir}
```

**Train Evaluation** (every epoch, max 200 samples):
```
[Epoch 3] Evaluating on TRAIN set (max 200 samples)...
[TRAIN] Overall: 65.50%
[TRAIN] Type correct: 78.20%
[TRAIN] BID correct: 60.30%
```

**Per-Epoch Checkpoint Saved**:
```
[Epoch 3] Saving checkpoint to train/checkpoints/{experiment_name}/{seed}/{data}/epoch3_state
```

**Epoch Summary**:
```
[Epoch 0] Total epoch time (approx): 125.3s (train+eval)
```

### 2.7 Training Complete

```
[Final] Test accuracy: 58.50%

Saving model to {save_dir}...
Metrics saved to {save_dir}/metrics_vanilla_seed42.csv
Training complete! Model saved to: {save_dir}

[BEST_TEST] Best test overall accuracy was 58.50% at epoch 8 (saved to {best_save_dir})
[BEST_INTRA] Best intra/val overall accuracy was 61.20% at epoch 6 (saved to {best_intra_save_dir})
```

### 2.8 Final Evaluation & Comparison

```
================================================================================
FINAL EVALUATION: Comparing Final Model vs Best Test Accuracy Model
================================================================================

[1/2] Evaluating FINAL model...

[FINAL] Evaluating on train set...
[FINAL] Evaluating on test set...
[FINAL] Evaluating on VAL set (intra)...

[2/2] Loading and evaluating BEST checkpoints...

[BEST_TEST] Evaluating on train set...
[BEST_TEST] Evaluating on test set...
[BEST_TEST] Evaluating on VAL set (intra)...

================================================================================
FINAL MODEL COMPARISON: Final State vs Best Test Accuracy State
================================================================================
Metric                          Final Model        Best Model
----------------------------------------------------------------------
Train Loss                           2.1234            2.0987
Test Loss                            2.3456            2.3012
Val Loss (intra)                     2.2891            2.2567
----------------------------------------------------------------------
--- Train Set ---
  Overall                           65.50%           66.20%
  Type correct                       78.20%           79.10%
  BID correct                        60.30%           61.50%
----------------------------------------------------------------------
--- Test Set ---
  Overall                           58.50%           59.20%
  Type correct                       70.30%           71.10%
  BID correct                        52.40%           53.20%
----------------------------------------------------------------------
--- Val Set (Intra) ---
  Overall                           61.20%           62.50%
  Type correct                       73.40%           74.20%
  BID correct                        55.60%           56.80%

Comparison results saved to: {save_dir}/final_comparison.json
```

**If best checkpoint not found**:
```
[WARNING] Best model checkpoint not found at {best_save_dir}, skipping comparison.

--- Final Model Results ---
Train Loss: 2.1234
Test Loss: 2.3456
Train Overall Accuracy: 65.50%
Test Overall Accuracy: 58.50%
Val Overall Accuracy (intra): 61.20%
```

### 2.9 Debug Output (if enabled)

```
================================================================================
DEBUG: Model Predictions vs Targets [TEST]
================================================================================

[0] ✓ Step step_5 | Goal: Create a new account...
    History (last 3): ['click("1234")', 'fill("5678", "John")', 'fill("5679", "Doe")']
    TARGET: click("9012")
    PRED:   click("9012")
    Parsed target: ('click', '9012', None)
    Parsed pred:   ('click', '9012', None)

[1] ✗ Step step_6 | Goal: Create a new account...
    History (last 4): ['click("1234")', 'fill("5678", "John")', 'fill("5679", "Doe")', 'click("9012")']
    TARGET: fill("3456", "john@example.com")
    PRED:   fill("3456", "jane@example.com")
    Parsed target: ('fill', '3456', 'john@example.com')
    Parsed pred:   ('fill', '3456', 'jane@example.com')

================================================================================

[DEBUG][TEST] Saved 50 full prompts to: {save_dir}/debug_prompts.json
```

---

## 3. EVALUATIONS PERFORMED

### 3.1 Pre-Training Evaluation (Baseline)

**When**: Before any training (epoch -1), unless `--skip_pretrain_evaluation`

**Evaluates**:
- **Train Loss**: Cross-entropy loss on training set
- **Test Loss**: Cross-entropy loss on test set
- **Val Loss** (if `--intra_evaluate`): Cross-entropy loss on validation set
- **Train Metrics**: Action accuracy metrics on training set (max 200 samples)
- **Test Metrics**: Action accuracy metrics on test set (max 200 samples)
- **Val Metrics** (if `--intra_evaluate`): Action accuracy metrics on validation set (max 200 samples)

**Purpose**: Establish baseline performance before training

### 3.2 Per-Epoch Evaluation (During Training)

**When**: At the end of each epoch

**Evaluates on Test Set**:
- **Test Loss**: Cross-entropy loss
- **Test Metrics**: Full action accuracy evaluation

**Evaluates on Validation Set** (if `--intra_evaluate`):
- **Val Loss**: Cross-entropy loss
- **Val Metrics**: Full action accuracy evaluation

**Evaluates on Train Set** (every epoch):
- **Train Metrics**: Action accuracy evaluation (max 200 samples for efficiency)
- Runs **every epoch** to track training progress

**Saves**:
- Per-epoch checkpoint to `train/checkpoints/{experiment_name}/{seed}/{data}/epoch{N}_state/`
- Best checkpoints when accuracy improves

**Purpose**: 
- Track progress during training
- Save best checkpoints based on test/val accuracy
- Detect overfitting
- Allow resuming from any epoch

### 3.3 Final Evaluation (After Training)

**When**: After all epochs complete

**Evaluates Final Model**:
- **Train Loss & Metrics**: Full evaluation on training set
- **Test Loss & Metrics**: Full evaluation on test set
- **Val Loss & Metrics** (if `--intra_evaluate`): Full evaluation on validation set

**Evaluates Best Test Model**:
- Loads best test accuracy checkpoint
- Re-evaluates on train/test/val sets
- Compares with final model

**Evaluates Best Intra Model** (if `--intra_evaluate`):
- Loads best validation accuracy checkpoint
- Re-evaluates on train/test/val sets
- Compares with final model

**Purpose**: 
- Compare final vs best checkpoints
- Determine which checkpoint performs best
- Generate comparison report

---

## 4. EVALUATION METRICS

Each evaluation computes the following metrics:

### 4.1 Loss Metrics
- **Train Loss**: Average cross-entropy loss on training set
- **Test Loss**: Average cross-entropy loss on test set
- **Val Loss**: Average cross-entropy loss on validation set (if applicable)

### 4.2 Action Accuracy Metrics

**Overall Metrics**:
- **Overall Accuracy**: Exact match of predicted action string
- **Type Accuracy**: Correctness of action type (click, fill, finish, etc.)

**Action-Specific Metrics**:
- **Click Accuracy**: Exact match for click actions
- **Fill Accuracy**: Exact match for fill actions
- **Other Accuracy**: Exact match for other actions (finish, select_option, etc.)

**Granular Metrics**:
- **BID Accuracy**: Correctness of UI element IDs (for click/fill/select_option)
- **Fill BID Accuracy**: Correctness of field IDs in fill actions
- **Fill Text Accuracy**: Correctness of fill text (given correct BID)

### 4.3 Evaluation Process

For each sample:
1. **Generate Prediction**: Model generates action string from prompt
2. **Parse Action**: Extract action type, BID, and text (if applicable)
3. **Compare**: Check exact match and component-wise correctness
4. **Aggregate**: Compute accuracy percentages per metric

**Sample Limit**: 
- Pre-training: Max 200 samples per split
- Final evaluation: All samples in each split
- Per-epoch: All samples in test/val sets

---

## 5. FILES GENERATED

### 5.1 Metrics CSV
**File**: `{save_dir}/metrics_{data}_seed{seed}.csv`

**Columns**:
- `epoch`: Epoch number (-1 for pre-train)
- `train_loss`: Training loss
- `test_loss`: Test loss
- `val_loss`: Validation loss (if applicable)
- `test_overall_accuracy`: Test overall accuracy
- `test_type_accuracy`: Test type accuracy
- `test_bid_accuracy`: Test BID accuracy
- `test_click_accuracy`: Test click accuracy
- `test_fill_accuracy`: Test fill accuracy
- ... (similar for train_* and val_* metrics)

### 5.2 Debug Prompts JSON
**File**: `{save_dir}/debug_prompts.json`

**Content**: Sample prompts, targets, predictions, and metadata for debugging

### 5.3 Final Comparison JSON
**File**: `{save_dir}/final_comparison.json`

**Content**: Comparison between FINAL, BEST_TEST, and BEST_INTRA models with all metrics

---

## Summary

- **4 Checkpoint Types**: Per-Epoch, Final, Best Test, Best Intra (optional)
- **New Directory Structure**: `train/checkpoints/{experiment_name}/{seed}/{data}/`
- **Results CSV**: `train/results/{seed}/metrics_{experiment_name}_{data}.csv`
- **Train Evaluation**: Every epoch (max 200 samples)
- **Comprehensive Prints**: Configuration, progress, metrics, comparisons
- **3 Evaluation Phases**: Pre-train, per-epoch, final comparison
- **10+ Metrics**: Loss, overall accuracy, type accuracy, BID accuracy, action-specific accuracies

---

## New Command Line Argument

### `--experiment_name`

**Type**: String  
**Default**: `"default_experiment"`  
**Description**: Name for the experiment. Used for organizing checkpoints and results.

**Example**:
```bash
python train_llama_il.py --experiment_name my_ablation_study --data vanilla --seed 42
```

This will create:
- Checkpoints: `train/checkpoints/my_ablation_study/42/vanilla/`
- Results: `train/results/42/metrics_my_ablation_study_vanilla.csv`

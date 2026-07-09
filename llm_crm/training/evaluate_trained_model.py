#!/usr/bin/env python3
"""
Evaluate a trained model checkpoint on train/val/test splits.

Usage:
    python evaluate_trained_model.py --experiment_name my_exp --seeds 1 2 3 --epoch 19 --data vanilla

Evaluations:
1. Full evaluation: accuracy metrics on the entire test set
2. Worst case evaluation: accuracy metrics per template, identifies worst performing template
"""

import argparse
import csv
import gc
import json
import math
import os
import fnmatch
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from train_llama_il
from train_llama_il import (
    WebArenaTrajectoryDataset,
    WebArenaTrajectorySubset,
    SYSTEM_PROMPT_BASE,
    evaluate_action_accuracy,
    get_model_path,
    get_data_path,
    build_prompt_and_target,
    parse_action,
)


def debug_model_weights(model, name="Model"):
    """Print statistics about model weights to verify they're loaded correctly."""
    print(f"\n[DEBUG] {name} weight statistics:")
    
    # Check if LoRA adapters are loaded
    lora_params = []
    for n, p in model.named_parameters():
        if "lora" in n.lower():
            lora_params.append((n, p))
    
    if lora_params:
        print(f"  Found {len(lora_params)} LoRA parameters")
        for n, p in lora_params[:5]:  # Show first 5
            print(f"    {n}: shape={p.shape}, mean={p.data.mean().item():.6f}, std={p.data.std().item():.6f}, requires_grad={p.requires_grad}")
    else:
        print("  WARNING: No LoRA parameters found!")
    
    # Check total trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")


def debug_single_prediction(model, tokenizer, sample, max_length=8192, max_history_actions=15):
    """Run a single prediction and show detailed output."""
    device = next(model.parameters()).device
    
    goal = sample["goal"]
    state = sample["state"]
    action = sample["action"]
    action_history = sample.get("action_history", [])
    
    prompt, target = build_prompt_and_target(
        goal=goal,
        state=state,
        action=action,
        action_history=action_history,
        system_prompt=SYSTEM_PROMPT_BASE,
        max_history_actions=max_history_actions,
    )
    
    print(f"\n[DEBUG] Single prediction test:")
    print(f"  Goal: {goal[:100]}...")
    print(f"  Target action: {target}")
    print(f"  Action history length: {len(action_history)}")
    
    # Tokenize
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    bos = torch.tensor([[tokenizer.bos_token_id]], device=device)
    input_ids = torch.cat([bos, enc["input_ids"].to(device)], dim=1)
    attention_mask = torch.cat(
        [torch.ones_like(bos), enc["attention_mask"].to(device)],
        dim=1,
    )
    
    print(f"  Input tokens: {input_ids.shape[1]}")
    
    # Truncate if needed
    if max_length and input_ids.shape[1] > max_length:
        input_ids = input_ids[:, -max_length:]
        attention_mask = attention_mask[:, -max_length:]
        print(f"  Truncated to: {input_ids.shape[1]}")
    
    # Generate
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id],
            pad_token_id=tokenizer.pad_token_id,
        )
    
    start = input_ids.shape[1]
    new_ids = out[0, start:]
    predicted = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    
    print(f"  Predicted: {predicted}")
    print(f"  Raw token IDs generated: {new_ids.tolist()[:20]}...")
    
    # Parse and compare
    pred_name, pred_arg1, pred_arg2 = parse_action(predicted)
    target_name, target_arg1, target_arg2 = parse_action(target)
    
    print(f"  Parsed target: ({target_name}, {target_arg1}, {target_arg2})")
    print(f"  Parsed pred:   ({pred_name}, {pred_arg1}, {pred_arg2})")
    
    # Check match
    is_match = (pred_name == target_name and pred_arg1 == target_arg1)
    if target_name == "fill":
        is_match = is_match and (pred_arg2 == target_arg2)
    print(f"  Match: {is_match}")
    
    return predicted, target, is_match


def debug_dataset_samples(dataset, num_samples=3):
    """Print sample information to verify dataset is correct."""
    print(f"\n[DEBUG] Dataset sample check (first {num_samples} samples):")
    print(f"  Total samples: {len(dataset.samples)}")
    
    for i, sample in enumerate(dataset.samples[:num_samples]):
        print(f"\n  Sample {i}:")
        print(f"    task_id: {sample.get('task_id')}")
        print(f"    task_template_id: {sample.get('task_template_id')}")
        print(f"    goal: {sample.get('goal', '')[:80]}...")
        print(f"    action: {sample.get('action')}")
        print(f"    history_len: {len(sample.get('action_history', []))}")


def evaluate_per_template(
    model,
    tokenizer,
    dataset: WebArenaTrajectoryDataset,
    max_samples_per_template: int = None,
    max_new_tokens: int = 64,
) -> Dict[str, Any]:
    """
    Evaluate action prediction per template (worst case analysis).
    Uses the same evaluate_action_accuracy function but splits by template.
    
    Returns:
        Dictionary with per-template metrics and worst template info
    """
    # Group samples by template
    samples_by_template = defaultdict(list)
    indices_by_template = defaultdict(list)
    
    for i, sample in enumerate(dataset.samples):
        template_id = sample.get("task_template_id")
        if template_id is not None:
            samples_by_template[str(template_id)].append(sample)
            indices_by_template[str(template_id)].append(i)
    
    template_results = {}
    
    for template_id in sorted(samples_by_template.keys()):
        template_indices = indices_by_template[template_id]
        
        # Create a subset for this template.
        # IMPORTANT: `indices_by_template` are indices into *this* dataset's `samples`
        # (which is already the TEST subset). We must therefore build a subset over
        # `dataset` itself, NOT over `dataset.base`, otherwise indices would point to
        # the wrong examples and corrupt the evaluation.
        template_subset = WebArenaTrajectorySubset(dataset, template_indices)
        
        # Limit samples if requested
        max_samples = min(len(template_subset), max_samples_per_template) if max_samples_per_template else len(template_subset)
        
        # Use the standard evaluation function
        metrics = evaluate_action_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=template_subset,
            max_samples=max_samples,
            max_new_tokens=max_new_tokens,
            split=f"TEMPLATE_{template_id}",
            debug=False,
        )
        
        template_results[template_id] = {
            "overall_accuracy": metrics["overall_accuracy"],
            "click_accuracy": metrics["click_accuracy"],
            "fill_accuracy": metrics["fill_accuracy"],
            "other_accuracy": metrics["other_accuracy"],
            "type_accuracy": metrics["type_accuracy"],
            "bid_accuracy": metrics["bid_accuracy"],
            "total_samples": len(template_subset.samples[:max_samples]),
        }
        
        print(f"  Template {template_id}: overall_acc={metrics['overall_accuracy']*100:.2f}% ({len(template_subset.samples[:max_samples])} samples)")
    
    # Find worst performing template by overall accuracy
    worst_template = None
    worst_accuracy = float("inf")
    
    for template_id, metrics in template_results.items():
        if metrics["total_samples"] > 0:
            if metrics["overall_accuracy"] < worst_accuracy:
                worst_accuracy = metrics["overall_accuracy"]
                worst_template = template_id
    
    return {
        "template_results": template_results,
        "worst_template": {
            "template_id": worst_template,
            "overall_accuracy": worst_accuracy,
            "metrics": template_results.get(worst_template, {}),
        },
        "num_templates": len(template_results),
    }


def load_checkpoint_and_data(
    experiment_name: str,
    seed: int,
    epoch: int,
    data: str = "vanilla",
    model_size: str = "1B",
    gpu_id: int = 0,
    split_mode: str = "template",
    ignore_templates: List[int] = None,
    test_template_ids: List[str] = None,
    max_length: int = 8192,
    max_history_actions: int = 15,
) -> Tuple[Any, Any, Any, Any, Any]:
    """
    Load checkpoint and recreate the train/val/test splits.
    
    Returns:
        Tuple of (model, tokenizer, train_dataset, val_dataset, test_dataset)
    """
    # Paths
    train_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(
        train_dir, "checkpoints", experiment_name, str(seed), data, f"epoch{epoch}_state"
    )
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_dir}")
    
    print(f"Loading checkpoint from: {checkpoint_dir}")
    
    # Get paths
    model_path = get_model_path(model_size)
    data_path = get_data_path(data)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    # Load base model on CPU first, then load LoRA adapter, then move to GPU.
    # IMPORTANT: Must use float32 (not bfloat16) because bfloat16 causes LoRA weights
    # to be corrupted (become zeros) when moving to GPU with .to(device).
    device = f"cuda:{gpu_id}"
    print(f"Loading base model from {model_path} with float32 (will move to GPU {gpu_id} after loading LoRA)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA adapter on CPU
    print(f"Loading LoRA adapter from {checkpoint_dir}...")
    
    # Check what files exist in checkpoint
    print(f"[DEBUG] Checkpoint directory contents:")
    for f in os.listdir(checkpoint_dir):
        fpath = os.path.join(checkpoint_dir, f)
        size = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
        print(f"  {f}: {size/1024:.1f} KB" if size > 0 else f"  {f}/")
    
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    
    # Move to GPU after loading LoRA adapter
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    # Debug: print LoRA weight statistics
    debug_model_weights(model, "Loaded model with LoRA")
    
    # Configure generation
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        # Don't set temperature to 0.0 as it causes warnings
        model.generation_config.top_p = 1.0
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    
    # Load dataset
    full_dataset = WebArenaTrajectoryDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_history_actions=max_history_actions,
        error_on_truncation=False,
        filter_successful=True,
        ignore_templates=ignore_templates,
        system_prompt=SYSTEM_PROMPT_BASE,
    )
    
    # Recreate the train/test split using the same seed
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_indices = []
    test_indices = []
    
    if split_mode == "random":
        n_samples = len(full_dataset.samples)
        perm = torch.randperm(n_samples, generator=g).tolist()
        n_test = max(1, int(0.2 * n_samples))
        test_indices = perm[:n_test]
        train_indices = perm[n_test:]
    else:
        # Template-based split
        all_template_ids = sorted(
            {s.get("task_template_id") for s in full_dataset.samples if s.get("task_template_id") is not None},
            key=lambda x: str(x),
        )
        
        if test_template_ids is not None and len(test_template_ids) > 0:
            available_by_str = {str(tid): tid for tid in all_template_ids}
            available_strs = sorted(available_by_str.keys())
            
            selectors = [str(x).strip() for x in test_template_ids if str(x).strip()]
            matched_strs = []
            for sel in selectors:
                if any(ch in sel for ch in ["*", "?", "[", "]"]):
                    matched_strs.extend([t for t in available_strs if fnmatch.fnmatch(t, sel)])
                else:
                    matched_strs.append(sel)
            
            test_template_ids_set = {available_by_str[s] for s in set(matched_strs) if s in available_by_str}
            train_template_ids = set(all_template_ids) - test_template_ids_set
        else:
            perm = torch.randperm(len(all_template_ids), generator=g).tolist()
            num_test_templates = math.ceil(len(all_template_ids) * 0.2)
            test_template_ids_set = {all_template_ids[i] for i in perm[:num_test_templates]}
            train_template_ids = {all_template_ids[i] for i in perm[num_test_templates:]}
        
        train_indices = [
            i for i, s in enumerate(full_dataset.samples)
            if s.get("task_template_id") in train_template_ids
        ]
        test_indices = [
            i for i, s in enumerate(full_dataset.samples)
            if s.get("task_template_id") in test_template_ids_set
        ]
        
        print(f"Train templates: {sorted(train_template_ids)}")
        print(f"Test templates: {sorted(test_template_ids_set)}")
    
    train_dataset = WebArenaTrajectorySubset(full_dataset, train_indices)
    test_dataset = WebArenaTrajectorySubset(full_dataset, test_indices)
    val_dataset = None  # For simplicity, not recreating val split here
    
    print(f"Loaded datasets: train={len(train_dataset)}, test={len(test_dataset)}")
    
    return model, tokenizer, train_dataset, val_dataset, test_dataset


def save_full_evaluation_csv(results: Dict[str, Any], output_path: str):
    """Save full evaluation results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    row = {
        "overall_accuracy": results["overall_accuracy"],
        "click_accuracy": results["click_accuracy"],
        "fill_accuracy": results["fill_accuracy"],
        "other_accuracy": results["other_accuracy"],
        "type_accuracy": results["type_accuracy"],
        "bid_accuracy": results["bid_accuracy"],
        "fill_bid_accuracy": results["fill_bid_accuracy"],
        "fill_text_accuracy": results["fill_text_accuracy"],
    }
    
    with open(output_path, "w", newline="") as f:
        fieldnames = list(row.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    
    print(f"Full evaluation saved to: {output_path}")


def save_worst_case_evaluation_csv(results: Dict[str, Any], output_path: str):
    """Save worst case (per-template) evaluation results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    rows = []
    
    # Per-template metrics
    for template_id, metrics in sorted(results["template_results"].items()):
        is_worst = (template_id == results["worst_template"]["template_id"])
        
        rows.append({
            "template_id": template_id,
            "overall_accuracy": metrics["overall_accuracy"],
            "click_accuracy": metrics["click_accuracy"],
            "fill_accuracy": metrics["fill_accuracy"],
            "other_accuracy": metrics["other_accuracy"],
            "type_accuracy": metrics["type_accuracy"],
            "bid_accuracy": metrics["bid_accuracy"],
            "total_samples": metrics["total_samples"],
            "is_worst": is_worst,
        })
    
    # Add summary row for worst template
    worst = results["worst_template"]
    rows.append({
        "template_id": f"WORST: {worst['template_id']}",
        "overall_accuracy": worst["overall_accuracy"],
        "click_accuracy": worst["metrics"].get("click_accuracy", ""),
        "fill_accuracy": worst["metrics"].get("fill_accuracy", ""),
        "other_accuracy": worst["metrics"].get("other_accuracy", ""),
        "type_accuracy": worst["metrics"].get("type_accuracy", ""),
        "bid_accuracy": worst["metrics"].get("bid_accuracy", ""),
        "total_samples": worst["metrics"].get("total_samples", ""),
        "is_worst": True,
    })
    
    with open(output_path, "w", newline="") as f:
        fieldnames = ["template_id", "overall_accuracy", "click_accuracy", "fill_accuracy", "other_accuracy", "type_accuracy", "bid_accuracy", "total_samples", "is_worst"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Worst case evaluation saved to: {output_path}")


def evaluate_single_seed(
    experiment_name: str,
    seed: int,
    epoch: int,
    data: str,
    model_size: str,
    gpu_id: int,
    split_mode: str,
    ignore_templates: List[int],
    test_template_ids: List[str],
    max_length: int,
    max_history_actions: int,
    max_eval_samples: int,
    skip_full: bool,
    skip_worst_case: bool,
    batch_size: int,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run evaluation for a single seed."""
    
    print("\n" + "=" * 80)
    print(f"EVALUATING SEED {seed}")
    print("=" * 80)
    
    # Load model and data
    model, tokenizer, train_dataset, val_dataset, test_dataset = load_checkpoint_and_data(
        experiment_name=experiment_name,
        seed=seed,
        epoch=epoch,
        data=data,
        model_size=model_size,
        gpu_id=gpu_id,
        split_mode=split_mode,
        ignore_templates=ignore_templates,
        test_template_ids=test_template_ids,
        max_length=max_length,
        max_history_actions=max_history_actions,
    )
    
    # Debug: print dataset info
    if debug:
        debug_dataset_samples(test_dataset, num_samples=3)
    
    # Debug: run a few single predictions to verify model works
    if debug:
        print("\n" + "-" * 60)
        print("DEBUG: Running manual predictions on first 5 test samples")
        print("-" * 60)
        for i in range(min(5, len(test_dataset.samples))):
            debug_single_prediction(
                model, tokenizer, test_dataset.samples[i],
                max_length=max_length, max_history_actions=max_history_actions
            )
    
    # Results directory
    train_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(train_dir, "results", str(seed))
    os.makedirs(results_dir, exist_ok=True)
    
    results = {"seed": seed}
    
    # Run evaluations
    if not skip_full:
        print("\n" + "-" * 60)
        print("FULL EVALUATION ON TEST SET")
        print("-" * 60)
        
        max_samples = max_eval_samples if max_eval_samples else len(test_dataset)
        
        # Save debug output path
        debug_save_path = os.path.join(results_dir, f"debug_prompts_{experiment_name}_{data}_epoch{epoch}.json") if debug else None
        
        full_results = evaluate_action_accuracy(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            max_samples=max_samples,
            split="TEST",
            debug=debug,
            debug_samples=30 if debug else 0,
            debug_save_path=debug_save_path or "debug_prompts.json",
        )
        
        print(f"\n--- Full Evaluation Results (Seed {seed}) ---")
        print(f"Overall Accuracy: {full_results['overall_accuracy']*100:.2f}%")
        print(f"Click Accuracy:   {full_results['click_accuracy']*100:.2f}%")
        print(f"Fill Accuracy:    {full_results['fill_accuracy']*100:.2f}%")
        print(f"Type Accuracy:    {full_results['type_accuracy']*100:.2f}%")
        print(f"BID Accuracy:     {full_results['bid_accuracy']*100:.2f}%")
        
        # Save CSV
        full_csv_path = os.path.join(results_dir, f"full_evaluation_{experiment_name}_{data}_epoch{epoch}.csv")
        save_full_evaluation_csv(full_results, full_csv_path)
        
        results["full"] = full_results
    
    if not skip_worst_case:
        print("\n" + "-" * 60)
        print("WORST CASE EVALUATION (PER TEMPLATE)")
        print("-" * 60)
        
        worst_case_results = evaluate_per_template(
            model=model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            max_samples_per_template=max_eval_samples,
        )
        
        print(f"\n--- Worst Case Results (Seed {seed}) ---")
        print(f"Number of templates evaluated: {worst_case_results['num_templates']}")
        
        worst = worst_case_results["worst_template"]
        print(f"\nWorst Template: {worst['template_id']}")
        print(f"  Overall Accuracy: {worst['overall_accuracy']*100:.2f}%")
        print(f"  Samples: {worst['metrics'].get('total_samples', 'N/A')}")
        
        # Save CSV
        worst_csv_path = os.path.join(results_dir, f"worst_case_evaluation_{experiment_name}_{data}_epoch{epoch}.csv")
        save_worst_case_evaluation_csv(worst_case_results, worst_csv_path)
        
        results["worst_case"] = worst_case_results
    
    # Cleanup
    del model
    del tokenizer
    del train_dataset
    del test_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model checkpoint")
    parser.add_argument("--experiment_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--seeds", type=int, nargs="+", required=True, help="Random seed(s) used for training")
    parser.add_argument("--epoch", type=int, default=19, help="Epoch of checkpoint to load (default: 19)")
    parser.add_argument("--data", type=str, default="vanilla", choices=["vanilla", "safe"], help="Data type")
    parser.add_argument("--model", type=str, default="1B", choices=["1B", "8B"], help="Model size")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation (currently used for data loading)")
    parser.add_argument("--split_mode", type=str, default="template", choices=["template", "random"], help="Split mode")
    parser.add_argument("--ignore_templates", type=int, nargs="+", default=[2002, 2007], help="Template IDs to ignore")
    parser.add_argument("--test_template_ids", type=str, nargs="+", default=None, help="Fixed test template IDs")
    parser.add_argument("--max_length", type=int, default=8192, help="Maximum sequence length")
    parser.add_argument("--max_history_actions", type=int, default=15, help="Max history actions")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Max samples to evaluate (None = all)")
    parser.add_argument("--skip_full", action="store_true", help="Skip full evaluation")
    parser.add_argument("--skip_worst_case", action="store_true", help="Skip worst case evaluation")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode: show model weights, sample predictions")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("EVALUATE TRAINED MODEL")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Seeds: {args.seeds}")
    print(f"Epoch: {args.epoch}")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    print(f"GPU: {args.gpu}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    
    all_results = []
    
    for seed in args.seeds:
        try:
            results = evaluate_single_seed(
                experiment_name=args.experiment_name,
                seed=seed,
                epoch=args.epoch,
                data=args.data,
                model_size=args.model,
                gpu_id=args.gpu,
                split_mode=args.split_mode,
                ignore_templates=args.ignore_templates,
                test_template_ids=args.test_template_ids,
                max_length=args.max_length,
                max_history_actions=args.max_history_actions,
                max_eval_samples=args.max_eval_samples,
                skip_full=args.skip_full,
                skip_worst_case=args.skip_worst_case,
                batch_size=args.batch_size,
                debug=args.debug,
            )
            all_results.append(results)
        except Exception as e:
            print(f"\n[ERROR] Seed {seed} failed: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
    
    # Print summary across all seeds
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL SEEDS")
        print("=" * 80)
        
        if not args.skip_full:
            accuracies = [r["full"]["overall_accuracy"] for r in all_results if "full" in r]
            if accuracies:
                import numpy as np
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f"\nFull Evaluation - Overall Accuracy:")
                print(f"  Seeds: {[r['seed'] for r in all_results if 'full' in r]}")
                print(f"  Accuracies: {[f'{a*100:.2f}%' for a in accuracies]}")
                print(f"  Mean: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
        
        if not args.skip_worst_case:
            worst_accs = [r["worst_case"]["worst_template"]["overall_accuracy"] for r in all_results if "worst_case" in r]
            worst_templates = [r["worst_case"]["worst_template"]["template_id"] for r in all_results if "worst_case" in r]
            if worst_accs:
                import numpy as np
                mean_worst = np.mean(worst_accs)
                std_worst = np.std(worst_accs)
                print(f"\nWorst Case Evaluation:")
                print(f"  Seeds: {[r['seed'] for r in all_results if 'worst_case' in r]}")
                print(f"  Worst Templates: {worst_templates}")
                print(f"  Worst Accuracies: {[f'{a*100:.2f}%' for a in worst_accs]}")
                print(f"  Mean Worst: {mean_worst*100:.2f}% ± {std_worst*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

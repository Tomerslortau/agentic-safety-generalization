#!/usr/bin/env python3
"""
Reproduce the seed-driven train/val split that --intra_evaluate --val_max_trajectories 2
would have produced, and evaluate a saved LoRA checkpoint on that VAL split.

Why: we trained seed 9 (and others) without --intra_evaluate, so the metrics CSV has
no val_overall_accuracy. The split is fully determined by --seed, so we can recompute
which 2 trajectories per template would have been the val set, load the saved LoRA
adapter for any epoch, and compute val accuracy offline.

Usage:
    python evaluate_val_split.py --seed 9 --data vanilla --epoch 19 --gpu 6
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from train_llama_il import (
    WebArenaTrajectoryDataset,
    WebArenaTrajectorySubset,
    SYSTEM_PROMPT_BASE,
    evaluate_action_accuracy,
    get_model_path,
    get_data_path,
)


def reconstruct_split_indices(
    full_dataset: WebArenaTrajectoryDataset,
    seed: int,
    sampling_trajectories_k: int = 7,
    val_max_trajectories: int = 2,
) -> Tuple[List[int], List[int], List[int], Dict[str, Tuple[int, int, int]]]:
    """
    Replicate train_llama_il.py's split logic exactly:
      - g = Generator(seed) -> template split (20% test, 80% train).
      - g_traj = Generator(seed + 10007) -> per template, randperm trajectories,
        keep first min(7, n).
      - g_val = Generator(seed + 10009) -> per train template, randperm of REMAINING,
        keep first min(2, n_remaining).

    Returns: train_indices, val_indices, test_indices, per_template_summary
    where per_template_summary[tid] = (n_total, n_train, n_val).
    """
    samples = full_dataset.samples

    all_template_ids = sorted(
        {s.get("task_template_id") for s in samples if s.get("task_template_id") is not None},
        key=lambda x: str(x),
    )

    # 1) Template split (matches train_llama_il.py:1773-1851)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(len(all_template_ids), generator=g).tolist()
    num_test_templates = math.ceil(len(all_template_ids) * 0.2)
    test_template_ids: Set[Any] = {all_template_ids[i] for i in perm[:num_test_templates]}
    train_template_ids: Set[Any] = {all_template_ids[i] for i in perm[num_test_templates:]}

    train_indices_pre = [
        i for i, s in enumerate(samples)
        if s.get("task_template_id") in train_template_ids
    ]
    test_indices = [
        i for i, s in enumerate(samples)
        if s.get("task_template_id") in test_template_ids
    ]

    # 2) Trajectory sampling for train (matches train_llama_il.py:1900-1949)
    by_template_traj: Dict[str, Dict[Tuple[str, str], List[int]]] = defaultdict(lambda: defaultdict(list))
    for i in train_indices_pre:
        s = samples[i]
        tid = str(s.get("task_template_id"))
        traj_key = (str(s.get("task_id")), str(s.get("trajectory_num")))
        by_template_traj[tid][traj_key].append(i)

    g_traj = torch.Generator()
    g_traj.manual_seed(int(seed) + 10007)

    train_indices: List[int] = []
    chosen_by_template: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for tid in sorted(by_template_traj.keys()):
        traj_keys = sorted(by_template_traj[tid].keys())
        n_traj = len(traj_keys)
        if n_traj == 0:
            continue
        k = min(sampling_trajectories_k, n_traj)
        perm_t = torch.randperm(n_traj, generator=g_traj).tolist()
        chosen = [traj_keys[j] for j in perm_t[:k]]
        for tk in chosen:
            train_indices.extend(by_template_traj[tid][tk])
            chosen_by_template[tid].add(tk)

    train_indices = sorted(set(train_indices))

    # 3) Val sampling from remaining (matches train_llama_il.py:2046-2073)
    g_val = torch.Generator()
    g_val.manual_seed(int(seed) + 10009)

    val_indices: List[int] = []
    per_template_summary: Dict[str, Tuple[int, int, int]] = {}
    for tid in sorted(by_template_traj.keys()):
        all_keys = sorted(by_template_traj[tid].keys())
        chosen = chosen_by_template.get(tid, set())
        remaining = [tk for tk in all_keys if tk not in chosen]
        n_all = len(all_keys)
        n_chosen = len(chosen)
        n_remaining = len(remaining)
        if n_remaining <= 0:
            per_template_summary[tid] = (n_all, n_chosen, 0)
            continue
        k_val = min(val_max_trajectories, n_remaining) if val_max_trajectories > 0 else n_remaining
        perm_v = torch.randperm(n_remaining, generator=g_val).tolist()
        kept = [remaining[j] for j in perm_v[:k_val]]
        for tk in kept:
            val_indices.extend(by_template_traj[tid][tk])
        per_template_summary[tid] = (n_all, n_chosen, len(kept))

    val_indices = sorted(set(val_indices))
    return train_indices, val_indices, test_indices, per_template_summary


def load_model(checkpoint_dir: str, model_size: str, gpu_id: int):
    model_path = get_model_path(model_size)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    device = f"cuda:{gpu_id}"
    print(f"[load] base from {model_path} (float32, then move to {device})")
    base = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    print(f"[load] LoRA from {checkpoint_dir}")
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    model = model.to(device)
    model.eval()

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.do_sample = False
        model.generation_config.top_p = 1.0
        model.generation_config.use_cache = True
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    return model, tokenizer, device


def evaluate_per_template(model, tokenizer, val_dataset: WebArenaTrajectorySubset, split: str = "VAL"):
    """Per-template breakdown of overall accuracy on the val subset."""
    by_tid: Dict[str, List[int]] = defaultdict(list)
    for local_idx, sample in enumerate(val_dataset.samples):
        tid = str(sample.get("task_template_id"))
        by_tid[tid].append(local_idx)

    rows = []
    for tid in sorted(by_tid.keys()):
        local_indices = by_tid[tid]
        # Build a sub-subset over the same base
        global_indices = [val_dataset.indices[li] for li in local_indices]
        sub = WebArenaTrajectorySubset(val_dataset.base, global_indices)
        m = evaluate_action_accuracy(
            model, tokenizer, sub,
            max_samples=len(sub),
            split=f"{split}/{tid}",
            debug=False,
        )
        rows.append({
            "template_id": tid,
            "overall_accuracy": m.get("overall_accuracy", 0.0),
            "click_accuracy":   m.get("click_accuracy", 0.0),
            "fill_accuracy":    m.get("fill_accuracy", 0.0),
            "other_accuracy":   m.get("other_accuracy", 0.0),
            "type_accuracy":    m.get("type_accuracy", 0.0),
            "bid_accuracy":     m.get("bid_accuracy", 0.0),
            "total_samples":    len(sub),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--data", choices=["vanilla", "safe"], required=True)
    p.add_argument("--epoch", type=int, default=19)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--experiment_name", type=str, default="7_trajectories_train")
    p.add_argument("--model_size", type=str, default="1B")
    p.add_argument("--sampling_trajectories_k", type=int, default=7)
    p.add_argument("--val_max_trajectories", type=int, default=2)
    p.add_argument("--max_length", type=int, default=8192)
    p.add_argument("--max_history_actions", type=int, default=15)
    p.add_argument("--out_dir", type=str, default=None,
                   help="Defaults to train/results/{seed}/")
    args = p.parse_args()

    train_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(
        train_dir, "checkpoints", args.experiment_name, str(args.seed),
        args.data, f"epoch{args.epoch}_state",
    )
    if not os.path.isdir(ckpt_dir):
        sys.exit(f"checkpoint not found: {ckpt_dir}")

    out_dir = args.out_dir or os.path.join(train_dir, "results", str(args.seed))
    os.makedirs(out_dir, exist_ok=True)

    # Load model + tokenizer + dataset
    model, tokenizer, device = load_model(ckpt_dir, args.model_size, args.gpu)
    data_path = get_data_path(args.data)
    full_dataset = WebArenaTrajectoryDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_history_actions=args.max_history_actions,
        error_on_truncation=False,
        filter_successful=True,
        ignore_templates=None,
        system_prompt=SYSTEM_PROMPT_BASE,
    )

    # Reconstruct splits
    train_indices, val_indices, test_indices, summary = reconstruct_split_indices(
        full_dataset, args.seed,
        sampling_trajectories_k=args.sampling_trajectories_k,
        val_max_trajectories=args.val_max_trajectories,
    )
    print(f"[split] seed={args.seed} train_steps={len(train_indices)} "
          f"val_steps={len(val_indices)} test_steps={len(test_indices)}")
    for tid in sorted(summary):
        n_all, n_chosen, n_val = summary[tid]
        print(f"  template_id={tid}: total={n_all} train_traj={n_chosen} val_traj={n_val}")

    if not val_indices:
        sys.exit("[ERROR] val_indices is empty; nothing to evaluate.")

    val_dataset = WebArenaTrajectorySubset(full_dataset, val_indices)
    print(f"[eval] running evaluate_action_accuracy on val_dataset (n={len(val_dataset)})")
    overall = evaluate_action_accuracy(
        model, tokenizer, val_dataset,
        max_samples=len(val_dataset),
        split="VAL",
        debug=False,
    )

    # Save overall CSV
    overall_path = os.path.join(
        out_dir,
        f"val_eval_{args.experiment_name}_{args.data}_epoch{args.epoch}.csv",
    )
    keys = sorted(overall.keys())
    with open(overall_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "data", "epoch", "n_val_steps"] + keys)
        w.writerow([args.seed, args.data, args.epoch, len(val_dataset)] + [overall[k] for k in keys])
    print(f"[out] overall: {overall_path}")

    # Per-template
    per_tid_rows = evaluate_per_template(model, tokenizer, val_dataset, split="VAL")
    per_tid_path = os.path.join(
        out_dir,
        f"val_per_template_{args.experiment_name}_{args.data}_epoch{args.epoch}.csv",
    )
    with open(per_tid_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "template_id", "overall_accuracy", "click_accuracy", "fill_accuracy",
            "other_accuracy", "type_accuracy", "bid_accuracy", "total_samples",
        ])
        w.writeheader()
        w.writerows(per_tid_rows)
    print(f"[out] per-template: {per_tid_path}")

    # Print headline
    print(f"\n=== seed={args.seed} {args.data} epoch{args.epoch} ===")
    print(f"  val_overall_accuracy = {overall.get('overall_accuracy', 0.0):.4f} (n={len(val_dataset)})")


if __name__ == "__main__":
    main()

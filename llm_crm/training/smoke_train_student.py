"""Weights-required training+eval smoke for the LLM student.

Fine-tunes the LLaMA student with LoRA for a few steps on the committed sample
trajectories (using the REAL `WebArenaTrajectoryDataset` and LoRA setup from
`train_llama_il.py`), then greedy-decodes one example and reports exact match.

This is NOT part of the offline `test_smoke.py` (which needs no weights). It requires
`transformers`, `peft`, and local LLaMA weights, and skips cleanly if any are missing.
It verifies that the actual student training + decoding path runs end to end; it does
NOT aim for accuracy (a few steps on a couple of samples will not match the target).

Usage:
    LLAMA_PATH=/path/to/Llama-3.2-1B-Instruct python smoke_train_student.py
"""

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) or '.'
sys.path.insert(0, HERE)
SAMPLE = os.path.join(HERE, '..', 'data_sample')


def main():
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        print(f"SKIP: missing dependency ({type(e).__name__}: {e}). "
              f"Install transformers + peft to run this smoke.")
        return

    model_path = (os.environ.get('LLAMA_PATH') or os.environ.get('LLAMA_1B_PATH')
                  or 'Llama-3.2-1B-Instruct')
    if not os.path.isdir(model_path):
        print(f"SKIP: LLaMA weights not found at '{model_path}'. Set LLAMA_PATH to a "
              f"local Llama-3.2-1B-Instruct checkpoint.")
        return

    from train_llama_il import WebArenaTrajectoryDataset, build_prompt_and_target

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}; model: {model_path}")

    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if device == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    lora = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.005,
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                      task_type='CAUSAL_LM')
    model = get_peft_model(model, lora).to(device)
    model.print_trainable_parameters()

    ds = WebArenaTrajectoryDataset(
        os.path.join(SAMPLE, 'safe_trajectories_common.json'),
        tok, max_length=1024, filter_successful=True)
    assert len(ds) > 0, "no training samples parsed from the sample trajectories"

    dl = DataLoader(ds, batch_size=1, shuffle=True, collate_fn=default_data_collator)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    # --- train a few steps ---
    model.train()
    losses = []
    for step, batch in enumerate(dl, start=1):
        inputs = {k: batch[k].to(device) for k in ('input_ids', 'attention_mask', 'labels')}
        out = model(**inputs)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
        print(f"train step {step}: loss={out.loss.item():.4f}")
        if step >= 3:
            break
    assert losses and all(math.isfinite(x) for x in losses), "non-finite training loss"

    # --- greedy-decode eval on one example ---
    s = ds.samples[0]
    prompt, target = build_prompt_and_target(
        s['goal'], s['state'], s['action'], action_history=s['action_history'])
    model.eval()
    ids = tok(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=32, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    pred = tok.decode(gen[0][ids['input_ids'].shape[1]:], skip_special_tokens=True).strip()
    exact = int(pred == target.strip())
    print(f"target : {target!r}")
    print(f"predict: {pred!r}")
    print(f"exact_match(after {len(losses)} steps) = {exact}")
    print(f"PASS: smoke_train_student (trained {len(losses)} LoRA steps + greedy-decode eval)")


if __name__ == '__main__':
    main()

"""Finite-sample (Q, x) -> u = -K*(Q) x policy imitation: LQR vs H-infinity.

The STUDENT (ResidualMLP) imitates the analytic teacher policy pi*(Q,x) = -K*(Q)x,
variant 'lqr' = unsafe teacher, 'hinf' = safe teacher. States are generated on the
fly per task. Reports normalized MSE on training tasks (train / in-task) and on
held-out tasks (cross-task). Produces the finite-sample linear rows of the table.

Self-contained (imports only the local ResidualMLP). Multi-GPU via
torch.multiprocessing; falls back to CPU/single-GPU automatically.
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import ResidualMLP

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    dimensions=[10, 15, 20],
    seeds=[14, 48, 99],
    train_frac=0.8,
    train_states_per_task=5,
    intask_test_states_per_task=2,
    data_root='data',
    hidden_dim=2048,
    num_layers=5,
    activation='relu',
    use_layernorm=False,
    lr=1e-5,
    min_lr=1e-7,
    batch_size=1024,
    max_epochs=3000,
    eval_every=25,
    patience=5,
    rel_tol=0.01,
    plateau_patience=30,
    plateau_factor=0.1,
    improvement_tol=1e-6,
    train_threshold=None,
)


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg, tag=""):
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}{msg}", flush=True)


# ── Dataset loading with on-the-fly state generation ─────────────────────────

def load_dataset(data_dir, n, train_states, intask_test_states, train_frac, state_seed=0, num_tasks=None):
    """Load gains from disk and generate states/actions on-the-fly.

    Returns (train, intask, crosstask) where each is (Q_flat, x, u_lqr, u_hinf).
    """
    tasks = np.load(os.path.join(data_dir, 'tasks.npz'))
    Q_all = tasks['Q']            # (num_tasks, n^2)
    K_lqr_all = tasks['K_lqr']    # (num_tasks, n, n)
    K_hinf_all = tasks['K_hinf']  # (num_tasks, n, n)

    total = len(Q_all)
    if num_tasks is not None and num_tasks < total:
        total = num_tasks
        Q_all = Q_all[:total]
        K_lqr_all = K_lqr_all[:total]
        K_hinf_all = K_hinf_all[:total]

    split = int(train_frac * total)
    train_idx = np.arange(split)
    cross_idx = np.arange(split, total)

    def _make_samples(task_indices, num_states, seed_offset):
        """Generate states and compute actions for a set of tasks."""
        Qs, xs, u_lqrs, u_hinfs = [], [], [], []
        for i in task_indices:
            rng = np.random.RandomState(state_seed + seed_offset + int(i))
            states = rng.normal(0, 1, (num_states, n))
            K_lqr = K_lqr_all[i]
            K_hinf = K_hinf_all[i]
            Q_flat = Q_all[i]

            # u = K x convention (sign baked into K from the solvers)
            u_lqr = (K_lqr @ states.T).T
            u_hinf = (K_hinf @ states.T).T

            Q_rep = np.tile(Q_flat, (num_states, 1))
            Qs.append(Q_rep)
            xs.append(states)
            u_lqrs.append(u_lqr)
            u_hinfs.append(u_hinf)

        return (np.concatenate(Qs), np.concatenate(xs),
                np.concatenate(u_lqrs), np.concatenate(u_hinfs))

    train_data = _make_samples(train_idx, train_states, seed_offset=0)
    intask_data = _make_samples(train_idx, intask_test_states, seed_offset=1_000_000)
    crosstask_data = _make_samples(cross_idx, train_states, seed_offset=2_000_000)

    return train_data, intask_data, crosstask_data


# ── Training ──────────────────────────────────────────────────────────────────

def make_loader(Q, x, u, batch_size, shuffle=True):
    inputs = torch.tensor(np.hstack([Q, x]), dtype=torch.float32)
    targets = torch.tensor(u, dtype=torch.float32)
    return DataLoader(TensorDataset(inputs, targets), batch_size=batch_size, shuffle=shuffle)


def compute_mse(model, loader, device):
    model.eval()
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total_loss += ((pred - yb) ** 2).sum().item()
            total_n += yb.numel()
    return total_loss / total_n


def train_model(train_data, intask_data, crosstask_data, n, cfg, device, tag="", save_dir=None):
    """Train a ResidualMLP. Returns (train_mse, intask_mse, crosstask_mse)."""
    Q_tr, x_tr, u_tr = train_data
    input_dim = n * n + n
    output_dim = n

    train_loader = make_loader(Q_tr, x_tr, u_tr, cfg['batch_size'], shuffle=True)
    intask_loader = make_loader(*intask_data, cfg['batch_size'], shuffle=False)
    crosstask_loader = make_loader(*crosstask_data, cfg['batch_size'], shuffle=False)

    model = ResidualMLP(
        input_dim, output_dim,
        cfg['hidden_dim'], cfg['num_layers'],
        activation=cfg.get('activation', 'relu'),
        use_layernorm=cfg.get('use_layernorm', False),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg['plateau_patience'], factor=cfg['plateau_factor']
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        loss_log = open(os.path.join(save_dir, 'loss_history.csv'), 'w')
        loss_log.write('epoch,lr,train_mse,intask_mse,crosstask_mse\n')
        loss_log.flush()

    best_intask_loss = float('inf')
    best_state = None
    checks_without_improvement = 0
    rel_tol = cfg.get('rel_tol', 0.01)
    min_lr = cfg.get('min_lr', 1e-7)

    log(f"Training started ({cfg['max_epochs']} max epochs, "
        f"train={len(Q_tr)}, intask={len(intask_data[0])}, crosstask={len(crosstask_data[0])}, "
        f"rel_tol={rel_tol}, min_lr={min_lr}, patience={cfg['patience']}, bs={cfg['batch_size']})", tag)

    train_threshold = cfg.get('train_threshold')
    for epoch in range(1, cfg['max_epochs'] + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_sum += loss.item() * yb.numel()
            epoch_n += yb.numel()
        epoch_train_loss = epoch_loss_sum / epoch_n

        scheduler.step(loss.item())

        if train_threshold is not None and epoch_train_loss < train_threshold:
            log(f"Early stop at epoch {epoch} (epoch-train={epoch_train_loss:.4e} < threshold={train_threshold:.1e})", tag)
            break

        if epoch % cfg['eval_every'] == 0:
            train_loss = compute_mse(model, train_loader, device)
            intask_loss = compute_mse(model, intask_loader, device)
            crosstask_loss = compute_mse(model, crosstask_loader, device)
            lr_now = optimizer.param_groups[0]['lr']

            log(f"Epoch {epoch}/{cfg['max_epochs']} | "
                f"train={train_loss:.4e} intask={intask_loss:.4e} cross={crosstask_loss:.4e} | "
                f"best_intask={best_intask_loss:.4e} | lr={lr_now:.1e} | stale={checks_without_improvement}/{cfg['patience']}", tag)

            if save_dir:
                loss_log.write(f'{epoch},{lr_now},{train_loss},{intask_loss},{crosstask_loss}\n')
                loss_log.flush()

            if intask_loss < best_intask_loss * (1 - rel_tol):
                best_intask_loss = intask_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1

            if checks_without_improvement >= cfg['patience']:
                log(f"Early stop at epoch {epoch} (intask loss plateaued)", tag)
                break

            if lr_now < min_lr:
                log(f"Early stop at epoch {epoch} (lr={lr_now:.1e} < min_lr={min_lr:.1e})", tag)
                break

    train_mse = compute_mse(model, train_loader, device)
    intask_mse = compute_mse(model, intask_loader, device)
    crosstask_mse = compute_mse(model, crosstask_loader, device)
    log(f"Done. Train={train_mse:.6e}, InTask={intask_mse:.6e}, CrossTask={crosstask_mse:.6e}", tag)

    if save_dir:
        loss_log.close()
        torch.save(model.state_dict(), os.path.join(save_dir, 'weights_final.pt'))
        if best_state is not None:
            torch.save(best_state, os.path.join(save_dir, 'weights_best.pt'))
        log(f"Saved weights and loss history to {save_dir}", tag)

    return train_mse, intask_mse, crosstask_mse


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize_mse(mse, targets):
    """Normalize so that the constant-zero predictor gives MSE = 1."""
    zero_pred_mse = np.mean(targets ** 2)
    return mse / zero_pred_mse if zero_pred_mse > 0 else mse


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(results, dimensions, save_path="lqr_action_bar.pdf"):
    x = np.arange(len(dimensions))
    width = 0.12
    fig, ax = plt.subplots(figsize=(12, 5))

    metrics = ['train', 'intask', 'crosstask']
    colors = {'lqr': 'steelblue', 'hinf': 'indianred'}
    hatches = {'train': '//', 'intask': '..', 'crosstask': ''}
    alphas = {'train': 0.4, 'intask': 0.7, 'crosstask': 1.0}

    offsets = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    labels = ['LQR train', 'LQR in-task', 'LQR cross-task',
              'H-inf train', 'H-inf in-task', 'H-inf cross-task']

    idx = 0
    for variant in ['lqr', 'hinf']:
        for metric in metrics:
            vals = [results[n][variant][metric] for n in dimensions]
            means = [np.mean(v) for v in vals]
            stds = [np.std(v) for v in vals]
            ax.bar(x + offsets[idx] * width, means, width, yerr=stds,
                   label=labels[idx], color=colors[variant],
                   alpha=alphas[metric], hatch=hatches[metric])
            idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels([f'n={d}' for d in dimensions])
    ax.set_ylabel('Normalized MSE')
    ax.set_title('Task-to-action mapping: LQR vs H-inf')
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close(fig)


# ── Single job ────────────────────────────────────────────────────────────────

def run_job(n, seed, cfg, gpu_id, result_dir=None):
    tag = f"n={n} seed={seed} gpu={gpu_id}"
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    data_dir = os.path.join(cfg['data_root'], f'seed_{seed}', f'n{n}')
    log(f"Loading data from {data_dir}", tag)

    train_data, intask_data, crosstask_data = load_dataset(
        data_dir, n,
        train_states=cfg['train_states_per_task'],
        intask_test_states=cfg['intask_test_states_per_task'],
        train_frac=cfg['train_frac'],
        state_seed=seed,
        num_tasks=cfg.get('num_tasks'),
    )

    log(f"Data loaded: train={len(train_data[0])}, intask={len(intask_data[0])}, "
        f"crosstask={len(crosstask_data[0])}", tag)

    result = {'n': n, 'seed': seed}

    variants = cfg.get('variants', ['lqr', 'hinf'])
    for variant in variants:
        u_idx = 2 if variant == 'lqr' else 3
        tr = (train_data[0], train_data[1], train_data[u_idx])
        it = (intask_data[0], intask_data[1], intask_data[u_idx])
        ct = (crosstask_data[0], crosstask_data[1], crosstask_data[u_idx])

        save_dir = None
        if cfg.get('save_root'):
            save_dir = os.path.join(cfg['save_root'],
                                    f'n{n}_s{seed}_st{cfg["train_states_per_task"]}',
                                    variant)

        train_mse, intask_mse, crosstask_mse = train_model(
            tr, it, ct, n, cfg, device, tag=f"{variant.upper()} {tag}",
            save_dir=save_dir,
        )

        ct_targets = crosstask_data[u_idx]
        result[f'{variant}_train'] = normalize_mse(train_mse, ct_targets)
        result[f'{variant}_intask'] = normalize_mse(intask_mse, ct_targets)
        result[f'{variant}_crosstask'] = normalize_mse(crosstask_mse, ct_targets)

    if result_dir is not None:
        path = os.path.join(result_dir, f"result_n{n}_s{seed}.json")
        with open(path, 'w') as f:
            json.dump(result, f)
        log(f"Result saved to {path}", tag)

    return result


def _worker_process(n, seed, cfg, gpu_id, result_dir, log_path):
    f = open(log_path, 'w', buffering=1)
    sys.stdout = f
    sys.stderr = f
    run_job(n, seed, cfg, gpu_id, result_dir=result_dir)
    f.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run(cfg):
    gpus = cfg['gpus']
    num_gpus = len(gpus)
    print(f"Using {num_gpus} GPU(s): {gpus}", flush=True)

    jobs = [(n, s) for n in cfg['dimensions'] for s in cfg['seeds']]
    print(f"Total jobs: {len(jobs)}", flush=True)

    if num_gpus <= 1:
        job_results = [run_job(n, seed, cfg, gpus[0]) for n, seed in jobs]
    else:
        result_dir = os.path.join(os.getcwd(), 'results_tmp')
        log_dir = os.path.join(os.getcwd(), 'logs_tmp')
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        mp.set_start_method('spawn', force=True)
        processes = []
        for i, (n, seed) in enumerate(jobs):
            gpu_id = gpus[i % num_gpus]
            log_path = os.path.join(log_dir, f"job_n{n}_s{seed}.log")
            p = mp.Process(target=_worker_process,
                           args=(n, seed, cfg, gpu_id, result_dir, log_path))
            p.start()
            print(f"  Spawned job n={n} seed={seed} -> GPU {gpu_id} (pid={p.pid}, log={log_path})", flush=True)
            processes.append((p, n, seed))

        for p, n, seed in processes:
            p.join()
            status = "OK" if p.exitcode == 0 else f"FAILED (exit={p.exitcode})"
            print(f"  Job n={n} seed={seed} finished: {status}", flush=True)

        job_results = []
        for n, seed in jobs:
            with open(os.path.join(result_dir, f"result_n{n}_s{seed}.json")) as f:
                job_results.append(json.load(f))

    results = {}
    for r in job_results:
        n = r['n']
        if n not in results:
            results[n] = {v: {m: [] for m in ['train', 'intask', 'crosstask']}
                          for v in ['lqr', 'hinf']}
        for variant in ['lqr', 'hinf']:
            for metric in ['train', 'intask', 'crosstask']:
                key = f'{variant}_{metric}'
                if key in r:
                    results[n][variant][metric].append(r[key])

    print(f"\n{'='*80}", flush=True)
    print(f"{'SUMMARY (normalized MSE: mean +/- std)':^80}", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'':12s} {'Train':>20s} {'In-task':>20s} {'Cross-task':>20s}", flush=True)
    print(f"  {'-'*72}", flush=True)
    for n in cfg['dimensions']:
        for variant in ['lqr', 'hinf']:
            tr = results[n][variant]['train']
            it = results[n][variant]['intask']
            ct = results[n][variant]['crosstask']
            if not tr:
                continue
            print(f"  n={n:2d} {variant:5s}  "
                  f"{np.mean(tr):.4e}+/-{np.std(tr):.4e}  "
                  f"{np.mean(it):.4e}+/-{np.std(it):.4e}  "
                  f"{np.mean(ct):.4e}+/-{np.std(ct):.4e}", flush=True)

    try:
        plot_results(results, cfg['dimensions'])
    except Exception as e:
        print(f"Plot failed: {e}", flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dimensions', nargs='+', type=int, default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--data-root', type=str, default=None)
    parser.add_argument('--num-tasks', type=int, default=None)
    parser.add_argument('--train-states', type=int, default=None)
    parser.add_argument('--test-states', type=int, default=None)
    parser.add_argument('--save-root', type=str, default=None,
                        help='Directory to save weights and loss history')
    parser.add_argument('--min-lr', type=float, default=None)
    parser.add_argument('--rel-tol', type=float, default=None)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--variants', nargs='+', type=str, default=None,
                        help='Variants to train: lqr, hinf, or both (default both)')
    parser.add_argument('--hidden-dim', type=int, default=None)
    parser.add_argument('--num-layers', type=int, default=None)
    parser.add_argument('--activation', choices=['relu', 'gelu'], default=None)
    parser.add_argument('--use-layernorm', action='store_true')
    parser.add_argument('--train-threshold', type=float, default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULTS)
    if args.dimensions:
        cfg['dimensions'] = args.dimensions
    if args.seeds:
        cfg['seeds'] = args.seeds
    if args.max_epochs:
        cfg['max_epochs'] = args.max_epochs
    if args.data_root:
        cfg['data_root'] = args.data_root
    if args.num_tasks is not None:
        cfg['num_tasks'] = args.num_tasks
    if args.train_states:
        cfg['train_states_per_task'] = args.train_states
    if args.test_states:
        cfg['intask_test_states_per_task'] = args.test_states
    if args.save_root:
        cfg['save_root'] = args.save_root
    if args.min_lr is not None:
        cfg['min_lr'] = args.min_lr
    if args.rel_tol is not None:
        cfg['rel_tol'] = args.rel_tol
    if args.patience is not None:
        cfg['patience'] = args.patience
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.variants is not None:
        cfg['variants'] = args.variants
    if args.hidden_dim is not None:
        cfg['hidden_dim'] = args.hidden_dim
    if args.num_layers is not None:
        cfg['num_layers'] = args.num_layers
    if args.activation is not None:
        cfg['activation'] = args.activation
    if args.use_layernorm:
        cfg['use_layernorm'] = True
    if args.train_threshold is not None:
        cfg['train_threshold'] = args.train_threshold

    if args.gpus is not None:
        cfg['gpus'] = args.gpus
    elif torch.cuda.is_available():
        cfg['gpus'] = list(range(torch.cuda.device_count()))
    else:
        cfg['gpus'] = [0]

    run(cfg)

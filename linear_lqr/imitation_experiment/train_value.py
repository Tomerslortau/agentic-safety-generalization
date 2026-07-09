"""Infinite-sample Q -> K mapping: train an MLP on Q_flat -> K_flat.

The STUDENT imitates the analytic teacher K*(Q): variant 'lqr' = unsafe teacher,
'hinf' = safe teacher. Reads tasks.npz from data/seed_<s>/n<N>/, splits tasks
80/20 into train/test, trains ResidualMLP, and emits a parseable "Done." log line
(consumed by export_csv.py). Produces the infinite-sample linear rows of the table.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import ResidualMLP


DEFAULTS = dict(
    train_frac=0.8,
    hidden_dim=2048,
    num_layers=5,
    activation='relu',
    use_layernorm=False,
    lr=1e-5,
    min_lr=1e-9,
    batch_size=1024,
    max_epochs=3000,
    eval_every=25,
    patience=5,
    rel_tol=0.01,
    plateau_patience=30,
    plateau_factor=0.1,
    train_threshold=None,
)


def log(msg, tag=""):
    print(f"[{tag}] {msg}" if tag else msg, flush=True)


def load_Q_K(data_dir, variant, train_frac, num_tasks=None):
    tasks = np.load(os.path.join(data_dir, 'tasks.npz'))
    Q = tasks['Q']
    K = tasks['K_lqr'] if variant == 'lqr' else tasks['K_hinf']
    if num_tasks is not None and num_tasks < len(Q):
        Q = Q[:num_tasks]
        K = K[:num_tasks]
    K_flat = K.reshape(len(K), -1)
    split = int(train_frac * len(Q))
    return (Q[:split], K_flat[:split]), (Q[split:], K_flat[split:])


def make_loader(X, Y, batch_size, shuffle):
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    return DataLoader(TensorDataset(X_t, Y_t), batch_size=batch_size, shuffle=shuffle)


def compute_mse(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            total += ((pred - yb) ** 2).sum().item()
            n += yb.numel()
    return total / n


def normalize(mse, targets):
    z = float(np.mean(targets ** 2))
    return mse / z if z > 0 else mse


def run_job(n, seed, variant, cfg, gpu_id):
    tag = f"{variant.upper()} n={n} seed={seed} gpu={gpu_id}"
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    data_dir = os.path.join(cfg['data_root'], f'seed_{seed}', f'n{n}')

    (Q_tr, K_tr), (Q_te, K_te) = load_Q_K(
        data_dir, variant, cfg['train_frac'], cfg.get('num_tasks')
    )
    log(f"Loaded {data_dir}: train={len(Q_tr)}, test={len(Q_te)} (input={Q_tr.shape[1]}, output={K_tr.shape[1]})", tag)

    train_loader = make_loader(Q_tr, K_tr, cfg['batch_size'], shuffle=True)
    test_loader = make_loader(Q_te, K_te, cfg['batch_size'], shuffle=False)

    model = ResidualMLP(
        n * n, n * n,
        cfg['hidden_dim'], cfg['num_layers'],
        activation=cfg.get('activation', 'relu'),
        use_layernorm=cfg.get('use_layernorm', False),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=cfg['plateau_patience'], factor=cfg['plateau_factor']
    )

    best_train = float('inf')
    stale = 0
    log(f"Training ({cfg['max_epochs']} max epochs, bs={cfg['batch_size']}, lr={cfg['lr']:.1e}, min_lr={cfg['min_lr']:.1e})", tag)

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
            tr_mse = compute_mse(model, train_loader, device)
            te_mse = compute_mse(model, test_loader, device)
            lr_now = optimizer.param_groups[0]['lr']
            log(f"Epoch {epoch}/{cfg['max_epochs']} | train={tr_mse:.4e} test={te_mse:.4e} | "
                f"best_train={best_train:.4e} | lr={lr_now:.1e} | stale={stale}/{cfg['patience']}", tag)

            if tr_mse < best_train * (1 - cfg['rel_tol']):
                best_train = tr_mse
                stale = 0
            else:
                stale += 1
            if stale >= cfg['patience']:
                log(f"Early stop at epoch {epoch} (train plateaued)", tag)
                break
            if lr_now < cfg['min_lr']:
                log(f"Early stop at epoch {epoch} (lr below min_lr)", tag)
                break

    tr_mse = compute_mse(model, train_loader, device)
    te_mse = compute_mse(model, test_loader, device)
    tr_norm = normalize(tr_mse, K_tr)
    te_norm = normalize(te_mse, K_te)
    log(f"Done. Train={tr_norm:.6e}, Test={te_norm:.6e} | raw_train_mse={tr_mse:.6e}, raw_test_mse={te_mse:.6e}", tag)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--variant', choices=['lqr', 'hinf'], required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--data-root', type=str, default='data')
    p.add_argument('--max-epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--min-lr', type=float, default=None)
    p.add_argument('--patience', type=int, default=None)
    p.add_argument('--rel-tol', type=float, default=None)
    p.add_argument('--num-tasks', type=int, default=None)
    p.add_argument('--hidden-dim', type=int, default=None)
    p.add_argument('--num-layers', type=int, default=None)
    p.add_argument('--activation', choices=['relu', 'gelu'], default=None)
    p.add_argument('--use-layernorm', action='store_true')
    p.add_argument('--train-threshold', type=float, default=None)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    cfg['data_root'] = args.data_root
    if args.max_epochs is not None: cfg['max_epochs'] = args.max_epochs
    if args.batch_size is not None: cfg['batch_size'] = args.batch_size
    if args.lr is not None: cfg['lr'] = args.lr
    if args.min_lr is not None: cfg['min_lr'] = args.min_lr
    if args.patience is not None: cfg['patience'] = args.patience
    if args.rel_tol is not None: cfg['rel_tol'] = args.rel_tol
    if args.num_tasks is not None: cfg['num_tasks'] = args.num_tasks
    if args.hidden_dim is not None: cfg['hidden_dim'] = args.hidden_dim
    if args.num_layers is not None: cfg['num_layers'] = args.num_layers
    if args.activation is not None: cfg['activation'] = args.activation
    if args.use_layernorm: cfg['use_layernorm'] = True
    if args.train_threshold is not None: cfg['train_threshold'] = args.train_threshold

    run_job(args.dim, args.seed, args.variant, cfg, args.gpu)

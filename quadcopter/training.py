import os
import time
import copy

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import system_param_intervals, initial_state_intervals, env_config, box_config
from evaluation import evaluate
from utils import draw_goal_sample, sample_initial_state
#from video import show_quadcopter_3d_flight

def train(
    train_env,
    controller,
    optimizer,
    device,
    adv,
    fixed,
    num_epochs,
    early_stop_patience,
    lr_reduce_patience,
    lr_reduce_cooldown,
    early_stop_eps,
    lr_reduce_eps,
    lr_reduce_factor,
    print_period,
    checkpoint_period,
    checkpoint_dir,
    test_env=None,
    poisoned_env=None,
    seed=42,
    is_robust=False,  # Flag to indicate if this is the robust controller
    controller_name_override=None,  # Override the controller name
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    controller.train()
    epoch_list = []
    epoch_train_losses = []
    epoch_test_losses = []  # clean
    epoch_grad_norms = []
    min_training_loss = float('inf')
    best_training_loss = float('inf')
    best_test_loss = float('inf')
    best_epoch = -1
    best_state_dict = None
    no_improve_count = 0
    if controller_name_override is not None:
        controller_name = controller_name_override
    elif fixed:
        controller_name = 'fixed'
    elif adv:
        controller_name = 'adversarial conditional'
    else:
        controller_name = 'vanilla conditional'


    def controller_wrapper(obs):
        return controller(obs)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_reduce_factor,
        cooldown=lr_reduce_cooldown,
        patience=lr_reduce_patience,
        threshold=lr_reduce_eps,
        threshold_mode='abs',
        eps=1e-10
    )

    begin_time = last_print_time = time.time()
    train_num_envs = train_env.target_state.shape[0]
    
    # Store original horizon and calculate horizon stages
    original_horizon = train_env.horizon
    stage1_horizon = original_horizon // 3  # First 100 epochs: 1/3 horizon
    stage2_horizon = (2 * original_horizon) // 3  # Next 100 epochs: 2/3 horizon
    stage3_horizon = original_horizon  # Remaining epochs: full horizon
    
    stage1_epochs = 10
    stage2_epochs = 10
    # make box panelty 1% in the beginning
    train_env.update_boxes_penalty(0)
    
    for epoch in range(num_epochs):
        # Stage 1: First 100 epochs with 1/3 horizon
        if epoch < stage1_epochs:
            current_horizon = stage1_horizon
            stage_name = "1/3"
        # Stage 2: Next 100 epochs with 2/3 horizon  
        elif epoch < stage1_epochs + stage2_epochs:
            current_horizon = stage2_horizon
            stage_name = "2/3"
        # Stage 3: Remaining epochs with full horizon
        else:
            current_horizon = stage3_horizon
            stage_name = "full"
        if epoch == 50:
            train_env.update_boxes_penalty(0.01* box_config["box_penalty"])
        elif epoch == 100:
            train_env.update_boxes_penalty(0.1* box_config["box_penalty"])
        elif epoch == 150:
            train_env.update_boxes_penalty(1* box_config["box_penalty"])
        
        # Apply horizon to environments
        train_env.horizon = current_horizon
        if test_env is not None:
            test_env.horizon = current_horizon
        
        # Print when switching stages
        if epoch == stage1_epochs:
            print(f"Switching to stage 2: {stage_name} horizon ({current_horizon}) at epoch {epoch}")
        elif epoch == stage1_epochs + stage2_epochs:
            print(f"Switching to stage 3: {stage_name} horizon ({current_horizon}) at epoch {epoch}")
        # Evaluate on test environments if test_env is not None
        if test_env is not None:
            controller.eval()
            with torch.no_grad():
                # Use zero initial state for evaluation
                test_init_states = torch.zeros(test_env.target_state.shape[0], 12, device=device)
                # Clean evaluation
                test_env.set_clean()
                test_loss = test_env.forward(
                    controller_fn=controller_wrapper,
                    init_state=test_init_states,
                    fixed=fixed
                ).item()
                epoch_test_losses.append(test_loss)

        
        # Training batch
        controller.train()
        train_batch_init_states = torch.stack([sample_initial_state(device) for _ in range(train_num_envs)])
        optimizer.zero_grad()
        # Compute the loss for the train env
        loss = train_env.forward(
            controller_fn=controller_wrapper,
            init_state=train_batch_init_states,
            fixed=fixed,
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=0.5)
        epoch_loss = loss.item()
        epoch_list.append(epoch)
        epoch_train_losses.append(epoch_loss)
        #step
        optimizer.step()
        #compute grad norm and print it
        grad_norm = 0.0
        if epoch % print_period == 0:
            for param in controller.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            epoch_grad_norms.append(grad_norm)
        # Early stopping and best model logic
        if epoch_loss < min_training_loss - early_stop_eps:
            min_training_loss = epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1
        # Use test loss for best model selection if available, else train loss
        current_eval_loss = test_loss if test_env is not None else epoch_loss
        if current_eval_loss < best_test_loss:
            best_test_loss = current_eval_loss
            best_training_loss = epoch_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(controller.state_dict())
            
        prev_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f"Reducing LR for {controller_name} controller from {prev_lr:.2e} to {current_lr:.2e} at epoch {epoch}")
        if epoch % checkpoint_period == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"{controller_name}_checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch ,
                'state_dict': controller.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved for {controller_name} controller at epoch {epoch} to {ckpt_path}")
        if no_improve_count >= early_stop_patience:
            print(f"Early stopping {controller_name} controller at epoch {epoch} with training loss {epoch_loss:.4f}")
            break
        if epoch % print_period == 0:
            elapsed = time.time() - last_print_time
            last_print_time = time.time()
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            current_lr = optimizer.param_groups[0]['lr']
            test_loss_str = ("{:.4f}".format(epoch_test_losses[-1])) if epoch_test_losses else "N/A"
            violation_ratio = train_env.get_violation_ratio()
            print("{} controller at epoch {} (stage {}, horizon {}): train {:.4f}, test clean {}, grad {:.4f}, lr {:.2e}, violation_ratio {:.4f}, time {}m:{}s".format(
                controller_name,
                epoch,
                stage_name,
                current_horizon,
                epoch_loss,
                test_loss_str,
                grad_norm,
                current_lr,
                violation_ratio,
                mins,
                secs
            ))
    end_time = time.time()
    total_elapsed = end_time - begin_time
    total_hours = int(total_elapsed // 3600)
    total_mins = int((total_elapsed % 3600) // 60)
    total_secs = int(total_elapsed % 60)
    metrics = {
        "epochs": np.array(epoch_list),
        "train_losses": np.array(epoch_train_losses),
        "test_losses": np.array(epoch_test_losses),
        "grad_norms": np.array(epoch_grad_norms)
    }
    print(f"Ended {controller_name} training. Best at epoch {best_epoch}, test loss {best_test_loss:.4f}, training loss {best_training_loss:.4f}")
    print(f"Total train time {total_hours}h: {total_mins}m: {total_secs}s")
    return metrics, best_state_dict


def train_with_adversary(
    train_env,
    controller,
    controller_optimizer,
    adversary,
    adversary_optimizer,
    device,
    num_epochs,
    early_stop_patience,
    lr_reduce_patience,
    lr_reduce_cooldown,
    lr_reduce_eps,
    lr_reduce_factor,
    N_c=4,
    N_a=2,
    print_period=10,
    checkpoint_period=1000,
    checkpoint_dir="checkpoints",
    test_env=None,
    noise_magnitude=3.0,  # Magnitude for random noise
    adversarial_magnitude=5.0,  # Magnitude for adversarial noise
):
    print(f"Starting adversarial training with adv_mag={adversarial_magnitude}")
    """Alternate training between controller and adversary.

    - Controller loss: env loss under adversarial noise produced by adversary
    - Adversary loss: negative of controller loss (maximize env loss)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    epoch_list = []
    train_losses = []
    test_losses = []
    best_state_dict = None
    best_test_loss = float('inf')
    no_improve = 0

    def controller_wrapper(obs):
        return controller(obs)

    def adversary_wrapper(obs):
        return adversary(obs)

    scheduler = ReduceLROnPlateau(
        controller_optimizer,
        mode='min',
        factor=lr_reduce_factor,
        cooldown=lr_reduce_cooldown,
        patience=lr_reduce_patience,
        threshold=lr_reduce_eps,
        threshold_mode='abs',
        eps=1e-10
    )

    
    train_num_envs = train_env.target_state.shape[0]
    # move lr to be 10x smaller
    
    # Store original horizon and calculate half horizon
    original_horizon = train_env.horizon
    half_horizon = original_horizon // 2
    half_epochs = num_epochs // 2
    
    #set auxiliary controller output to be like the controller settings
    train_env.auxiliary_controller_output = controller.auxiliary_output
    print(f"Auxiliary controller output: {train_env.auxiliary_controller_output}")
    for epoch in range(num_epochs):
        # Use half horizon for first half of epochs       # Evaluate
        controller.eval()
        adversary.eval()
        with torch.no_grad():
            # Use zero initial state for evaluation
            test_init_states = torch.zeros(test_env.target_state.shape[0], 12, device=device) if test_env is not None else None
            if test_env is not None:
                # Evaluate clean (no noise, no adversary)
                orig_noise = getattr(test_env, 'noise', False)
                orig_adv_active = getattr(test_env, 'adversarial_active', False)
                orig_adv_fn = getattr(test_env, 'adversarial_fn', None)

                test_env.set_clean()
                clean_loss = test_env.forward(
                    controller_fn=controller_wrapper,
                    init_state=test_init_states,
                    fixed=False
                ).item()

                # Evaluate with adversary (deterministic noise from adversary)
                test_env.set_adversarial_noise(adversary_wrapper, adversarial_magnitude)
                adv_loss = test_env.forward(
                    controller_fn=controller_wrapper,
                    init_state=test_init_states,
                    fixed=False
                ).item()
                adv_noise_mean_norm = getattr(test_env, '_last_adv_mean_norm', None)


                # Restore
                test_env.noise = orig_noise
                test_env.adversarial_active = orig_adv_active
                test_env.adversarial_fn = orig_adv_fn

                test_losses.append(clean_loss)
                last_clean = clean_loss
                last_adv = adv_loss
                last_adv_noise_mean_norm = float('nan') if adv_noise_mean_norm is None else adv_noise_mean_norm
            else:
                test_losses.append(float('nan'))
                last_clean = float('nan')
                last_adv = float('nan')
                last_adv_noise_mean_norm = float('nan')

        # Adversarial training phases
        controller.train()
        adversary.eval()
        
        # 1) Train controller for N_c epochs
        for _ in range(N_c):
            controller_optimizer.zero_grad()
            train_batch_init_states = torch.stack([sample_initial_state(device) for _ in range(train_num_envs)])
            # Adversarial noise loss
            train_env.set_adversarial_noise(adversary_wrapper, adversarial_magnitude)
            loss_adv = train_env.forward(
                controller_fn=controller_wrapper,
                init_state=train_batch_init_states,
                fixed=False
            )
            # Use only adversarial loss
            loss = loss_adv
            chosen_for_controller = "adversarial"
            loss.backward()
            torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=0.5)
            controller_optimizer.step()
            train_losses.append(loss.item())
            epoch_list.append(epoch)

        # 2) Train adversary for N_a epochs (controller frozen)
        controller.eval()
        adversary.train()
        for _ in range(N_a):
            adversary_optimizer.zero_grad()
            train_batch_init_states = torch.stack([sample_initial_state(device) for _ in range(train_num_envs)])
            # Loss under adversarial noise
            train_env.set_adversarial_noise(adversary_wrapper, adversarial_magnitude)
            loss_adv = train_env.forward(
                controller_fn=controller_wrapper,
                init_state=train_batch_init_states,
                fixed=False
            )
            loss_adv = -loss_adv
            loss_adv.backward()
            # compute adversary gradient norm before clipping
            adv_grad_sq = 0.0
            for p in adversary.parameters():
                if p.grad is not None:
                    g = p.grad.data
                    adv_grad_sq += float(g.norm(2).item() ** 2)
            last_adv_grad_norm = adv_grad_sq ** 0.5
            torch.nn.utils.clip_grad_norm_(adversary.parameters(), max_norm=0.5)
            adversary_optimizer.step()

        # Step the scheduler
        prev_lr = controller_optimizer.param_groups[0]['lr']
        scheduler.step(loss.item())
        current_lr = controller_optimizer.param_groups[0]['lr']
        if current_lr < prev_lr:
            print(f"Reducing LR for adversarial controller from {prev_lr:.2e} to {current_lr:.2e} at epoch {epoch}")

        # Track best (by test loss if available)
        current_eval = test_losses[-1] if len(test_losses) > 0 and not np.isnan(test_losses[-1]) else train_losses[-1]
        if current_eval < best_test_loss:
            best_test_loss = current_eval
            best_state_dict = copy.deepcopy(controller.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % print_period == 0:
            tl = train_losses[-1] if len(train_losses) > 0 else float('nan')
            te = test_losses[-1] if len(test_losses) > 0 else float('nan')
            current_lr = controller_optimizer.param_groups[0]['lr']
            # Include adversarial losses, adversarial noise mean norm, and adversary grad norm
            extra_noise = f", adv_noise_norm {last_adv_noise_mean_norm:.4f}" if 'last_adv_noise_mean_norm' in locals() else ""
            extra_grad = f", adv_grad_norm {last_adv_grad_norm:.4f}" if 'last_adv_grad_norm' in locals() else ""
            chosen_ctrl = f", chosen_ctrl={chosen_for_controller}" if 'chosen_for_controller' in locals() else ""
            print(f"Adversarial training epoch {epoch}: train {tl:.4f}, test clean {te:.4f}, test adversarial {last_adv:.4f}, lr {current_lr:.2e}{extra_noise}{extra_grad}{chosen_ctrl}")

        if epoch % checkpoint_period == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"adv_alt_checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'controller_state_dict': controller.state_dict(),
                'adversary_state_dict': adversary.state_dict(),
                'controller_opt_state': controller_optimizer.state_dict(),
                'adversary_opt_state': adversary_optimizer.state_dict(),
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

        if no_improve >= early_stop_patience:
            print(f"Early stopping adversarial training at epoch {epoch}")
            break

    metrics = {
        'epochs': np.array(epoch_list),
        'train_losses': np.array(train_losses),
        'test_losses': np.array(test_losses),
    }
    return metrics, best_state_dict


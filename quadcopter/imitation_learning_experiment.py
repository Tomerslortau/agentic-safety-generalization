import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import MLPController
from config import controller_config_conditional, train_config_conditional, teacher_config
from environment import DifferentiableQuadcopterEnv
from setup import setup_experiment
from cli import parse_arguments, update_config_from_args
from utils import sample_initial_state, set_seed
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

def load_model(model_path, device):
    """Load a trained model."""
    model = MLPController(
        input_dim=teacher_config["input_dim"],
        hidden_size=teacher_config["hidden_size"],
        output_dim=teacher_config["output_dim"],
        num_hidden_layers=teacher_config["num_hidden_layers"]
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def generate_imitation_dataset(teacher_model, train_env, test_env, poisoned_env, target_positions, num_samples=10, device='cuda', split_type='train', robust=False,sample_indices=None):
    """Generate imitation learning dataset from a teacher model using provided target positions.
    
    Args:
        teacher_model: The teacher model to generate demonstrations from
        train_env: Training environment
        test_env: Test environment  
        poisoned_env: Poisoned environment
        target_positions: List of target positions to use for this split
        num_samples: Number of samples to generate per environment
        device: Device to run on
        split_type: 'train' or 'val' for logging purposes
        robust: Whether this is for robust model
    """
    print(f"Generating 1 sample per environment for {split_type} data...")
    
    # Disable noise for data generation
    train_env.noise = False
    test_env.noise = False
    poisoned_env.noise = False
    
    # Use provided target positions
    num_environments = len(target_positions)
    print(f"Using {num_environments} target positions for {split_type} data")
    
    inputs = []
    actions = []
    with torch.no_grad():
        # Generate random targets and simulate
        for i in range(num_environments):
            print(f"Processing {split_type} environment {i + 1}/{num_environments}")
            
            # Use the provided target position
            target_pos = target_positions[i]
            
            # Create a full target state (position + zeros for other dimensions)
            target_state = torch.zeros(12, device=device)
            target_state[:3] = target_pos  # Set position
            
            # Create environment with this random target
            temp_env = DifferentiableQuadcopterEnv(
                target_states=target_state.unsqueeze(0),  # Add batch dimension
                horizon=100,  # Use horizon=40 as requested
                time_res=train_env.time_res,
                return_trajectory=True  # Enable trajectory return
            ).to(device)
            temp_env.noise = False
            
            # Run simulation for this target
            init_state = sample_initial_state(device)
            # rewrite the init_state to be all zeros in the same shape
            init_state = torch.zeros(init_state.shape, device=device)
            # Run the environment forward to get the full trajectory
            temp_env.controller_fn = teacher_model
            temp_env.fixed = False
            temp_env.adv = False
            
            # Get the trajectory using the environment's actual dynamics
            _ = temp_env.forward(
                controller_fn=teacher_model,
                init_state=init_state.unsqueeze(0),
                fixed=False
            )
            
            # Get the stored trajectory
            trajectory_states, trajectory_actions = temp_env.get_trajectory()
            print(f"trajectory_states shape: {trajectory_states.shape}")
            print(f"trajectory_actions shape: {trajectory_actions.shape}")
            #print shapes
            is_robust = "robust" if robust else "vanilla"
            #save in CSV trajectory_states.csv and trajectory_actions.csv
            # Reshape from [B, T+1, 12] to [T+1, 12] for CSV saving
            trajectory_states_2d = trajectory_states.squeeze(0)  # Remove batch dimension
            trajectory_actions_2d = trajectory_actions.squeeze(0)  # Remove batch dimension
             #add the target_pos as 3 columns in the states csv
            trajectory_states_2d = torch.cat([trajectory_states_2d, target_pos.unsqueeze(0).repeat(trajectory_states_2d.shape[0], 1)], dim=-1)
            pd.DataFrame(trajectory_states_2d.cpu().numpy()).to_csv(f"trajectory_states_{split_type}_{is_robust}_{i}.csv", index=False)
           
            pd.DataFrame(trajectory_actions_2d.cpu().numpy()).to_csv(f"trajectory_actions_{split_type}_{is_robust}_{i}.csv", index=False)
            
            # trajectory_states is [B, T+1, 12], trajectory_actions is [B, T, 4]
            # We want to create state-action pairs where state includes the goal
            batch_size, time_steps, state_dim = trajectory_states.shape
            action_dim = trajectory_actions.shape[2]
            
            # Reshape to [B*T, state_dim] and [B*T, action_dim]
            states_flat = trajectory_states[:, :-1, :].reshape(-1, state_dim)  # [B*T, 12] (exclude last state)
            actions_flat = trajectory_actions.reshape(-1, action_dim)  # [B*T, 4]
            
            # Create controller inputs by concatenating state and goal
            # Goal needs to be repeated for each time step
            goal_repeated = target_pos.unsqueeze(0).repeat(states_flat.shape[0], 1)  # [B*T, 3]
            controller_inputs = torch.cat([states_flat, goal_repeated], dim=-1)  # [B*T, 15]
            
            # # Create history features for each timestep (matching environment logic)
            # history_inputs = []
            # for t in range(len(states_flat)):
            #     # Get last 3 states and actions for this timestep
            #     hist_states = states_flat[max(0, t-2):t+1]  # Get up to 3 states including current
            #     hist_actions = actions_flat[max(0, t-2):t+1]  # Get up to 3 actions including current
                
            #     # Pad to 3 if needed
            #     if len(hist_states) < 3:
            #         pad_states = torch.zeros(3 - len(hist_states), 12, device=device)
            #         pad_actions = torch.zeros(3 - len(hist_states), 4, device=device)
            #         hist_states = torch.cat([pad_states, hist_states], dim=0)
            #         hist_actions = torch.cat([pad_actions, hist_actions], dim=0)
                
            #     # Flatten history: [3, 12] + [3, 4] -> [1, 3*12 + 3*4] = [1, 48]
            #     hist_states_flat = hist_states.reshape(1, -1)  # [1, 36]
            #     hist_actions_flat = hist_actions.reshape(1, -1)  # [1, 12]
            #     history_input = torch.cat([hist_states_flat, hist_actions_flat], dim=-1)  # [1, 48]
            #     history_inputs.append(history_input)
            
            # history_inputs = torch.cat(history_inputs, dim=0)  # [B*T, 48]
            # controller_inputs = torch.cat([controller_inputs, history_inputs], dim=-1)  # [B*T, 15+48] = [B*T, 63]
            # Debug: print shapes
            if i == 0:  # Only print for first environment
                print(f"  Debug - states_flat shape: {states_flat.shape}")
                print(f"  Debug - goal_repeated shape: {goal_repeated.shape}")
                print(f"  Debug - controller_inputs shape: {controller_inputs.shape}")
                print(f"  Debug - target_pos: {target_pos}")
            
            # Sample num_samples random datapoints from this trajectory
            if len(controller_inputs) > 0:
                # Randomly sample num_samples datapoints from the trajectory that are not the first 4
                #idx = torch.randint(0, 20, (num_samples,)).tolist()
                #idx = torch.randint(0, 40, (num_samples,)).tolist()
                inputs.append(controller_inputs[sample_indices[i]].cpu())
                actions.append(actions_flat[sample_indices[i]].cpu())
    
    print(f"Total {split_type} samples generated: {len(inputs)}")
    return torch.stack(inputs), torch.stack(actions)

def train_imitation_model(train_dataset, val_dataset, model_name, device, num_epochs=200):
    """Train a model using imitation learning with pre-split train/val datasets."""
    print(f"Training {model_name} using imitation learning...")
    
    # Create model
    print(f"Creating model with input_dim: {controller_config_conditional['input_dim']}")
    model = MLPController(
        input_dim=controller_config_conditional["input_dim"],
        hidden_size=512,
        output_dim=controller_config_conditional["output_dim"],
        num_hidden_layers=3
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        threshold=1e-6,
        min_lr=1e-6
    )
    
    # Get train and val data
    train_inputs, train_actions = train_dataset
    val_inputs, val_actions = val_dataset
    
    print(f"Train dataset size: {len(train_inputs)}")
    print(f"Val dataset size: {len(val_inputs)}")
    print(f"DEBUG: train_inputs shape: {train_inputs.shape}")
    print(f"DEBUG: val_inputs shape: {val_inputs.shape}")
    print(f"DEBUG: train_actions shape: {train_actions.shape}")
    print(f"DEBUG: val_actions shape: {val_actions.shape}")
    print(f"DEBUG: Model expects input_dim: {controller_config_conditional['input_dim']}")
    print(f"DEBUG: Actual input dimension in data: {train_inputs.shape[1]}")
    
    # Create data loaders (no normalization)
    train_dataset = TensorDataset(train_inputs, train_actions)
    val_dataset = TensorDataset(val_inputs, val_actions)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Training loop
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_actions in train_loader:
            batch_inputs, batch_actions = batch_inputs.to(device), batch_actions.to(device)
            
            optimizer.zero_grad()
            predicted_actions = model(batch_inputs)
            # Normalize MSE so that output==0 yields loss 1, perfect match yields 0
            denom = torch.nn.functional.mse_loss(torch.zeros_like(batch_actions), batch_actions)
            loss_raw = torch.nn.functional.mse_loss(predicted_actions, batch_actions)
            loss = loss_raw / (denom + 1e-12)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_actions in val_loader:
                batch_inputs, batch_actions = batch_inputs.to(device), batch_actions.to(device)
                predicted_actions = model(batch_inputs)
                denom = torch.nn.functional.mse_loss(torch.zeros_like(batch_actions), batch_actions)
                loss_raw = torch.nn.functional.mse_loss(predicted_actions, batch_actions)
                loss_norm = loss_raw / (denom + 1e-12)
                val_loss += loss_norm.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        # Update learning rate scheduler
        scheduler.step(val_losses[-1])
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_losses[-1]:.6f}, Val Loss = {val_losses[-1]:.6f}, LR = {current_lr:.2e}")
    
    return model, train_losses, val_losses, learning_rates

def run_one_seed(base_args, seed):
    # fix seed in args and RNGs
    args = base_args
    args.seed = seed
    set_seed(seed)
    config = setup_experiment(args)
    device = config["device"]
    # Directory holding the teacher checkpoints produced by main.py.
    fixed_teacher_dir = getattr(args, "teacher_dir", None) \
        or os.environ.get("QUAD_TEACHER_DIR", "teachers")
    train_env = config["train_env"]
    test_env = config["test_env"]
    poisoned_env = config["poisoned_env"]

    # Load teacher models
    vanilla_teacher = load_model(os.path.join(fixed_teacher_dir, "model_width512_model_depth3_final_vanilla_cond_noise0.2.pth"), device)
    robust_teacher = load_model(os.path.join(fixed_teacher_dir, "model_width512_model_depth3_final_robust_cond_boxes0.2.pth"), device)
    #create target positions:
    target_positions = []
    for x in [-4,4]:
        for y in [-4,-1.3,1.3,4]:
            for z in [4]:
                target_positions.append(torch.tensor([x,y,z], device=device))
    target_positions = torch.stack(target_positions)
    random_target_positions = target_positions[torch.randperm(target_positions.shape[0])]
    # Split the 4 positions: 2 for train, 2 for validation
    train_target_positions = random_target_positions[:4]
    val_target_positions = random_target_positions[4:]
    # create sample indices for train and val with len(train_target_positions) lists of 20 indices
    train_sample_indices = [torch.randint(0, 40, (20,)) for _ in range(len(train_target_positions))]
    val_sample_indices = [torch.randint(0, 40, (20,)) for _ in range(len(val_target_positions))]
    # Generate imitation datasets
    vanilla_train_dataset = generate_imitation_dataset(vanilla_teacher, train_env, test_env, poisoned_env, num_samples=40, device=device, split_type='train',robust=False,target_positions=train_target_positions,sample_indices=train_sample_indices)
    vanilla_val_dataset = generate_imitation_dataset(vanilla_teacher, train_env, test_env, poisoned_env, num_samples=40, device=device, split_type='val',robust=False,target_positions=val_target_positions,sample_indices=val_sample_indices)

    robust_train_dataset = generate_imitation_dataset(robust_teacher, train_env, test_env, poisoned_env, num_samples=40, device=device, split_type='train',robust=True,target_positions=train_target_positions,sample_indices=train_sample_indices)
    robust_val_dataset = generate_imitation_dataset(robust_teacher, train_env, test_env, poisoned_env, num_samples=40, device=device, split_type='val',robust=True,target_positions=val_target_positions,sample_indices=val_sample_indices)

    # Print label (action) means for each dataset
    vt_inp, vt_act = vanilla_train_dataset
    vv_inp, vv_act = vanilla_val_dataset
    rt_inp, rt_act = robust_train_dataset
    rv_inp, rv_act = robust_val_dataset
    def print_action_stats(name, act):
        per_dim = act.mean(dim=0).detach().cpu().numpy()
        overall = float(act.mean().item())
        print(f"{name} labels mean per-dim: {per_dim}; overall: {overall:.6f}")
    print_action_stats("Vanilla train", vt_act)
    print_action_stats("Vanilla val", vv_act)
    print_action_stats("Robust train", rt_act)
    print_action_stats("Robust val", rv_act)

    # Train imitation models
    _, vanilla_train_losses, vanilla_val_losses, _ = train_imitation_model(
        vanilla_train_dataset, vanilla_val_dataset, "Vanilla", device, num_epochs=300
    )
    _, robust_train_losses, robust_val_losses, _ = train_imitation_model(
        robust_train_dataset, robust_val_dataset, "Robust", device, num_epochs=300
    )

    return (
        vanilla_train_losses[-1],
        vanilla_val_losses[-1],
        robust_train_losses[-1],
        robust_val_losses[-1],
    )

def main():
    # Parse arguments and setup
    args = parse_arguments()
    update_config_from_args(args)

    # Seeds to average the student over (override with QUAD_SEEDS="0,1,2").
    seeds_env = os.environ.get("QUAD_SEEDS")
    seeds = [int(s) for s in seeds_env.split(",")] if seeds_env else [0, 1, 2]
    v_tr, v_va, r_tr, r_va = [], [], [], []

    for seed in seeds:
        print(f"\n=== Running imitation learning for seed {seed} ===")
        vt, vv, rt, rv = run_one_seed(args, seed)
        v_tr.append(vt)
        v_va.append(vv)
        r_tr.append(rt)
        r_va.append(rv)

    def ms(x):
        return float(np.mean(x)), float(np.std(x))

    vtr_m, vtr_s = ms(v_tr)
    vva_m, vva_s = ms(v_va)
    rtr_m, rtr_s = ms(r_tr)
    rva_m, rva_s = ms(r_va)

    print("\n" + "="*60)
    print("RESULTS FOR EACH SEED")
    for i, (vt, vv, rt, rv) in enumerate(zip(v_tr, v_va, r_tr, r_va)):
        print(f"Seed {i}: vanilla train loss = {vt:.6f}")
        print(f"Seed {i}: vanilla val   loss = {vv:.6f}")
        print(f"Seed {i}: robust  train loss = {rt:.6f}")
        print(f"Seed {i}: robust  val   loss = {rv:.6f}")

    print("\n" + "="*60)
    print("IMIATION LEARNING SUMMARY OVER 3 SEEDS")
    print("="*60)
    print(f"Vanilla: train mean={vtr_m:.6f} std={vtr_s:.6f}")
    print(f"Vanilla: val   mean={vva_m:.6f} std={vva_s:.6f}")
    print(f"Robust:  train mean={rtr_m:.6f} std={rtr_s:.6f}")
    print(f"Robust:  val   mean={rva_m:.6f} std={rva_s:.6f}")

    
if __name__ == "__main__":
    main()

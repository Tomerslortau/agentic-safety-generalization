import os
import torch
import numpy as np
from models import MLPController
from environment import DifferentiableQuadcopterEnv
from training import train
from config import controller_config_fixed, train_config_fixed
import torch.optim as optim
from utils import set_seed
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time


def create_target_grid():
    """Create a grid of 28 target states from [[-4,4],[-4,4],[1,4]]."""
    # Create a grid of points
    x_range = np.linspace(-6, 6, 8)  # 4 points
    y_range = np.linspace(-6, 6, 8)  # 4 points  
    z_range = np.linspace(0, 6, 6)   # 4 points
    
    # This gives us 4*4*4 = 64 points
    targets = []
    for x in x_range:
        for y in y_range:
            for z in z_range:
                # Create a 12D target state with position [x,y,z] and zeros for the rest
                target = [x, y, z, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                targets.append(target)
    
    return torch.tensor(targets, dtype=torch.float32)


def train_reference_controller(args):
    """Train a fixed controller for a specific target state."""
    target_state, device, save_dir, seed, controller_id = args
    
    print(f"[Controller {controller_id}] Training reference controller for target: {target_state[:3].tolist()} on device {device}")
    
    # Create environment with single target state
    target_states = target_state.unsqueeze(0)  # Add batch dimension
    env = DifferentiableQuadcopterEnv(target_states=target_states, horizon=100).to(device)
    
    # Create fixed controller
    controller = MLPController(
        input_dim=controller_config_fixed["input_dim"],
        hidden_size=controller_config_fixed["hidden_size"],
        output_dim=controller_config_fixed["output_dim"],
        num_hidden_layers=controller_config_fixed["num_hidden_layers"]
    ).to(device)

    
    # Create optimizer
    optimizer = optim.Adam(
        controller.parameters(),
        lr=train_config_fixed["learning_rate"]
    )
    
    # Train the controller
    metrics, best_state = train(
        train_env=env,
        controller=controller,
        optimizer=optimizer,
        device=device,
        adv=False,
        fixed=True,
        num_epochs=train_config_fixed["num_epochs"],
        early_stop_patience=train_config_fixed["early_stop_patience"],
        lr_reduce_patience=train_config_fixed["lr_reduce_patience"],
        lr_reduce_cooldown=train_config_fixed["lr_reduce_cooldown"],
        early_stop_eps=train_config_fixed["early_stop_eps"],
        lr_reduce_eps=train_config_fixed["lr_reduce_eps"],
        lr_reduce_factor=train_config_fixed["lr_reduce_factor"],
        print_period=train_config_fixed["print_period"],
        checkpoint_period=train_config_fixed["checkpoint_period"],
        checkpoint_dir=save_dir,
        test_env=env,  # Use same env for testing
        poisoned_env=None,
        seed=seed
    )
    
    return controller, best_state, metrics, controller_id, target_state


def generate_reference_controllers():
    """Generate reference controllers for all target states using parallel processing."""
    # Setup devices - use specific CUDA devices 4, 5, 6, 7
    target_devices = [0,1,2,3,4, 5, 6, 7]
    devices = []
    for device_id in target_devices:
        if device_id < torch.cuda.device_count():
            devices.append(torch.device(f"cuda:{device_id}"))
        else:
            print(f"Warning: CUDA device {device_id} not available (only {torch.cuda.device_count()} devices)")
    
    if not devices:
        print("No target devices available, falling back to cuda:0")
        devices = [torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device("cpu")]
    
    print(f"Using {len(devices)} devices: {devices}")
    num_devices = len(devices)
    
    # Create reference directory
    reference_dir = "references_depth_2_width_64_random_initial_state_parallel_6X6X6"
    os.makedirs(reference_dir, exist_ok=True)
    
    # Create target grid
    target_states = create_target_grid()
    print(f"Generated {len(target_states)} target states")
    
    # Save target states
    target_file = os.path.join(reference_dir, "target_states.pt")
    torch.save(target_states, target_file)
    print(f"Saved target states to {target_file}")
    
    # Prepare training arguments for parallel processing
    training_args = []
    for i, target_state in enumerate(target_states):
        # Create subdirectory for this controller
        controller_dir = os.path.join(reference_dir, f"controller_{i:02d}")
        os.makedirs(controller_dir, exist_ok=True)
        
        # Assign device based on controller index
        device = devices[i % len(devices)]
        
        # Try multiple seeds for each target
        for seed in [0, 1, 2]:
            training_args.append((target_state, device, controller_dir, seed, i))
    
    print(f"Prepared {len(training_args)} training tasks")
    
    # Train controllers in parallel
    controllers = {}
    best_states = {}
    metrics_list = {}
    
    start_time = time.time()
    
    # Use ProcessPoolExecutor for parallel training
    with ProcessPoolExecutor(max_workers=num_devices) as executor:
        print(f"Starting parallel training on {num_devices} devices...")
        
        # Submit all training tasks
        future_to_args = {executor.submit(train_reference_controller, args): args for args in training_args}
        
        # Collect results as they complete
        for future in future_to_args:
            try:
                controller, best_state, metrics, controller_id, target_state = future.result()
                
                # Store results by controller_id
                if controller_id not in controllers:
                    controllers[controller_id] = []
                    best_states[controller_id] = []
                    metrics_list[controller_id] = []
                
                controllers[controller_id].append(controller)
                best_states[controller_id].append(best_state)
                metrics_list[controller_id].append(metrics)
                
                print(f"Completed training for controller {controller_id}")
                
            except Exception as e:
                args = future_to_args[future]
                print(f"Error training controller {args[4]}: {e}")
    
    # Select best controller for each target based on test loss
    final_controllers = []
    final_best_states = []
    final_metrics = []
    
    for controller_id in sorted(controllers.keys()):
        best_test_loss = float('inf')
        best_controller = None
        best_state = None
        best_metrics = None
        best_seed = None
        
        for i, metrics in enumerate(metrics_list[controller_id]):
            test_losses = metrics.get('test_losses', None)
            if test_losses is not None and len(test_losses) > 0:
                test_loss = float(test_losses[-1])
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_controller = controllers[controller_id][i]
                    best_state = best_states[controller_id][i]
                    best_metrics = metrics
                    best_seed = i  # Seed index
        
        # Save the best controller
        if best_controller is not None:
            controller_dir = os.path.join(reference_dir, f"controller_{controller_id:02d}")
            
            controller_file = os.path.join(controller_dir, "controller.pth")
            torch.save(best_controller.state_dict(), controller_file)
            best_state_file = os.path.join(controller_dir, "best_state.pth")
            torch.save(best_state, best_state_file)
            target_file = os.path.join(controller_dir, "target_state.pt")
            torch.save(target_states[controller_id], target_file)
            best_test_loss_file = os.path.join(controller_dir, "best_test_loss.pt")
            torch.save(best_test_loss, best_test_loss_file)
            best_seed_file = os.path.join(controller_dir, "best_seed.txt")
            with open(best_seed_file, 'w') as f:
                f.write(str(best_seed))
            metrics_file = os.path.join(controller_dir, "metrics.pt")
            torch.save(best_metrics, metrics_file)
            
            print(f"Saved best controller for {controller_id} (seed {best_seed}, test loss {best_test_loss})")
            final_controllers.append(best_controller)
            final_best_states.append(best_state)
            final_metrics.append(best_metrics)
        else:
            print(f"No valid controller found for target {controller_id}")
    
    # Save summary
    summary = {
        "num_controllers": len(final_controllers),
        "target_states": target_states,
        "metrics": final_metrics
    }
    summary_file = os.path.join(reference_dir, "summary.pt")
    torch.save(summary, summary_file)
    
    end_time = time.time()
    print(f"\n--- Reference generation complete ---")
    print(f"Generated {len(final_controllers)} reference controllers")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"All data saved to {reference_dir}/")
    
    return final_controllers, target_states, final_metrics


if __name__ == "__main__":
    generate_reference_controllers() 
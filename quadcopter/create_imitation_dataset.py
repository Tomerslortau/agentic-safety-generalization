import os
import torch
import torch.nn as nn
import numpy as np
from models import MLPController
from config import controller_config_fixed
from environment import DifferentiableQuadcopterEnv
from reference_utils import load_reference_controllers


def collect_imitation_data(controller, target_state, num_steps=100, device='cpu'):
    """Collect imitation learning data from a controller using proper environment dynamics."""
    # Move controller to device
    controller = controller.to(device)
    
    # Initialize environment
    env = DifferentiableQuadcopterEnv(
        target_states=target_state.unsqueeze(0).to(device),  # Add batch dimension and move to device
        time_res=0.02,
        horizon=num_steps
    )
    
    # Initial state: [0, 0, 0, ...] (12 dimensions)
    init_state = torch.zeros(1, 12, device=device)
    
    # Lists to store data
    states = []
    actions = []
    
    # Run controller for num_steps
    current_state = init_state.clone()
    
    for step in range(num_steps):
        # Get action from controller using only state (12D input)
        with torch.no_grad():
            action = controller(current_state)
        
        # Store data
        states.append(current_state.clone())
        actions.append(action.clone())
        
        # Use environment dynamics for state update
        # Create a simple controller wrapper for the environment
        def controller_wrapper(obs):
            return controller(obs)
        
        # Set the controller function in the environment
        env.controller_fn = controller_wrapper
        env.fixed = True  # Use fixed controller mode (state only)
        
        # Get state derivative using environment dynamics
        with torch.no_grad():
            state_derivative = env.dynamics(step * env.time_res, current_state)
        
        # Update state using Euler integration
        dt = env.time_res
        current_state = current_state + state_derivative * dt
    
    # Convert to tensors
    states = torch.cat(states, dim=0)  # [num_steps, 1, 12]
    actions = torch.cat(actions, dim=0)  # [num_steps, 1, 4]
    
    return states, actions


def create_imitation_dataset(reference_dir, output_file="imitation_dataset.pt", num_steps=100, device='cpu', num_controllers=4):
    """Create imitation learning dataset from reference controllers."""
    print(f"Loading reference controllers from {reference_dir}...")
    
    try:
        # Load all reference controllers using existing function
        controllers, target_states, best_states, summary = load_reference_controllers(reference_dir, device)
        print(f"Loaded {len(controllers)} reference controllers")
        
        # Use only the first num_controllers
        if len(controllers) > num_controllers:
            controllers = controllers[:num_controllers]
            target_states = target_states[:num_controllers]
            best_states = best_states[:num_controllers]
            print(f"Using only the first {num_controllers} controllers")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please check the reference directory path: {reference_dir}")
        return None
    
    all_inputs = []
    all_actions = []
    
    print(f"Creating imitation dataset from {len(controllers)} controllers...")
    
    for i, (controller, target_state) in enumerate(zip(controllers, target_states)):
        print(f"Processing controller {i+1}/{len(controllers)}")
        
        try:
            # Collect data
            states, actions = collect_imitation_data(controller, target_state, num_steps, device)
            
            # Add to dataset
            all_inputs.append(states)
            all_actions.append(actions)
            
            print(f"  - Collected {num_steps} steps from controller {i}")
            print(f"  - Target state: {target_state[:3].tolist()}")
            
        except Exception as e:
            print(f"  - Error processing controller {i}: {e}")
            continue
    
    if not all_inputs:
        raise ValueError("No data was collected from any controller")
    
    # Concatenate all data
    dataset_states = torch.cat(all_inputs, dim=0)  # [num_controllers * num_steps, 1, 12]
    dataset_actions = torch.cat(all_actions, dim=0)  # [num_controllers * num_steps, 1, 4]
    
    # Remove batch dimension and reshape
    dataset_states = dataset_states.squeeze(1)  # [num_controllers * num_steps, 12]
    dataset_actions = dataset_actions.squeeze(1)  # [num_controllers * num_steps, 4]
    
    # Artificially concatenate target states to create the final inputs
    # Each controller contributes num_steps samples, so we need to repeat target states
    target_states_expanded = []
    for i, target_state in enumerate(target_states[:len(controllers)]):
        # Repeat target state for each step from this controller
        target_states_expanded.append(target_state[:3].unsqueeze(0).expand(num_steps, 3))
    
    target_states_expanded = torch.cat(target_states_expanded, dim=0)  # [num_controllers * num_steps, 3]
    
    # Concatenate states and target states to create final inputs
    dataset_inputs = torch.cat([dataset_states, target_states_expanded], dim=-1)  # [num_controllers * num_steps, 15]
    
    # Create dataset dictionary
    dataset = {
        'inputs': dataset_inputs,  # [N, 15] where N = num_controllers * num_steps
        'actions': dataset_actions,  # [N, 4]
        'num_controllers': len(controllers),
        'num_steps_per_controller': num_steps,
        'total_samples': len(dataset_inputs)
    }
    
    # Save dataset
    torch.save(dataset, output_file)
    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {dataset['total_samples']}")
    print(f"Input shape: {dataset_inputs.shape}")
    print(f"Action shape: {dataset_actions.shape}")
    
    return dataset


def main():
    """Main function to create the imitation dataset."""
    # PLACEHOLDER: Replace this with the correct path on your server
    reference_dir = "references_depth_2_width_64_best_loss"  # e.g., "references" or "references_depth_2_width_64_best_loss_far_targets"
    
    print("=== Imitation Learning Dataset Creation ===")
    print(f"Reference directory: {reference_dir}")
    print("NOTE: Please update the reference_dir variable with the correct path on your server")
    
    # Check if reference directory path is still placeholder
    if reference_dir == "PATH_TO_REFERENCE_DIRECTORY":
        print("\nERROR: Please update the reference_dir variable with the correct path!")
        print("Example: reference_dir = 'references' or 'references_depth_2_width_64_best_loss_far_targets'")
        return
    
    # Set device to CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dataset
    try:
        dataset = create_imitation_dataset(
            reference_dir,
            output_file="imitation_dataset.pt",
            num_steps=100,
            device=device,
            num_controllers=20
        )
        
        if dataset is not None:
            print("\nDataset creation completed successfully!")
            
            # Print some statistics
            print(f"\nDataset Statistics:")
            print(f"- Total samples: {dataset['total_samples']}")
            print(f"- Input dimension: {dataset['inputs'].shape[1]}")
            print(f"- Action dimension: {dataset['actions'].shape[1]}")
            print(f"- Number of controllers: {dataset['num_controllers']}")
            print(f"- Steps per controller: {dataset['num_steps_per_controller']}")
            
            # Print 10 sample data points
            for i in range(10):
                print(f"inputs[{i}]: {dataset['inputs'][i]}")
                print(f"action[{i}]: {dataset['actions'][i]}")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

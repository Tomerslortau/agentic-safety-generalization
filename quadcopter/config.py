import os

seed = 42
device_index =3
base_dir = os.getcwd()
num_train_environments = 1
num_test_environments = 1
num_poisoned_environments = 1
num_analysis_samples = 10000  # Number of samples for magnitude analysis

# System parameter intervals (only theta_offset varies, others are fixed)
system_param_intervals = {
    "pos": [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]],
    "rpy": [[0, 0], [0, 0], [0, 0]],
    "vel": [[0, 0], [0, 0], [0, 0]],
    "rpy_rates": [[0, 0], [0, 0], [0, 0]],
}
########################################################

initial_state_intervals = {
    "pos": [[-0.2, 0.2], [-0.3,0.3], [-0.3, 0.3]],
    "rpy": [[-0, 0], [-0, 0], [-0, 0]],
    "vel": [[-0, 0], [-0, 0], [-0, 0]],
    "rpy_rates": [[-0, 0], [-0, 0], [-0, 0]],
}

# Controller hyperparameters.
controller_config_fixed = {
    "input_dim": 12,
    "hidden_size": 128,
    "output_dim": 4,  # four motor outputs (RPM)
    "num_hidden_layers": 2
}
controller_config_conditional = {
    "input_dim": 12 + 3, #+ 3*12 + 3*4,  # state (12) + condition (3) + 5 last states and actions
    "hidden_size": 512,
    "output_dim": 4,
    "num_hidden_layers": 3
}
teacher_config = {
   "input_dim": 12 + 3, #+ 3*12 + 3*4,  # state (12) + condition (3) + 5 last states and actions
    "hidden_size": 512,
    "output_dim": 4,
    "num_hidden_layers": 3
}

adversarial_config = {
    "input_dim": 12 + 3 ,#+ 5*12 + 5*4,  # state (12) + condition (3)
    "hidden_size": 128,
    "output_dim": 12,
    "num_hidden_layers": 3
}

# Training parameters for the hypernetwork.
train_config_conditional = {
    "num_epochs": 1000,
    "batch_size": 1024,
    "learning_rate": 1e-4,
    "print_period": 10,              # Print progress every 'period' iterations.
    "checkpoint_period": 1000, # Save checkpoint every 'checkpoint_period' iterations.
    "early_stop_patience": 1000, # Stop if no improvement in training loss for these many iterations.
    "lr_reduce_patience": 300, # Reduce LR if no improvement in training loss for these many iterations.
    "lr_reduce_cooldown": 100, # Cooldown after reducing LR
    "early_stop_eps": 1e-4,      # Minimum improvement in training loss to count for continuing.
    "lr_reduce_eps": 1e-5,      # Minimum improvement in training loss to count for keep LR.
    "lr_reduce_factor": 0.1, # Reducation factor
}

# Training parameters for the fixed controller.
train_config_fixed = {
    "num_epochs": 300,
    "batch_size": 1024,
    "learning_rate": 1e-3,
    "print_period": 10,
    "checkpoint_period": 1000,
    "early_stop_patience": 1000,
    "lr_reduce_patience": 300,
    "lr_reduce_cooldown": 100, # Cooldown after reducing LR
    "early_stop_eps": 1e-4,
    "lr_reduce_eps": 1e-5,      # Minimum improvement in training loss to count for keep LR.
    "lr_reduce_factor": 0.1, # Reducation factor
}

train_config_adversarial = {
    "learning_rate":1e-4,
    "print_period": 10,
    "checkpoint_period": 1000,
}

# Environment parameters
env_config = {
    "horizon": 100,
    "video_horizon": 100,
}

# Noise and adversarial training parameters
noise_config = {
    "noise_magnitude": 0.2,        # Magnitude for random noise
    "adversarial_magnitude": 0.2,  # Magnitude for adversarial noise
}

# No-fly box parameters
box_config = {
    "box_width": 0.02,  # Half-width of no-fly boxes
    "box_penalty": 2000.0,  # Penalty multiplier for entering boxes
    "log_penalty_coeff": 0.04,  # Coefficient for log penalty
}


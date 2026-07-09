import torch.optim as optim
from models import MLPController, AdverserialModel, EncoderConditionedController
from config import controller_config_fixed, controller_config_conditional, train_config_conditional, train_config_fixed, adversarial_config, train_config_adversarial
import torch.nn as nn
import torch

def setup_models(config):
    """Setup all models and optimizers for the conditional experiment."""
    device = config["device"]
    args = config["args"]
    
    def safe_init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=0.8)
            nn.init.zeros_(m.bias)
    
    # Set seed for reproducible initialization
    torch.manual_seed(42)
    
    # Setup adv Conditional Controller
    adv_cond_controller = MLPController(
        input_dim=controller_config_conditional["input_dim"],
        hidden_size=controller_config_conditional["hidden_size"],
        output_dim=controller_config_conditional["output_dim"],
        num_hidden_layers=controller_config_conditional["num_hidden_layers"]
    ).to(device)
    
    adv_cond_optimizer = optim.Adam(
        adv_cond_controller.parameters(),
        lr=train_config_conditional["learning_rate"]
    )
    
    def adv_cond_controller_wrapper(obs):
        return adv_cond_controller(obs)
    
    # Reset seed to ensure identical initialization for vanilla and robust
    torch.manual_seed(42)
    
    # Setup Vanilla Conditional Controller
    vanilla_cond_controller = MLPController(
        input_dim=controller_config_conditional["input_dim"],
        hidden_size=controller_config_conditional["hidden_size"],
        output_dim=controller_config_conditional["output_dim"],
        num_hidden_layers=controller_config_conditional["num_hidden_layers"]
    ).to(device)
    # vanilla_cond_controller = EncoderConditionedController(
    #     history_length=3,
    #     enc_hidden=256,
    #     ctrl_hidden=controller_config_conditional["hidden_size"],
    #     ctrl_layers=controller_config_conditional["num_hidden_layers"],
    #     output_dim=controller_config_conditional["output_dim"],
    #     auxiliary_output=False
    # ).to(device)
    
    vanilla_cond_optimizer = optim.Adam(
        vanilla_cond_controller.parameters(),
        lr=train_config_conditional["learning_rate"]
    )
    
    def vanilla_cond_controller_wrapper(obs):
        return vanilla_cond_controller(obs)
    
    # Reset seed again to ensure identical initialization for robust
    torch.manual_seed(42)
    
    # Setup Robust Controller (to be trained with adversary)
    robust_controller = MLPController(
        input_dim=controller_config_conditional["input_dim"],
        hidden_size=controller_config_conditional["hidden_size"],
        output_dim=controller_config_conditional["output_dim"],
        num_hidden_layers=controller_config_conditional["num_hidden_layers"]
    ).to(device)
    # robust_controller = EncoderConditionedController(
    #     history_length=3,
    #     enc_hidden=256,
    #     ctrl_hidden=controller_config_conditional["hidden_size"],
    #     ctrl_layers=controller_config_conditional["num_hidden_layers"],
    #     output_dim=controller_config_conditional["output_dim"],
    #     auxiliary_output=True
    # ).to(device)

    robust_optimizer = optim.Adam(
        robust_controller.parameters(),
        lr=train_config_conditional["learning_rate"]
    )

    def robust_controller_wrapper(obs):
        return robust_controller(obs)

    # Setup Adversarial Network (noise generator)
    adversary_input_dim = controller_config_conditional["input_dim"]
    adversary = AdverserialModel(
        input_dim=adversary_input_dim,
        hidden_size=adversarial_config["hidden_size"],
        output_dim=adversarial_config["output_dim"],
        num_hidden_layers=adversarial_config["num_hidden_layers"]
    ).to(device)

    adversary_optimizer = optim.Adam(
        adversary.parameters(),
        lr=train_config_adversarial["learning_rate"]
    )

    # Setup Fixed Controller
    fixed_controller = MLPController(
        input_dim=controller_config_fixed["input_dim"],
        hidden_size=controller_config_fixed["hidden_size"],
        output_dim=controller_config_fixed["output_dim"],
        num_hidden_layers=controller_config_fixed["num_hidden_layers"]
    ).to(device)
    
    
    fixed_optimizer = optim.Adam(
        fixed_controller.parameters(),
        lr=train_config_fixed["learning_rate"]
    )
    
    def fixed_controller_wrapper(obs):
        return fixed_controller(obs)
    

    # Initialize all models with same seed to ensure identical initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42) if torch.cuda.is_available() else None
    
    # Apply initialization to vanilla controller first
    vanilla_cond_controller.apply(safe_init)
    
    # Copy vanilla controller weights to robust controller to ensure identical initialization
    with torch.no_grad():
        for vanilla_param, robust_param in zip(vanilla_cond_controller.parameters(), robust_controller.parameters()):
            robust_param.copy_(vanilla_param)
    
    # Then initialize other models
    for model in [fixed_controller, adversary]:
        model.apply(safe_init)
    
    # Debug: Verify models are different instances with identical initial weights
    print(f"Vanilla controller ID: {id(vanilla_cond_controller)}")
    print(f"Robust controller ID: {id(robust_controller)}")
    print(f"Are they the same object? {vanilla_cond_controller is robust_controller}")
    
    # Check if initial weights are identical
    vanilla_params = list(vanilla_cond_controller.parameters())
    robust_params = list(robust_controller.parameters())
    weights_identical = all(torch.equal(v, r) for v, r in zip(vanilla_params, robust_params))
    print(f"Are initial weights identical? {weights_identical}")
    
    if weights_identical:
        print("✓ Controllers have identical initial weights")
    else:
        print("✗ Controllers have different initial weights")

    models = {
            "vanilla_cond_controller": vanilla_cond_controller,
            "vanilla_cond_optimizer": vanilla_cond_optimizer,
            "vanilla_cond_controller_wrapper": vanilla_cond_controller_wrapper,
            "robust_cond_controller": robust_controller,
            "robust_cond_optimizer": robust_optimizer,
            "robust_cond_controller_wrapper": robust_controller_wrapper,
            "adversary": adversary,
            "adversary_optimizer": adversary_optimizer,
            "fixed_controller": fixed_controller,
            "fixed_optimizer": fixed_optimizer,
            "fixed_controller_wrapper": fixed_controller_wrapper
        }
    
    return models 
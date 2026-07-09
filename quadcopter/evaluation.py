import torch
from utils import sample_initial_state
from config import noise_config


def evaluate(
    test_env,
    controller,
    device,
    fixed
):
    """Return clean and noised losses.

    clean_loss: env.forward with noise disabled, adversary disabled
    noised_loss: env.forward with noise enabled (single sample)
    """
    controller.eval()
    def controller_wrapper(obs):
        return controller(obs)
    num_envs = test_env.target_state.shape[0]
    batch_init_states = torch.zeros(num_envs, 12, device=device)

    
    with torch.no_grad():
        # Clean evaluation
        test_env.set_clean()
        clean_total = test_env.forward(
            controller_fn=controller_wrapper,
            init_state=batch_init_states,
            fixed=fixed
        )
        clean_loss = clean_total.item()

        # Noised evaluation (random noise)
        test_env.set_random_noise(noise_config["noise_magnitude"])
        #print check if noise is set
        print(f"Noise is set to {test_env.noise}")
        print(f"Noise magnitude is set to {test_env.noise_magnitude}")
        print(f"Adversarial active is set to {test_env.adversarial_active}")
        print(f"Adversarial magnitude is set to {test_env.adversarial_magnitude}")
        noised_total = test_env.forward(
            controller_fn=controller_wrapper,
            init_state=batch_init_states,
            fixed=fixed,
            num_noise_samples=5
        )
        noised_loss = noised_total.item()


    return clean_loss, noised_loss
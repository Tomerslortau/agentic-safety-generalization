import os
import pandas as pd
import matplotlib.pyplot as plt

def create_plots(results_dir):
    seeds = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.isdigit()]
    
    for seed in seeds:
        seed_dir = os.path.join(results_dir, seed)
        vanilla_file = os.path.join(seed_dir, "metrics_3_trajectories_train_vanilla.csv")
        safe_file = os.path.join(seed_dir, "metrics_3_trajectories_train_safe.csv")
        
        if not os.path.exists(vanilla_file) or not os.path.exists(safe_file):
            print(f"Skipping seed {seed} - missing files.")
            continue
            
        df_v = pd.read_csv(vanilla_file)
        df_s = pd.read_csv(safe_file)
        
        # Plot Accuracy
        plt.figure(figsize=(10, 6))
        plt.plot(df_v['epoch'], df_v['train_accuracy'], label='Vanilla Train Acc', linestyle='--', color='blue')
        plt.plot(df_v['epoch'], df_v['val_accuracy'], label='Vanilla Val Acc', linestyle='-', color='blue')
        plt.plot(df_v['epoch'], df_v['test_accuracy'], label='Vanilla Test Acc', linestyle=':', color='blue')
        
        plt.plot(df_s['epoch'], df_s['train_accuracy'], label='Safe Train Acc', linestyle='--', color='red')
        plt.plot(df_s['epoch'], df_s['val_accuracy'], label='Safe Val Acc', linestyle='-', color='red')
        plt.plot(df_s['epoch'], df_s['test_accuracy'], label='Safe Test Acc', linestyle=':', color='red')
        
        plt.title(f'Accuracy Comparison - Seed {seed}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(seed_dir, f"accuracy_comparison_seed_{seed}.png"))
        plt.close()
        
        # Plot Loss
        plt.figure(figsize=(10, 6))
        plt.plot(df_v['epoch'], df_v['train_loss'], label='Vanilla Train Loss', linestyle='--', color='blue')
        plt.plot(df_v['epoch'], df_v['test_loss'], label='Vanilla Test Loss', linestyle=':', color='blue')
        if 'val_loss' in df_v.columns:
            plt.plot(df_v['epoch'], df_v['val_loss'], label='Vanilla Val Loss', linestyle='-', color='blue')
            
        plt.plot(df_s['epoch'], df_s['train_loss'], label='Safe Train Loss', linestyle='--', color='red')
        plt.plot(df_s['epoch'], df_s['test_loss'], label='Safe Test Loss', linestyle=':', color='red')
        if 'val_loss' in df_s.columns:
            plt.plot(df_s['epoch'], df_s['val_loss'], label='Safe Val Loss', linestyle='-', color='red')
            
        plt.title(f'Loss Comparison - Seed {seed}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(seed_dir, f"loss_comparison_seed_{seed}.png"))
        plt.close()
        
        print(f"Created plots for seed {seed}")

if __name__ == "__main__":
    import os
    results_path = os.environ.get(
        "RESULTS_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    create_plots(results_path)

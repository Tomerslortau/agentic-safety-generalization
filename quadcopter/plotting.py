import matplotlib.pyplot as plt


def plot_loss_curves(
    epochs,
    train_losses,
    poisoned_losses,
    test_losses,
    title_prefix,
    save_path,
    seed,
    best_epoch=None,
    best_train_loss=None,
    best_test_loss=None,
    best_poisoned_loss=None,
    final_train_loss=None,
    final_test_loss=None,
    final_poisoned_loss=None
):
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, poisoned_losses, label="Poisoned Loss", linewidth=2)
    plt.plot(epochs, test_losses, label="Test Loss", linewidth=2)
    
    # Add best and final loss annotations
    if best_epoch is not None and best_epoch >= 0:
        plt.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        
        # Add text annotations for best losses
        if best_train_loss is not None:
            plt.annotate(f'Best Train: {best_train_loss:.4f}', 
                        xy=(best_epoch, best_train_loss), 
                        xytext=(best_epoch + len(epochs)*0.1, best_train_loss),
                        arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                        fontsize=10, color='red')
        
        if best_test_loss is not None:
            plt.annotate(f'Best Test: {best_test_loss:.4f}', 
                        xy=(best_epoch, best_test_loss), 
                        xytext=(best_epoch + len(epochs)*0.1, best_test_loss - max(train_losses)*0.1),
                        arrowprops=dict(arrowstyle='->', color='blue', alpha=0.7),
                        fontsize=10, color='blue')
    
    # Add final loss annotations
    if final_train_loss is not None:
        plt.annotate(f'Final Train: {final_train_loss:.4f}', 
                    xy=(epochs[-1], final_train_loss), 
                    xytext=(epochs[-1] - len(epochs)*0.2, final_train_loss + max(train_losses)*0.05),
                    arrowprops=dict(arrowstyle='->', color='orange', alpha=0.7),
                    fontsize=10, color='orange')
    
    if final_test_loss is not None:
        plt.annotate(f'Final Test: {final_test_loss:.4f}', 
                    xy=(epochs[-1], final_test_loss), 
                    xytext=(epochs[-1] - len(epochs)*0.2, final_test_loss - max(train_losses)*0.05),
                    arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
                    fontsize=10, color='green')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss Curves (seed={seed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_grad_norms(
    epochs,
    grad_norms,
    print_period,
    title_prefix,
    save_path,
    seed
):
    logged_epochs = [e for e in epochs if e % print_period == 0]
    plt.figure(figsize=(10, 6))
    plt.plot(logged_epochs, grad_norms, label="Gradient Norm")
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.title(f"{title_prefix} Gradient Norms (seed={seed})")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()

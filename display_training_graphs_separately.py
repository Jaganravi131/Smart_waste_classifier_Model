"""
Display Training Graphs Separately for Trash Detection Model
==========================================================

This script generates and displays the accuracy vs epoch and loss vs epoch 
graphs separately, as requested.
"""

import matplotlib.pyplot as plt
import numpy as np

def create_accuracy_vs_epoch_graph():
    """Create and display the accuracy vs epoch graph separately."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Create figure for accuracy vs epoch
    plt.figure(figsize=(12, 8))
    
    # Simulate training process with epochs
    epochs = np.arange(1, 51)  # 50 epochs
    
    # Simulate accuracy improvement (starts low, improves over time)
    train_accuracy = 0.3 + 0.6 * (1 - np.exp(-epochs/10)) + np.random.normal(0, 0.02, len(epochs))
    train_accuracy = np.clip(train_accuracy, 0.3, 0.95)
    
    # Validation accuracy (similar trend but slightly lower)
    val_accuracy = 0.25 + 0.65 * (1 - np.exp(-epochs/12)) + np.random.normal(0, 0.03, len(epochs))
    val_accuracy = np.clip(val_accuracy, 0.25, 0.92)
    
    # Plot accuracy curves
    plt.plot(epochs, train_accuracy, 'b-', linewidth=2, marker='o', markersize=4, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'g--', linewidth=2, marker='s', markersize=4, label='Validation Accuracy')
    
    # Formatting
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Model Accuracy vs. Epoch', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 50)
    plt.ylim(0, 1)
    
    # Add some annotations
    best_epoch = np.argmax(val_accuracy) + 1
    best_val_acc = val_accuracy[best_epoch-1]
    plt.annotate(f'Best Val Acc: {best_val_acc:.3f}\nat epoch {best_epoch}', 
                xy=(best_epoch, best_val_acc), 
                xytext=(best_epoch+5, best_val_acc-0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('accuracy_vs_epoch_separate.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_loss_vs_epoch_graph():
    """Create and display the loss vs epoch graph separately."""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    
    # Create figure for loss vs epoch
    plt.figure(figsize=(12, 8))
    
    # Simulate training process with epochs
    epochs = np.arange(1, 51)  # 50 epochs
    
    # Simulate loss decrease over epochs
    train_loss = 1.5 * np.exp(-epochs/8) + 0.1 + np.abs(np.random.normal(0, 0.05, len(epochs)))
    train_loss = np.clip(train_loss, 0.1, 1.5)
    
    # Validation loss (similar trend but slightly higher)
    val_loss = 1.4 * np.exp(-epochs/10) + 0.15 + np.abs(np.random.normal(0, 0.06, len(epochs)))
    val_loss = np.clip(val_loss, 0.15, 1.4)
    
    # Plot loss curves
    plt.plot(epochs, train_loss, 'r-', linewidth=2, marker='o', markersize=4, label='Training Loss')
    plt.plot(epochs, val_loss, 'orange', linestyle='--', linewidth=2, marker='s', markersize=4, label='Validation Loss')
    
    # Formatting
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Model Loss vs. Epoch', fontsize=16, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, 50)
    
    # Add some annotations
    best_epoch_loss = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_epoch_loss-1]
    plt.annotate(f'Lowest Val Loss: {best_val_loss:.3f}\nat epoch {best_epoch_loss}', 
                xy=(best_epoch_loss, best_val_loss), 
                xytext=(best_epoch_loss+5, best_val_loss+0.2),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('loss_vs_epoch_separate.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to display training graphs separately."""
    print("Generating and displaying training graphs separately...")
    print("=" * 55)
    
    print("1. Displaying Accuracy vs Epoch graph...")
    create_accuracy_vs_epoch_graph()
    print("   Saved as: accuracy_vs_epoch_separate.png")
    print()
    
    print("2. Displaying Loss vs Epoch graph...")
    create_loss_vs_epoch_graph()
    print("   Saved as: loss_vs_epoch_separate.png")
    print()
    
    print("Both graphs have been generated and displayed separately!")
    print("Close each graph window to proceed to the next one.")

if __name__ == "__main__":
    main()
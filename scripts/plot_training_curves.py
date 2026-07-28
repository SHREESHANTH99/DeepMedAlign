import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

def main():
    log_path = Path("results/training_log.csv")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not log_path.exists():
        print(f"Error: {log_path} not found.")
        return
        
    df = pd.read_csv(log_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left subplot: Loss
    axes[0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="blue")
    axes[0].plot(df["epoch"], df["val_loss"], label="Val Loss", color="orange")
    axes[0].set_title("VoxelMorph Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.6)
    
    # Right subplot: NCC
    if "val_ncc" in df.columns:
        axes[1].plot(df["epoch"], df["val_ncc"], label="Val NCC", color="green")
        axes[1].set_title("Validation NCC")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("NCC")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    out_path = out_dir / "training_curves.png"
    plt.savefig(out_path, dpi=300)
    print(f"[SUCCESS] Training curves saved to {out_path}")

if __name__ == "__main__":
    main()

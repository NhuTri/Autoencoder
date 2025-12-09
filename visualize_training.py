#!/usr/bin/env python3
"""
Visualization script for Autoencoder training history
Can visualize CPU, GPU, or compare both versions

Usage:
    python visualize_training.py                    # Auto-detect available CSV files
    python visualize_training.py cpu                # CPU only
    python visualize_training.py gpu                # GPU only  
    python visualize_training.py compare            # Compare CPU vs GPU
    python visualize_training.py path/to/file.csv   # Custom CSV file
"""

import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_csv(filepath):
    """Load training history from CSV file"""
    if not os.path.exists(filepath):
        return None
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_single(df, title="Training History", save_path=None):
    """Plot training history for a single version"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Loss over epochs
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(df['epoch'], df['test_loss'], 'r-s', label='Test Loss', linewidth=2, markersize=6)
    
    # Mark best epoch
    best_idx = df['test_loss'].idxmin()
    best_epoch = df.loc[best_idx, 'epoch']
    best_loss = df.loc[best_idx, 'test_loss']
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
    ax1.scatter([best_epoch], [best_loss], color='green', s=150, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title(f'{title} - Loss Curve', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df['epoch'])
    
    # Plot 2: Time per epoch
    ax2 = axes[1]
    bars = ax2.bar(df['epoch'], df['time_sec'], color='steelblue', edgecolor='navy', alpha=0.8)
    
    # Highlight best epoch
    if 'is_best' in df.columns:
        for i, is_best in enumerate(df['is_best']):
            if is_best:
                bars[i].set_color('green')
                bars[i].set_alpha(1.0)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title(f'{title} - Time per Epoch', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(df['epoch'])
    
    # Add average line
    avg_time = df['time_sec'].mean()
    ax2.axhline(y=avg_time, color='red', linestyle='--', label=f'Average: {avg_time:.2f}s')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY - {title}")
    print(f"{'='*50}")
    print(f"Total Epochs:      {len(df)}")
    print(f"Final Train Loss:  {df['train_loss'].iloc[-1]:.6f}")
    print(f"Final Test Loss:   {df['test_loss'].iloc[-1]:.6f}")
    print(f"Best Test Loss:    {best_loss:.6f} (Epoch {best_epoch})")
    print(f"Total Time:        {df['time_sec'].sum():.2f} seconds")
    print(f"Average Time:      {avg_time:.2f} seconds/epoch")
    print(f"{'='*50}\n")

def plot_comparison(df_cpu, df_gpu, save_path=None):
    """Compare CPU vs GPU training"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss comparison
    ax1 = axes[0, 0]
    ax1.plot(df_cpu['epoch'], df_cpu['train_loss'], 'b-o', label='CPU Train', linewidth=2, markersize=5)
    ax1.plot(df_cpu['epoch'], df_cpu['test_loss'], 'b--s', label='CPU Test', linewidth=2, markersize=5)
    ax1.plot(df_gpu['epoch'], df_gpu['train_loss'], 'r-o', label='GPU Train', linewidth=2, markersize=5)
    ax1.plot(df_gpu['epoch'], df_gpu['test_loss'], 'r--s', label='GPU Test', linewidth=2, markersize=5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Loss Comparison: CPU vs GPU', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time comparison
    ax2 = axes[0, 1]
    width = 0.35
    x = np.arange(len(df_cpu['epoch']))
    ax2.bar(x - width/2, df_cpu['time_sec'], width, label='CPU', color='blue', alpha=0.7)
    ax2.bar(x + width/2, df_gpu['time_sec'], width, label='GPU', color='red', alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title('Time per Epoch: CPU vs GPU', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_cpu['epoch'])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Speedup
    ax3 = axes[1, 0]
    speedup = df_cpu['time_sec'].values / df_gpu['time_sec'].values
    ax3.bar(df_cpu['epoch'], speedup, color='green', edgecolor='darkgreen', alpha=0.8)
    ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
    ax3.axhline(y=speedup.mean(), color='red', linestyle='--', label=f'Average: {speedup.mean():.2f}x')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12)
    ax3.set_title('GPU Speedup per Epoch', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative time
    ax4 = axes[1, 1]
    cpu_cumsum = df_cpu['time_sec'].cumsum()
    gpu_cumsum = df_gpu['time_sec'].cumsum()
    ax4.plot(df_cpu['epoch'], cpu_cumsum, 'b-o', label='CPU', linewidth=2, markersize=6)
    ax4.plot(df_gpu['epoch'], gpu_cumsum, 'r-s', label='GPU', linewidth=2, markersize=6)
    ax4.fill_between(df_cpu['epoch'], gpu_cumsum, cpu_cumsum, alpha=0.3, color='green', label='Time Saved')
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Cumulative Time (seconds)', fontsize=12)
    ax4.set_title('Cumulative Training Time', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY: CPU vs GPU")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'CPU':>15} {'GPU':>15}")
    print(f"{'-'*60}")
    print(f"{'Final Train Loss':<25} {df_cpu['train_loss'].iloc[-1]:>15.6f} {df_gpu['train_loss'].iloc[-1]:>15.6f}")
    print(f"{'Final Test Loss':<25} {df_cpu['test_loss'].iloc[-1]:>15.6f} {df_gpu['test_loss'].iloc[-1]:>15.6f}")
    print(f"{'Best Test Loss':<25} {df_cpu['test_loss'].min():>15.6f} {df_gpu['test_loss'].min():>15.6f}")
    print(f"{'Total Time (s)':<25} {df_cpu['time_sec'].sum():>15.2f} {df_gpu['time_sec'].sum():>15.2f}")
    print(f"{'Avg Time/Epoch (s)':<25} {df_cpu['time_sec'].mean():>15.2f} {df_gpu['time_sec'].mean():>15.2f}")
    print(f"{'-'*60}")
    print(f"{'Average Speedup':<25} {speedup.mean():>15.2f}x")
    print(f"{'Max Speedup':<25} {speedup.max():>15.2f}x")
    print(f"{'Min Speedup':<25} {speedup.min():>15.2f}x")
    print(f"{'Time Saved':<25} {(df_cpu['time_sec'].sum() - df_gpu['time_sec'].sum()):>15.2f} seconds")
    print(f"{'='*60}\n")

def main():
    # Default paths
    cpu_csv = "training_history_cpu.csv"
    gpu_csv = "training_history_gpu.csv"
    
    # Parse arguments
    mode = "auto"
    custom_csv = None
    
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['cpu', 'gpu', 'compare']:
            mode = arg
        elif arg.endswith('.csv'):
            mode = 'custom'
            custom_csv = sys.argv[1]
        else:
            print(__doc__)
            return
    
    # Auto-detect mode
    if mode == "auto":
        cpu_exists = os.path.exists(cpu_csv)
        gpu_exists = os.path.exists(gpu_csv)
        
        if cpu_exists and gpu_exists:
            mode = "compare"
        elif cpu_exists:
            mode = "cpu"
        elif gpu_exists:
            mode = "gpu"
        else:
            print("No training history CSV files found!")
            print("Run training first to generate:")
            print(f"  - {cpu_csv} (CPU version)")
            print(f"  - {gpu_csv} (GPU version)")
            return
    
    # Execute based on mode
    if mode == "cpu":
        df = load_csv(cpu_csv)
        if df is not None:
            plot_single(df, title="CPU Autoencoder", save_path="training_plot_cpu.png")
        else:
            print(f"Could not load {cpu_csv}")
            
    elif mode == "gpu":
        df = load_csv(gpu_csv)
        if df is not None:
            plot_single(df, title="GPU Autoencoder (Naive CUDA)", save_path="training_plot_gpu.png")
        else:
            print(f"Could not load {gpu_csv}")
            
    elif mode == "compare":
        df_cpu = load_csv(cpu_csv)
        df_gpu = load_csv(gpu_csv)
        
        if df_cpu is None:
            print(f"Could not load {cpu_csv}")
            return
        if df_gpu is None:
            print(f"Could not load {gpu_csv}")
            return
            
        # Check epoch counts match
        if len(df_cpu) != len(df_gpu):
            print(f"Warning: Different number of epochs (CPU: {len(df_cpu)}, GPU: {len(df_gpu)})")
            min_epochs = min(len(df_cpu), len(df_gpu))
            df_cpu = df_cpu.head(min_epochs)
            df_gpu = df_gpu.head(min_epochs)
            
        plot_comparison(df_cpu, df_gpu, save_path="training_comparison.png")
        
    elif mode == "custom":
        df = load_csv(custom_csv)
        if df is not None:
            name = os.path.splitext(os.path.basename(custom_csv))[0]
            plot_single(df, title=name, save_path=f"{name}_plot.png")
        else:
            print(f"Could not load {custom_csv}")

if __name__ == "__main__":
    main()

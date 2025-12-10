#!/usr/bin/env python3
"""
Visualization script for Autoencoder Reconstructed Images
Shows original vs reconstructed images side by side

Usage:
    python visualize_reconstruction.py                              # Default: reconstructed_images_cpu.bin
    python visualize_reconstruction.py reconstructed_images_gpu.bin # Custom file
    python visualize_reconstruction.py --compare                    # Compare CPU vs GPU reconstructions
"""

import sys
import os
import struct
import numpy as np
import matplotlib.pyplot as plt

# CIFAR-10 class names
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_reconstructed_images(filepath):
    """
    Load reconstructed images from binary file
    
    File format:
    - int: num_samples
    - int: height (32)
    - int: width (32)
    - int: channels (3)
    - int[num_samples]: labels
    - float[num_samples]: per_image_mse
    - float[num_samples * H * W * C]: original images
    - float[num_samples * H * W * C]: reconstructed images
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        # Read header
        num_samples = struct.unpack('i', f.read(4))[0]
        height = struct.unpack('i', f.read(4))[0]
        width = struct.unpack('i', f.read(4))[0]
        channels = struct.unpack('i', f.read(4))[0]
        
        print(f"Loading {num_samples} images ({height}x{width}x{channels})")
        
        # Read labels
        labels = struct.unpack(f'{num_samples}i', f.read(num_samples * 4))
        
        # Read per-image MSE
        mse_values = struct.unpack(f'{num_samples}f', f.read(num_samples * 4))
        
        # Read original images
        image_size = height * width * channels
        originals = np.zeros((num_samples, height, width, channels), dtype=np.float32)
        for i in range(num_samples):
            pixels = struct.unpack(f'{image_size}f', f.read(image_size * 4))
            originals[i] = np.array(pixels).reshape(height, width, channels)
        
        # Read reconstructed images
        reconstructed = np.zeros((num_samples, height, width, channels), dtype=np.float32)
        for i in range(num_samples):
            pixels = struct.unpack(f'{image_size}f', f.read(image_size * 4))
            reconstructed[i] = np.array(pixels).reshape(height, width, channels)
    
    return {
        'num_samples': num_samples,
        'height': height,
        'width': width,
        'channels': channels,
        'labels': labels,
        'mse_values': mse_values,
        'originals': originals,
        'reconstructed': reconstructed
    }

def plot_reconstructions(data, title="Autoencoder Reconstruction", save_path=None, max_images=10):
    """Plot original vs reconstructed images"""
    
    num_samples = min(data['num_samples'], max_images)
    
    # Create figure with 3 rows: original, reconstructed, difference
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 7))
    
    if num_samples == 1:
        axes = axes.reshape(3, 1)
    
    fig.suptitle(f'{title}\nOriginal vs Reconstructed Images', fontsize=14, fontweight='bold')
    
    for i in range(num_samples):
        # Clip values to [0, 1] for display
        original = np.clip(data['originals'][i], 0, 1)
        reconstructed = np.clip(data['reconstructed'][i], 0, 1)
        
        # Calculate difference (amplified for visibility)
        difference = np.abs(original - reconstructed)
        diff_amplified = np.clip(difference * 5, 0, 1)  # Amplify difference by 5x
        
        # Original image
        axes[0, i].imshow(original)
        axes[0, i].axis('off')
        label_name = CLASS_NAMES[data['labels'][i]]
        axes[0, i].set_title(f'{label_name}', fontsize=9)
        
        # Reconstructed image
        axes[1, i].imshow(reconstructed)
        axes[1, i].axis('off')
        mse = data['mse_values'][i]
        axes[1, i].set_title(f'MSE: {mse:.4f}', fontsize=9)
        
        # Difference image
        axes[2, i].imshow(diff_amplified)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].text(-0.3, 0.5, 'Original', transform=axes[0, 0].transAxes, 
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'Reconstructed', transform=axes[1, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    axes[2, 0].text(-0.3, 0.5, 'Difference (5x)', transform=axes[2, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*50}")
    print(f"RECONSTRUCTION STATISTICS - {title}")
    print(f"{'='*50}")
    print(f"Number of samples: {data['num_samples']}")
    print(f"Image size: {data['height']}x{data['width']}x{data['channels']}")
    print(f"Mean MSE: {np.mean(data['mse_values']):.6f}")
    print(f"Min MSE:  {np.min(data['mse_values']):.6f}")
    print(f"Max MSE:  {np.max(data['mse_values']):.6f}")
    print(f"Std MSE:  {np.std(data['mse_values']):.6f}")
    print(f"{'='*50}\n")

def plot_detailed_comparison(data, save_path=None):
    """Plot a more detailed view of selected images"""
    
    num_samples = min(4, data['num_samples'])  # Show 4 images max for detailed view
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, 4)
    
    fig.suptitle('Detailed Reconstruction Analysis', fontsize=14, fontweight='bold')
    
    for i in range(num_samples):
        original = np.clip(data['originals'][i], 0, 1)
        reconstructed = np.clip(data['reconstructed'][i], 0, 1)
        difference = np.abs(original - reconstructed)
        
        # Original
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f'Original\n{CLASS_NAMES[data["labels"][i]]}')
        axes[i, 0].axis('off')
        
        # Reconstructed
        axes[i, 1].imshow(reconstructed)
        axes[i, 1].set_title(f'Reconstructed\nMSE: {data["mse_values"][i]:.4f}')
        axes[i, 1].axis('off')
        
        # Difference (color)
        axes[i, 2].imshow(np.clip(difference * 3, 0, 1))  # Amplify 3x
        axes[i, 2].set_title('Difference (3x)')
        axes[i, 2].axis('off')
        
        # Difference heatmap
        diff_gray = np.mean(difference, axis=2)
        im = axes[i, 3].imshow(diff_gray, cmap='hot', vmin=0, vmax=0.2)
        axes[i, 3].set_title('Error Heatmap')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        name = os.path.splitext(save_path)[0]
        detailed_path = f"{name}_detailed.png"
        plt.savefig(detailed_path, dpi=150, bbox_inches='tight')
        print(f"Detailed plot saved to: {detailed_path}")
    
    plt.show()

def compare_cpu_gpu(cpu_file, gpu_file, save_path=None):
    """Compare CPU and GPU reconstructions side by side"""
    
    cpu_data = load_reconstructed_images(cpu_file)
    gpu_data = load_reconstructed_images(gpu_file)
    
    if cpu_data is None or gpu_data is None:
        return
    
    num_samples = min(cpu_data['num_samples'], gpu_data['num_samples'], 5)
    
    fig, axes = plt.subplots(4, num_samples, figsize=(2 * num_samples, 10))
    
    if num_samples == 1:
        axes = axes.reshape(4, 1)
    
    fig.suptitle('CPU vs GPU Reconstruction Comparison', fontsize=14, fontweight='bold')
    
    for i in range(num_samples):
        # Original (same for both)
        original = np.clip(cpu_data['originals'][i], 0, 1)
        cpu_recon = np.clip(cpu_data['reconstructed'][i], 0, 1)
        gpu_recon = np.clip(gpu_data['reconstructed'][i], 0, 1)
        
        # Original
        axes[0, i].imshow(original)
        axes[0, i].axis('off')
        axes[0, i].set_title(CLASS_NAMES[cpu_data['labels'][i]], fontsize=9)
        
        # CPU Reconstructed
        axes[1, i].imshow(cpu_recon)
        axes[1, i].axis('off')
        axes[1, i].set_title(f'MSE: {cpu_data["mse_values"][i]:.4f}', fontsize=9)
        
        # GPU Reconstructed
        axes[2, i].imshow(gpu_recon)
        axes[2, i].axis('off')
        axes[2, i].set_title(f'MSE: {gpu_data["mse_values"][i]:.4f}', fontsize=9)
        
        # Difference between CPU and GPU
        diff = np.abs(cpu_recon - gpu_recon)
        axes[3, i].imshow(np.clip(diff * 10, 0, 1))
        axes[3, i].axis('off')
    
    # Row labels
    axes[0, 0].text(-0.3, 0.5, 'Original', transform=axes[0, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    axes[1, 0].text(-0.3, 0.5, 'CPU Recon', transform=axes[1, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    axes[2, 0].text(-0.3, 0.5, 'GPU Recon', transform=axes[2, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    axes[3, 0].text(-0.3, 0.5, 'Diff (10x)', transform=axes[3, 0].transAxes,
                    fontsize=11, fontweight='bold', va='center', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print("RECONSTRUCTION COMPARISON: CPU vs GPU")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'CPU':>15} {'GPU':>15}")
    print(f"{'-'*60}")
    print(f"{'Mean MSE':<25} {np.mean(cpu_data['mse_values']):>15.6f} {np.mean(gpu_data['mse_values']):>15.6f}")
    print(f"{'Min MSE':<25} {np.min(cpu_data['mse_values']):>15.6f} {np.min(gpu_data['mse_values']):>15.6f}")
    print(f"{'Max MSE':<25} {np.max(cpu_data['mse_values']):>15.6f} {np.max(gpu_data['mse_values']):>15.6f}")
    print(f"{'Std MSE':<25} {np.std(cpu_data['mse_values']):>15.6f} {np.std(gpu_data['mse_values']):>15.6f}")
    print(f"{'='*60}\n")

def main():
    cpu_file = "reconstructed_images_cpu.bin"
    gpu_file = "reconstructed_images_gpu.bin"
    
    # Parse arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg == '--compare' or arg == '-c':
            if os.path.exists(cpu_file) and os.path.exists(gpu_file):
                compare_cpu_gpu(cpu_file, gpu_file, save_path="reconstruction_comparison.png")
            else:
                print("Error: Need both CPU and GPU reconstruction files for comparison")
                if not os.path.exists(cpu_file):
                    print(f"  Missing: {cpu_file}")
                if not os.path.exists(gpu_file):
                    print(f"  Missing: {gpu_file}")
            return
        
        elif arg == '--help' or arg == '-h':
            print(__doc__)
            return
        
        elif arg.endswith('.bin'):
            # Custom file
            data = load_reconstructed_images(sys.argv[1])
            if data is not None:
                name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
                plot_reconstructions(data, title=name, save_path=f"{name}_visualization.png")
                plot_detailed_comparison(data, save_path=f"{name}_visualization.png")
            return
    
    # Default: auto-detect
    if os.path.exists(cpu_file):
        data = load_reconstructed_images(cpu_file)
        if data is not None:
            plot_reconstructions(data, title="CPU Autoencoder", save_path="reconstruction_cpu.png")
            plot_detailed_comparison(data, save_path="reconstruction_cpu.png")
    
    elif os.path.exists(gpu_file):
        data = load_reconstructed_images(gpu_file)
        if data is not None:
            plot_reconstructions(data, title="GPU Autoencoder", save_path="reconstruction_gpu.png")
            plot_detailed_comparison(data, save_path="reconstruction_gpu.png")
    
    else:
        print("No reconstruction files found!")
        print(f"Run the autoencoder training first to generate:")
        print(f"  - {cpu_file} (CPU version)")
        print(f"  - {gpu_file} (GPU version)")
        print("\nUsage:")
        print(__doc__)

if __name__ == "__main__":
    main()

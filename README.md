# 🧠 Autoencoder CPU Baseline - CIFAR-10

**CSC14120 - Parallel Programming Final Project**

Convolutional Autoencoder implementation for unsupervised feature learning on CIFAR-10 dataset.

## 📋 Project Overview

This is **Phase 1: CPU Baseline Implementation** with detailed profiling to identify bottlenecks for GPU optimization.

### Architecture
```
Input (32×32×3)
    ↓
[Encoder]
    Conv2D (3→256) + ReLU + MaxPool2D → (16×16×256)
    Conv2D (256→128) + ReLU + MaxPool2D → (8×8×128) [LATENT]
    ↓
[Decoder]  
    Conv2D (128→128) + ReLU + UpSample2D → (16×16×128)
    Conv2D (128→256) + ReLU + UpSample2D → (32×32×256)
    Conv2D (256→3) → (32×32×3)
    ↓
Output (32×32×3)
```

**Total Parameters:** ~751,875

## 🗂️ Project Structure

```
AutoencoderCpu/
├── src/
│   ├── common.h        # Tensor4D, Timer, Profiler, MemoryTracker
│   ├── cifar10.h       # CIFAR-10 dataset loader
│   ├── layers_cpu.h    # Conv2D, ReLU, MaxPool2D, UpSample2D, MSELoss
│   ├── optimizer.h     # SGD, Adam optimizers
│   ├── autoencoder.h   # Autoencoder model
│   └── main.cpp        # Main entry point
├── Makefile
├── Autoencoder_CPU_Colab_Drive.ipynb  # Google Colab notebook
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- C++17 compatible compiler (g++ or clang++)
- Make

### Build
```bash
make clean && make
```

### Download CIFAR-10
```bash
make download-data
```
Or manually:
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
```

### Run
```bash
./autoencoder_cpu <data_path> [epochs] [batch_size] [max_samples] [optimizer]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| data_path | Path to CIFAR-10 binary folder | ./cifar-10-batches-bin |
| epochs | Number of training epochs | 3 |
| batch_size | Batch size | 32 |
| max_samples | Limit training samples (0 = ALL) | 0 |
| optimizer | `adam` or `sgd` | adam |

### Examples
```bash
# Quick test (100 samples, 2 epochs, Adam)
./autoencoder_cpu ./cifar-10-batches-bin 2 32 100 adam

# Compare SGD vs Adam
./autoencoder_cpu ./cifar-10-batches-bin 2 32 100 sgd

# More samples
./autoencoder_cpu ./cifar-10-batches-bin 3 32 500 adam

# Full dataset (slow!)
./autoencoder_cpu ./cifar-10-batches-bin 5 32 0 adam
```

## ☁️ Run on Google Colab

1. Upload this repository to your Google Drive
2. Open `Autoencoder_CPU_Colab_Drive.ipynb` with Google Colab
3. Follow the instructions in the notebook

## 📊 Output

The program outputs:
- **Training progress:** Loss per epoch, time per epoch
- **Profiler Report:** Time breakdown for each operation (Conv2D, ReLU, MaxPool, etc.)
- **Memory Report:** Memory usage for weights, activations, gradients

### Sample Output
```
=== AUTOENCODER CPU BASELINE ===
Config: epochs=2, batch=32, samples=100, optimizer=Adam

Epoch 1/2 | Loss: 0.089234 | Time: 52.31s
Epoch 2/2 | Loss: 0.062773 | Time: 51.89s

========== PROFILER REPORT ==========
Operation              Total (ms)    Calls    Percentage
Conv2D_backward         65432.10       40        63.12%
Conv2D_forward          38012.45       40        36.68%
...
======================================

========== MEMORY USAGE ==========
Weights:     2.87 MB
Activations: 15.23 MB
...
==================================
```

## 🔍 Key Findings

- **Conv2D operations** consume >99% of total time
- Forward pass: ~37%, Backward pass: ~63%
- This identifies the main target for GPU optimization in Phase 2

## 📝 License

This project is for educational purposes (CSC14120 - HCMUS).

## 👥 Authors

- Student: [Your Name]
- Course: CSC14120 - Parallel Programming
- University: HCMUS

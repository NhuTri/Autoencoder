# 🚀 Autoencoder GPU (Naive) - CIFAR-10

**CSC14120 - Parallel Programming Final Project**

**Phase 2: Naive GPU Implementation** - Conv2D accelerated with CUDA

## 📋 Overview

This is the GPU-accelerated version of the Autoencoder, where **Conv2D operations** are parallelized using CUDA. Since Conv2D consumes >99% of training time (as identified in Phase 1), this is the primary target for GPU optimization.

### What's GPU-accelerated?
| Layer | Implementation | Notes |
|-------|---------------|-------|
| **Conv2D** | ✅ CUDA | Each thread computes 1 output pixel |
| ReLU | CPU | Negligible time |
| MaxPool2D | CPU | Negligible time |
| UpSample2D | CPU | Negligible time |
| MSELoss | CPU | Negligible time |

## 🏗️ Architecture

```
Input (32×32×3)
    ↓
[Encoder] - GPU Conv2D
    Conv2D(3→256) + ReLU + MaxPool → (16×16×256)
    Conv2D(256→128) + ReLU + MaxPool → (8×8×128) [LATENT]
    ↓
[Decoder] - GPU Conv2D
    Conv2D(128→128) + ReLU + UpSample → (16×16×128)
    Conv2D(128→256) + ReLU + UpSample → (32×32×256)
    Conv2D(256→3) → (32×32×3)
    ↓
Output (32×32×3)
```

## 🗂️ Project Structure

```
AutoencoderGpu/
├── src/
│   ├── common.h         # Tensor4D, Timer, Profiler
│   ├── cifar10.h        # Dataset loader
│   ├── optimizer.h      # SGD, Adam
│   ├── conv2d_gpu.cu    # 🆕 CUDA Conv2D kernels
│   ├── layers_gpu.h     # Layers with GPU Conv2D
│   ├── autoencoder.h    # Model definition
│   └── main.cu          # Entry point
├── Makefile
└── README.md
```

## 🔧 Requirements

- **CUDA Toolkit** (11.0+)
- **NVIDIA GPU** with compute capability 6.1+
- **g++** with C++17 support

## 🚀 Quick Start

### 1. Check CUDA Installation
```bash
make check-cuda
```

### 2. Build
```bash
make clean && make
```

### 3. Download CIFAR-10
```bash
make download-data
```

### 4. Run
```bash
# Quick test (100 samples)
./autoencoder_gpu ./cifar-10-batches-bin 3 32 100 adam

# Full dataset
./autoencoder_gpu ./cifar-10-batches-bin 5 32 0 adam
```

## 📊 Usage

```
./autoencoder_gpu <data_path> [epochs] [batch_size] [max_samples] [optimizer]
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| data_path | Path to CIFAR-10 | Required |
| epochs | Training epochs | 5 |
| batch_size | Batch size | 32 |
| max_samples | Limit samples (0=all) | 0 |
| optimizer | `adam` or `sgd` | adam |

## ☁️ Run on Google Colab

1. Upload `AutoencoderGpu/` folder to Google Drive
2. Open new Colab notebook with **GPU runtime**
3. Run:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project
%cd /content/drive/MyDrive/AutoencoderGpu

# Check GPU
!nvidia-smi

# Download data
!make download-data

# Build
!make clean && make

# Run
!./autoencoder_gpu ./cifar-10-batches-bin 3 32 100 adam
```

## 🔥 CUDA Implementation Details

### Naive Approach (Phase 2)
- **One thread per output pixel**
- Each thread computes the full convolution sum for one `(n, oh, ow, oc)` position
- Simple but effective for small to medium networks

```cpp
// Each thread computes one output
int idx = blockIdx.x * blockDim.x + threadIdx.x;
// ... decode to (n, oh, ow, oc)
float sum = bias[oc];
for (kh, kw, ic) {
    sum += input[...] * weights[...];
}
output[idx] = sum;
```

### Backward Pass
- **Weight gradients**: Uses `atomicAdd` since multiple threads contribute
- **Input gradients**: Each thread computes gradient for one input position

## 📈 Expected Speedup

| Metric | CPU (Phase 1) | GPU Naive (Phase 2) | Speedup |
|--------|--------------|---------------------|---------|
| Conv2D forward | ~37s | ~2-5s | 7-18x |
| Conv2D backward | ~63s | ~3-8s | 8-20x |
| **Total time** | ~100s | ~10-20s | **5-10x** |

*Results vary by GPU. Tested on RTX 3060 / Tesla T4*

## 🔮 Future Optimizations (Phase 3)

For better performance, consider:
1. **Shared Memory Tiling** - Reduce global memory access
2. **im2col + cuBLAS GEMM** - Convert to matrix multiplication
3. **cuDNN Integration** - Use optimized library

## 📝 Comparing with CPU Baseline

Run both versions with same parameters:

```bash
# CPU (in AutoencoderCpu/)
./autoencoder_cpu ./cifar-10-batches-bin 3 32 100 adam

# GPU (in AutoencoderGpu/)
./autoencoder_gpu ./cifar-10-batches-bin 3 32 100 adam
```

Compare:
- Total training time
- Conv2D_forward time
- Conv2D_backward time

## 📄 License

Educational project for CSC14120 - HCMUS

## 👥 Authors

- Student: [Your Name]
- Course: CSC14120 - Parallel Programming
- University: HCMUS

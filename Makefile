# Makefile for Autoencoder GPU (CUDA)

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -std=c++17 -O3 -arch=sm_75 -Xcompiler -Wall

# For different GPU architectures, change -arch flag:
# GTX 10xx series: sm_61
# RTX 20xx series: sm_75
# RTX 30xx series: sm_86
# RTX 40xx series: sm_89
# Google Colab T4: sm_75
# Google Colab V100: sm_70
# Google Colab A100: sm_80

# Directories
SRC_DIR = src
BUILD_DIR = build

# Target
TARGET = autoencoder_gpu

# Source files
CUDA_SRC = $(SRC_DIR)/main.cu

# Header files
HEADERS = $(SRC_DIR)/common.h \
          $(SRC_DIR)/cifar10.h \
          $(SRC_DIR)/optimizer.h \
          $(SRC_DIR)/conv2d_gpu.cu \
          $(SRC_DIR)/layers_gpu.h \
          $(SRC_DIR)/autoencoder.h

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(CUDA_SRC) $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(CUDA_SRC)
	@echo "Build complete: $(TARGET)"

# Clean
clean:
	rm -f $(TARGET)
	rm -rf $(BUILD_DIR)
	@echo "Clean complete"

# Download CIFAR-10 dataset
download-data:
	@echo "Downloading CIFAR-10 dataset..."
	@if [ ! -d "cifar-10-batches-bin" ]; then \
		wget -q https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz; \
		tar -xzf cifar-10-binary.tar.gz; \
		rm cifar-10-binary.tar.gz; \
		echo "Download complete!"; \
	else \
		echo "CIFAR-10 already exists."; \
	fi

# Run with default settings
run: $(TARGET)
	./$(TARGET) ./cifar-10-batches-bin 3 32 100 adam

# Run quick test
test: $(TARGET)
	./$(TARGET) ./cifar-10-batches-bin 2 32 50 adam

# Run full dataset
run-full: $(TARGET)
	./$(TARGET) ./cifar-10-batches-bin 5 32 0 adam

# Check CUDA
check-cuda:
	@echo "Checking CUDA installation..."
	@which nvcc || (echo "NVCC not found! Please install CUDA toolkit." && exit 1)
	@nvcc --version
	@echo ""
	@nvidia-smi || echo "nvidia-smi not available"

# Help
help:
	@echo "Autoencoder GPU Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all           Build the project (default)"
	@echo "  clean         Remove compiled files"
	@echo "  download-data Download CIFAR-10 dataset"
	@echo "  run           Run with 100 samples, 3 epochs"
	@echo "  test          Quick test with 50 samples, 2 epochs"
	@echo "  run-full      Run with full dataset"
	@echo "  check-cuda    Check CUDA installation"
	@echo ""
	@echo "Usage after build:"
	@echo "  ./autoencoder_gpu <data_path> [epochs] [batch_size] [max_samples] [optimizer]"

.PHONY: all clean download-data run test run-full check-cuda help

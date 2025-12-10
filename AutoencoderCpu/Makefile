# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -march=native

# Source directory
SRC_DIR = src

# Build directory
BUILD_DIR = build

# Target executable
TARGET = autoencoder_cpu

# Source files
SOURCES = $(SRC_DIR)/main.cpp

# Header files (for dependency tracking)
HEADERS = $(SRC_DIR)/common.h \
          $(SRC_DIR)/cifar10.h \
          $(SRC_DIR)/layers_cpu.h \
          $(SRC_DIR)/optimizer.h \
          $(SRC_DIR)/autoencoder.h \
          $(SRC_DIR)/trainer.h

# Object files
OBJECTS = $(BUILD_DIR)/main.o

# Default target
all: $(BUILD_DIR) $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Link
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile main.cpp
$(BUILD_DIR)/main.o: $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Run with default parameters (requires cifar-10 data)
run: $(TARGET)
	./$(TARGET) ./data/cifar-10-batches-bin 5 32

# Run with fewer epochs for quick testing
test: $(TARGET)
	./$(TARGET) ./data/cifar-10-batches-bin 1 32

# Download CIFAR-10 dataset
download-data:
	@echo "Downloading CIFAR-10 dataset..."
	mkdir -p data
	cd data && curl -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
	cd data && tar -xzf cifar-10-binary.tar.gz
	@echo "Dataset downloaded to data/cifar-10-batches-bin"

# Debug build
debug: CXXFLAGS = -std=c++17 -g -Wall -Wextra -DDEBUG
debug: clean all

# Help
help:
	@echo "Available targets:"
	@echo "  all           - Build the project (default)"
	@echo "  clean         - Remove build files"
	@echo "  run           - Build and run with 5 epochs"
	@echo "  test          - Build and run with 1 epoch (quick test)"
	@echo "  download-data - Download CIFAR-10 dataset"
	@echo "  debug         - Build with debug symbols"
	@echo "  help          - Show this help message"

.PHONY: all clean run test download-data debug help

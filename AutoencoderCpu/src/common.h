/**
 * @file common.h
 * @brief Common definitions and includes for the Autoencoder project
 */

#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <map>

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

using DataType = float;

// ============================================================================
// TENSOR CLASS - Simple 4D tensor for neural network operations
// ============================================================================

/**
 * @class Tensor4D
 * @brief A simple 4D tensor class for storing neural network data
 * 
 * Dimensions: (batch, height, width, channels) - NHWC format
 * This format is intuitive and matches how images are typically stored
 */
class Tensor4D {
public:
    int batch;      // N - number of samples
    int height;     // H - spatial height
    int width;      // W - spatial width  
    int channels;   // C - number of channels

    std::vector<DataType> data;

    // Default constructor
    Tensor4D() : batch(0), height(0), width(0), channels(0) {}

    // Constructor with dimensions
    Tensor4D(int n, int h, int w, int c) 
        : batch(n), height(h), width(w), channels(c) {
        data.resize(n * h * w * c, 0.0f);
    }

    // Get total number of elements
    size_t size() const {
        return batch * height * width * channels;
    }

    // Get memory size in bytes
    size_t memorySize() const {
        return size() * sizeof(DataType);
    }

    // Access element at (n, h, w, c)
    DataType& at(int n, int h, int w, int c) {
        return data[((n * height + h) * width + w) * channels + c];
    }

    const DataType& at(int n, int h, int w, int c) const {
        return data[((n * height + h) * width + w) * channels + c];
    }

    // Reshape tensor (total size must remain the same)
    void reshape(int n, int h, int w, int c) {
        assert(n * h * w * c == (int)data.size());
        batch = n;
        height = h;
        width = w;
        channels = c;
    }

    // Fill tensor with a value
    void fill(DataType value) {
        std::fill(data.begin(), data.end(), value);
    }

    // Fill tensor with random values (Xavier initialization)
    void randomInit(float scale = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& val : data) {
            val = dist(gen);
        }
    }

    // Copy from another tensor
    void copyFrom(const Tensor4D& other) {
        batch = other.batch;
        height = other.height;
        width = other.width;
        channels = other.channels;
        data = other.data;
    }

    // Print tensor info
    void printInfo(const std::string& name = "") const {
        std::cout << name << " Tensor: (" << batch << ", " << height 
                  << ", " << width << ", " << channels << ") - "
                  << (memorySize() / 1024.0 / 1024.0) << " MB" << std::endl;
    }
};

// ============================================================================
// TIMER CLASS - For profiling
// ============================================================================

/**
 * @class Timer
 * @brief Simple timer class for measuring execution time
 */
class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
    bool running;

public:
    Timer() : running(false) {}

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
        running = true;
    }

    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
        running = false;
    }

    // Get elapsed time in milliseconds
    double elapsedMs() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : endTime;
        return std::chrono::duration<double, std::milli>(end - startTime).count();
    }

    // Get elapsed time in seconds
    double elapsedSec() const {
        return elapsedMs() / 1000.0;
    }
};

// ============================================================================
// PROFILER CLASS - For detailed timing of each operation
// ============================================================================

/**
 * @class Profiler
 * @brief Collects timing statistics for different operations
 */
class Profiler {
public:
    struct Stats {
        double totalTime;    // Total time in ms
        int callCount;       // Number of calls
        
        Stats() : totalTime(0.0), callCount(0) {}
        
        double avgTime() const {
            return callCount > 0 ? totalTime / callCount : 0.0;
        }
    };

    std::map<std::string, Stats> stats;

    void addTime(const std::string& name, double timeMs) {
        stats[name].totalTime += timeMs;
        stats[name].callCount++;
    }

    void reset() {
        stats.clear();
    }

    void printReport() const {
        std::cout << "\n========== PROFILER REPORT ==========\n";
        std::cout << std::left << std::setw(20) << "Operation"
                  << std::right << std::setw(15) << "Total (ms)"
                  << std::setw(15) << "Calls"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(15) << "Percentage" << "\n";
        std::cout << std::string(80, '-') << "\n";

        // Calculate total time
        double total = 0.0;
        for (const auto& [name, stat] : stats) {
            total += stat.totalTime;
        }

        // Sort by total time (descending)
        std::vector<std::pair<std::string, Stats>> sorted(stats.begin(), stats.end());
        std::sort(sorted.begin(), sorted.end(), 
            [](const auto& a, const auto& b) { return a.second.totalTime > b.second.totalTime; });

        for (const auto& [name, stat] : sorted) {
            double percentage = (total > 0) ? (stat.totalTime / total * 100.0) : 0.0;
            std::cout << std::left << std::setw(20) << name
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << stat.totalTime
                      << std::setw(15) << stat.callCount
                      << std::setw(15) << std::fixed << std::setprecision(4) << stat.avgTime()
                      << std::setw(14) << std::fixed << std::setprecision(2) << percentage << "%\n";
        }

        std::cout << std::string(80, '-') << "\n";
        std::cout << std::left << std::setw(20) << "TOTAL"
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << total
                  << std::setw(15) << "" << std::setw(15) << "" << std::setw(15) << "100.00%\n";
        std::cout << "======================================\n";
    }
};

// Global profiler instance
extern Profiler gProfiler;

// ============================================================================
// MEMORY TRACKER - For tracking memory usage
// ============================================================================

/**
 * @class MemoryTracker
 * @brief Tracks memory allocation for the project
 */
class MemoryTracker {
public:
    size_t weightsMemory;
    size_t activationsMemory;
    size_t gradientsMemory;
    size_t dataMemory;

    MemoryTracker() : weightsMemory(0), activationsMemory(0), 
                      gradientsMemory(0), dataMemory(0) {}

    void addWeights(size_t bytes) { weightsMemory += bytes; }
    void addActivations(size_t bytes) { activationsMemory += bytes; }
    void addGradients(size_t bytes) { gradientsMemory += bytes; }
    void addData(size_t bytes) { dataMemory += bytes; }

    size_t totalMemory() const {
        return weightsMemory + activationsMemory + gradientsMemory + dataMemory;
    }

    void printReport() const {
        std::cout << "\n========== MEMORY USAGE REPORT ==========\n";
        std::cout << "Weights Memory:      " << std::fixed << std::setprecision(2) 
                  << (weightsMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << "Activations Memory:  " << std::fixed << std::setprecision(2)
                  << (activationsMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << "Gradients Memory:    " << std::fixed << std::setprecision(2)
                  << (gradientsMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << "Data Memory:         " << std::fixed << std::setprecision(2)
                  << (dataMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << std::string(40, '-') << "\n";
        std::cout << "TOTAL Memory:        " << std::fixed << std::setprecision(2)
                  << (totalMemory() / 1024.0 / 1024.0) << " MB\n";
        std::cout << "=========================================\n";
    }
};

// Global memory tracker instance
extern MemoryTracker gMemoryTracker;

#endif // COMMON_H

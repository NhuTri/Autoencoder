/**
 * @file common.h
 * @brief Common definitions and includes for the Autoencoder GPU project
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
 */
class Tensor4D {
public:
    int batch;      // N - number of samples
    int height;     // H - spatial height
    int width;      // W - spatial width  
    int channels;   // C - number of channels

    std::vector<DataType> data;

    Tensor4D() : batch(0), height(0), width(0), channels(0) {}

    Tensor4D(int n, int h, int w, int c) 
        : batch(n), height(h), width(w), channels(c) {
        data.resize(n * h * w * c, 0.0f);
    }

    size_t size() const { return batch * height * width * channels; }
    size_t memorySize() const { return size() * sizeof(DataType); }

    DataType& at(int n, int h, int w, int c) {
        return data[((n * height + h) * width + w) * channels + c];
    }

    const DataType& at(int n, int h, int w, int c) const {
        return data[((n * height + h) * width + w) * channels + c];
    }

    void reshape(int n, int h, int w, int c) {
        assert(n * h * w * c == (int)data.size());
        batch = n; height = h; width = w; channels = c;
    }

    void fill(DataType value) { std::fill(data.begin(), data.end(), value); }

    void randomInit(float scale = 1.0f) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& val : data) val = dist(gen);
    }

    void copyFrom(const Tensor4D& other) {
        batch = other.batch; height = other.height;
        width = other.width; channels = other.channels;
        data = other.data;
    }

    void printInfo(const std::string& name = "") const {
        std::cout << name << " Tensor: (" << batch << ", " << height 
                  << ", " << width << ", " << channels << ") - "
                  << (memorySize() / 1024.0 / 1024.0) << " MB" << std::endl;
    }
};

// ============================================================================
// TIMER CLASS - For profiling
// ============================================================================

class Timer {
private:
    std::chrono::high_resolution_clock::time_point startTime, endTime;
    bool running;
public:
    Timer() : running(false) {}
    void start() { startTime = std::chrono::high_resolution_clock::now(); running = true; }
    void stop() { endTime = std::chrono::high_resolution_clock::now(); running = false; }
    double elapsedMs() const {
        auto end = running ? std::chrono::high_resolution_clock::now() : endTime;
        return std::chrono::duration<double, std::milli>(end - startTime).count();
    }
    double elapsedSec() const { return elapsedMs() / 1000.0; }
};

// ============================================================================
// PROFILER CLASS - For detailed timing of each operation
// ============================================================================

class Profiler {
public:
    struct Stats {
        double totalTime = 0; int callCount = 0;
        double avgTime() const { return callCount > 0 ? totalTime / callCount : 0; }
    };
    std::map<std::string, Stats> stats;

    void addTime(const std::string& name, double timeMs) {
        stats[name].totalTime += timeMs;
        stats[name].callCount++;
    }

    void reset() { stats.clear(); }

    void printReport() const {
        std::cout << "\n========== PROFILER REPORT ==========\n";
        std::cout << std::left << std::setw(25) << "Operation"
                  << std::right << std::setw(15) << "Total (ms)"
                  << std::setw(10) << "Calls"
                  << std::setw(15) << "Avg (ms)"
                  << std::setw(12) << "Percentage" << "\n";
        std::cout << std::string(77, '-') << "\n";

        double total = 0;
        for (const auto& [name, stat] : stats) total += stat.totalTime;

        std::vector<std::pair<std::string, Stats>> sorted(stats.begin(), stats.end());
        std::sort(sorted.begin(), sorted.end(), 
            [](const auto& a, const auto& b) { return a.second.totalTime > b.second.totalTime; });

        for (const auto& [name, stat] : sorted) {
            double pct = (total > 0) ? (stat.totalTime / total * 100.0) : 0.0;
            std::cout << std::left << std::setw(25) << name
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << stat.totalTime
                      << std::setw(10) << stat.callCount
                      << std::setw(15) << std::fixed << std::setprecision(4) << stat.avgTime()
                      << std::setw(11) << std::fixed << std::setprecision(2) << pct << "%\n";
        }
        std::cout << std::string(77, '-') << "\n";
        std::cout << std::left << std::setw(25) << "TOTAL"
                  << std::right << std::setw(15) << std::fixed << std::setprecision(2) << total
                  << std::setw(10) << "" << std::setw(15) << "" << std::setw(12) << "100.00%\n";
        std::cout << "======================================\n";
    }
};

extern Profiler gProfiler;

// ============================================================================
// MEMORY TRACKER
// ============================================================================

class MemoryTracker {
public:
    size_t weightsMemory = 0, activationsMemory = 0, gradientsMemory = 0, dataMemory = 0;
    size_t gpuMemory = 0;  // Track GPU memory

    void addWeights(size_t bytes) { weightsMemory += bytes; }
    void addActivations(size_t bytes) { activationsMemory += bytes; }
    void addGradients(size_t bytes) { gradientsMemory += bytes; }
    void addData(size_t bytes) { dataMemory += bytes; }
    void addGPU(size_t bytes) { gpuMemory += bytes; }

    size_t totalMemory() const { return weightsMemory + activationsMemory + gradientsMemory + dataMemory; }

    void printReport() const {
        std::cout << "\n========== MEMORY USAGE REPORT ==========\n";
        std::cout << "CPU Memory:\n";
        std::cout << "  Weights:      " << std::fixed << std::setprecision(2) << (weightsMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  Activations:  " << std::fixed << std::setprecision(2) << (activationsMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  Gradients:    " << std::fixed << std::setprecision(2) << (gradientsMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  Data:         " << std::fixed << std::setprecision(2) << (dataMemory / 1024.0 / 1024.0) << " MB\n";
        std::cout << "  TOTAL CPU:    " << std::fixed << std::setprecision(2) << (totalMemory() / 1024.0 / 1024.0) << " MB\n";
        if (gpuMemory > 0) {
            std::cout << "\nGPU Memory:\n";
            std::cout << "  TOTAL GPU:    " << std::fixed << std::setprecision(2) << (gpuMemory / 1024.0 / 1024.0) << " MB\n";
        }
        std::cout << "=========================================\n";
    }
};

extern MemoryTracker gMemoryTracker;

#endif // COMMON_H

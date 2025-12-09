/**
 * @file main.cu
 * @brief Main entry point for GPU Autoencoder training
 * 
 * Phase 2: Naive GPU Implementation - Conv2D accelerated with CUDA
 * Features: Train/Test evaluation per epoch, automatic best model saving
 */

#include "common.h"
#include "cifar10.h"
#include "autoencoder.h"

#include <iostream>
#include <string>
#include <limits>
#include <cuda_runtime.h>

// Global instances
Profiler gProfiler;
MemoryTracker gMemoryTracker;

// Epoch statistics structure
struct EpochStats {
    int epoch;
    double trainLoss;
    double testLoss;
    double epochTime;
    bool isBest;
};

void printGPUInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found!" << std::endl;
        exit(1);
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\n========== GPU INFORMATION ==========\n";
    std::cout << "Device:          " << prop.name << "\n";
    std::cout << "Compute:         " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global Memory:   " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n";
    std::cout << "Shared Memory:   " << (prop.sharedMemPerBlock / 1024) << " KB/block\n";
    std::cout << "Max Threads:     " << prop.maxThreadsPerBlock << "/block\n";
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "=====================================\n\n";
}

void printUsage(const char* name) {
    std::cout << "Usage: " << name << " <cifar10_path> [epochs] [batch_size] [max_samples] [optimizer]\n\n";
    std::cout << "Arguments:\n";
    std::cout << "  cifar10_path   Path to CIFAR-10 binary data\n";
    std::cout << "  epochs         Number of epochs (default: 5)\n";
    std::cout << "  batch_size     Batch size (default: 32)\n";
    std::cout << "  max_samples    Max training samples, 0=all (default: 0)\n";
    std::cout << "  optimizer      'adam' or 'sgd' (default: adam)\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << name << " ./cifar-10-batches-bin 3 32 100 adam\n";
    std::cout << "  " << name << " ./cifar-10-batches-bin 5 32 0 sgd\n";
}

// Evaluate loss on a dataset (forward-only, no gradients)
double evaluateLoss(Autoencoder& model, CIFAR10Dataset& dataset, 
                    bool isTrainSet, int batchSize) {
    int numImages = isTrainSet ? dataset.getNumTrainImages() : dataset.getNumTestImages();
    int numBatches = (numImages + batchSize - 1) / batchSize;
    double totalLoss = 0.0;
    
    for (int b = 0; b < numBatches; b++) {
        int start = b * batchSize;
        int currBatch = std::min(batchSize, numImages - start);
        
        Tensor4D batch = isTrainSet ? 
            dataset.getTrainBatch(start, currBatch) : 
            dataset.getTestBatch(start, currBatch);
        
        // Forward only
        model.forward(batch);
        
        // Compute MSE loss
        const Tensor4D& output = model.getOutput();
        double batchLoss = 0.0;
        int total = currBatch * 32 * 32 * 3;
        for (int i = 0; i < total; i++) {
            double diff = output.data[i] - batch.data[i];
            batchLoss += diff * diff;
        }
        totalLoss += batchLoss / total;
    }
    
    return totalLoss / numBatches;
}

int main(int argc, char* argv[]) {
    std::cout << "============================================================\n";
    std::cout << "   AUTOENCODER GPU (NAIVE) - Phase 2: CUDA Conv2D\n";
    std::cout << "============================================================\n";

    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    // Parse arguments
    std::string dataPath = argv[1];
    int epochs = (argc >= 3) ? std::stoi(argv[2]) : 5;
    int batchSize = (argc >= 4) ? std::stoi(argv[3]) : 32;
    int maxSamples = (argc >= 5) ? std::stoi(argv[4]) : 0;
    std::string optStr = (argc >= 6) ? argv[5] : "adam";
    bool useAdam = (optStr == "adam" || optStr == "Adam");

    std::string bestModelPath = "autoencoder_gpu_best.bin";
    std::string finalModelPath = "autoencoder_gpu_final.bin";

    // Print GPU info
    printGPUInfo();

    std::cout << "Configuration:\n";
    std::cout << "  Data path:    " << dataPath << "\n";
    std::cout << "  Epochs:       " << epochs << "\n";
    std::cout << "  Batch size:   " << batchSize << "\n";
    std::cout << "  Max samples:  " << (maxSamples > 0 ? std::to_string(maxSamples) : "ALL") << "\n";
    std::cout << "  Optimizer:    " << (useAdam ? "Adam" : "SGD") << "\n";
    std::cout << "  Best model:   " << bestModelPath << "\n\n";

    // Load dataset
    std::cout << "Step 1: Loading CIFAR-10...\n";
    std::cout << std::string(50, '-') << "\n";
    
    CIFAR10Dataset dataset;
    if (!dataset.load(dataPath, maxSamples)) {
        std::cerr << "Failed to load dataset!" << std::endl;
        return 1;
    }
    std::cout << "\n";

    // Create model
    std::cout << "Step 2: Creating Autoencoder model...\n";
    std::cout << std::string(50, '-') << "\n";
    
    Autoencoder model;

    // Training
    std::cout << "Step 3: Training with Train/Test Evaluation...\n";
    std::cout << std::string(50, '-') << "\n";

    int numImages = dataset.getNumTrainImages();
    int numBatches = (numImages + batchSize - 1) / batchSize;
    
    AdamOptimizer adamOpt(0.001f);
    
    gProfiler.reset();
    Timer totalTimer;
    totalTimer.start();
    
    std::vector<EpochStats> allStats;
    double bestTestLoss = std::numeric_limits<double>::max();
    int bestEpoch = 0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Timer epochTimer;
        epochTimer.start();
        
        dataset.shuffleTrainData();
        double epochLoss = 0;
        
        // Training phase
        for (int b = 0; b < numBatches; b++) {
            int start = b * batchSize;
            int currBatch = std::min(batchSize, numImages - start);
            
            Tensor4D batch = dataset.getTrainBatch(start, currBatch);
            model.forward(batch);
            epochLoss += model.backward(batch);
            
            if (useAdam) {
                adamOpt.step();
                model.updateWeightsAdam(adamOpt);
            } else {
                model.updateWeights(0.001f);
            }
            
            // Progress
            if ((b + 1) % 10 == 0 || b == numBatches - 1) {
                std::cout << "\r  Epoch " << (epoch + 1) << "/" << epochs 
                          << " | Batch " << (b + 1) << "/" << numBatches
                          << " | Loss: " << std::fixed << std::setprecision(6) 
                          << (epochLoss / (b + 1)) << std::flush;
            }
        }
        std::cout << "\n";
        
        // Evaluate on train and test sets
        std::cout << "  Evaluating on train set..." << std::flush;
        double trainLoss = evaluateLoss(model, dataset, true, batchSize);
        std::cout << " done.\n";
        
        std::cout << "  Evaluating on test set..." << std::flush;
        double testLoss = evaluateLoss(model, dataset, false, batchSize);
        std::cout << " done.\n";
        
        epochTimer.stop();
        
        // Check if best model
        bool isBest = (testLoss < bestTestLoss);
        if (isBest) {
            bestTestLoss = testLoss;
            bestEpoch = epoch + 1;
            model.saveWeights(bestModelPath);
            std::cout << "  >> New best model saved! (test_loss: " << std::setprecision(6) << testLoss << ")\n";
        }
        
        // Store stats
        EpochStats stats;
        stats.epoch = epoch + 1;
        stats.trainLoss = trainLoss;
        stats.testLoss = testLoss;
        stats.epochTime = epochTimer.elapsedSec();
        stats.isBest = isBest;
        allStats.push_back(stats);
        
        std::cout << "[Epoch " << (epoch + 1) << "/" << epochs << "] "
                  << "Train: " << std::fixed << std::setprecision(6) << trainLoss
                  << " | Test: " << testLoss
                  << " | Time: " << std::setprecision(2) << epochTimer.elapsedSec() << "s"
                  << (isBest ? " *BEST*" : "") << "\n\n";
    }
    
    totalTimer.stop();

    // Save final model
    model.saveWeights(finalModelPath);

    // Reports
    std::cout << "\nStep 4: Reports\n";
    std::cout << std::string(50, '-') << "\n";

    // Training summary
    std::cout << "\n==================== TRAINING SUMMARY ====================\n";
    std::cout << std::left << std::setw(8) << "Epoch" 
              << std::right << std::setw(15) << "Train Loss"
              << std::setw(15) << "Test Loss"
              << std::setw(12) << "Time (s)"
              << std::setw(8) << "Best" << "\n";
    std::cout << std::string(58, '-') << "\n";
    for (const auto& s : allStats) {
        std::cout << std::left << std::setw(8) << s.epoch
                  << std::right << std::setw(15) << std::fixed << std::setprecision(6) << s.trainLoss
                  << std::setw(15) << s.testLoss
                  << std::setw(12) << std::setprecision(2) << s.epochTime
                  << std::setw(8) << (s.isBest ? "*" : "") << "\n";
    }
    std::cout << "==========================================================\n";

    gProfiler.printReport();
    gMemoryTracker.printReport();

    // Final summary
    std::cout << "\n================== FINAL SUMMARY ==================\n";
    std::cout << "Total training time: " << std::fixed << std::setprecision(2) 
              << totalTimer.elapsedSec() << " seconds\n";
    std::cout << "Average time/epoch:  " << std::setprecision(2) 
              << (totalTimer.elapsedSec() / epochs) << " seconds\n";
    std::cout << "\n";
    std::cout << "Final train loss:    " << std::setprecision(6) << allStats.back().trainLoss << "\n";
    std::cout << "Final test loss:     " << std::setprecision(6) << allStats.back().testLoss << "\n";
    std::cout << "\n";
    std::cout << "Best epoch:          " << bestEpoch << "\n";
    std::cout << "Best test loss:      " << std::setprecision(6) << bestTestLoss << "\n";
    std::cout << "Best model saved to: " << bestModelPath << "\n";
    std::cout << "Final model saved to: " << finalModelPath << "\n";
    std::cout << "===================================================\n";

    // Speedup analysis hint
    std::cout << "\n========= GPU vs CPU COMPARISON =========\n";
    std::cout << "To compare speedup, run the CPU baseline with\n";
    std::cout << "the same parameters and compare:\n";
    std::cout << "  - Total training time\n";
    std::cout << "  - Conv2D_forward and Conv2D_backward times\n";
    std::cout << "\nExpected speedup for Conv2D: 5-20x (naive)\n";
    std::cout << "For better speedup, implement:\n";
    std::cout << "  - Shared memory tiling\n";
    std::cout << "  - im2col + cuBLAS GEMM\n";
    std::cout << "  - cuDNN integration\n";
    std::cout << "=========================================\n";

    return 0;
}

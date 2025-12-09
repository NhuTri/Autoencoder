/**
 * @file main.cpp
 * @brief Main entry point for CPU Autoencoder training
 * 
 * This program trains a convolutional autoencoder on CIFAR-10 dataset
 * and provides detailed profiling information for each operation.
 * 
 * Usage: ./autoencoder_cpu <path_to_cifar10_data> [epochs] [batch_size]
 */

#include "common.h"
#include "cifar10.h"
#include "autoencoder.h"
#include "trainer.h"

#include <iostream>
#include <string>

// Define global instances
Profiler gProfiler;
MemoryTracker gMemoryTracker;

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " <cifar10_data_path> [epochs] [batch_size] [max_samples] [optimizer]\n";
    std::cout << "\n";
    std::cout << "Arguments:\n";
    std::cout << "  cifar10_data_path  Path to CIFAR-10 data directory\n";
    std::cout << "                     (containing data_batch_1.bin, ..., test_batch.bin)\n";
    std::cout << "  epochs             Number of training epochs (default: 5)\n";
    std::cout << "  batch_size         Training batch size (default: 32)\n";
    std::cout << "  max_samples        Max training samples to load (default: 0 = all)\n";
    std::cout << "                     Test samples will be 20% of training samples\n";
    std::cout << "  optimizer          Optimizer: 'adam' or 'sgd' (default: adam)\n";
    std::cout << "\n";
    std::cout << "Example:\n";
    std::cout << "  " << programName << " ./cifar-10-batches-bin 10 32           # Full dataset, Adam\n";
    std::cout << "  " << programName << " ./cifar-10-batches-bin 3 32 500 adam   # 500 samples, Adam\n";
    std::cout << "  " << programName << " ./cifar-10-batches-bin 3 32 500 sgd    # 500 samples, SGD\n";
}

int main(int argc, char* argv[]) {
    std::cout << "==========================================================\n";
    std::cout << "   AUTOENCODER CPU BASELINE - CIFAR-10 Feature Learning\n";
    std::cout << "==========================================================\n\n";

    // Parse command line arguments
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string dataPath = argv[1];
    int epochs = (argc >= 3) ? std::stoi(argv[2]) : 5;
    int batchSize = (argc >= 4) ? std::stoi(argv[3]) : 32;
    int maxSamples = (argc >= 5) ? std::stoi(argv[4]) : 0;  // 0 = load all
    std::string optimizerStr = (argc >= 6) ? argv[5] : "adam";
    bool useAdam = (optimizerStr == "adam" || optimizerStr == "Adam" || optimizerStr == "ADAM");

    std::cout << "Configuration:\n";
    std::cout << "  Data path:    " << dataPath << "\n";
    std::cout << "  Epochs:       " << epochs << "\n";
    std::cout << "  Batch size:   " << batchSize << "\n";
    std::cout << "  Max samples:  " << (maxSamples > 0 ? std::to_string(maxSamples) : "ALL") << "\n";
    std::cout << "  Optimizer:    " << (useAdam ? "Adam" : "SGD") << "\n";
    std::cout << "\n";

    // ========== STEP 1: Load CIFAR-10 Dataset ==========
    std::cout << "Step 1: Loading CIFAR-10 dataset...\n";
    std::cout << std::string(50, '-') << "\n";
    
    CIFAR10Dataset dataset;
    if (!dataset.load(dataPath, maxSamples)) {
        std::cerr << "Failed to load CIFAR-10 dataset from: " << dataPath << std::endl;
        return 1;
    }
    std::cout << "\n";

    // ========== STEP 2: Create Autoencoder Model ==========
    std::cout << "Step 2: Creating Autoencoder model...\n";
    std::cout << std::string(50, '-') << "\n";
    
    Autoencoder model;

    // ========== STEP 3: Training ==========
    std::cout << "Step 3: Training Autoencoder...\n";
    std::cout << std::string(50, '-') << "\n";

    TrainingConfig config;
    config.batchSize = batchSize;
    config.epochs = epochs;
    config.learningRate = 0.001f;
    config.shuffle = true;
    config.printEvery = 50;
    config.saveModelPath = "autoencoder_weights.bin";
    config.useAdam = useAdam;

    Trainer trainer(model, dataset, config);
    
    // Reset profiler before training
    gProfiler.reset();
    
    // Train the model
    trainer.train();

    // ========== STEP 4: Print Reports ==========
    std::cout << "\nStep 4: Generating reports...\n";
    std::cout << std::string(50, '-') << "\n";

    // Training report
    trainer.printTrainingReport();

    // Profiler report (detailed timing per operation)
    gProfiler.printReport();

    // Memory report
    gMemoryTracker.printReport();

    // Export training history to CSV
    trainer.exportToCSV("training_history_cpu.csv");

    // ========== SUMMARY ==========
    std::cout << "\n============== FINAL SUMMARY ==============\n";
    std::cout << "Training completed successfully!\n";
    std::cout << "Final reconstruction loss: " << std::fixed << std::setprecision(6) 
              << trainer.getFinalLoss() << "\n";
    std::cout << "Total training time: " << std::fixed << std::setprecision(2)
              << trainer.getTotalTrainingTime() << " seconds\n";
    std::cout << "Model saved to: " << config.saveModelPath << "\n";
    std::cout << "============================================\n";

    // ========== Analysis for CUDA Optimization ==========
    std::cout << "\n========= OPTIMIZATION ANALYSIS =========\n";
    std::cout << "Based on profiling, these operations should be\n";
    std::cout << "prioritized for GPU parallelization:\n\n";
    
    // Get top time-consuming operations
    std::vector<std::pair<std::string, double>> operations;
    double totalTime = 0;
    for (const auto& [name, stat] : gProfiler.stats) {
        operations.push_back({name, stat.totalTime});
        totalTime += stat.totalTime;
    }
    std::sort(operations.begin(), operations.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });

    std::cout << "Priority  Operation            Time%    Parallelization Potential\n";
    std::cout << std::string(70, '-') << "\n";
    
    int priority = 1;
    for (const auto& [name, time] : operations) {
        if (priority > 5) break;
        
        double percentage = (totalTime > 0) ? (time / totalTime * 100.0) : 0.0;
        std::string potential;
        
        if (name.find("Conv2D") != std::string::npos) {
            potential = "HIGH - Matrix ops, tiling, shared memory";
        } else if (name.find("MaxPool") != std::string::npos) {
            potential = "MEDIUM - Simple parallel reduction";
        } else if (name.find("ReLU") != std::string::npos) {
            potential = "HIGH - Embarrassingly parallel";
        } else if (name.find("UpSample") != std::string::npos) {
            potential = "MEDIUM - Simple element copy";
        } else if (name.find("MSE") != std::string::npos) {
            potential = "HIGH - Parallel reduction";
        } else {
            potential = "Varies";
        }

        std::cout << std::setw(8) << priority << "  "
                  << std::left << std::setw(20) << name 
                  << std::right << std::setw(7) << std::fixed << std::setprecision(1) << percentage << "%   "
                  << potential << "\n";
        priority++;
    }
    
    std::cout << std::string(70, '-') << "\n";
    std::cout << "\nRecommendation: Focus on Conv2D operations first as they\n";
    std::cout << "typically consume >90% of training time and have excellent\n";
    std::cout << "parallelization potential using techniques like:\n";
    std::cout << "  - im2col transformation + matrix multiplication\n";
    std::cout << "  - Shared memory tiling\n";
    std::cout << "  - cuDNN library integration\n";
    std::cout << "==========================================\n";

    return 0;
}

/**
 * @file trainer.h
 * @brief Training loop for the Autoencoder with detailed profiling
 */

#ifndef TRAINER_H
#define TRAINER_H

#include "common.h"
#include "cifar10.h"
#include "autoencoder.h"
#include <iomanip>

/**
 * @struct TrainingConfig
 * @brief Configuration for training
 */
struct TrainingConfig {
    int batchSize = 32;
    int epochs = 20;
    float learningRate = 0.001f;
    bool shuffle = true;
    int printEvery = 100;        // Print progress every N batches
    std::string saveModelPath = "autoencoder_weights.bin";
    bool useAdam = true;         // Use Adam optimizer (true) or SGD (false)
    float adamBeta1 = 0.9f;      // Adam beta1 parameter
    float adamBeta2 = 0.999f;    // Adam beta2 parameter
};

/**
 * @struct EpochStats
 * @brief Statistics for a single epoch
 */
struct EpochStats {
    int epoch;
    double avgLoss;
    double epochTime;           // in seconds
    int numBatches;
};

/**
 * @class Trainer
 * @brief Handles training loop with profiling
 */
class Trainer {
private:
    Autoencoder& model;
    CIFAR10Dataset& dataset;
    TrainingConfig config;

    std::vector<EpochStats> trainingHistory;
    Timer totalTimer;

public:
    Trainer(Autoencoder& m, CIFAR10Dataset& d, const TrainingConfig& cfg = TrainingConfig())
        : model(m), dataset(d), config(cfg) {}

    /**
     * @brief Run training loop
     */
    void train() {
        std::cout << "\n========== TRAINING STARTED ==========\n";
        std::cout << "Batch size: " << config.batchSize << std::endl;
        std::cout << "Epochs: " << config.epochs << std::endl;
        std::cout << "Learning rate: " << config.learningRate << std::endl;
        std::cout << "Optimizer: " << (config.useAdam ? "Adam" : "SGD") << std::endl;
        std::cout << "========================================\n\n";

        int numImages = dataset.getNumTrainImages();
        int numBatches = (numImages + config.batchSize - 1) / config.batchSize;  // Round up
        
        // Warn if batch size is larger than dataset
        if (config.batchSize > numImages) {
            std::cout << "Warning: Batch size (" << config.batchSize 
                      << ") is larger than dataset (" << numImages 
                      << "). Using batch size = " << numImages << std::endl;
        }

        // Create Adam optimizer if needed
        AdamOptimizer adamOptimizer(config.learningRate, config.adamBeta1, config.adamBeta2);

        totalTimer.start();

        for (int epoch = 0; epoch < config.epochs; epoch++) {
            Timer epochTimer;
            epochTimer.start();

            // Shuffle data at the beginning of each epoch
            if (config.shuffle) {
                dataset.shuffleTrainData();
            }

            double epochLoss = 0.0;
            int batchCount = 0;

            for (int b = 0; b < numBatches; b++) {
                int startIdx = b * config.batchSize;
                int currentBatchSize = std::min(config.batchSize, numImages - startIdx);
                
                // Get batch
                Tensor4D batch = dataset.getTrainBatch(startIdx, currentBatchSize);

                // Forward pass
                Tensor4D output = model.forward(batch);

                // Backward pass (target = input for autoencoder)
                DataType loss = model.backward(batch);
                epochLoss += loss;
                batchCount++;

                // Update weights using chosen optimizer
                if (config.useAdam) {
                    adamOptimizer.step();  // Increment timestep
                    model.updateWeightsAdam(adamOptimizer);
                } else {
                    model.updateWeights(config.learningRate);
                }

                // Print progress
                if ((b + 1) % config.printEvery == 0) {
                    std::cout << "  Epoch " << (epoch + 1) << "/" << config.epochs
                              << " | Batch " << (b + 1) << "/" << numBatches
                              << " | Loss: " << std::fixed << std::setprecision(6) 
                              << (epochLoss / batchCount) << "\r" << std::flush;
                }
            }

            epochTimer.stop();

            // Store epoch statistics
            EpochStats stats;
            stats.epoch = epoch + 1;
            stats.avgLoss = epochLoss / batchCount;
            stats.epochTime = epochTimer.elapsedSec();
            stats.numBatches = batchCount;
            trainingHistory.push_back(stats);

            // Print epoch summary
            std::cout << "\n[Epoch " << std::setw(2) << (epoch + 1) << "/" << config.epochs << "] "
                      << "Loss: " << std::fixed << std::setprecision(6) << stats.avgLoss
                      << " | Time: " << std::fixed << std::setprecision(2) << stats.epochTime << "s"
                      << " | " << std::fixed << std::setprecision(2) 
                      << (stats.epochTime / stats.numBatches * 1000) << " ms/batch\n";
        }

        totalTimer.stop();

        std::cout << "\n========== TRAINING COMPLETED ==========\n";
        std::cout << "Total training time: " << std::fixed << std::setprecision(2) 
                  << totalTimer.elapsedSec() << " seconds\n";
        std::cout << "=========================================\n";

        // Save model
        model.saveWeights(config.saveModelPath);
    }

    /**
     * @brief Print detailed training report
     */
    void printTrainingReport() {
        std::cout << "\n============= TRAINING REPORT =============\n";
        
        // Epoch-by-epoch summary
        std::cout << std::left << std::setw(8) << "Epoch"
                  << std::right << std::setw(15) << "Loss"
                  << std::setw(15) << "Time (s)"
                  << std::setw(15) << "ms/batch" << "\n";
        std::cout << std::string(53, '-') << "\n";

        for (const auto& stats : trainingHistory) {
            std::cout << std::left << std::setw(8) << stats.epoch
                      << std::right << std::setw(15) << std::fixed << std::setprecision(6) << stats.avgLoss
                      << std::setw(15) << std::fixed << std::setprecision(2) << stats.epochTime
                      << std::setw(15) << std::fixed << std::setprecision(2) 
                      << (stats.epochTime / stats.numBatches * 1000) << "\n";
        }

        // Summary statistics
        std::cout << std::string(53, '-') << "\n";
        
        double totalTime = 0, avgTime = 0;
        for (const auto& stats : trainingHistory) {
            totalTime += stats.epochTime;
        }
        avgTime = totalTime / trainingHistory.size();

        std::cout << "Total Training Time:  " << std::fixed << std::setprecision(2) 
                  << totalTime << " seconds\n";
        std::cout << "Average Time/Epoch:   " << std::fixed << std::setprecision(2) 
                  << avgTime << " seconds\n";
        std::cout << "Final Loss:           " << std::fixed << std::setprecision(6) 
                  << trainingHistory.back().avgLoss << "\n";
        std::cout << "=============================================\n";
    }

    /**
     * @brief Get the final reconstruction loss
     */
    DataType getFinalLoss() const {
        return trainingHistory.empty() ? 0.0f : trainingHistory.back().avgLoss;
    }

    /**
     * @brief Get total training time in seconds
     */
    double getTotalTrainingTime() const {
        return totalTimer.elapsedSec();
    }
};

#endif // TRAINER_H

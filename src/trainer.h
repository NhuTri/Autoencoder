/**
 * @file trainer.h
 * @brief Training loop with train/test evaluation and best model saving
 */

#ifndef TRAINER_H
#define TRAINER_H

#include "common.h"
#include "cifar10.h"
#include "autoencoder.h"
#include <iomanip>
#include <limits>

/**
 * @struct TrainingConfig
 */
struct TrainingConfig {
    int batchSize = 32;
    int epochs = 20;
    float learningRate = 0.001f;
    bool shuffle = true;
    int printEvery = 100;
    std::string saveModelPath = "autoencoder_weights.bin";
    std::string bestModelPath = "autoencoder_best.bin";
    bool useAdam = true;
    float adamBeta1 = 0.9f;
    float adamBeta2 = 0.999f;
};

/**
 * @struct EpochStats
 */
struct EpochStats {
    int epoch;
    double trainLoss;
    double testLoss;
    double epochTime;
    int numBatches;
    bool isBest;
};

/**
 * @class Trainer
 * @brief Handles training with train/test evaluation and best model tracking
 */
class Trainer {
private:
    Autoencoder& model;
    CIFAR10Dataset& dataset;
    TrainingConfig config;

    std::vector<EpochStats> trainingHistory;
    Timer totalTimer;
    
    double bestTestLoss = std::numeric_limits<double>::max();
    int bestEpoch = 0;

public:
    Trainer(Autoencoder& m, CIFAR10Dataset& d, const TrainingConfig& cfg = TrainingConfig())
        : model(m), dataset(d), config(cfg) {}

    /**
     * @brief Evaluate model on train or test set (forward only)
     */
    double evaluate(bool isTrainSet) {
        int numImages = isTrainSet ? dataset.getNumTrainImages() : dataset.getNumTestImages();
        int numBatches = (numImages + config.batchSize - 1) / config.batchSize;
        
        double totalLoss = 0.0;
        int batchCount = 0;
        
        for (int b = 0; b < numBatches; b++) {
            int startIdx = b * config.batchSize;
            int currentBatchSize = std::min(config.batchSize, numImages - startIdx);
            
            Tensor4D batch = isTrainSet ? 
                dataset.getTrainBatch(startIdx, currentBatchSize) :
                dataset.getTestBatch(startIdx, currentBatchSize);
            
            Tensor4D output = model.forward(batch);
            
            // Compute MSE loss manually (no backward)
            double loss = 0.0;
            for (size_t i = 0; i < output.data.size(); i++) {
                double diff = output.data[i] - batch.data[i];
                loss += diff * diff;
            }
            loss /= output.data.size();
            
            totalLoss += loss;
            batchCount++;
        }
        
        return totalLoss / batchCount;
    }

    /**
     * @brief Run training loop with train/test evaluation
     */
    void train() {
        std::cout << "\n========== TRAINING STARTED ==========\n";
        std::cout << "Batch size:     " << config.batchSize << std::endl;
        std::cout << "Epochs:         " << config.epochs << std::endl;
        std::cout << "Learning rate:  " << config.learningRate << std::endl;
        std::cout << "Optimizer:      " << (config.useAdam ? "Adam" : "SGD") << std::endl;
        std::cout << "Train samples:  " << dataset.getNumTrainImages() << std::endl;
        std::cout << "Test samples:   " << dataset.getNumTestImages() << std::endl;
        std::cout << "========================================\n\n";

        int numImages = dataset.getNumTrainImages();
        int numBatches = (numImages + config.batchSize - 1) / config.batchSize;

        AdamOptimizer adamOptimizer(config.learningRate, config.adamBeta1, config.adamBeta2);

        totalTimer.start();

        for (int epoch = 0; epoch < config.epochs; epoch++) {
            Timer epochTimer;
            epochTimer.start();

            if (config.shuffle) {
                dataset.shuffleTrainData();
            }

            double epochLoss = 0.0;
            int batchCount = 0;

            // Training loop
            for (int b = 0; b < numBatches; b++) {
                int startIdx = b * config.batchSize;
                int currentBatchSize = std::min(config.batchSize, numImages - startIdx);
                
                Tensor4D batch = dataset.getTrainBatch(startIdx, currentBatchSize);
                model.forward(batch);
                DataType loss = model.backward(batch);
                epochLoss += loss;
                batchCount++;

                if (config.useAdam) {
                    adamOptimizer.step();
                    model.updateWeightsAdam(adamOptimizer);
                } else {
                    model.updateWeights(config.learningRate);
                }

                if ((b + 1) % config.printEvery == 0) {
                    std::cout << "  Epoch " << (epoch + 1) << "/" << config.epochs
                              << " | Batch " << (b + 1) << "/" << numBatches
                              << " | Loss: " << std::fixed << std::setprecision(6) 
                              << (epochLoss / batchCount) << "\r" << std::flush;
                }
            }

            // Evaluate on train and test sets
            double trainLoss = evaluate(true);
            double testLoss = evaluate(false);

            epochTimer.stop();

            // Check if best model
            bool isBest = testLoss < bestTestLoss;
            if (isBest) {
                bestTestLoss = testLoss;
                bestEpoch = epoch + 1;
                model.saveWeights(config.bestModelPath);
            }

            // Store stats
            EpochStats stats;
            stats.epoch = epoch + 1;
            stats.trainLoss = trainLoss;
            stats.testLoss = testLoss;
            stats.epochTime = epochTimer.elapsedSec();
            stats.numBatches = batchCount;
            stats.isBest = isBest;
            trainingHistory.push_back(stats);

            // Print epoch summary
            std::cout << "\n[Epoch " << std::setw(2) << (epoch + 1) << "/" << config.epochs << "] "
                      << "Train: " << std::fixed << std::setprecision(6) << trainLoss
                      << " | Test: " << std::setprecision(6) << testLoss
                      << " | Time: " << std::setprecision(2) << stats.epochTime << "s";
            if (isBest) std::cout << " *Best*";
            std::cout << "\n";
        }

        totalTimer.stop();
        model.saveWeights(config.saveModelPath);

        std::cout << "\n========== TRAINING COMPLETED ==========\n";
        std::cout << "Total time:     " << std::fixed << std::setprecision(2) 
                  << totalTimer.elapsedSec() << " seconds\n";
        std::cout << "Best epoch:     " << bestEpoch << "\n";
        std::cout << "Best test loss: " << std::setprecision(6) << bestTestLoss << "\n";
        std::cout << "Best model:     " << config.bestModelPath << "\n";
        std::cout << "Final model:    " << config.saveModelPath << "\n";
        std::cout << "=========================================\n";
    }

    /**
     * @brief Print detailed training report
     */
    void printTrainingReport() {
        std::cout << "\n================= TRAINING REPORT =================\n";
        
        std::cout << std::left << std::setw(8) << "Epoch"
                  << std::right << std::setw(14) << "Train Loss"
                  << std::setw(14) << "Test Loss"
                  << std::setw(12) << "Time (s)"
                  << std::setw(8) << "Best" << "\n";
        std::cout << std::string(56, '-') << "\n";

        for (const auto& stats : trainingHistory) {
            std::cout << std::left << std::setw(8) << stats.epoch
                      << std::right << std::setw(14) << std::fixed << std::setprecision(6) << stats.trainLoss
                      << std::setw(14) << std::setprecision(6) << stats.testLoss
                      << std::setw(12) << std::setprecision(2) << stats.epochTime
                      << std::setw(8) << (stats.isBest ? "*" : "") << "\n";
        }

        std::cout << std::string(56, '-') << "\n";
        
        double totalTime = 0;
        for (const auto& stats : trainingHistory) totalTime += stats.epochTime;

        std::cout << "Total Training Time:   " << std::fixed << std::setprecision(2) << totalTime << " seconds\n";
        std::cout << "Average Time/Epoch:    " << std::setprecision(2) << (totalTime / trainingHistory.size()) << " seconds\n";
        std::cout << "Final Train Loss:      " << std::setprecision(6) << trainingHistory.back().trainLoss << "\n";
        std::cout << "Final Test Loss:       " << std::setprecision(6) << trainingHistory.back().testLoss << "\n";
        std::cout << "Best Test Loss:        " << std::setprecision(6) << bestTestLoss << " (Epoch " << bestEpoch << ")\n";
        std::cout << "===================================================\n";
    }

    /**
     * @brief Export training history to CSV file for visualization
     */
    void exportToCSV(const std::string& filename = "training_history_cpu.csv") {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Warning: Could not open " << filename << " for writing.\n";
            return;
        }
        
        // Write header
        file << "epoch,train_loss,test_loss,time_sec,is_best\n";
        
        // Write data
        for (const auto& stats : trainingHistory) {
            file << stats.epoch << ","
                 << std::fixed << std::setprecision(8) << stats.trainLoss << ","
                 << stats.testLoss << ","
                 << std::setprecision(4) << stats.epochTime << ","
                 << (stats.isBest ? 1 : 0) << "\n";
        }
        
        file.close();
        std::cout << "Training history exported to: " << filename << "\n";
    }

    DataType getFinalLoss() const {
        return trainingHistory.empty() ? 0.0f : trainingHistory.back().trainLoss;
    }

    DataType getFinalTestLoss() const {
        return trainingHistory.empty() ? 0.0f : trainingHistory.back().testLoss;
    }

    DataType getBestTestLoss() const { return bestTestLoss; }
    int getBestEpoch() const { return bestEpoch; }
    double getTotalTrainingTime() const { return totalTimer.elapsedSec(); }
};

#endif // TRAINER_H

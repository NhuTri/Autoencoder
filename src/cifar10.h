/**
 * @file cifar10.h
 * @brief CIFAR-10 dataset loading (same as CPU version)
 */

#ifndef CIFAR10_H
#define CIFAR10_H

#include "common.h"
#include <string>
#include <vector>
#include <fstream>

class CIFAR10Dataset {
public:
    static constexpr int IMAGE_HEIGHT = 32, IMAGE_WIDTH = 32, IMAGE_CHANNELS = 3;
    static constexpr int IMAGE_SIZE = 3072, NUM_CLASSES = 10, IMAGES_PER_BATCH = 10000;
    static constexpr int NUM_TRAIN_BATCHES = 5, TOTAL_TRAIN_IMAGES = 50000, TOTAL_TEST_IMAGES = 10000;

private:
    std::vector<float> trainImages, testImages;
    std::vector<int> trainLabels, testLabels;
    bool loaded = false;
    int numTrainImages = 0, numTestImages = 0;

public:
    bool load(const std::string& path, int maxTrainSamples = 0, float testRatio = 0.2f) {
        int targetTrain = (maxTrainSamples > 0 && maxTrainSamples < TOTAL_TRAIN_IMAGES) ? maxTrainSamples : TOTAL_TRAIN_IMAGES;
        int targetTest = (maxTrainSamples > 0) ? std::max(1, (int)(targetTrain * testRatio)) : TOTAL_TEST_IMAGES;
        targetTest = std::min(targetTest, TOTAL_TEST_IMAGES);
        
        std::cout << "Loading CIFAR-10 from: " << path << std::endl;
        if (maxTrainSamples > 0) std::cout << "  [LIMITED] Train: " << targetTrain << ", Test: " << targetTest << std::endl;

        trainImages.resize(targetTrain * IMAGE_SIZE);
        trainLabels.resize(targetTrain);
        
        int loadedTrain = 0;
        for (int batch = 1; batch <= NUM_TRAIN_BATCHES && loadedTrain < targetTrain; batch++) {
            std::string filename = path + "/data_batch_" + std::to_string(batch) + ".bin";
            int toLoad = std::min(IMAGES_PER_BATCH, targetTrain - loadedTrain);
            if (!loadBatch(filename, trainImages.data() + loadedTrain * IMAGE_SIZE, trainLabels.data() + loadedTrain, toLoad)) return false;
            loadedTrain += toLoad;
            std::cout << "  Loaded batch " << batch << " (" << toLoad << " images)" << std::endl;
        }
        numTrainImages = loadedTrain;

        testImages.resize(targetTest * IMAGE_SIZE);
        testLabels.resize(targetTest);
        if (!loadBatch(path + "/test_batch.bin", testImages.data(), testLabels.data(), targetTest)) return false;
        numTestImages = targetTest;
        
        loaded = true;
        gMemoryTracker.addData((trainImages.size() + testImages.size()) * sizeof(float));
        std::cout << "Dataset loaded! Train: " << numTrainImages << ", Test: " << numTestImages << std::endl;
        return true;
    }

    Tensor4D getTrainBatch(int startIdx, int batchSize) const {
        Tensor4D batch(batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
        for (int i = 0; i < batchSize; i++) {
            const float* src = trainImages.data() + (startIdx + i) * IMAGE_SIZE;
            for (int h = 0; h < IMAGE_HEIGHT; h++)
                for (int w = 0; w < IMAGE_WIDTH; w++)
                    for (int c = 0; c < IMAGE_CHANNELS; c++)
                        batch.at(i, h, w, c) = src[(c * IMAGE_HEIGHT + h) * IMAGE_WIDTH + w];
        }
        return batch;
    }

    Tensor4D getTestBatch(int startIdx, int batchSize) const {
        Tensor4D batch(batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
        for (int i = 0; i < batchSize; i++) {
            const float* src = testImages.data() + (startIdx + i) * IMAGE_SIZE;
            for (int h = 0; h < IMAGE_HEIGHT; h++)
                for (int w = 0; w < IMAGE_WIDTH; w++)
                    for (int c = 0; c < IMAGE_CHANNELS; c++)
                        batch.at(i, h, w, c) = src[(c * IMAGE_HEIGHT + h) * IMAGE_WIDTH + w];
        }
        return batch;
    }

    void shuffleTrainData() {
        std::vector<int> idx(numTrainImages);
        std::iota(idx.begin(), idx.end(), 0);
        std::random_device rd; std::mt19937 gen(rd());
        std::shuffle(idx.begin(), idx.end(), gen);
        std::vector<float> tmpImg(trainImages.size());
        std::vector<int> tmpLbl(trainLabels.size());
        for (int i = 0; i < numTrainImages; i++) {
            std::copy(trainImages.begin() + idx[i] * IMAGE_SIZE, trainImages.begin() + (idx[i] + 1) * IMAGE_SIZE, tmpImg.begin() + i * IMAGE_SIZE);
            tmpLbl[i] = trainLabels[idx[i]];
        }
        trainImages = std::move(tmpImg); trainLabels = std::move(tmpLbl);
    }

    int getNumTrainImages() const { return numTrainImages; }
    int getNumTestImages() const { return numTestImages; }

    std::vector<int> getTestLabels(int startIdx, int batchSize) const {
        return std::vector<int>(testLabels.begin() + startIdx, testLabels.begin() + startIdx + batchSize);
    }

private:
    bool loadBatch(const std::string& filename, float* images, int* labels, int numImages) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) { std::cerr << "Cannot open: " << filename << std::endl; return false; }
        std::vector<unsigned char> buffer(1 + IMAGE_SIZE);
        for (int i = 0; i < numImages; i++) {
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
            labels[i] = buffer[0];
            for (int j = 0; j < IMAGE_SIZE; j++) images[i * IMAGE_SIZE + j] = buffer[1 + j] / 255.0f;
        }
        return true;
    }
};

#endif // CIFAR10_H

/**
 * @file cifar10.h
 * @brief CIFAR-10 dataset loading and preprocessing
 */

#ifndef CIFAR10_H
#define CIFAR10_H

#include "common.h"
#include <string>
#include <vector>
#include <fstream>

/**
 * @class CIFAR10Dataset
 * @brief Handles loading and preprocessing of CIFAR-10 dataset
 * 
 * CIFAR-10 binary format:
 * - Each image is 3073 bytes: 1 byte label + 3072 bytes pixel data
 * - Pixel data is stored as: 1024 red + 1024 green + 1024 blue
 * - Images are 32x32 pixels
 */
class CIFAR10Dataset {
public:
    // Dataset specifications
    static constexpr int IMAGE_HEIGHT = 32;
    static constexpr int IMAGE_WIDTH = 32;
    static constexpr int IMAGE_CHANNELS = 3;
    static constexpr int IMAGE_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNELS; // 3072
    static constexpr int NUM_CLASSES = 10;
    static constexpr int IMAGES_PER_BATCH = 10000;
    static constexpr int NUM_TRAIN_BATCHES = 5;
    static constexpr int TOTAL_TRAIN_IMAGES = NUM_TRAIN_BATCHES * IMAGES_PER_BATCH; // 50000
    static constexpr int TOTAL_TEST_IMAGES = IMAGES_PER_BATCH; // 10000

    // Class names
    static const std::vector<std::string> classNames;

private:
    std::string dataPath;
    
    // Training data
    std::vector<DataType> trainImages;  // Normalized to [0, 1]
    std::vector<int> trainLabels;
    
    // Test data  
    std::vector<DataType> testImages;   // Normalized to [0, 1]
    std::vector<int> testLabels;

    bool loaded;
    int numTrainImages;  // Actual number of training images loaded
    int numTestImages;   // Actual number of test images loaded

public:
    CIFAR10Dataset() : loaded(false), numTrainImages(0), numTestImages(0) {}

    /**
     * @brief Load CIFAR-10 dataset from binary files
     * @param path Path to CIFAR-10 data directory (containing data_batch_1.bin, etc.)
     * @param maxTrainSamples Maximum number of training samples to load (0 = all)
     * @param testRatio Ratio of test samples relative to training (default 0.2 = 20%)
     * @return true if loading successful
     */
    bool load(const std::string& path, int maxTrainSamples = 0, float testRatio = 0.2f) {
        dataPath = path;
        
        // Determine actual number of samples to load
        int targetTrainImages = (maxTrainSamples > 0 && maxTrainSamples < TOTAL_TRAIN_IMAGES) 
                                ? maxTrainSamples : TOTAL_TRAIN_IMAGES;
        int targetTestImages = (maxTrainSamples > 0) 
                               ? std::max(1, static_cast<int>(targetTrainImages * testRatio))
                               : TOTAL_TEST_IMAGES;
        
        // Make sure test images don't exceed available
        targetTestImages = std::min(targetTestImages, TOTAL_TEST_IMAGES);
        
        std::cout << "Loading CIFAR-10 dataset from: " << path << std::endl;
        if (maxTrainSamples > 0) {
            std::cout << "  [LIMITED MODE] Train: " << targetTrainImages 
                      << ", Test: " << targetTestImages << " (" << (testRatio * 100) << "%)" << std::endl;
        }

        // Reserve memory for training data
        trainImages.resize(targetTrainImages * IMAGE_SIZE);
        trainLabels.resize(targetTrainImages);
        
        // Load training batches
        int loadedTrainImages = 0;
        for (int batch = 1; batch <= NUM_TRAIN_BATCHES && loadedTrainImages < targetTrainImages; batch++) {
            std::string filename = path + "/data_batch_" + std::to_string(batch) + ".bin";
            int toLoad = std::min(IMAGES_PER_BATCH, targetTrainImages - loadedTrainImages);
            
            if (!loadBatch(filename, 
                          trainImages.data() + loadedTrainImages * IMAGE_SIZE,
                          trainLabels.data() + loadedTrainImages,
                          toLoad)) {
                std::cerr << "Failed to load training batch " << batch << std::endl;
                return false;
            }
            loadedTrainImages += toLoad;
            std::cout << "  Loaded training batch " << batch << " (" << toLoad << " images)" << std::endl;
            
            if (loadedTrainImages >= targetTrainImages) break;
        }
        numTrainImages = loadedTrainImages;

        // Reserve memory for test data
        testImages.resize(targetTestImages * IMAGE_SIZE);
        testLabels.resize(targetTestImages);

        // Load test batch
        std::string testFilename = path + "/test_batch.bin";
        if (!loadBatch(testFilename, testImages.data(), testLabels.data(), targetTestImages)) {
            std::cerr << "Failed to load test batch" << std::endl;
            return false;
        }
        numTestImages = targetTestImages;
        std::cout << "  Loaded test batch (" << targetTestImages << " images)" << std::endl;

        loaded = true;
        
        // Track memory usage
        gMemoryTracker.addData(trainImages.size() * sizeof(DataType));
        gMemoryTracker.addData(testImages.size() * sizeof(DataType));
        gMemoryTracker.addData(trainLabels.size() * sizeof(int));
        gMemoryTracker.addData(testLabels.size() * sizeof(int));

        std::cout << "CIFAR-10 dataset loaded successfully!" << std::endl;
        std::cout << "  Training images: " << numTrainImages << std::endl;
        std::cout << "  Test images: " << numTestImages << std::endl;
        
        return true;
    }

    /**
     * @brief Get a batch of training images as a Tensor4D
     * @param startIdx Starting index in training set
     * @param batchSize Number of images to get
     * @return Tensor4D with shape (batchSize, 32, 32, 3)
     */
    Tensor4D getTrainBatch(int startIdx, int batchSize) const {
        assert(loaded && "Dataset not loaded!");
        assert(startIdx >= 0 && startIdx + batchSize <= TOTAL_TRAIN_IMAGES);

        Tensor4D batch(batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
        
        for (int i = 0; i < batchSize; i++) {
            int imgIdx = startIdx + i;
            const DataType* src = trainImages.data() + imgIdx * IMAGE_SIZE;
            
            // Copy image data (already in HWC format after preprocessing)
            for (int h = 0; h < IMAGE_HEIGHT; h++) {
                for (int w = 0; w < IMAGE_WIDTH; w++) {
                    for (int c = 0; c < IMAGE_CHANNELS; c++) {
                        batch.at(i, h, w, c) = src[(c * IMAGE_HEIGHT + h) * IMAGE_WIDTH + w];
                    }
                }
            }
        }
        
        return batch;
    }

    /**
     * @brief Get a batch of test images as a Tensor4D
     */
    Tensor4D getTestBatch(int startIdx, int batchSize) const {
        assert(loaded && "Dataset not loaded!");
        assert(startIdx >= 0 && startIdx + batchSize <= TOTAL_TEST_IMAGES);

        Tensor4D batch(batchSize, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS);
        
        for (int i = 0; i < batchSize; i++) {
            int imgIdx = startIdx + i;
            const DataType* src = testImages.data() + imgIdx * IMAGE_SIZE;
            
            for (int h = 0; h < IMAGE_HEIGHT; h++) {
                for (int w = 0; w < IMAGE_WIDTH; w++) {
                    for (int c = 0; c < IMAGE_CHANNELS; c++) {
                        batch.at(i, h, w, c) = src[(c * IMAGE_HEIGHT + h) * IMAGE_WIDTH + w];
                    }
                }
            }
        }
        
        return batch;
    }

    /**
     * @brief Get training labels for a batch
     */
    std::vector<int> getTrainLabels(int startIdx, int batchSize) const {
        assert(loaded && "Dataset not loaded!");
        return std::vector<int>(trainLabels.begin() + startIdx, 
                               trainLabels.begin() + startIdx + batchSize);
    }

    /**
     * @brief Get test labels for a batch
     */
    std::vector<int> getTestLabels(int startIdx, int batchSize) const {
        assert(loaded && "Dataset not loaded!");
        return std::vector<int>(testLabels.begin() + startIdx,
                               testLabels.begin() + startIdx + batchSize);
    }

    /**
     * @brief Shuffle training data
     */
    void shuffleTrainData() {
        assert(loaded && "Dataset not loaded!");
        
        std::vector<int> indices(numTrainImages);
        std::iota(indices.begin(), indices.end(), 0);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(indices.begin(), indices.end(), gen);

        // Create temporary copies
        std::vector<DataType> tempImages(trainImages.size());
        std::vector<int> tempLabels(trainLabels.size());

        for (int i = 0; i < numTrainImages; i++) {
            int srcIdx = indices[i];
            std::copy(trainImages.begin() + srcIdx * IMAGE_SIZE,
                     trainImages.begin() + (srcIdx + 1) * IMAGE_SIZE,
                     tempImages.begin() + i * IMAGE_SIZE);
            tempLabels[i] = trainLabels[srcIdx];
        }

        trainImages = std::move(tempImages);
        trainLabels = std::move(tempLabels);
    }

    int getNumTrainImages() const { return numTrainImages; }
    int getNumTestImages() const { return numTestImages; }
    bool isLoaded() const { return loaded; }

private:
    /**
     * @brief Load a single CIFAR-10 binary batch file
     */
    bool loadBatch(const std::string& filename, DataType* images, int* labels, int numImages) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return false;
        }

        // Read each image
        std::vector<unsigned char> buffer(1 + IMAGE_SIZE); // 1 byte label + 3072 bytes image
        
        for (int i = 0; i < numImages; i++) {
            file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
            
            if (!file) {
                std::cerr << "Error reading image " << i << " from " << filename << std::endl;
                return false;
            }

            // First byte is label
            labels[i] = static_cast<int>(buffer[0]);

            // Remaining bytes are pixel values (R, G, B channels)
            // Normalize from [0, 255] to [0, 1]
            DataType* imgPtr = images + i * IMAGE_SIZE;
            for (int j = 0; j < IMAGE_SIZE; j++) {
                imgPtr[j] = static_cast<DataType>(buffer[1 + j]) / 255.0f;
            }
        }

        file.close();
        return true;
    }
};

// Initialize static member
const std::vector<std::string> CIFAR10Dataset::classNames = {
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
};

#endif // CIFAR10_H

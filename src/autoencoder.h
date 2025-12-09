/**
 * @file autoencoder.h
 * @brief Autoencoder model for CIFAR-10 feature learning
 * 
 * Architecture (from project specification):
 * 
 * ENCODER:
 *   INPUT: (32, 32, 3)
 *   Conv2D(256 filters, 3×3) + ReLU → (32, 32, 256)
 *   MaxPool2D(2×2) → (16, 16, 256)
 *   Conv2D(128 filters, 3×3) + ReLU → (16, 16, 128)
 *   MaxPool2D(2×2) → (8, 8, 128)
 *   LATENT: (8, 8, 128) = 8192 features
 * 
 * DECODER:
 *   Conv2D(128 filters, 3×3) + ReLU → (8, 8, 128)
 *   UpSample2D(2×2) → (16, 16, 128)
 *   Conv2D(256 filters, 3×3) + ReLU → (16, 16, 256)
 *   UpSample2D(2×2) → (32, 32, 256)
 *   Conv2D(3 filters, 3×3) → (32, 32, 3)
 *   OUTPUT: (32, 32, 3)
 */

#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "common.h"
#include "layers_cpu.h"
#include "optimizer.h"

class Autoencoder {
public:
    // ========== ENCODER LAYERS ==========
    Conv2D enc_conv1;      // (32,32,3) -> (32,32,256)
    ReLU enc_relu1;
    MaxPool2D enc_pool1;   // (32,32,256) -> (16,16,256)
    
    Conv2D enc_conv2;      // (16,16,256) -> (16,16,128)
    ReLU enc_relu2;
    MaxPool2D enc_pool2;   // (16,16,128) -> (8,8,128) = LATENT

    // ========== DECODER LAYERS ==========
    Conv2D dec_conv1;      // (8,8,128) -> (8,8,128)
    ReLU dec_relu1;
    UpSample2D dec_up1;    // (8,8,128) -> (16,16,128)
    
    Conv2D dec_conv2;      // (16,16,128) -> (16,16,256)
    ReLU dec_relu2;
    UpSample2D dec_up2;    // (16,16,256) -> (32,32,256)
    
    Conv2D dec_conv3;      // (32,32,256) -> (32,32,3)

    // Loss function
    MSELoss lossFn;

    // Cached intermediate activations for backward pass
    Tensor4D input_cached;
    Tensor4D enc_conv1_out, enc_relu1_out, enc_pool1_out;
    Tensor4D enc_conv2_out, enc_relu2_out, enc_pool2_out;
    Tensor4D dec_conv1_out, dec_relu1_out, dec_up1_out;
    Tensor4D dec_conv2_out, dec_relu2_out, dec_up2_out;
    Tensor4D dec_conv3_out;

    /**
     * @brief Constructor - Initialize all layers
     */
    Autoencoder()
        : enc_conv1(3, 256, 3, 1, 1),       // in=3, out=256, kernel=3, pad=1, stride=1
          enc_pool1(2, 2),                   // pool=2, stride=2
          enc_conv2(256, 128, 3, 1, 1),     // in=256, out=128
          enc_pool2(2, 2),
          dec_conv1(128, 128, 3, 1, 1),     // in=128, out=128
          dec_up1(2),                        // scale=2
          dec_conv2(128, 256, 3, 1, 1),     // in=128, out=256
          dec_up2(2),
          dec_conv3(256, 3, 3, 1, 1)        // in=256, out=3
    {
        std::cout << "Autoencoder initialized!" << std::endl;
        printArchitecture();
    }

    /**
     * @brief Print network architecture summary
     */
    void printArchitecture() {
        std::cout << "\n========== AUTOENCODER ARCHITECTURE ==========\n";
        std::cout << "ENCODER:\n";
        std::cout << "  Input:      (N, 32, 32, 3)\n";
        std::cout << "  Conv2D:     (N, 32, 32, 256)   params: " << (3*3*3*256 + 256) << "\n";
        std::cout << "  ReLU\n";
        std::cout << "  MaxPool:    (N, 16, 16, 256)\n";
        std::cout << "  Conv2D:     (N, 16, 16, 128)   params: " << (3*3*256*128 + 128) << "\n";
        std::cout << "  ReLU\n";
        std::cout << "  MaxPool:    (N, 8, 8, 128)     <- LATENT (8192 features)\n";
        std::cout << "\nDECODER:\n";
        std::cout << "  Conv2D:     (N, 8, 8, 128)     params: " << (3*3*128*128 + 128) << "\n";
        std::cout << "  ReLU\n";
        std::cout << "  UpSample:   (N, 16, 16, 128)\n";
        std::cout << "  Conv2D:     (N, 16, 16, 256)   params: " << (3*3*128*256 + 256) << "\n";
        std::cout << "  ReLU\n";
        std::cout << "  UpSample:   (N, 32, 32, 256)\n";
        std::cout << "  Conv2D:     (N, 32, 32, 3)     params: " << (3*3*256*3 + 3) << "\n";
        std::cout << "  Output:     (N, 32, 32, 3)\n";
        
        int totalParams = (3*3*3*256 + 256) +      // enc_conv1
                          (3*3*256*128 + 128) +    // enc_conv2
                          (3*3*128*128 + 128) +    // dec_conv1
                          (3*3*128*256 + 256) +    // dec_conv2
                          (3*3*256*3 + 3);         // dec_conv3
        std::cout << "\nTotal Parameters: " << totalParams << " (~" 
                  << (totalParams * sizeof(DataType) / 1024.0 / 1024.0) << " MB)\n";
        std::cout << "===============================================\n\n";
    }

    /**
     * @brief Full forward pass through encoder and decoder
     * @param input Input images (N, 32, 32, 3)
     * @return Reconstructed images (N, 32, 32, 3)
     */
    Tensor4D forward(const Tensor4D& input) {
        input_cached.copyFrom(input);

        // ========== ENCODER ==========
        enc_conv1_out = enc_conv1.forward(input);
        enc_relu1_out = enc_relu1.forward(enc_conv1_out);
        enc_pool1_out = enc_pool1.forward(enc_relu1_out);

        enc_conv2_out = enc_conv2.forward(enc_pool1_out);
        enc_relu2_out = enc_relu2.forward(enc_conv2_out);
        enc_pool2_out = enc_pool2.forward(enc_relu2_out);  // LATENT

        // ========== DECODER ==========
        dec_conv1_out = dec_conv1.forward(enc_pool2_out);
        dec_relu1_out = dec_relu1.forward(dec_conv1_out);
        dec_up1_out = dec_up1.forward(dec_relu1_out);

        dec_conv2_out = dec_conv2.forward(dec_up1_out);
        dec_relu2_out = dec_relu2.forward(dec_conv2_out);
        dec_up2_out = dec_up2.forward(dec_relu2_out);

        dec_conv3_out = dec_conv3.forward(dec_up2_out);

        return dec_conv3_out;
    }

    /**
     * @brief Encode input images to latent representation
     * @param input Input images (N, 32, 32, 3)
     * @return Latent features (N, 8, 8, 128) - can be flattened to (N, 8192)
     */
    Tensor4D encode(const Tensor4D& input) {
        auto x = enc_conv1.forward(input);
        x = enc_relu1.forward(x);
        x = enc_pool1.forward(x);

        x = enc_conv2.forward(x);
        x = enc_relu2.forward(x);
        x = enc_pool2.forward(x);

        return x;
    }

    /**
     * @brief Backward pass to compute gradients
     * @param target Target images for reconstruction loss
     * @return Loss value
     */
    DataType backward(const Tensor4D& target) {
        // Compute loss
        DataType loss = lossFn.forward(dec_conv3_out, target);
        
        // Get gradient of loss
        Tensor4D grad = lossFn.backward();

        // ========== DECODER BACKWARD ==========
        grad = dec_conv3.backward(grad);
        
        grad = dec_up2.backward(grad);
        grad = dec_relu2.backward(grad);
        grad = dec_conv2.backward(grad);

        grad = dec_up1.backward(grad);
        grad = dec_relu1.backward(grad);
        grad = dec_conv1.backward(grad);

        // ========== ENCODER BACKWARD ==========
        grad = enc_pool2.backward(grad);
        grad = enc_relu2.backward(grad);
        grad = enc_conv2.backward(grad);

        grad = enc_pool1.backward(grad);
        grad = enc_relu1.backward(grad);
        grad = enc_conv1.backward(grad);

        return loss;
    }

    /**
     * @brief Update all weights using SGD
     * @param learningRate Learning rate for gradient descent
     */
    void updateWeights(float learningRate) {
        enc_conv1.updateWeights(learningRate);
        enc_conv2.updateWeights(learningRate);
        dec_conv1.updateWeights(learningRate);
        dec_conv2.updateWeights(learningRate);
        dec_conv3.updateWeights(learningRate);
    }

    /**
     * @brief Update all weights using Adam optimizer
     * @param optimizer Adam optimizer instance
     */
    void updateWeightsAdam(AdamOptimizer& optimizer) {
        // Each conv layer has 2 parameter sets: weights and bias
        // We use unique IDs for each: layer_idx * 2 for weights, layer_idx * 2 + 1 for bias
        
        optimizer.update(enc_conv1.weights.data, enc_conv1.gradWeights.data, 0);
        optimizer.update(enc_conv1.bias, enc_conv1.gradBias, 1);
        
        optimizer.update(enc_conv2.weights.data, enc_conv2.gradWeights.data, 2);
        optimizer.update(enc_conv2.bias, enc_conv2.gradBias, 3);
        
        optimizer.update(dec_conv1.weights.data, dec_conv1.gradWeights.data, 4);
        optimizer.update(dec_conv1.bias, dec_conv1.gradBias, 5);
        
        optimizer.update(dec_conv2.weights.data, dec_conv2.gradWeights.data, 6);
        optimizer.update(dec_conv2.bias, dec_conv2.gradBias, 7);
        
        optimizer.update(dec_conv3.weights.data, dec_conv3.gradWeights.data, 8);
        optimizer.update(dec_conv3.bias, dec_conv3.gradBias, 9);
    }

    /**
     * @brief Save model weights to file
     */
    bool saveWeights(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open file for saving: " << filename << std::endl;
            return false;
        }

        auto saveConvLayer = [&file](const Conv2D& layer) {
            file.write(reinterpret_cast<const char*>(layer.weights.data.data()),
                      layer.weights.memorySize());
            file.write(reinterpret_cast<const char*>(layer.bias.data()),
                      layer.bias.size() * sizeof(DataType));
        };

        saveConvLayer(enc_conv1);
        saveConvLayer(enc_conv2);
        saveConvLayer(dec_conv1);
        saveConvLayer(dec_conv2);
        saveConvLayer(dec_conv3);

        file.close();
        std::cout << "Weights saved to: " << filename << std::endl;
        return true;
    }

    /**
     * @brief Load model weights from file
     */
    bool loadWeights(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Cannot open file for loading: " << filename << std::endl;
            return false;
        }

        auto loadConvLayer = [&file](Conv2D& layer) {
            file.read(reinterpret_cast<char*>(layer.weights.data.data()),
                     layer.weights.memorySize());
            file.read(reinterpret_cast<char*>(layer.bias.data()),
                     layer.bias.size() * sizeof(DataType));
        };

        loadConvLayer(enc_conv1);
        loadConvLayer(enc_conv2);
        loadConvLayer(dec_conv1);
        loadConvLayer(dec_conv2);
        loadConvLayer(dec_conv3);

        file.close();
        std::cout << "Weights loaded from: " << filename << std::endl;
        return true;
    }

    /**
     * @brief Extract features from all images (for SVM)
     * @param dataset CIFAR10 dataset
     * @param isTrainSet true for training set, false for test set
     * @param batchSize Batch size for processing
     * @return Feature matrix (numImages, 8192)
     */
    std::vector<std::vector<DataType>> extractFeatures(
        const CIFAR10Dataset& dataset, bool isTrainSet, int batchSize = 100) {
        
        int numImages = isTrainSet ? dataset.getNumTrainImages() : dataset.getNumTestImages();
        int numBatches = (numImages + batchSize - 1) / batchSize;
        
        std::vector<std::vector<DataType>> features(numImages);
        
        Timer timer;
        timer.start();

        for (int b = 0; b < numBatches; b++) {
            int startIdx = b * batchSize;
            int currentBatchSize = std::min(batchSize, numImages - startIdx);
            
            Tensor4D batch = isTrainSet ? 
                dataset.getTrainBatch(startIdx, currentBatchSize) :
                dataset.getTestBatch(startIdx, currentBatchSize);
            
            Tensor4D latent = encode(batch);
            
            // Flatten latent to features
            int featureSize = latent.height * latent.width * latent.channels;
            for (int i = 0; i < currentBatchSize; i++) {
                features[startIdx + i].resize(featureSize);
                for (int h = 0; h < latent.height; h++) {
                    for (int w = 0; w < latent.width; w++) {
                        for (int c = 0; c < latent.channels; c++) {
                            int idx = (h * latent.width + w) * latent.channels + c;
                            features[startIdx + i][idx] = latent.at(i, h, w, c);
                        }
                    }
                }
            }

            if ((b + 1) % 10 == 0 || b == numBatches - 1) {
                std::cout << "\rExtracting features: " << (b + 1) << "/" << numBatches 
                          << " batches" << std::flush;
            }
        }

        timer.stop();
        std::cout << "\nFeature extraction completed in " << timer.elapsedSec() << " seconds\n";
        
        return features;
    }
};

#endif // AUTOENCODER_H

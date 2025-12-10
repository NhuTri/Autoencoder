/**
 * @file autoencoder.h
 * @brief Autoencoder model with GPU-accelerated Conv2D
 */

#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "common.h"
#include "layers_gpu.h"
#include "optimizer.h"

class Autoencoder {
public:
    // ========== ENCODER LAYERS ==========
    Conv2D enc_conv1{3, 256, 3, 1, 1};      // (32,32,3) -> (32,32,256)
    ReLU enc_relu1;
    MaxPool2D enc_pool1{2, 2};              // -> (16,16,256)
    
    Conv2D enc_conv2{256, 128, 3, 1, 1};    // -> (16,16,128)
    ReLU enc_relu2;
    MaxPool2D enc_pool2{2, 2};              // -> (8,8,128) = LATENT

    // ========== DECODER LAYERS ==========
    Conv2D dec_conv1{128, 128, 3, 1, 1};    // (8,8,128) -> (8,8,128)
    ReLU dec_relu1;
    UpSample2D dec_up1{2};                  // -> (16,16,128)
    
    Conv2D dec_conv2{128, 256, 3, 1, 1};    // -> (16,16,256)
    ReLU dec_relu2;
    UpSample2D dec_up2{2};                  // -> (32,32,256)
    
    Conv2D dec_conv3{256, 3, 3, 1, 1};      // -> (32,32,3)

    MSELoss lossFn;
    Tensor4D dec_conv3_out;

    // Get output after forward pass
    const Tensor4D& getOutput() const { return dec_conv3_out; }

    Autoencoder() {
        std::cout << "\n========== AUTOENCODER (GPU) ==========\n";
        std::cout << "Conv2D layers: GPU-accelerated (CUDA)\n";
        std::cout << "Other layers:  CPU\n";
        std::cout << "\nArchitecture:\n";
        std::cout << "  Input(32×32×3) → Conv(256) → Pool → Conv(128) → Pool\n";
        std::cout << "  → LATENT(8×8×128) → Conv(128) → Up → Conv(256) → Up\n";
        std::cout << "  → Conv(3) → Output(32×32×3)\n";
        std::cout << "\nTotal params: ~751,875\n";
        std::cout << "========================================\n\n";
    }

    Tensor4D forward(const Tensor4D& input) {
        // Encoder
        auto x = enc_relu1.forward(enc_conv1.forward(input));
        x = enc_pool1.forward(x);
        x = enc_relu2.forward(enc_conv2.forward(x));
        x = enc_pool2.forward(x);  // LATENT
        
        // Decoder
        x = dec_relu1.forward(dec_conv1.forward(x));
        x = dec_up1.forward(x);
        x = dec_relu2.forward(dec_conv2.forward(x));
        x = dec_up2.forward(x);
        dec_conv3_out = dec_conv3.forward(x);
        
        return dec_conv3_out;
    }

    Tensor4D encode(const Tensor4D& input) {
        auto x = enc_relu1.forward(enc_conv1.forward(input));
        x = enc_pool1.forward(x);
        x = enc_relu2.forward(enc_conv2.forward(x));
        return enc_pool2.forward(x);
    }

    float backward(const Tensor4D& target) {
        float loss = lossFn.forward(dec_conv3_out, target);
        auto grad = lossFn.backward();
        
        // Decoder backward
        grad = dec_conv3.backward(grad);
        grad = dec_up2.backward(grad);
        grad = dec_relu2.backward(grad);
        grad = dec_conv2.backward(grad);
        grad = dec_up1.backward(grad);
        grad = dec_relu1.backward(grad);
        grad = dec_conv1.backward(grad);
        
        // Encoder backward
        grad = enc_pool2.backward(grad);
        grad = enc_relu2.backward(grad);
        grad = enc_conv2.backward(grad);
        grad = enc_pool1.backward(grad);
        grad = enc_relu1.backward(grad);
        enc_conv1.backward(grad);
        
        return loss;
    }

    void updateWeights(float lr) {
        enc_conv1.updateWeights(lr);
        enc_conv2.updateWeights(lr);
        dec_conv1.updateWeights(lr);
        dec_conv2.updateWeights(lr);
        dec_conv3.updateWeights(lr);
    }

    void updateWeightsAdam(AdamOptimizer& opt) {
        // Update each conv layer's weights and bias
        opt.update(enc_conv1.getWeights(), enc_conv1.getGradWeights(), 0);
        opt.update(enc_conv1.getBias(), enc_conv1.getGradBias(), 1);
        enc_conv1.syncWeightsToDevice();
        
        opt.update(enc_conv2.getWeights(), enc_conv2.getGradWeights(), 2);
        opt.update(enc_conv2.getBias(), enc_conv2.getGradBias(), 3);
        enc_conv2.syncWeightsToDevice();
        
        opt.update(dec_conv1.getWeights(), dec_conv1.getGradWeights(), 4);
        opt.update(dec_conv1.getBias(), dec_conv1.getGradBias(), 5);
        dec_conv1.syncWeightsToDevice();
        
        opt.update(dec_conv2.getWeights(), dec_conv2.getGradWeights(), 6);
        opt.update(dec_conv2.getBias(), dec_conv2.getGradBias(), 7);
        dec_conv2.syncWeightsToDevice();
        
        opt.update(dec_conv3.getWeights(), dec_conv3.getGradWeights(), 8);
        opt.update(dec_conv3.getBias(), dec_conv3.getGradBias(), 9);
        dec_conv3.syncWeightsToDevice();
    }

    bool saveWeights(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file) return false;
        
        auto saveConv = [&file](Conv2D& layer) {
            auto& w = layer.getWeights();
            auto& b = layer.getBias();
            file.write(reinterpret_cast<const char*>(w.data()), w.size() * sizeof(float));
            file.write(reinterpret_cast<const char*>(b.data()), b.size() * sizeof(float));
        };
        
        saveConv(enc_conv1); saveConv(enc_conv2);
        saveConv(dec_conv1); saveConv(dec_conv2); saveConv(dec_conv3);
        
        std::cout << "Weights saved to: " << filename << std::endl;
        return true;
    }

    bool loadWeights(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        
        auto loadConv = [&file](Conv2D& layer) {
            auto& w = layer.getWeights();
            auto& b = layer.getBias();
            file.read(reinterpret_cast<char*>(w.data()), w.size() * sizeof(float));
            file.read(reinterpret_cast<char*>(b.data()), b.size() * sizeof(float));
            layer.syncWeightsToDevice();
        };
        
        loadConv(enc_conv1); loadConv(enc_conv2);
        loadConv(dec_conv1); loadConv(dec_conv2); loadConv(dec_conv3);
        
        std::cout << "Weights loaded from: " << filename << std::endl;
        return true;
    }
};

#endif // AUTOENCODER_H

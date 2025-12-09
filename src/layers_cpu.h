/**
 * @file layers_cpu.h
 * @brief CPU implementations of neural network layers
 * 
 * All layers follow the same pattern:
 * - forward(): Compute output from input
 * - backward(): Compute gradients with respect to input and weights
 */

#ifndef LAYERS_CPU_H
#define LAYERS_CPU_H

#include "common.h"
#include <cmath>
#include <map>

// ============================================================================
// CONV2D LAYER - Convolutional Layer
// ============================================================================

/**
 * @class Conv2D
 * @brief 2D Convolution layer with padding and stride support
 * 
 * Input:  (N, H_in, W_in, C_in)
 * Output: (N, H_out, W_out, C_out)
 * Weights: (kernel_h, kernel_w, C_in, C_out)
 * Bias: (C_out)
 */
class Conv2D {
public:
    int inChannels;
    int outChannels;
    int kernelSize;
    int padding;
    int stride;

    Tensor4D weights;   // (kernel_h, kernel_w, C_in, C_out)
    std::vector<DataType> bias;  // (C_out)
    
    // Gradients
    Tensor4D gradWeights;
    std::vector<DataType> gradBias;

    // Cache for backward pass
    Tensor4D inputCache;

    Conv2D(int inCh, int outCh, int kSize = 3, int pad = 1, int str = 1)
        : inChannels(inCh), outChannels(outCh), kernelSize(kSize), 
          padding(pad), stride(str) {
        
        // Initialize weights with Xavier/He initialization
        weights = Tensor4D(kernelSize, kernelSize, inChannels, outChannels);
        float scale = std::sqrt(2.0f / (kernelSize * kernelSize * inChannels));
        weights.randomInit(scale);

        // Initialize bias to zero
        bias.resize(outChannels, 0.0f);

        // Initialize gradients
        gradWeights = Tensor4D(kernelSize, kernelSize, inChannels, outChannels);
        gradBias.resize(outChannels, 0.0f);

        // Track memory
        gMemoryTracker.addWeights(weights.memorySize() + bias.size() * sizeof(DataType));
        gMemoryTracker.addGradients(gradWeights.memorySize() + gradBias.size() * sizeof(DataType));
    }

    /**
     * @brief Forward pass: compute convolution
     * 
     * For each output pixel (n, oh, ow, oc):
     *   output[n,oh,ow,oc] = sum over (kh, kw, ic) of:
     *       input[n, oh*stride-pad+kh, ow*stride-pad+kw, ic] * weights[kh,kw,ic,oc]
     *   + bias[oc]
     */
    Tensor4D forward(const Tensor4D& input) {
        Timer timer;
        timer.start();

        // Cache input for backward pass
        inputCache.copyFrom(input);

        int N = input.batch;
        int H_in = input.height;
        int W_in = input.width;
        
        // Calculate output dimensions
        int H_out = (H_in + 2 * padding - kernelSize) / stride + 1;
        int W_out = (W_in + 2 * padding - kernelSize) / stride + 1;

        Tensor4D output(N, H_out, W_out, outChannels);
        gMemoryTracker.addActivations(output.memorySize());

        // Convolution
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    for (int oc = 0; oc < outChannels; oc++) {
                        DataType sum = bias[oc];
                        
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                // Check bounds (padding with zeros)
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    for (int ic = 0; ic < inChannels; ic++) {
                                        sum += input.at(n, ih, iw, ic) * 
                                               weights.at(kh, kw, ic, oc);
                                    }
                                }
                            }
                        }
                        
                        output.at(n, oh, ow, oc) = sum;
                    }
                }
            }
        }

        timer.stop();
        gProfiler.addTime("Conv2D_forward", timer.elapsedMs());

        return output;
    }

    /**
     * @brief Backward pass: compute gradients
     * 
     * @param gradOutput Gradient from next layer (N, H_out, W_out, C_out)
     * @return Gradient with respect to input (N, H_in, W_in, C_in)
     */
    Tensor4D backward(const Tensor4D& gradOutput) {
        Timer timer;
        timer.start();

        int N = inputCache.batch;
        int H_in = inputCache.height;
        int W_in = inputCache.width;
        int H_out = gradOutput.height;
        int W_out = gradOutput.width;

        // Initialize gradients
        Tensor4D gradInput(N, H_in, W_in, inChannels);
        gradWeights.fill(0.0f);
        std::fill(gradBias.begin(), gradBias.end(), 0.0f);

        // Compute gradients
        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    for (int oc = 0; oc < outChannels; oc++) {
                        DataType grad = gradOutput.at(n, oh, ow, oc);
                        
                        // Gradient for bias
                        gradBias[oc] += grad;
                        
                        for (int kh = 0; kh < kernelSize; kh++) {
                            for (int kw = 0; kw < kernelSize; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                
                                if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                                    for (int ic = 0; ic < inChannels; ic++) {
                                        // Gradient for weights
                                        gradWeights.at(kh, kw, ic, oc) += 
                                            inputCache.at(n, ih, iw, ic) * grad;
                                        
                                        // Gradient for input
                                        gradInput.at(n, ih, iw, ic) += 
                                            weights.at(kh, kw, ic, oc) * grad;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        timer.stop();
        gProfiler.addTime("Conv2D_backward", timer.elapsedMs());

        return gradInput;
    }

    /**
     * @brief Update weights using SGD
     */
    void updateWeights(float learningRate) {
        for (size_t i = 0; i < weights.data.size(); i++) {
            weights.data[i] -= learningRate * gradWeights.data[i];
        }
        for (size_t i = 0; i < bias.size(); i++) {
            bias[i] -= learningRate * gradBias[i];
        }
    }

    /**
     * @brief Get flattened gradients for optimizer
     */
    std::vector<DataType> getGradients() const {
        std::vector<DataType> grads;
        grads.reserve(weights.data.size() + bias.size());
        grads.insert(grads.end(), gradWeights.data.begin(), gradWeights.data.end());
        grads.insert(grads.end(), gradBias.begin(), gradBias.end());
        return grads;
    }

    /**
     * @brief Get flattened weights for optimizer
     */
    std::vector<DataType>& getWeightsFlat() {
        // We need to combine weights and bias into a single vector
        // But for efficiency, we'll update them separately
        return weights.data;
    }

    std::vector<DataType>& getBias() {
        return bias;
    }

    std::vector<DataType>& getGradWeightsFlat() {
        return gradWeights.data;
    }

    std::vector<DataType>& getGradBias() {
        return gradBias;
    }
};


// ============================================================================
// RELU ACTIVATION
// ============================================================================

/**
 * @class ReLU
 * @brief Rectified Linear Unit activation: f(x) = max(0, x)
 */
class ReLU {
public:
    Tensor4D maskCache;  // Cache which elements were positive

    Tensor4D forward(const Tensor4D& input) {
        Timer timer;
        timer.start();

        Tensor4D output(input.batch, input.height, input.width, input.channels);
        maskCache = Tensor4D(input.batch, input.height, input.width, input.channels);

        for (size_t i = 0; i < input.data.size(); i++) {
            if (input.data[i] > 0) {
                output.data[i] = input.data[i];
                maskCache.data[i] = 1.0f;
            } else {
                output.data[i] = 0.0f;
                maskCache.data[i] = 0.0f;
            }
        }

        gMemoryTracker.addActivations(output.memorySize());

        timer.stop();
        gProfiler.addTime("ReLU_forward", timer.elapsedMs());

        return output;
    }

    Tensor4D backward(const Tensor4D& gradOutput) {
        Timer timer;
        timer.start();

        Tensor4D gradInput(gradOutput.batch, gradOutput.height, 
                          gradOutput.width, gradOutput.channels);

        for (size_t i = 0; i < gradOutput.data.size(); i++) {
            gradInput.data[i] = gradOutput.data[i] * maskCache.data[i];
        }

        timer.stop();
        gProfiler.addTime("ReLU_backward", timer.elapsedMs());

        return gradInput;
    }
};


// ============================================================================
// MAX POOLING LAYER
// ============================================================================

/**
 * @class MaxPool2D
 * @brief 2D Max Pooling layer
 * 
 * Reduces spatial dimensions by taking maximum value in each window
 */
class MaxPool2D {
public:
    int poolSize;
    int stride;

    // Cache for backward pass - stores indices of max values
    Tensor4D maxIndicesH;
    Tensor4D maxIndicesW;
    int cachedH_in, cachedW_in;

    MaxPool2D(int size = 2, int str = 2) : poolSize(size), stride(str) {}

    Tensor4D forward(const Tensor4D& input) {
        Timer timer;
        timer.start();

        int N = input.batch;
        int H_in = input.height;
        int W_in = input.width;
        int C = input.channels;

        cachedH_in = H_in;
        cachedW_in = W_in;

        int H_out = (H_in - poolSize) / stride + 1;
        int W_out = (W_in - poolSize) / stride + 1;

        Tensor4D output(N, H_out, W_out, C);
        maxIndicesH = Tensor4D(N, H_out, W_out, C);
        maxIndicesW = Tensor4D(N, H_out, W_out, C);

        gMemoryTracker.addActivations(output.memorySize());

        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    for (int c = 0; c < C; c++) {
                        DataType maxVal = -std::numeric_limits<DataType>::infinity();
                        int maxH = 0, maxW = 0;

                        for (int ph = 0; ph < poolSize; ph++) {
                            for (int pw = 0; pw < poolSize; pw++) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                
                                DataType val = input.at(n, ih, iw, c);
                                if (val > maxVal) {
                                    maxVal = val;
                                    maxH = ih;
                                    maxW = iw;
                                }
                            }
                        }

                        output.at(n, oh, ow, c) = maxVal;
                        maxIndicesH.at(n, oh, ow, c) = static_cast<DataType>(maxH);
                        maxIndicesW.at(n, oh, ow, c) = static_cast<DataType>(maxW);
                    }
                }
            }
        }

        timer.stop();
        gProfiler.addTime("MaxPool2D_forward", timer.elapsedMs());

        return output;
    }

    Tensor4D backward(const Tensor4D& gradOutput) {
        Timer timer;
        timer.start();

        int N = gradOutput.batch;
        int H_out = gradOutput.height;
        int W_out = gradOutput.width;
        int C = gradOutput.channels;

        Tensor4D gradInput(N, cachedH_in, cachedW_in, C);
        gradInput.fill(0.0f);

        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    for (int c = 0; c < C; c++) {
                        int maxH = static_cast<int>(maxIndicesH.at(n, oh, ow, c));
                        int maxW = static_cast<int>(maxIndicesW.at(n, oh, ow, c));
                        
                        gradInput.at(n, maxH, maxW, c) += gradOutput.at(n, oh, ow, c);
                    }
                }
            }
        }

        timer.stop();
        gProfiler.addTime("MaxPool2D_backward", timer.elapsedMs());

        return gradInput;
    }
};


// ============================================================================
// UPSAMPLING LAYER (Nearest Neighbor)
// ============================================================================

/**
 * @class UpSample2D
 * @brief 2D Upsampling using nearest neighbor interpolation
 * 
 * Doubles spatial dimensions by copying each pixel to a 2x2 block
 */
class UpSample2D {
public:
    int scaleFactor;
    int cachedH_in, cachedW_in;

    UpSample2D(int scale = 2) : scaleFactor(scale) {}

    Tensor4D forward(const Tensor4D& input) {
        Timer timer;
        timer.start();

        int N = input.batch;
        int H_in = input.height;
        int W_in = input.width;
        int C = input.channels;

        cachedH_in = H_in;
        cachedW_in = W_in;

        int H_out = H_in * scaleFactor;
        int W_out = W_in * scaleFactor;

        Tensor4D output(N, H_out, W_out, C);

        gMemoryTracker.addActivations(output.memorySize());

        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    // Map output coordinates to input coordinates
                    int ih = oh / scaleFactor;
                    int iw = ow / scaleFactor;
                    
                    for (int c = 0; c < C; c++) {
                        output.at(n, oh, ow, c) = input.at(n, ih, iw, c);
                    }
                }
            }
        }

        timer.stop();
        gProfiler.addTime("UpSample2D_forward", timer.elapsedMs());

        return output;
    }

    Tensor4D backward(const Tensor4D& gradOutput) {
        Timer timer;
        timer.start();

        int N = gradOutput.batch;
        int H_out = gradOutput.height;
        int W_out = gradOutput.width;
        int C = gradOutput.channels;

        Tensor4D gradInput(N, cachedH_in, cachedW_in, C);
        gradInput.fill(0.0f);

        for (int n = 0; n < N; n++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int ih = oh / scaleFactor;
                    int iw = ow / scaleFactor;
                    
                    for (int c = 0; c < C; c++) {
                        // Sum gradients that were copied during forward
                        gradInput.at(n, ih, iw, c) += gradOutput.at(n, oh, ow, c);
                    }
                }
            }
        }

        timer.stop();
        gProfiler.addTime("UpSample2D_backward", timer.elapsedMs());

        return gradInput;
    }
};


// ============================================================================
// MSE LOSS FUNCTION
// ============================================================================

/**
 * @class MSELoss
 * @brief Mean Squared Error loss for reconstruction
 * 
 * Loss = (1/N) * sum((output - target)^2)
 */
class MSELoss {
public:
    Tensor4D outputCache;
    Tensor4D targetCache;

    /**
     * @brief Compute MSE loss value
     */
    DataType forward(const Tensor4D& output, const Tensor4D& target) {
        Timer timer;
        timer.start();

        assert(output.size() == target.size());
        
        outputCache.copyFrom(output);
        targetCache.copyFrom(target);

        DataType loss = 0.0f;
        for (size_t i = 0; i < output.data.size(); i++) {
            DataType diff = output.data[i] - target.data[i];
            loss += diff * diff;
        }
        loss /= output.data.size();

        timer.stop();
        gProfiler.addTime("MSELoss_forward", timer.elapsedMs());

        return loss;
    }

    /**
     * @brief Compute gradient of loss with respect to output
     * 
     * d(MSE)/d(output) = 2 * (output - target) / N
     */
    Tensor4D backward() {
        Timer timer;
        timer.start();

        Tensor4D gradOutput(outputCache.batch, outputCache.height,
                           outputCache.width, outputCache.channels);

        DataType scale = 2.0f / outputCache.data.size();
        for (size_t i = 0; i < outputCache.data.size(); i++) {
            gradOutput.data[i] = scale * (outputCache.data[i] - targetCache.data[i]);
        }

        timer.stop();
        gProfiler.addTime("MSELoss_backward", timer.elapsedMs());

        return gradOutput;
    }
};

#endif // LAYERS_CPU_H

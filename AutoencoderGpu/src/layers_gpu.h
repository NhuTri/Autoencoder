/**
 * @file layers_gpu.h
 * @brief Neural network layers with GPU-accelerated Conv2D
 * 
 * Conv2D uses CUDA kernels, other layers remain on CPU for Phase 2 (Naive GPU)
 */

#ifndef LAYERS_GPU_H
#define LAYERS_GPU_H

#include "common.h"
#include "conv2d_gpu.cu"
#include <cmath>
#include <limits>

// ============================================================================
// CONV2D LAYER - GPU ACCELERATED WRAPPER
// ============================================================================

/**
 * @class Conv2D
 * @brief Wrapper around Conv2DGPU that integrates with Tensor4D
 */
class Conv2D {
public:
    Conv2DGPU gpuConv;
    
    int inChannels, outChannels, kernelSize, padding, stride;
    
    // Cache for backward pass
    Tensor4D inputCache;
    
    Conv2D(int inCh, int outCh, int kSize = 3, int pad = 1, int str = 1)
        : gpuConv(inCh, outCh, kSize, pad, str),
          inChannels(inCh), outChannels(outCh), kernelSize(kSize), 
          padding(pad), stride(str) 
    {
        gMemoryTracker.addWeights(kSize * kSize * inCh * outCh * sizeof(float) + outCh * sizeof(float));
        gMemoryTracker.addGPU(gpuConv.gpuMemoryUsed);
    }
    
    Tensor4D forward(const Tensor4D& input) {
        Timer timer;
        timer.start();
        
        inputCache.copyFrom(input);
        
        int N = input.batch;
        int H_in = input.height;
        int W_in = input.width;
        int H_out = gpuConv.getOutputHeight(H_in);
        int W_out = gpuConv.getOutputWidth(W_in);
        
        Tensor4D output(N, H_out, W_out, outChannels);
        gMemoryTracker.addActivations(output.memorySize());
        
        // Call GPU implementation
        gpuConv.forward(input.data.data(), output.data.data(), N, H_in, W_in);
        
        timer.stop();
        gProfiler.addTime("Conv2D_forward_GPU", timer.elapsedMs());
        
        return output;
    }
    
    Tensor4D backward(const Tensor4D& gradOutput) {
        Timer timer;
        timer.start();
        
        int N = inputCache.batch;
        int H_in = inputCache.height;
        int W_in = inputCache.width;
        
        Tensor4D gradInput(N, H_in, W_in, inChannels);
        
        // Call GPU implementation
        gpuConv.backward(inputCache.data.data(), gradOutput.data.data(), 
                        gradInput.data.data(), N, H_in, W_in);
        
        timer.stop();
        gProfiler.addTime("Conv2D_backward_GPU", timer.elapsedMs());
        
        return gradInput;
    }
    
    void updateWeights(float learningRate) {
        gpuConv.updateWeightsSGD(learningRate);
    }
    
    // For Adam optimizer
    std::vector<float>& getWeights() { return gpuConv.getWeights(); }
    std::vector<float>& getBias() { return gpuConv.getBias(); }
    std::vector<float>& getGradWeights() { return gpuConv.getGradWeights(); }
    std::vector<float>& getGradBias() { return gpuConv.getGradBias(); }
    void syncWeightsToDevice() { gpuConv.syncWeightsToDevice(); }
};


// ============================================================================
// RELU ACTIVATION - CPU (simple operation, not bottleneck)
// ============================================================================

class ReLU {
public:
    Tensor4D maskCache;

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
// MAX POOLING LAYER - CPU
// ============================================================================

class MaxPool2D {
public:
    int poolSize, stride;
    Tensor4D maxIndicesH, maxIndicesW;
    int cachedH_in, cachedW_in;

    MaxPool2D(int size = 2, int str = 2) : poolSize(size), stride(str) {}

    Tensor4D forward(const Tensor4D& input) {
        Timer timer;
        timer.start();

        int N = input.batch, H_in = input.height, W_in = input.width, C = input.channels;
        cachedH_in = H_in; cachedW_in = W_in;

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
                        float maxVal = -std::numeric_limits<float>::infinity();
                        int maxH = 0, maxW = 0;

                        for (int ph = 0; ph < poolSize; ph++) {
                            for (int pw = 0; pw < poolSize; pw++) {
                                int ih = oh * stride + ph;
                                int iw = ow * stride + pw;
                                
                                float val = input.at(n, ih, iw, c);
                                if (val > maxVal) {
                                    maxVal = val; maxH = ih; maxW = iw;
                                }
                            }
                        }

                        output.at(n, oh, ow, c) = maxVal;
                        maxIndicesH.at(n, oh, ow, c) = maxH;
                        maxIndicesW.at(n, oh, ow, c) = maxW;
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

        Tensor4D gradInput(gradOutput.batch, cachedH_in, cachedW_in, gradOutput.channels);
        gradInput.fill(0.0f);

        for (int n = 0; n < gradOutput.batch; n++) {
            for (int oh = 0; oh < gradOutput.height; oh++) {
                for (int ow = 0; ow < gradOutput.width; ow++) {
                    for (int c = 0; c < gradOutput.channels; c++) {
                        int maxH = (int)maxIndicesH.at(n, oh, ow, c);
                        int maxW = (int)maxIndicesW.at(n, oh, ow, c);
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
// UPSAMPLING LAYER - CPU
// ============================================================================

class UpSample2D {
public:
    int scaleFactor;
    int cachedH_in, cachedW_in;

    UpSample2D(int scale = 2) : scaleFactor(scale) {}

    Tensor4D forward(const Tensor4D& input) {
        Timer timer;
        timer.start();

        cachedH_in = input.height; cachedW_in = input.width;
        int H_out = input.height * scaleFactor;
        int W_out = input.width * scaleFactor;

        Tensor4D output(input.batch, H_out, W_out, input.channels);
        gMemoryTracker.addActivations(output.memorySize());

        for (int n = 0; n < input.batch; n++) {
            for (int oh = 0; oh < H_out; oh++) {
                for (int ow = 0; ow < W_out; ow++) {
                    int ih = oh / scaleFactor;
                    int iw = ow / scaleFactor;
                    for (int c = 0; c < input.channels; c++) {
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

        Tensor4D gradInput(gradOutput.batch, cachedH_in, cachedW_in, gradOutput.channels);
        gradInput.fill(0.0f);

        for (int n = 0; n < gradOutput.batch; n++) {
            for (int oh = 0; oh < gradOutput.height; oh++) {
                for (int ow = 0; ow < gradOutput.width; ow++) {
                    int ih = oh / scaleFactor;
                    int iw = ow / scaleFactor;
                    for (int c = 0; c < gradOutput.channels; c++) {
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
// MSE LOSS - CPU
// ============================================================================

class MSELoss {
public:
    Tensor4D outputCache, targetCache;

    float forward(const Tensor4D& output, const Tensor4D& target) {
        Timer timer;
        timer.start();

        outputCache.copyFrom(output);
        targetCache.copyFrom(target);

        float loss = 0.0f;
        for (size_t i = 0; i < output.data.size(); i++) {
            float diff = output.data[i] - target.data[i];
            loss += diff * diff;
        }
        loss /= output.data.size();

        timer.stop();
        gProfiler.addTime("MSELoss_forward", timer.elapsedMs());

        return loss;
    }

    Tensor4D backward() {
        Timer timer;
        timer.start();

        Tensor4D grad(outputCache.batch, outputCache.height,
                     outputCache.width, outputCache.channels);

        float scale = 2.0f / outputCache.data.size();
        for (size_t i = 0; i < outputCache.data.size(); i++) {
            grad.data[i] = scale * (outputCache.data[i] - targetCache.data[i]);
        }

        timer.stop();
        gProfiler.addTime("MSELoss_backward", timer.elapsedMs());

        return grad;
    }
};

#endif // LAYERS_GPU_H

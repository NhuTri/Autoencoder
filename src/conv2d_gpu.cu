/**
 * @file conv2d_gpu.cu
 * @brief CUDA implementation of Conv2D layer (Naive GPU Implementation)
 * 
 * Phase 2: Each CUDA thread computes ONE output pixel
 * This is a straightforward parallelization without advanced optimizations.
 */

#ifndef CONV2D_GPU_CU
#define CONV2D_GPU_CU

#include <cuda_runtime.h>
#include <iostream>

// ============================================================================
// CUDA ERROR CHECKING MACRO
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// CONV2D FORWARD KERNEL - Naive Implementation
// ============================================================================

/**
 * @brief Naive Conv2D forward kernel
 * 
 * Each thread computes one output pixel at position (n, oh, ow, oc)
 * Input: NHWC format, Output: NHWC format, Weights: (kH, kW, inC, outC)
 */
__global__ void conv2d_forward_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int kernelSize, int padding, int stride)
{
    // Calculate output position from thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * H_out * W_out * C_out;
    
    if (idx >= total_outputs) return;
    
    // Decode linear index to (n, oh, ow, oc)
    int oc = idx % C_out;
    int temp = idx / C_out;
    int ow = temp % W_out;
    temp = temp / W_out;
    int oh = temp % H_out;
    int n = temp / H_out;
    
    // Compute convolution for this output pixel
    float sum = bias[oc];
    
    for (int kh = 0; kh < kernelSize; kh++) {
        for (int kw = 0; kw < kernelSize; kw++) {
            int ih = oh * stride - padding + kh;
            int iw = ow * stride - padding + kw;
            
            // Check bounds (padding with zeros)
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                for (int ic = 0; ic < C_in; ic++) {
                    // Input index: NHWC format
                    int input_idx = ((n * H_in + ih) * W_in + iw) * C_in + ic;
                    // Weight index: (kh, kw, ic, oc)
                    int weight_idx = ((kh * kernelSize + kw) * C_in + ic) * C_out + oc;
                    
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}

// ============================================================================
// CONV2D BACKWARD INPUT KERNEL - Naive Implementation
// ============================================================================

/**
 * @brief Naive Conv2D backward kernel for input gradients
 * 
 * Computes gradient with respect to input for backpropagation
 */
__global__ void conv2d_backward_input_kernel(
    const float* __restrict__ gradOutput,
    const float* __restrict__ weights,
    float* __restrict__ gradInput,
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int kernelSize, int padding, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = N * H_in * W_in * C_in;
    
    if (idx >= total_inputs) return;
    
    // Decode to (n, ih, iw, ic)
    int ic = idx % C_in;
    int temp = idx / C_in;
    int iw = temp % W_in;
    temp = temp / W_in;
    int ih = temp % H_in;
    int n = temp / H_in;
    
    float sum = 0.0f;
    
    // For each output position that uses this input
    for (int kh = 0; kh < kernelSize; kh++) {
        for (int kw = 0; kw < kernelSize; kw++) {
            // Calculate which output position this input contributes to
            int oh_check = ih + padding - kh;
            int ow_check = iw + padding - kw;
            
            // Check if this maps to a valid output position
            if (oh_check >= 0 && oh_check % stride == 0 && ow_check >= 0 && ow_check % stride == 0) {
                int oh = oh_check / stride;
                int ow = ow_check / stride;
                
                if (oh < H_out && ow < W_out) {
                    for (int oc = 0; oc < C_out; oc++) {
                        int grad_out_idx = ((n * H_out + oh) * W_out + ow) * C_out + oc;
                        int weight_idx = ((kh * kernelSize + kw) * C_in + ic) * C_out + oc;
                        
                        sum += gradOutput[grad_out_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }
    
    gradInput[idx] = sum;
}

// ============================================================================
// CONV2D BACKWARD WEIGHTS KERNEL - Naive Implementation
// ============================================================================

/**
 * @brief Naive Conv2D backward kernel for weight gradients
 * 
 * Uses atomicAdd since multiple threads contribute to same weight gradient
 */
__global__ void conv2d_backward_weights_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gradOutput,
    float* __restrict__ gradWeights,
    float* __restrict__ gradBias,
    int N, int H_in, int W_in, int C_in,
    int H_out, int W_out, int C_out,
    int kernelSize, int padding, int stride)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * H_out * W_out * C_out;
    
    if (idx >= total_outputs) return;
    
    // Decode to (n, oh, ow, oc)
    int oc = idx % C_out;
    int temp = idx / C_out;
    int ow = temp % W_out;
    temp = temp / W_out;
    int oh = temp % H_out;
    int n = temp / H_out;
    
    float grad = gradOutput[idx];
    
    // Accumulate gradient for bias
    atomicAdd(&gradBias[oc], grad);
    
    // Accumulate gradients for weights
    for (int kh = 0; kh < kernelSize; kh++) {
        for (int kw = 0; kw < kernelSize; kw++) {
            int ih = oh * stride - padding + kh;
            int iw = ow * stride - padding + kw;
            
            if (ih >= 0 && ih < H_in && iw >= 0 && iw < W_in) {
                for (int ic = 0; ic < C_in; ic++) {
                    int input_idx = ((n * H_in + ih) * W_in + iw) * C_in + ic;
                    int weight_idx = ((kh * kernelSize + kw) * C_in + ic) * C_out + oc;
                    
                    atomicAdd(&gradWeights[weight_idx], input[input_idx] * grad);
                }
            }
        }
    }
}

// ============================================================================
// CONV2D GPU WRAPPER CLASS
// ============================================================================

class Conv2DGPU {
public:
    int inChannels, outChannels, kernelSize, padding, stride;
    
    // Device pointers
    float *d_weights, *d_bias;
    float *d_gradWeights, *d_gradBias;
    float *d_input, *d_output;
    float *d_gradInput, *d_gradOutput;
    
    // Host weights (for initialization and retrieval)
    std::vector<float> h_weights, h_bias;
    std::vector<float> h_gradWeights, h_gradBias;
    
    // Cached dimensions
    int cached_N, cached_H_in, cached_W_in, cached_H_out, cached_W_out;
    
    // Memory tracking
    size_t gpuMemoryUsed = 0;
    
    Conv2DGPU(int inCh, int outCh, int kSize = 3, int pad = 1, int str = 1)
        : inChannels(inCh), outChannels(outCh), kernelSize(kSize), 
          padding(pad), stride(str),
          d_weights(nullptr), d_bias(nullptr), 
          d_gradWeights(nullptr), d_gradBias(nullptr),
          d_input(nullptr), d_output(nullptr),
          d_gradInput(nullptr), d_gradOutput(nullptr) 
    {
        int weightSize = kernelSize * kernelSize * inChannels * outChannels;
        
        // Initialize host weights with He initialization
        h_weights.resize(weightSize);
        h_bias.resize(outChannels, 0.0f);
        h_gradWeights.resize(weightSize, 0.0f);
        h_gradBias.resize(outChannels, 0.0f);
        
        float scale = std::sqrt(2.0f / (kernelSize * kernelSize * inChannels));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, scale);
        for (auto& val : h_weights) val = dist(gen);
        
        // Allocate device memory for weights and bias
        CUDA_CHECK(cudaMalloc(&d_weights, weightSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_bias, outChannels * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradWeights, weightSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradBias, outChannels * sizeof(float)));
        
        // Copy weights to device
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), outChannels * sizeof(float), cudaMemcpyHostToDevice));
        
        gpuMemoryUsed = (weightSize * 2 + outChannels * 2) * sizeof(float);
    }
    
    ~Conv2DGPU() {
        if (d_weights) cudaFree(d_weights);
        if (d_bias) cudaFree(d_bias);
        if (d_gradWeights) cudaFree(d_gradWeights);
        if (d_gradBias) cudaFree(d_gradBias);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_gradInput) cudaFree(d_gradInput);
        if (d_gradOutput) cudaFree(d_gradOutput);
    }
    
    /**
     * @brief Allocate input/output buffers for given batch size
     */
    void allocateBuffers(int N, int H_in, int W_in) {
        cached_N = N;
        cached_H_in = H_in;
        cached_W_in = W_in;
        cached_H_out = (H_in + 2 * padding - kernelSize) / stride + 1;
        cached_W_out = (W_in + 2 * padding - kernelSize) / stride + 1;
        
        size_t inputSize = N * H_in * W_in * inChannels;
        size_t outputSize = N * cached_H_out * cached_W_out * outChannels;
        
        // Free old buffers if they exist
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_gradInput) cudaFree(d_gradInput);
        if (d_gradOutput) cudaFree(d_gradOutput);
        
        CUDA_CHECK(cudaMalloc(&d_input, inputSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, outputSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradInput, inputSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradOutput, outputSize * sizeof(float)));
        
        gpuMemoryUsed += (inputSize + outputSize) * 2 * sizeof(float);
    }
    
    /**
     * @brief Forward pass on GPU
     */
    void forward(const float* h_input, float* h_output, int N, int H_in, int W_in) {
        // Allocate buffers if dimensions changed
        if (N != cached_N || H_in != cached_H_in || W_in != cached_W_in) {
            allocateBuffers(N, H_in, W_in);
        }
        
        size_t inputSize = N * H_in * W_in * inChannels;
        size_t outputSize = N * cached_H_out * cached_W_out * outChannels;
        
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
        
        // Launch kernel
        int blockSize = 256;
        int numBlocks = (outputSize + blockSize - 1) / blockSize;
        
        conv2d_forward_kernel<<<numBlocks, blockSize>>>(
            d_input, d_weights, d_bias, d_output,
            N, H_in, W_in, inChannels,
            cached_H_out, cached_W_out, outChannels,
            kernelSize, padding, stride
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy output back to host
        CUDA_CHECK(cudaMemcpy(h_output, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    /**
     * @brief Backward pass on GPU
     */
    void backward(const float* h_input, const float* h_gradOutput, 
                  float* h_gradInput, int N, int H_in, int W_in) {
        
        size_t inputSize = N * H_in * W_in * inChannels;
        size_t outputSize = N * cached_H_out * cached_W_out * outChannels;
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gradOutput, h_gradOutput, outputSize * sizeof(float), cudaMemcpyHostToDevice));
        
        // Zero gradients
        int weightSize = kernelSize * kernelSize * inChannels * outChannels;
        CUDA_CHECK(cudaMemset(d_gradWeights, 0, weightSize * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gradBias, 0, outChannels * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_gradInput, 0, inputSize * sizeof(float)));
        
        int blockSize = 256;
        
        // Compute weight gradients
        int numBlocksOutput = (outputSize + blockSize - 1) / blockSize;
        conv2d_backward_weights_kernel<<<numBlocksOutput, blockSize>>>(
            d_input, d_gradOutput, d_gradWeights, d_gradBias,
            N, H_in, W_in, inChannels,
            cached_H_out, cached_W_out, outChannels,
            kernelSize, padding, stride
        );
        
        // Compute input gradients
        int numBlocksInput = (inputSize + blockSize - 1) / blockSize;
        conv2d_backward_input_kernel<<<numBlocksInput, blockSize>>>(
            d_gradOutput, d_weights, d_gradInput,
            N, H_in, W_in, inChannels,
            cached_H_out, cached_W_out, outChannels,
            kernelSize, padding, stride
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy gradients back to host
        CUDA_CHECK(cudaMemcpy(h_gradInput, d_gradInput, inputSize * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_gradWeights.data(), d_gradWeights, weightSize * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_gradBias.data(), d_gradBias, outChannels * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    /**
     * @brief Update weights on device using SGD
     */
    void updateWeightsSGD(float learningRate) {
        // Update on host and copy back
        int weightSize = kernelSize * kernelSize * inChannels * outChannels;
        for (int i = 0; i < weightSize; i++) {
            h_weights[i] -= learningRate * h_gradWeights[i];
        }
        for (int i = 0; i < outChannels; i++) {
            h_bias[i] -= learningRate * h_gradBias[i];
        }
        
        // Copy updated weights to device
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), outChannels * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    // Getters for optimizer integration
    std::vector<float>& getWeights() { return h_weights; }
    std::vector<float>& getBias() { return h_bias; }
    std::vector<float>& getGradWeights() { return h_gradWeights; }
    std::vector<float>& getGradBias() { return h_gradBias; }
    
    void syncWeightsToDevice() {
        int weightSize = kernelSize * kernelSize * inChannels * outChannels;
        CUDA_CHECK(cudaMemcpy(d_weights, h_weights.data(), weightSize * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), outChannels * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    int getOutputHeight(int H_in) const { return (H_in + 2 * padding - kernelSize) / stride + 1; }
    int getOutputWidth(int W_in) const { return (W_in + 2 * padding - kernelSize) / stride + 1; }
};

#endif // CONV2D_GPU_CU

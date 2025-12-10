/**
 * @file optimizer.h
 * @brief Optimizers for neural network training
 * 
 * Implements:
 * - SGD (Stochastic Gradient Descent)
 * - Adam (Adaptive Moment Estimation)
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "common.h"
#include <cmath>

// ============================================================================
// BASE OPTIMIZER CLASS
// ============================================================================

class Optimizer {
public:
    float learningRate;
    
    Optimizer(float lr = 0.001f) : learningRate(lr) {}
    virtual ~Optimizer() = default;
    
    virtual void update(std::vector<DataType>& weights, 
                       const std::vector<DataType>& gradients,
                       int paramId) = 0;
    
    virtual void reset() = 0;
};

// ============================================================================
// SGD OPTIMIZER
// ============================================================================

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(float lr = 0.001f) : Optimizer(lr) {}
    
    void update(std::vector<DataType>& weights,
               const std::vector<DataType>& gradients,
               int /*paramId*/) override {
        for (size_t i = 0; i < weights.size(); i++) {
            weights[i] -= learningRate * gradients[i];
        }
    }
    
    void reset() override {}
};

// ============================================================================
// ADAM OPTIMIZER
// ============================================================================

/**
 * @class AdamOptimizer
 * @brief Adam optimizer with adaptive learning rates
 * 
 * Adam combines:
 * - Momentum: exponential moving average of gradients (m)
 * - RMSprop: exponential moving average of squared gradients (v)
 * 
 * Update rule:
 *   m = beta1 * m + (1 - beta1) * gradient
 *   v = beta2 * v + (1 - beta2) * gradient^2
 *   m_hat = m / (1 - beta1^t)  # bias correction
 *   v_hat = v / (1 - beta2^t)  # bias correction
 *   weight = weight - lr * m_hat / (sqrt(v_hat) + epsilon)
 */
class AdamOptimizer : public Optimizer {
public:
    float beta1;      // Exponential decay rate for first moment (momentum)
    float beta2;      // Exponential decay rate for second moment (RMSprop)
    float epsilon;    // Small constant for numerical stability
    int timestep;     // Current timestep for bias correction
    
    // First moment (momentum) for each parameter set
    std::map<int, std::vector<DataType>> m;
    // Second moment (squared gradients) for each parameter set
    std::map<int, std::vector<DataType>> v;
    
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps), timestep(0) {}
    
    void update(std::vector<DataType>& weights,
               const std::vector<DataType>& gradients,
               int paramId) override {
        
        // Initialize moment vectors if not exists
        if (m.find(paramId) == m.end()) {
            m[paramId].resize(weights.size(), 0.0f);
            v[paramId].resize(weights.size(), 0.0f);
        }
        
        std::vector<DataType>& m_t = m[paramId];
        std::vector<DataType>& v_t = v[paramId];
        
        // Note: timestep should be incremented once per batch, not per parameter
        // We'll handle this in the caller
        
        // Bias correction factors
        float bias_correction1 = 1.0f - std::pow(beta1, timestep);
        float bias_correction2 = 1.0f - std::pow(beta2, timestep);
        
        for (size_t i = 0; i < weights.size(); i++) {
            // Update biased first moment estimate
            m_t[i] = beta1 * m_t[i] + (1.0f - beta1) * gradients[i];
            
            // Update biased second raw moment estimate
            v_t[i] = beta2 * v_t[i] + (1.0f - beta2) * gradients[i] * gradients[i];
            
            // Compute bias-corrected estimates
            float m_hat = m_t[i] / bias_correction1;
            float v_hat = v_t[i] / bias_correction2;
            
            // Update weights
            weights[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
    
    void step() {
        timestep++;
    }
    
    void reset() override {
        timestep = 0;
        m.clear();
        v.clear();
    }
};

#endif // OPTIMIZER_H

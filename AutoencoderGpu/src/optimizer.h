/**
 * @file optimizer.h
 * @brief Optimizers for neural network training
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "common.h"
#include <cmath>

class Optimizer {
public:
    float learningRate;
    Optimizer(float lr = 0.001f) : learningRate(lr) {}
    virtual ~Optimizer() = default;
    virtual void update(std::vector<float>& weights, const std::vector<float>& gradients, int paramId) = 0;
    virtual void reset() = 0;
};

class SGDOptimizer : public Optimizer {
public:
    SGDOptimizer(float lr = 0.001f) : Optimizer(lr) {}
    void update(std::vector<float>& weights, const std::vector<float>& gradients, int) override {
        for (size_t i = 0; i < weights.size(); i++) weights[i] -= learningRate * gradients[i];
    }
    void reset() override {}
};

class AdamOptimizer : public Optimizer {
public:
    float beta1, beta2, epsilon;
    int timestep = 0;
    std::map<int, std::vector<float>> m, v;
    
    AdamOptimizer(float lr = 0.001f, float b1 = 0.9f, float b2 = 0.999f, float eps = 1e-8f)
        : Optimizer(lr), beta1(b1), beta2(b2), epsilon(eps) {}
    
    void update(std::vector<float>& weights, const std::vector<float>& gradients, int paramId) override {
        if (m.find(paramId) == m.end()) {
            m[paramId].resize(weights.size(), 0.0f);
            v[paramId].resize(weights.size(), 0.0f);
        }
        
        auto& m_t = m[paramId];
        auto& v_t = v[paramId];
        
        float bc1 = 1.0f - std::pow(beta1, timestep);
        float bc2 = 1.0f - std::pow(beta2, timestep);
        
        for (size_t i = 0; i < weights.size(); i++) {
            m_t[i] = beta1 * m_t[i] + (1.0f - beta1) * gradients[i];
            v_t[i] = beta2 * v_t[i] + (1.0f - beta2) * gradients[i] * gradients[i];
            float m_hat = m_t[i] / bc1;
            float v_hat = v_t[i] / bc2;
            weights[i] -= learningRate * m_hat / (std::sqrt(v_hat) + epsilon);
        }
    }
    
    void step() { timestep++; }
    void reset() override { timestep = 0; m.clear(); v.clear(); }
};

#endif // OPTIMIZER_H

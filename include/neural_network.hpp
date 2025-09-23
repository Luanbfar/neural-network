#pragma once

#include "layer.hpp"
#include <vector>
#include <memory>

using namespace layers;

namespace neural_network
{
    class NeuralNetwork
    {
    private:
        shared_ptr<InputLayer> inputLayer;
        vector<shared_ptr<HiddenLayer>> hiddenLayers;
        shared_ptr<OutputLayer> outputLayer;

        void createConnections();
        int calculateHiddenLayerSize(int inputSize, int outputSize) const;
        void updateLayer(shared_ptr<InputLayer> layer, float learningRate);
        void updateLayer(shared_ptr<HiddenLayer> layer, float learningRate);
        void updateLayer(shared_ptr<OutputLayer> layer, float learningRate);
        void applyGradients(float learningRate);

    public:
        NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount);

        NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount, int hiddenLayerSize);

        void resetNetwork();
        vector<float> forward(const vector<float> &inputs);
        float calculateLoss(const vector<float> &expected);
        void backpropagate(vector<float> &expected, float learningRate);
        void train(vector<vector<float>> &trainingData, int batchSize = 32);

        int getInputSize() const { return inputLayer ? inputLayer->getNodeCount() : 0; }
        int getOutputSize() const { return outputLayer ? outputLayer->getNodeCount() : 0; }
        int getHiddenLayerCount() const { return hiddenLayers.size(); }
    };
}
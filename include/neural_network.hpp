#pragma once

#include "layer.hpp"
#include <vector>
#include <memory>

using namespace layers;

namespace neural_network
{
    struct DeltaSnapshot
    {
        int epoch;
        int sample;
        vector<float> inputDeltas;
        vector<vector<float>> hiddenDeltas; // One vector per hidden layer
        vector<float> outputDeltas;
        vector<float> inputWeights; // Sample of weights
        vector<vector<float>> hiddenWeights;
        vector<float> inputBiases;
        vector<vector<float>> hiddenBiases;
        vector<float> outputBiases;
        float loss;
    };

    class NeuralNetwork
    {
    private:
        shared_ptr<InputLayer> inputLayer;
        vector<shared_ptr<HiddenLayer>> hiddenLayers;
        shared_ptr<OutputLayer> outputLayer;

        vector<DeltaSnapshot> deltaHistory;
        bool trackDeltas;
        int currentEpoch;
        int currentSample;

        void createConnections();
        int calculateHiddenLayerSize(int inputSize, int outputSize) const;
        void updateLayer(shared_ptr<InputLayer> layer, float learningRate);
        void updateLayer(shared_ptr<HiddenLayer> layer, float learningRate);
        void updateLayer(shared_ptr<OutputLayer> layer, float learningRate);
        void applyGradients(float learningRate);
        void backpropagate(vector<float> &expected, float learningRate);

        void captureDeltas(float loss);

    public:
        NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount);

        NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount, int hiddenLayerSize);

        void resetNetwork();
        vector<float> forward(const vector<float> &inputs);
        float calculateLoss(const vector<float> &expected);
        void train(vector<vector<float>> &trainingData, int batchSize = 32);

        void enableDeltaTracking() { trackDeltas = true; }
        void disableDeltaTracking() { trackDeltas = false; }
        void clearDeltaHistory() { deltaHistory.clear(); }
        void exportDeltasToCSV(const string &filename);
        void setEpoch(int epoch) { currentEpoch = epoch; }

        int getInputSize() const { return inputLayer ? inputLayer->getNodeCount() : 0; }
        int getOutputSize() const { return outputLayer ? outputLayer->getNodeCount() : 0; }
        int getHiddenLayerCount() const { return hiddenLayers.size(); }
    };
}
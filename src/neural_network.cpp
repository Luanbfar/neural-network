#include "neural_network.hpp"
#include <stdexcept>
#include <algorithm>
#include <math.h>
#include <fstream>

using namespace neural_network;
using namespace layers;

namespace neural_network
{
    NeuralNetwork::NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount)
    {
        if (inputSize <= 0 || outputSize <= 0 || hiddenLayerCount < 0)
        {
            throw invalid_argument("Invalid network dimensions");
        }

        inputLayer = make_shared<InputLayer>(inputSize);
        outputLayer = make_shared<OutputLayer>(outputSize);

        int hiddenSize = calculateHiddenLayerSize(inputSize, outputSize);
        hiddenLayers.reserve(hiddenLayerCount);

        for (int i = 0; i < hiddenLayerCount; ++i)
        {
            hiddenLayers.push_back(make_shared<HiddenLayer>(hiddenSize));
        }

        createConnections();

        trackDeltas = false;
        currentEpoch = 0;
        currentSample = 0;
    }

    NeuralNetwork::NeuralNetwork(int inputSize, int outputSize, int hiddenLayerCount, int hiddenLayerSize)
    {
        if (inputSize <= 0 || outputSize <= 0 || hiddenLayerCount < 0 || hiddenLayerSize <= 0)
        {
            throw invalid_argument("Invalid network dimensions");
        }

        inputLayer = make_shared<InputLayer>(inputSize);
        outputLayer = make_shared<OutputLayer>(outputSize);

        hiddenLayers.reserve(hiddenLayerCount);
        for (int i = 0; i < hiddenLayerCount; ++i)
        {
            hiddenLayers.push_back(make_shared<HiddenLayer>(hiddenLayerSize));
        }

        createConnections();

        trackDeltas = false;
        currentEpoch = 0;
        currentSample = 0;
    }

    void NeuralNetwork::createConnections()
    {
        if (!inputLayer || !outputLayer)
        {
            throw runtime_error("Input and output layers must be initialized");
        }

        if (hiddenLayers.empty())
        {
            inputLayer->attachLayer(outputLayer);
        }
        else
        {
            inputLayer->attachLayer(hiddenLayers[0]);

            for (size_t i = 0; i < hiddenLayers.size() - 1; ++i)
            {
                hiddenLayers[i]->attachLayer(hiddenLayers[i + 1]);
            }

            hiddenLayers.back()->attachLayer(outputLayer);
        }
    }

    int NeuralNetwork::calculateHiddenLayerSize(int inputSize, int outputSize) const
    {
        return max(1, (inputSize + outputSize) * 2 / 3);
    }

    void NeuralNetwork::updateLayer(shared_ptr<InputLayer> layer, float learningRate)
    {
        for (auto &edge : layer->edges)
        {
            edge->weight -= learningRate * edge->sourceNode->value * edge->targetNode->delta;
        }
        for (auto &node : layer->nodes)
        {
            node->bias -= learningRate * node->delta;
        }
    }

    void NeuralNetwork::updateLayer(shared_ptr<HiddenLayer> layer, float learningRate)
    {
        for (auto &edge : layer->edges)
        {
            edge->weight -= learningRate * edge->sourceNode->value * edge->targetNode->delta;
        }
        for (auto &node : layer->nodes)
        {
            node->bias -= learningRate * node->delta;
        }
    }

    void NeuralNetwork::updateLayer(shared_ptr<OutputLayer> layer, float learningRate)
    {
        for (auto &node : layer->nodes)
        {
            node->bias -= learningRate * node->delta;
        }
    }

    void NeuralNetwork::applyGradients(float learningRate)
    {
        this->updateLayer(this->inputLayer, learningRate);

        for (auto &hiddenLayer : this->hiddenLayers)
        {
            this->updateLayer(hiddenLayer, learningRate);
        }

        this->updateLayer(this->outputLayer, learningRate);
    }

    void NeuralNetwork::backpropagate(vector<float> &expected, float learningRate)
    {
        for (size_t i = 0; i < outputLayer->nodes.size(); ++i)
        {
            outputLayer->nodes[i]->delta = outputLayer->nodes[i]->value - expected[i];
        }

        for (size_t l = hiddenLayers.size(); l-- > 0;)
        {
            for (auto &node : hiddenLayers[l]->nodes)
            {
                float sum = 0.0f;
                for (auto &edge : hiddenLayers[l]->edges)
                {
                    if (edge->sourceNode == node)
                    {
                        sum += edge->weight * edge->targetNode->delta;
                    }
                }
                node->delta = sum * node->reluDerivative();
            }
        }

        for (auto &node : inputLayer->nodes)
        {
            float sum = 0.0f;
            for (auto &edge : inputLayer->edges)
            {
                if (edge->sourceNode == node)
                {
                    sum += edge->weight * edge->targetNode->delta;
                }
            }
            node->delta = sum;
        }

        this->applyGradients(learningRate);
    }

    void NeuralNetwork::captureDeltas(float loss)
    {
        if (!trackDeltas)
            return;

        DeltaSnapshot snapshot;
        snapshot.epoch = currentEpoch;
        snapshot.sample = currentSample;
        snapshot.loss = loss;

        // Capture input layer deltas
        for (const auto &node : inputLayer->nodes)
        {
            snapshot.inputDeltas.push_back(node->delta);
        }

        // Capture sample of input weights (first 5 edges)
        int weightSample = inputLayer->edges.size();
        for (int i = 0; i < weightSample; ++i)
        {
            snapshot.inputWeights.push_back(inputLayer->edges[i]->weight);
        }

        // Capture hidden layer deltas
        for (const auto &hiddenLayer : hiddenLayers)
        {
            vector<float> layerDeltas;
            vector<float> layerWeights;

            for (const auto &node : hiddenLayer->nodes)
            {
                layerDeltas.push_back(node->delta);
            }

            // Sample weights
            int hiddenWeightSample = hiddenLayer->edges.size();
            for (int i = 0; i < hiddenWeightSample; ++i)
            {
                layerWeights.push_back(hiddenLayer->edges[i]->weight);
            }

            snapshot.hiddenDeltas.push_back(layerDeltas);
            snapshot.hiddenWeights.push_back(layerWeights);
        }

        // Capture output layer deltas
        for (const auto &node : outputLayer->nodes)
        {
            snapshot.outputDeltas.push_back(node->delta);
        }

        deltaHistory.push_back(snapshot);
        currentSample++;
    }

    void NeuralNetwork::resetNetwork()
    {
        if (this->inputLayer)
            this->inputLayer->resetValues();

        for (auto &hiddenLayer : this->hiddenLayers)
        {
            if (hiddenLayer)
                hiddenLayer->resetValues();
        }

        if (this->outputLayer)
            this->outputLayer->resetValues();
    }

    vector<float> NeuralNetwork::forward(const vector<float> &inputs)
    {
        if (!this->inputLayer || !this->outputLayer)
        {
            throw runtime_error("Network not properly initialized");
        }

        this->resetNetwork();

        this->inputLayer->setInputValues(inputs);

        this->inputLayer->forward();

        for (auto &hiddenLayer : this->hiddenLayers)
        {
            hiddenLayer->forward();
        }

        return outputLayer->getOutput();
    }

    float NeuralNetwork::calculateLoss(const vector<float> &expected)
    {
        if (expected.size() != this->outputLayer->nodes.size())
        {
            throw invalid_argument("Expected output size doesn't match network output size");
        }

        float totalLoss = 0.0f;

        for (size_t i = 0; i < expected.size(); ++i)
        {
            float actual = this->outputLayer->nodes[i]->value;

            totalLoss -= expected[i] * log(actual) + (1.0f - expected[i]) * log(1.0f - actual);
        }

        return totalLoss / expected.size();
    }

    void NeuralNetwork::train(vector<vector<float>> &trainingData, int batchSize)
    {
        currentSample = 0;

        for (size_t i = 0; i < trainingData.size(); i += batchSize)
        {
            size_t end = min(i + batchSize, trainingData.size());

            for (size_t j = i; j < end; ++j)
            {
                auto &row = trainingData[j];
                vector<float> inputs(row.begin(), row.begin() + this->getInputSize());
                vector<float> targets(row.begin() + this->getInputSize(), row.end());

                this->forward(inputs);
                float loss = this->calculateLoss(targets);
                this->backpropagate(targets, 0.001f);

                // Capture deltas after backpropagation
                captureDeltas(loss);
            }
        }
    }

    void NeuralNetwork::exportDeltasToCSV(const string &filename)
    {
        ofstream file(filename);
        if (!file.is_open())
        {
            throw runtime_error("Could not open file for delta export: " + filename);
        }

        // Write header
        file << "epoch,sample,loss,";

        // Input deltas
        for (size_t i = 0; i < inputLayer->nodes.size(); ++i)
        {
            file << "input_delta_" << i << ",";
        }

        // Hidden deltas
        for (size_t layer = 0; layer < hiddenLayers.size(); ++layer)
        {
            for (size_t i = 0; i < hiddenLayers[layer]->nodes.size(); ++i)
            {
                file << "hidden" << layer << "_delta_" << i << ",";
            }
        }

        // Output deltas
        for (size_t i = 0; i < outputLayer->nodes.size(); ++i)
        {
            file << "output_delta_" << i << ",";
        }

        // Sample weights
        file << "input_weight_0,hidden0_weight_0";
        file << "\n";

        // Write data
        for (const auto &snapshot : deltaHistory)
        {
            // Write leading columns
            file << snapshot.epoch << "," << snapshot.sample << "," << snapshot.loss;

            // Input deltas (add a leading comma for the first item)
            for (float delta : snapshot.inputDeltas)
            {
                file << "," << delta;
            }

            // Hidden deltas (add a leading comma for the first item)
            for (const auto &layerDeltas : snapshot.hiddenDeltas)
            {
                for (float delta : layerDeltas)
                {
                    file << "," << delta;
                }
            }

            // Output deltas (add a leading comma for the first item)
            for (float delta : snapshot.outputDeltas)
            {
                file << "," << delta;
            }

            // Sample weights (add a leading comma for the first item)
            if (!snapshot.inputWeights.empty())
                file << "," << snapshot.inputWeights[0];
            else
                file << ",0"; // Note: Comma is now before the value

            // Second sample weight (add a leading comma for the second item)
            if (!snapshot.hiddenWeights.empty() && !snapshot.hiddenWeights[0].empty())
                file << "," << snapshot.hiddenWeights[0][0];
            else
                file << ",0"; // Note: Comma is now before the value

            file << "\n";
        }

        file.close();
        printf("Deltas exported to %s (%zu snapshots)\n", filename.c_str(), deltaHistory.size());
    }
}
#include "neural_network.hpp"
#include <stdexcept>
#include <algorithm>
#include <math.h>

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
        const float epsilon = 1e-15f;

        for (size_t i = 0; i < expected.size(); ++i)
        {
            float actual = this->outputLayer->nodes[i]->value;
            actual = max(epsilon, min(1.0f - epsilon, actual));

            totalLoss -= expected[i] * log(actual) + (1.0f - expected[i]) * log(1.0f - actual);
        }

        return totalLoss / expected.size();
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

    void NeuralNetwork::train(vector<vector<float>> &trainingData, int batchSize)
    {
        for (size_t i = 0; i < trainingData.size(); i += batchSize)
        {
            size_t end = min(i + batchSize, trainingData.size());

            for (size_t j = i; j < end; ++j)
            {
                auto &row = trainingData[j];
                vector<float> inputs(row.begin(), row.begin() + this->getInputSize());
                vector<float> targets(row.begin() + this->getInputSize(), row.end());

                this->forward(inputs);
                this->backpropagate(targets, 0.01f);
            }
        }
    }
}
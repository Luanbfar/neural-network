#include "layer.hpp"
#include <random>
#include <stdexcept>
#include <algorithm>

using namespace std;
using namespace nodes;

namespace layers
{
    float Layer::generateRandomNormalizedValues(const float range[2])
    {
        static random_device rd;
        static mt19937 gen(rd());
        uniform_real_distribution<float> dist(range[0], range[1]);
        return dist(gen);
    }

    void Layer::initializeNodes(int nodeCount)
    {
        this->nodes.clear();
        this->nodes.reserve(nodeCount);

        float biasRange[2] = {-0.5f, 0.5f};
        for (int i = 0; i < nodeCount; ++i)
        {
            float bias = this->generateRandomNormalizedValues(biasRange);
            shared_ptr<Node> node = make_shared<Node>(0.0f, bias);
            this->nodes.push_back(node);
        }
    }

    void Layer::resetValues()
    {
        for (const auto &node : this->nodes)
        {
            node->reset();
        }
    }

    InputLayer::InputLayer(int nodeCount)
    {
        if (nodeCount <= 0)
            throw invalid_argument("Node count must be positive");
        initializeNodes(nodeCount);
    }

    void InputLayer::initializeNodes(int nodeCount)
    {
        this->nodes.clear();
        this->nodes.reserve(nodeCount);

        for (int i = 0; i < nodeCount; i++)
        {
            shared_ptr<Node> node = make_shared<Node>(0.0f);
            this->nodes.push_back(node);
        }
    }

    void InputLayer::initializeEdges(shared_ptr<Layer> nextLayer)
    {
        if (!nextLayer || nextLayer->nodes.empty())
            throw invalid_argument("Cannot attach to null or empty layer");

        this->edges.clear();
        this->edges.reserve(nodes.size() * nextLayer->nodes.size());

        float weightRange[2] = {-0.5f, 0.5f};
        for (const auto &sourceNode : this->nodes)
        {
            for (const auto &targetNode : nextLayer->nodes)
            {
                float weight = generateRandomNormalizedValues(weightRange);
                this->edges.push_back(make_shared<Edge>(sourceNode, targetNode, weight));
            }
        }
    }

    void InputLayer::setInputValues(const vector<float> &values)
    {
        if (values.size() != this->nodes.size())
            throw invalid_argument("Input size doesn't match layer size");

        for (size_t i = 0; i < values.size(); ++i)
            this->nodes[i]->value = values[i];
    }

    void InputLayer::attachLayer(shared_ptr<Layer> nextLayer)
    {
        this->initializeEdges(nextLayer);
    }

    void InputLayer::forward()
    {
        for (const auto &edge : this->edges)
        {
            edge->targetNode->value += edge->sourceNode->value * edge->weight;
        }
    }

    HiddenLayer::HiddenLayer(int nodeCount)
    {
        if (nodeCount <= 0)
            throw invalid_argument("Node count must be positive");
        initializeNodes(nodeCount);
    }

    void HiddenLayer::initializeEdges(shared_ptr<Layer> nextLayer)
    {
        if (!nextLayer || nextLayer->nodes.empty())
            throw invalid_argument("Cannot attach to null or empty layer");

        this->edges.clear();
        this->edges.reserve(nodes.size() * nextLayer->nodes.size());

        float weightRange[2] = {-0.5f, 0.5f};
        for (const auto &sourceNode : this->nodes)
        {
            for (const auto &targetNode : nextLayer->nodes)
            {
                float weight = generateRandomNormalizedValues(weightRange);
                this->edges.push_back(make_shared<Edge>(sourceNode, targetNode, weight));
            }
        }
    }

    void HiddenLayer::attachLayer(shared_ptr<Layer> nextLayer)
    {
        this->initializeEdges(nextLayer);
    }

    void HiddenLayer::processNodes()
    {
        for (const auto &node : this->nodes)
        {
            node->addBias();
            node->relu();
        }
    }

    void HiddenLayer::forward()
    {
        this->processNodes();
        for (const auto &edge : this->edges)
        {
            edge->targetNode->value += edge->sourceNode->value * edge->weight;
        }
    }

    OutputLayer::OutputLayer(int nodeCount)
    {
        if (nodeCount <= 0)
            throw invalid_argument("Node count must be positive");
        initializeNodes(nodeCount);
    }

    void OutputLayer::processNodes()
    {
        for (const auto &node : this->nodes)
        {
            node->addBias();
            node->sigmoid();
        }
    }

    vector<float> OutputLayer::getOutput()
    {
        this->processNodes();
        vector<float> outputs;
        outputs.reserve(this->nodes.size());
        for (const auto &node : this->nodes)
            outputs.push_back(node->value);
        return outputs;
    }

}

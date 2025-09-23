#include "node.hpp"
#include <cmath>
#include <random>

using namespace nodes;
using namespace std;

namespace nodes
{
    Edge::Edge(shared_ptr<Node> sourceNode, shared_ptr<Node> targetNode, float weight)
    {
        this->sourceNode = sourceNode;
        this->targetNode = targetNode;
        this->weight = weight;
    }

    Node::Node()
    {
        this->value = 0.0f;
        this->bias = 0.0f;
        this->delta = 0.0f;
    }

    Node::Node(float value, float bias)
    {
        this->value = value;
        this->bias = bias;
        this->delta = 0.0f;
    }

    void Node::sigmoid()
    {
        this->value = 1.0f / (1.0f + exp(-this->value));
    }

    float Node::sigmoidDerivative()
    {
        float s = this->value;
        return s * (1 - s);
    }

    void Node::relu()
    {
        this->value = (this->value > 0.0f) ? this->value : 0.01f * this->value;
    }

    float Node::reluDerivative()
    {
        return (this->value > 0.0f) ? 1.0f : 0.01f;
    }

    void Node::addBias()
    {
        this->value += this->bias;
    }

    void Node::reset()
    {
        this->value = 0.0f;
    }
}
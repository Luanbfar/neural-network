#pragma once

#include <memory>
#include <vector>

using namespace std;

namespace nodes
{
    class Node;
    class Edge;

    class Edge
    {
    public:
        shared_ptr<Node> sourceNode;
        shared_ptr<Node> targetNode;
        float weight;
        int paramIndex;

        Edge(shared_ptr<Node> sourceNode, shared_ptr<Node> targetNode, float weight);
    };

    class Node
    {
    public:
        float value;
        float bias;
        float delta;
        int biasIndex;

        Node();
        Node(float value, float bias = 0.0f);

        void sigmoid();
        float sigmoidDerivative();
        void relu();
        float reluDerivative();
        void addBias();
        void reset();
    };
}
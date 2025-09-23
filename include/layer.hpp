#pragma once

#include <vector>
#include <memory>
#include "node.hpp"

using namespace std;
using namespace nodes;

namespace layers
{
    class Layer;
    class InputLayer;
    class HiddenLayer;
    class OutputLayer;

    class Layer
    {
    protected:
        virtual void initializeNodes(int nodeCount);
        float generateRandomNormalizedValues(const float range[2]);

    public:
        vector<shared_ptr<Node>> nodes;

        virtual ~Layer() = default;
        Layer() = default;

        void resetValues();

        size_t getNodeCount() const { return nodes.size(); }
    };

    class InputLayer : public Layer
    {
    private:
        void initializeNodes(int nodeCount) override;
        void initializeEdges(shared_ptr<Layer> nextLayer);

    public:
        vector<shared_ptr<Edge>> edges;

        InputLayer(int nodeCount);

        void setInputValues(const vector<float> &values);
        void attachLayer(shared_ptr<Layer> nextLayer);
        void forward();
    };

    class HiddenLayer : public Layer
    {
    private:
        void initializeEdges(shared_ptr<Layer> nextLayer);

    public:
        vector<shared_ptr<Edge>> edges;

        HiddenLayer(int nodeCount);

        void attachLayer(shared_ptr<Layer> nextLayer);
        void processNodes();
        void forward();
    };

    class OutputLayer : public Layer
    {
    public:
        OutputLayer(int nodeCount);

        void processNodes();
        vector<float> getOutput();
    };
}
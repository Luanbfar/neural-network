#include "normalizer.hpp"
#include <stdexcept>
#include <cstdio>
#include <sstream>
#include <array>

using namespace std;

Normalizer::Normalizer(const vector<float> maxValues)
{
    this->maxValues = maxValues;
}

vector<float> Normalizer::normalize(vector<float> &features) const
{
    vector<float> normalized(features.size());

    for (size_t i = 0; i < features.size(); ++i)
    {
        float actual = features[i];
        float maxVal = maxValues[i];

        float normalizedValue = min(max(actual / maxVal, 0.0f), 1.0f);

        normalized[i] = normalizedValue;
    }
    return normalized;
}

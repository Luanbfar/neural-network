#pragma once
#include <string>
#include <vector>

using namespace std;

class Normalizer
{
private:
    vector<float> maxValues;

public:
    Normalizer(const vector<float> maxValues);

    vector<float> normalize(vector<float> &features) const;
};

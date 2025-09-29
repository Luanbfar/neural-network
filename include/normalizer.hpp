#pragma once
#include <string>
#include <vector>

using namespace std;

class Normalizer
{
public:
    explicit Normalizer(const string &scriptPath);

    vector<float> normalize(float age, float weight, float height) const;

private:
    string scriptPath; // Armazena o caminho para o script
};

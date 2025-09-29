#include "normalizer.hpp"
#include <stdexcept>
#include <cstdio>
#include <sstream>
#include <array>

using namespace std;

Normalizer::Normalizer(const string &scriptPath)
{
    this->scriptPath = scriptPath;
}

vector<float> Normalizer::normalize(float age, float weight, float height) const
{
    string command = "python3 " + this->scriptPath + " --normalize " +
                     to_string(age) + " " +
                     to_string(weight) + " " +
                     to_string(height);

    FILE *pipe = popen(command.c_str(), "r");
    if (!pipe)
    {
        throw runtime_error("popen() falhou ao tentar executar o script: " + this->scriptPath);
    }

    array<char, 128> buffer;
    string result_str = "";
    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr)
    {
        result_str += buffer.data();
    }

    int exit_code = pclose(pipe);
    if (exit_code != 0)
    {
        throw runtime_error("O script Python terminou com um código de erro.");
    }

    if (result_str.empty())
    {
        throw runtime_error("O script Python não retornou nenhuma saída.");
    }

    vector<float> values;
    stringstream ss(result_str);
    string item;
    while (getline(ss, item, ','))
    {
        values.push_back(stof(item));
    }

    if (values.size() != 3)
    {
        throw runtime_error("A saída do script de normalização está em um formato inesperado.");
    }

    return values;
}
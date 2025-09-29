/**
 * @file main.cpp
 * @brief Implementação principal da rede neural para predição de risco cardiovascular
 *
 * Este arquivo contém a implementação completa de treinamento e inferência
 * de uma rede neural para predizer probabilidade de doença cardiovascular
 * baseada em dados normalizados de idade, peso e altura.
 *
 * Características da rede:
 * - 3 neurônios de entrada (idade, peso, altura normalizados)
 * - 2 camadas ocultas com ativação ReLU
 * - 1 neurônio de saída com ativação sigmoid (probabilidade CVD)
 * - Função de perda: entropia cruzada binária
 * - Otimizador: gradiente descendente com taxa de aprendizado fixa
 */

#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include "neural_network.hpp"
#include "normalizer.hpp"

using namespace std;
using namespace neural_network;

/**
 * @brief Lê dados de um arquivo CSV e retorna uma matriz de valores float
 *
 * Formato esperado do CSV:
 * - Primeira linha: cabeçalho (ignorado)
 * - Colunas: age_norm,weight_norm,height_norm,cvd_prob
 * - Valores normalizados entre 0 e 1
 *
 * @param filename Caminho para o arquivo CSV
 * @return vector<vector<float>> Matriz com os dados lidos, cada linha representa uma amostra
 */
vector<vector<float>> lerCSV(const string &filename)
{
    vector<vector<float>> dados;
    ifstream arquivo(filename);

    if (!arquivo.is_open())
    {
        printf("Erro: Não foi possível abrir o arquivo %s\n", filename.c_str());
        return dados;
    }

    string linha;

    // Pula a linha do cabeçalho
    getline(arquivo, linha);

    // Lê as linhas de dados
    while (getline(arquivo, linha))
    {
        vector<float> linha_dados;
        stringstream ss(linha);
        string celula;

        // Separa cada valor por vírgula
        while (getline(ss, celula, ','))
        {
            linha_dados.push_back(stof(celula));
        }

        // Só adiciona linhas com 4 colunas (3 entradas + 1 saída)
        if (linha_dados.size() == 4)
        {
            dados.push_back(linha_dados);
        }
    }

    arquivo.close();
    printf("Carregadas %zu amostras de %s\n", dados.size(), filename.c_str());

    return dados;
}

/**
 * @brief Avalia a performance da rede neural em um conjunto de dados
 *
 * Calcula métricas de avaliação:
 * - Perda média (entropia cruzada binária)
 * - Erro absoluto médio
 * - Número de predições dentro de uma margem de erro aceitável
 *
 * @param rede Ponteiro para a rede neural treinada
 * @param dados Conjunto de dados para avaliação
 * @param nome_conjunto Nome do conjunto (ex: "teste", "validação")
 */
void avaliarRede(NeuralNetwork *rede, const vector<vector<float>> &dados, const string &nome_conjunto)
{
    if (dados.empty())
    {
        printf("Conjunto %s vazio, pulando avaliação.\n", nome_conjunto.c_str());
        return;
    }

    float perda_total = 0.0f;
    float erro_absoluto_total = 0.0f;
    int predicoes_proximas = 0;
    const float margem_erro = 0.1f; // 10% de margem de erro aceitável

    printf("\nAvaliando conjunto %s (%zu amostras)...\n", nome_conjunto.c_str(), dados.size());

    for (const auto &amostra : dados)
    {
        // Separa entradas (3 primeiros valores) e alvo (último valor)
        vector<float> entradas(amostra.begin(), amostra.begin() + 3);
        vector<float> alvo = {amostra.back()};

        // Inferência
        vector<float> saida = rede->forward(entradas);

        // Calcula métricas
        float perda = rede->calculateLoss(alvo);
        float erro_abs = abs(saida[0] - alvo[0]);

        perda_total += perda;
        erro_absoluto_total += erro_abs;

        if (erro_abs <= margem_erro)
            predicoes_proximas++;
    }

    // Calcula médias
    float perda_media = perda_total / dados.size();
    float erro_medio = erro_absoluto_total / dados.size();
    float precisao = (float)predicoes_proximas / dados.size() * 100.0f;

    printf("Resultados %s:\n", nome_conjunto.c_str());
    printf("  Perda média: %.4f\n", perda_media);
    printf("  Erro absoluto médio: %.4f\n", erro_medio);
    printf("  Predições dentro de %.0f%%: %.1f%% (%d/%zu)\n",
           margem_erro * 100, precisao, predicoes_proximas, dados.size());
}

/**
 * @brief Demonstra predições individuais da rede neural
 *
 * Mostra exemplos específicos de predições para verificar
 * o comportamento da rede em diferentes tipos de entrada
 *
 * @param rede Ponteiro para a rede neural treinada
 * @param dados_teste Conjunto de dados para demonstração
 * @param num_exemplos Número de exemplos para mostrar (máximo 10)
 */
void demonstrarPredicoes(NeuralNetwork *rede, const vector<vector<float>> &dados_teste, int num_exemplos = 5)
{
    printf("\n--- Exemplos de Predições ---\n");
    printf("Formato: [idade_norm, peso_norm, altura_norm] -> Predito vs Real\n");

    int exemplos_mostrados = min(num_exemplos, (int)dados_teste.size());

    for (int i = 0; i < exemplos_mostrados; ++i)
    {
        const auto &amostra = dados_teste[i];
        vector<float> entradas(amostra.begin(), amostra.begin() + 3);
        float alvo = amostra.back();

        vector<float> saida = rede->forward(entradas);

        printf("Exemplo %d: [%.3f, %.3f, %.3f] -> %.4f vs %.4f (erro: %.4f)\n",
               i + 1, entradas[0], entradas[1], entradas[2],
               saida[0], alvo, abs(saida[0] - alvo));
    }
}

/**
 * @brief Função principal - implementa treinamento e avaliação completos
 *
 * Fluxo de execução:
 * 1. Carrega dados de treinamento, teste e validação
 * 2. Cria e inicializa a rede neural
 * 3. Executa treinamento com monitoramento de progresso
 * 4. Avalia performance nos conjuntos de teste e validação
 * 5. Demonstra predições individuais
 *
 * @return int Código de saída (0 = sucesso, 1 = erro)
 */
int main()
{
    printf("=== Sistema de Predição de Risco Cardiovascular ===\n");
    printf("Rede Neural em C++ - Treinamento e Inferência\n\n");

    // =====================================
    // 1. CARREGAMENTO DOS DADOS
    // =====================================
    printf("Carregando conjuntos de dados...\n");

    vector<vector<float>> dados_treinamento = lerCSV("data/training_data.csv");
    vector<vector<float>> dados_teste = lerCSV("data/test_data.csv");
    vector<vector<float>> dados_validacao = lerCSV("data/validation_data.csv");

    if (dados_treinamento.empty())
    {
        printf("Erro: Não foi possível carregar dados de treinamento.\n");
        printf("Certifique-se de que o arquivo data/training_data.csv existe.\n");
        return 1;
    }

    printf("\nResumo dos dados:\n");
    printf("  Treinamento: %zu amostras\n", dados_treinamento.size());
    printf("  Teste: %zu amostras\n", dados_teste.size());
    printf("  Validação: %zu amostras\n", dados_validacao.size());

    // =====================================
    // 2. CRIAÇÃO DA REDE NEURAL
    // =====================================
    printf("\nCriando rede neural...\n");

    Normalizer *normalizer = new Normalizer("process_data.py");

    // Arquitetura: 3 entradas -> 2 camadas ocultas de 8 neurônios -> 1 saída
    const int entradas = 3; // idade, peso, altura (normalizados)
    const int saidas = 1;   // probabilidade CVD
    const int camadas_ocultas = 4;
    const int neuronios_por_camada = 4;

    NeuralNetwork *rede = new NeuralNetwork(entradas, saidas, camadas_ocultas, neuronios_por_camada);

    printf("Arquitetura da rede:\n");
    printf("  Entradas: %d neurônios\n", rede->getInputSize());
    printf("  Camadas ocultas: %d (com %d neurônios cada)\n", camadas_ocultas, neuronios_por_camada);
    printf("  Saídas: %d neurônio\n", rede->getOutputSize());

    // =====================================
    // 3. TREINAMENTO DA REDE
    // =====================================
    printf("\nIniciando treinamento...\n");

    const int epocas = 1000;
    const int tamanho_lote = 32;
    const int intervalo_relatorio = 100;

    auto inicio_treinamento = chrono::high_resolution_clock::now();

    for (int epoca = 0; epoca < epocas; ++epoca)
    {
        // Treina uma época
        rede->train(dados_treinamento, tamanho_lote);

        // Relatório de progresso
        if (epoca % intervalo_relatorio == 0)
        {
            // Calcula perda média em uma amostra do conjunto de treinamento
            float perda_media = 0.0f;
            int amostras_teste = min(100, (int)dados_treinamento.size());

            for (int i = 0; i < amostras_teste; ++i)
            {
                const auto &amostra = dados_treinamento[i];
                vector<float> entradas(amostra.begin(), amostra.begin() + 3);
                vector<float> alvo = {amostra.back()};

                rede->forward(entradas);
                perda_media += rede->calculateLoss(alvo);
            }
            perda_media /= amostras_teste;

            printf("Época %d/%d: Perda média = %.4f\n", epoca, epocas, perda_media);
        }
    }

    auto fim_treinamento = chrono::high_resolution_clock::now();
    auto duracao = chrono::duration_cast<chrono::milliseconds>(fim_treinamento - inicio_treinamento);

    printf("Treinamento concluído em %.2f segundos.\n", duracao.count() / 1000.0);

    // =====================================
    // 4. AVALIAÇÃO DA REDE
    // =====================================
    printf("\n=== AVALIAÇÃO DE PERFORMANCE ===\n");

    // Avalia nos diferentes conjuntos de dados
    avaliarRede(rede, dados_teste, "teste");
    avaliarRede(rede, dados_validacao, "validação");

    // =====================================
    // 5. DEMONSTRAÇÃO DE PREDIÇÕES
    // =====================================
    if (!dados_teste.empty())
    {
        demonstrarPredicoes(rede, dados_teste, 8);
    }

    // =====================================
    // 6. TESTE COM DADOS PERSONALIZADOS
    // =====================================
    printf("\n--- Teste com Dados Personalizados ---\n");

    // Exemplo: pessoa jovem, peso normal, altura média
    vector<float> exemplo1 = {25.0f, 70.0f, 175.0f}; // 25 anos, 70kg, 175cm
    vector<float> exemplo1Normalizado = normalizer->normalize(exemplo1[0], exemplo1[1], exemplo1[2]);
    vector<float> resultado1 = rede->forward(exemplo1Normalizado);
    printf("Pessoa jovem (25a, 70kg, 175cm): Risco CVD = %.2f%%\n", resultado1[0] * 100);

    // Exemplo: pessoa mais velha, sobrepeso, altura média
    vector<float> exemplo2 = {60.0f, 90.0f, 170.0f}; // 60 anos, 90kg, 170cm
    vector<float> exemplo2Normalizado = normalizer->normalize(exemplo2[0], exemplo2[1], exemplo2[2]);
    vector<float> resultado2 = rede->forward(exemplo2Normalizado);
    printf("Pessoa mais velha (60a, 90kg, 170cm): Risco CVD = %.2f%%\n", resultado2[0] * 100);

    // Exemplo: pai
    vector<float> exemplo3 = {57.0f, 79.0f, 170.0f}; // 57 anos, 79kg, 170cm
    vector<float> exemplo3Normalizado = normalizer->normalize(exemplo3[0], exemplo3[1], exemplo3[2]);
    vector<float> resultado3 = rede->forward(exemplo3Normalizado);
    printf("Pessoa (57a, 79kg, 170cm): Risco CVD = %.2f%%\n", resultado3[0] * 100);

    // Exemplo: mãe
    vector<float> exemplo4 = {50.0f, 58.0f, 159.0f}; // 50 anos, 58kg, 159cm
    vector<float> exemplo4Normalizado = normalizer->normalize(exemplo4[0], exemplo4[1], exemplo4[2]);
    vector<float> resultado4 = rede->forward(exemplo4Normalizado);
    printf("Pessoa (50a, 58kg, 159cm): Risco CVD = %.2f%%\n", resultado4[0] * 100);

    // =====================================
    // 7. LIMPEZA E FINALIZAÇÃO
    // =====================================
    delete rede;

    printf("\n=== Execução Finalizada com Sucesso ===\n");
    printf("A rede neural foi treinada e avaliada.\n");
    printf("Para melhorar a performance, considere:\n");
    printf("  - Aumentar o número de épocas de treinamento\n");
    printf("  - Ajustar a taxa de aprendizado\n");
    printf("  - Modificar a arquitetura da rede\n");
    printf("  - Usar técnicas de regularização\n");

    return 0;
}
#!/usr/bin/env python3
"""
Visualização de Deltas da Rede Neural

Este script analisa e plota os valores de delta durante o treinamento
para identificar problemas como vanishing gradients, exploding gradients,
ou saturação da rede.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_delta_distribution(df, output_file='delta_distribution.png'):
    """Plota a distribuição dos deltas por camada"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Distribuição de Deltas por Camada', fontsize=16)
    
    # Input deltas
    input_cols = [col for col in df.columns if col.startswith('input_delta')]
    if input_cols:
        input_deltas = df[input_cols].values.flatten()
        axes[0, 0].hist(input_deltas, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Input Layer Deltas')
        axes[0, 0].set_xlabel('Delta Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    
    # Hidden layer 0 deltas
    hidden0_cols = [col for col in df.columns if col.startswith('hidden0_delta')]
    if hidden0_cols:
        hidden0_deltas = df[hidden0_cols].values.flatten()
        axes[0, 1].hist(hidden0_deltas, bins=50, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 1].set_title('Hidden Layer 0 Deltas')
        axes[0, 1].set_xlabel('Delta Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=1)
    
    # Hidden layer 1 deltas
    hidden1_cols = [col for col in df.columns if col.startswith('hidden1_delta')]
    if hidden1_cols:
        hidden1_deltas = df[hidden1_cols].values.flatten()
        axes[1, 0].hist(hidden1_deltas, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 0].set_title('Hidden Layer 1 Deltas')
        axes[1, 0].set_xlabel('Delta Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    
    # Output deltas
    output_cols = [col for col in df.columns if col.startswith('output_delta')]
    if output_cols:
        output_deltas = df[output_cols].values.flatten()
        axes[1, 1].hist(output_deltas, bins=50, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].set_title('Output Layer Deltas')
        axes[1, 1].set_xlabel('Delta Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfico de distribuição salvo em: {output_file}")
    plt.close()


def plot_delta_evolution(df, output_file='delta_evolution.png'):
    """Plota como os deltas evoluem ao longo do treinamento"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle('Evolução dos Deltas Durante o Treinamento', fontsize=16)
    
    # Média dos deltas absolutos por epoch
    epochs = df['epoch'].unique()
    
    input_cols = [col for col in df.columns if col.startswith('input_delta')]
    hidden0_cols = [col for col in df.columns if col.startswith('hidden0_delta')]
    hidden1_cols = [col for col in df.columns if col.startswith('hidden1_delta')]
    output_cols = [col for col in df.columns if col.startswith('output_delta')]
    
    avg_input = []
    avg_hidden0 = []
    avg_hidden1 = []
    avg_output = []
    
    for epoch in epochs:
        epoch_data = df[df['epoch'] == epoch]
        
        if input_cols:
            avg_input.append(np.abs(epoch_data[input_cols].values).mean())
        if hidden0_cols:
            avg_hidden0.append(np.abs(epoch_data[hidden0_cols].values).mean())
        if hidden1_cols:
            avg_hidden1.append(np.abs(epoch_data[hidden1_cols].values).mean())
        if output_cols:
            avg_output.append(np.abs(epoch_data[output_cols].values).mean())
    
    # Plot 1: Magnitude média dos deltas
    if avg_input:
        axes[0].plot(epochs, avg_input, label='Input Layer', marker='o', markersize=3)
    if avg_hidden0:
        axes[0].plot(epochs, avg_hidden0, label='Hidden Layer 0', marker='s', markersize=3)
    if avg_hidden1:
        axes[0].plot(epochs, avg_hidden1, label='Hidden Layer 1', marker='^', markersize=3)
    if avg_output:
        axes[0].plot(epochs, avg_output, label='Output Layer', marker='d', markersize=3)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Magnitude Média de |Delta|')
    axes[0].set_title('Magnitude Média dos Deltas por Camada')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Plot 2: Loss over time
    avg_loss = [df[df['epoch'] == epoch]['loss'].mean() for epoch in epochs]
    axes[1].plot(epochs, avg_loss, color='red', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss Médio')
    axes[1].set_title('Evolução da Loss')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Ratio between layers (gradient flow)
    if avg_hidden0 and avg_output:
        ratio = np.array(avg_hidden0) / (np.array(avg_output) + 1e-10)
        axes[2].plot(epochs, ratio, color='purple', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Ratio Hidden0/Output')
        axes[2].set_title('Fluxo de Gradiente (Ratio entre Camadas)')
        axes[2].axhline(1.0, color='red', linestyle='--', label='Ideal (ratio=1)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfico de evolução salvo em: {output_file}")
    plt.close()


def plot_weight_evolution(df, output_file='weight_evolution.png'):
    """Plota como os pesos evoluem"""
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    fig.suptitle('Evolução dos Pesos', fontsize=16)
    
    epochs = df['epoch'].unique()
    
    # Input weights
    input_weight_avg = [df[df['epoch'] == e]['input_weight_0'].mean() for e in epochs]
    axes[0].plot(epochs, input_weight_avg, marker='o', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Peso Médio')
    axes[0].set_title('Input Layer Weight Sample')
    axes[0].grid(True, alpha=0.3)
    
    # Hidden weights
    hidden_weight_avg = [df[df['epoch'] == e]['hidden0_weight_0'].mean() for e in epochs]
    axes[1].plot(epochs, hidden_weight_avg, marker='s', markersize=3, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Peso Médio')
    axes[1].set_title('Hidden Layer 0 Weight Sample')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Gráfico de pesos salvo em: {output_file}")
    plt.close()


def print_statistics(df):
    """Imprime estatísticas sobre os deltas"""
    print("\n" + "="*60)
    print("ESTATÍSTICAS DOS DELTAS")
    print("="*60)
    
    delta_cols = [col for col in df.columns if 'delta' in col]
    
    for col in delta_cols:
        values = df[col].values
        print(f"\n{col}:")
        print(f"  Média: {values.mean():.6f}")
        print(f"  Desvio padrão: {values.std():.6f}")
        print(f"  Mín: {values.min():.6f}")
        print(f"  Máx: {values.max():.6f}")
        print(f"  Zeros: {(np.abs(values) < 1e-10).sum()} ({(np.abs(values) < 1e-10).sum()/len(values)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Visualizar deltas da rede neural')
    parser.add_argument('--input', default='deltas.csv', help='Arquivo CSV com os deltas')
    parser.add_argument('--output-dir', default='.', help='Diretório para salvar os gráficos')
    args = parser.parse_args()
    
    # Carrega dados
    print(f"Carregando dados de: {args.input}")
    df = pd.read_csv(args.input)
    
    print(f"Total de amostras: {len(df)}")
    print(f"Épocas: {df['epoch'].min()} - {df['epoch'].max()}")
    
    # Gera visualizações
    print("\nGerando visualizações...")
    
    plot_delta_distribution(df, f'{args.output_dir}/delta_distribution.png')
    plot_delta_evolution(df, f'{args.output_dir}/delta_evolution.png')
    plot_weight_evolution(df, f'{args.output_dir}/weight_evolution.png')
    
    # Imprime estatísticas
    print_statistics(df)
    
    print("\n" + "="*60)
    print("ANÁLISE COMPLETA")
    print("="*60)
    print("\nVerifique os gráficos para identificar:")
    print("  1. Vanishing gradients: deltas muito próximos de zero")
    print("  2. Exploding gradients: deltas muito grandes")
    print("  3. Saturação: deltas não mudam ao longo do tempo")
    print("  4. Dead neurons: deltas sempre zero")
    print("  5. Gradient flow issues: ratio entre camadas muito diferente de 1")


if __name__ == "__main__":
    main()
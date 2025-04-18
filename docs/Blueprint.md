# Análise e Representação Topológica de Sons Fisiológicos com Classificação Neural e Mineração de Dados

---

## Visão Geral

Este documento consolida a arquitetura conceitual e técnica do sistema computacional para análise, visualização e classificação de sons fisiológicos, baseado na geração de curvas topológicas (windings) extraídas das componentes senoidais das frequências dominantes das ondas sonoras.



## Objetivo Geral

Desenvolver um sistema capaz de:

- Captar sons fisiológicos reais (pulmão, coração, intestino);
- Gerar windings tridimensionais padronizadas como representação visual;
- Classificar o tipo e o estado do som com base em vetores extraídos das windings;
- Traduzir os sons em experiências sensoriais;
- Aplicar data mining para descoberta de padrões e suporte à análise.



## Pipeline Geral

```
[Som captado]
      ↓
[Pré-processamento e extração de features (FFT, MFCC, etc.)]
      ↓
[Formação da winding 3D]
      ↓
[Extração de descritores topológicos]
      ↓
[Classificação via MLP (tipo e estado do som)]
      ↓
[Mineração de dados: clusterização, árvores, correlações]
      ↓
[Saída sensorial: forma 3D padronizada + cor + vibração]
```

## Processamento de Sinais: Técnicas Escolhidas

| Técnica | Função | Justificativa |
|--------|--------|----------------|
| **FFT / STFT** | Análise espectral | Essencial para obter as componentes senoidais e gerar as windings (base do sistema). Pode ser em tempo real com STFT. |
| **MFCC** | Extração perceptual | Muito útil como feature complementar para alimentar o MLP e os modelos de data mining. Especialmente relevante para sons com textura, como intestinais ou respiratórios. |
| **Filtros Binomiais / Kalman** | Suavização e previsão | Úteis para suavizar as curvas das windings ou o sinal antes da FFT. Mantém a limpeza e estabilidade visual no modo real-time. |
| **Wavelets** | Análise multiescalar | Detecta padrões localizados e transientes |



Essas técnicas alimentam os vetores de entrada que serão usados em algoritmos de data science e redes neurais.



## Data Science

O módulo de Data Science é responsável por organizar os dados extraídos dos sons fisiológicos, aplicar técnicas de mineração e descobrir padrões relevantes que ajudem no diagnóstico e na visualização topológica.

### Etapas Principais
O modelo de classificação principal escolhido foi o Random Forest, devido à sua capacidade interpretativa e robustez. Contudo, também é interessante explorar a aplicação de técnicas de clustering (k-Means e DBSCAN), com o intuito de investigar padrões emergentes não supervisionados nos dados extraídos das windings.

#### Modelo Principal: Random Forest

- Justifica bem a classificação supervisionada
- Permite ver importância das variáveis
- Serve como base para explicar o comportamento dos dados

#### Modelo Complementar: k-Means ou DBSCAN

- Usado antes ou depois da classificação
- Útil para detectar padrões sem supervisão
- Validar a existência de grupos naturais
- Explorar estrutura oculta nos dados



## Redes Neurais 

Utilizaremos **redes neurais clássicas MLP** para **reconhecer padrões, prever formas e traduzir representações sonoras em respostas sensoriais.**


#### Modelo Principal: MLP Clássico
- Entrada: vetores de features das windings;
- Saída: tipo ou estado do som.

#### Modelo Alternativo (exploratório): MLP com outra configuração
- Comparar MLP raso com MLP com camada intermediária
- MLP om diferentes funções  de ativação
- Regularização L2 ou variação de `learning_rate`

## Comparação Entre Modelos

|Modelo | Tipo | Finalidade | Métrica Principal
--------|-------|-----------|------------------
Random Forest | Supervisionado | Classificação + interpretação | Acurácia, Importância de variáveis
K-Means | Não Supervisionado |  Descoberta de padrões | Silhouette Score
DBSCAN | Não Supervisionado |  Densidade e outliers | Cluster labeling, ruído detectado
MLP (NN) | Supervisionado | Classificação final | Acurácia, F1-score

## Bases de Dados Utilizadas

- [HLS-CMDS: Heart and Lung Sounds Dataset Recorded from a Clinical Manikin using Digital Stethoscope](https://data.mendeley.com/datasets/8972jxbpmp/2)
- [ICBHI 2017 Challenge - Respiratory Sound Database](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
- [Bowel Sounds Signal](https://figshare.com/articles/media/Bowel_sounds_signal/28595741)


Módulo | Objetivo | Tarefas | Entregável | Observações
-------|----------|---------|-------------|-----------
Índice Topológico Sintético (ITS) | Criar vetor de descritores morfológicos a partir das windings | - Definir métricas (curvatura, torção, cruzamentos, entropia, etc.)- Calcular vetores para cada winding- Normalizar e salvar vetores como .csv | its_vectors.csv + módulo its_extractor.py | Base para o MLP e análise de importância de variáveis
Embedding em Espaço de Fase (Takens) | Gerar curvas no espaço de fase para análise de ciclos e atratores | - Definir τ e dimensão (d)- Construir vetores do tipo [x(t), x(t+τ), x(t+2τ)]- Visualizar curvas e extrair features | takens_embedding.py + vetores .csv | Complementar às windings baseadas em FFT/Wavelet
Topo-Sequência (Assinatura Temporal Topológica) | Capturar a evolução das formas ao longo do tempo | - Dividir sinal em janelas móveis- Gerar uma winding por janela- Extrair ITS por janela- Concatenar vetores para cada som | topo_sequence.csv + módulo topo_sequence.py | Permite detectar transições morfológicas (ex: crises, eventos)
Mapeamento Sensorial Adaptativo | Traduzir propriedades morfológicas em visualização e resposta sensorial | - Definir regras de mapeamento (ex: torção → vibração forte)- Implementar visualização em VisPy- Aplicar coloração dinâmica com base em intensidade/entropia | visualizer.py atualizado + mapper_rules.json | Mostra em tempo real a interpretação simbólica dos sinais
Vetor Híbrido (Winding + MFCC + Wavelet) | Combinar descritores topológicos e espectrais | - Extrair MFCCs e coeficientes Wavelet- Concatenar com ITS- Treinar modelos com vetor combinado | hybrid_vector.csv + MLP e RF treinados com esses dados | Ideal para testes comparativos de desempenho e análise de importância de features


## Outputs Esperados

- Visualizações sensoriais interativas (3D, forma, cor)
- Tipo de som + classificação/estado

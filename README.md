# PhysioSound-Miner

[Português](#portugues) | [English](#english)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#)

Um framework robusto para extração de características multidomínio e classificação de sons fisiológicos, com foco em fonocardiogramas.

---

<a name="portugues"></a>

## PhysioSound-Miner

### Visão Geral

O **PhysioSound-Miner** é um projeto de pesquisa e desenvolvimento focado na análise computacional de sons fisiológicos, com aplicação principal na detecção de sopros cardíacos a partir de fonocardiogramas (FCG). A metodologia integra técnicas clássicas de processamento de sinais com abordagens inovadoras de análise topológica e dinâmica para criar um vetor de características rico e discriminativo.

Este repositório contém o código-fonte para o pré-processamento de sinais, extração de características, treinamento de modelos de *machine learning* e interpretabilidade (XAI), conforme apresentado no artigo:

> **Classificação de Sons Cardíacos com Aprendizado Estatístico e Neural: Detecção de Sopros por Modelagem Morfológica, Topológica e Recorrente de Sinais de Ausculta.**

### Principais Funcionalidades

* **Extração Multidomínio:** Combina características dos domínios espectral, temporal, dinâmico e topológico.
* **Análise Topológica Inovadora:** Implementação do **Índice Topológico Sintético (ITS)** baseado em Curvas *Winding*, uma contribuição original para a caracterização de FCGs.
* **Análise de Recorrência (RQA):** Captura a dinâmica não-linear e a complexidade dos sinais.
* **Pipeline Modular:** Código organizado em módulos para pré-processamento, extração de características e utilitários, facilitando a reutilização e expansão.
* **Modelagem e Interpretabilidade:** Notebooks com pipelines completos para treinamento, validação rigorosa (`StratifiedGroupKFold`) e interpretabilidade de modelos com SHAP.
* **Reprodutibilidade:** Inclui arquivos de ambiente (`env.yml`, `requirements.txt`) para garantir a fácil replicação dos experimentos.

### Estrutura do Projeto

```
PHYSIOSOUND-MINER/
│
├── core/
│   ├── signal/
│   │   ├── features/             # Módulos de extração (its.py, mfcc.py, takens_rqa.py...)
│   │   └── signal_processing/    # Módulos de processamento (preprocessing.py...)
│   └── utils/                    # Utilitários (audio_io.py, plot_utils.py...)
│
├── data/
│   └── datasets/                 # Datasets brutos e processados
│
├── models/                       # Modelos treinados e salvos
│   ├── meu_pipeline_completo.pkl
│   └── label_encoder.pkl
│
├── notebooks/                    # Notebooks para análise e modelagem
│   └── modelagem_normal_vs_sopro_Kalman.ipynb
│
├── env.yml                       # Arquivo de ambiente Conda
├── requirements.txt              # Lista de dependências pip
└── README.md 
```

### Instalação

Para configurar o ambiente e instalar as dependências, escolha uma das opções abaixo.

**Usando Conda (Recomendado)**

```bash
git clone [https://github.com/seu-usuario/physiosound-miner.git](https://github.com/seu-usuario/physiosound-miner.git)
cd physiosound-miner
```


#### Crie o ambiente Conda a partir do arquivo .yml
```
conda env create -f env.yml
```

#### Ative o ambiente
```
conda activate physiosound
```

### Como Usar

1. **Extração de Características**

O framework foi projetado para ser modular. Você pode importar e usar os módulos de extração de características em seus próprios scripts.

```
from core.signal.features import its, rqa
from core.utils import audio_io

fs, signal_data = audio_io.load_audio('caminho/para/seu/audio.wav')

its_features = its.calculate_its_features(signal_data, fs)
rqa_features = rqa.calculate_rqa_features(signal_data, m=3, tau=4, threshold=0.1)

print("Características ITS:", its_features)
```

2. **Treinamento e Avaliação de Modelos**

A análise completa, desde a exploração dos dados até a avaliação final dos modelos, está documentada em
`notebooks/base_vertical/modelagem_normal_vs_sopro_Kalman.ipynb`

Para executar o notebook, inicie o Jupyter Lab ou Jupyter Notebook a partir do ambiente ativado:

3. **Uso do Modelo Pré-treinado**

Um pipeline completo e treinado está disponível no diretório core/models/. Ele pode ser usado para fazer predições em novos dados de áudio (após a extração das características correspondentes).


### Como Citar

Se você utilizar este código ou a metodologia em sua pesquisa, por favor, cite nosso trabalho:

```
@inproceedings{Sette2025,
  author    = {Sette, Paloma F. L.},
  title     = {Classificação de Sons Cardíacos com Aprendizado Estatístico e Neural: Detecção de Sopros por Modelagem Morfológica, Topológica e Recorrente de Sinais de Ausculta},
  year      = {2025},
  howpublished = {\url{https://github.com/palomaflsette/physiosound-miner}},
  address   = {Rio de Janeiro, Brasil},
  publisher = {}
}
```

----

# PhysioSound-Miner

[Português](#portugues) | [English](#english)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#)

A robust framework for multi-domain feature extraction and classification of physiological sounds, focusing on phonocardiograms.

---

<a name="english"></a>

## PhysioSound-Miner

### Overview

**PhysioSound-Miner** is a research and development project focused on computational analysis of physiological sounds, with primary application in heart murmur detection from phonocardiograms (PCG). The methodology integrates classical signal processing techniques with innovative topological and dynamical analysis approaches to create a rich and discriminative feature vector.

This repository contains the source code for signal preprocessing, feature extraction, machine learning model training, and interpretability (XAI), as presented in the paper:

> **Heart Sound Classification with Statistical and Neural Learning: Murmur Detection through Morphological, Topological and Recurrent Modeling of Auscultation Signals.**

### Key Features

* **Multi-domain Extraction:** Combines features from spectral, temporal, dynamic, and topological domains.
* **Innovative Topological Analysis:** Implementation of the **Synthetic Topological Index (STI)** based on Winding Curves, an original contribution for PCG characterization.
* **Recurrence Analysis (RQA):** Captures non-linear dynamics and signal complexity.
* **Modular Pipeline:** Code organized into modules for preprocessing, feature extraction, and utilities, facilitating reuse and expansion.
* **Modeling and Interpretability:** Notebooks with complete pipelines for training, rigorous validation (`StratifiedGroupKFold`), and model interpretability with SHAP.
* **Reproducibility:** Includes environment files (`env.yml`, `requirements.txt`) to ensure easy experiment replication.

### Project Structure

```
PHYSIOSOUND-MINER/
│
├── core/
│   ├── signal/
│   │   ├── features/             # Extraction modules (its.py, mfcc.py, takens_rqa.py...)
│   │   └── signal_processing/    # Processing modules (preprocessing.py...)
│   └── utils/                    # Utilities (audio_io.py, plot_utils.py...)
│
├── data/
│   └── datasets/                 # Raw and processed datasets
│
├── models/                       # Trained and saved models
│   ├── meu_pipeline_completo.pkl
│   └── label_encoder.pkl
│
├── notebooks/                    # Analysis and modeling notebooks
│   └── modelagem_normal_vs_sopro_Kalman.ipynb
│
├── env.yml                       # Conda environment file
├── requirements.txt              # pip dependencies list
└── README.md 
```

### Installation

To set up the environment and install dependencies, choose one of the options below.

**Using Conda (Recommended)**

```bash
git clone https://github.com/your-username/physiosound-miner.git
cd physiosound-miner
```

#### Create the Conda environment from the .yml file

```
conda env create -f env.yml
```

#### Activate the environment

```
conda activate physiosound
```

### How to Use

1. **Feature Extraction**

The framework is designed to be modular. You can import and use the feature extraction modules in your own scripts.

```python
from core.signal.features import its, rqa
from core.utils import audio_io

fs, signal_data = audio_io.load_audio('path/to/your/audio.wav')

its_features = its.calculate_its_features(signal_data, fs)
rqa_features = rqa.calculate_rqa_features(signal_data, m=3, tau=4, threshold=0.1)

print("ITS Features:", its_features)
```

2. **Model Training and Evaluation**

The complete analysis, from data exploration to final model evaluation, is documented in
`notebooks/base_vertical/modelagem_normal_vs_sopro_Kalman.ipynb`

To run the notebook, start Jupyter Lab or Jupyter Notebook from the activated environment:

3. **Using the Pre-trained Model**

A complete and trained pipeline is available in the core/models/ directory. It can be used to make predictions on new audio data (after extracting the corresponding features).

### How to Cite

If you use this code or methodology in your research, please cite our work:

```bibtex
@inproceedings{Sette2025,
  author    = {Sette, Paloma F. L.},
  title     = {Heart Sound Classification with Statistical and Neural Learning: Murmur Detection through Morphological, Topological and Recurrent Modeling of Auscultation Signals},
  year      = {2025},
  howpublished = {\url{https://github.com/palomaflsette/physiosound-miner}},
  address   = {Rio de Janeiro, Brasil},
  publisher = {}
}
```

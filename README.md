# PhysioSound-Miner

[Português](#portugues) | [English](#english)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active-success.svg)](#)

Um framework robusto para extração de características multidomínio e classificação de sons fisiológicos, com foco em fonocardiogramas.

---

<a name="portugues"></a>

## PhysioSound-Miner (Português)

### Visão Geral

O **PhysioSound-Miner** é um projeto de pesquisa e desenvolvimento em Python focado na análise computacional de sons fisiológicos, com aplicação principal na detecção de sopros cardíacos a partir de fonocardiogramas (FCG). A metodologia integra técnicas clássicas de processamento de sinais com abordagens inovadoras de análise topológica e dinâmica para criar um vetor de características rico e discriminativo.

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
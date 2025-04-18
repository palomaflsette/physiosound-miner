# Pipeline de Pré-Processamento de Sinais Fisiológicos

Este documento descreve todas as etapas envolvidas no pré-processamento dos sinais fisiológicos para o projeto.

---

## 1. Aquisição do Som

- **Fonte:** microfone piezoelétrico ou arquivos `.wav`
- **Formato padrão:** 44.1 kHz, mono, 16-bit PCM
- **Duração típica por amostra:** entre 5 e 30 segundos

---

## 2. Pré-processamento Básico do Sinal

- **Normalização de amplitude**
  - Escala o sinal entre -1 e 1 para padronização

- **Remoção de ruído de fundo (opcional)**
  - Filtro passa-baixa ou band-pass (20–2000 Hz)

- **Suavização com filtro Kalman ou Binomial**
  - Reduz ruído de alta frequência sem distorcer forma geral do sinal

---

## 3. Segmentação do Sinal (opcional)

- Divide o som em **janelas deslizantes** de 2s (com sobreposição de 50%)
- Cada janela será tratada como uma amostra independente para análise topológica

---

## 4. Extração Espectral

### 4.1. **FFT / STFT**

- Calcula espectro de frequência do sinal
- Detecta picos harmônicos usados para formar a **winding principal**

### 4.2. **MFCC (Mel-Frequency Cepstral Coefficients)**

- Extrai coeficientes perceptuais (13 ou 20)
- Representa o som de forma aproximada ao ouvido humano

### 4.3. **Wavelet Transform**

- Decompõe o sinal em múltiplas escalas temporais
- Gera coeficientes de energia ou médias por faixa

---

## 5. Construção de Representações Geométricas

### 5.1. **Winding baseada em FFT**

- Combina componentes senoidais como coordenadas `x(t), y(t), z(t)`
- Cria curva tridimensional única para cada som ou janela

### 5.2. **(Opcional) Winding baseada em Wavelet**

- Alternativa com base em coeficientes wavelet

### 5.3. **(Opcional) Espaço de Fase (Takens Embedding)**

- Curva do tipo `[x(t), x(t+τ), x(t+2τ)]` para capturar dinâmica interna

---

## 6. Cálculo do ITS (Índice Topológico Sintético)

Para cada winding gerada:

- Curvatura média
- Comprimento total da curva
- Número de cruzamentos (self-intersections)
- Torção total (variação angular)
- Entropia do trajeto
- Variância radial
- Compressibilidade (opcional)

→ Vetor ITS = `[c₁, c₂, ..., cₙ]`

---

## 7. Construção do Vetor Final (Híbrido)

- Concatena:
  - Vetor ITS
  - MFCCs
  - Wavelet (coeficientes ou energia por escala)

→ Vetor final `X = [ITS₁..n, MFCC₁..13, Wavelet₁..k]`

Esse vetor será usado como entrada nos modelos (MLP, Random Forest, etc.)

---

## 8. Armazenamento Estruturado

- Vetores exportados em `.csv` ou `.parquet`
- Nome dos arquivos reflete:
  - Tipo de som
  - Origem (pulmão, intestino, etc.)
  - Estado (normal, anômalo)
  - Janela (se segmentado)
- Windings podem ser salvas como:
  - Curvas `.npy`

---

## Resultado do Pré-processamento

Ao final dessas etapas, teremos:

- Vetores prontos para classificação e mineração de dados
- Curvas 3D para visualização
- Dados organizados para treinamento, teste e demonstração interativa

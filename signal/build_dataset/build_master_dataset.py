import pandas as pd
import chardet
import sys
import os
sys.path.append(os.path.abspath("../.."))

RAW_FEATURES_DIR = "../../datasets/raw/"
METADATA_DIR = "../../data/raw/HeartANDLung_Sounds_Dataset"
FINAL_DATASET_PATH = "../../datasets/final/features_master.csv"
# Carrega os metadados dos três conjuntos


def load_metadata():

    def detectar_encoding(caminho):
        with open(caminho, 'rb') as f:
            result = chardet.detect(f.read(10000))
        return result['encoding']

    arquivos = ["HS.csv", "LS.csv", "Mix.csv"]
    bases = []

    for arquivo in arquivos:
        caminho = os.path.join(METADATA_DIR, arquivo)
        encoding_detectado = detectar_encoding(caminho)
        print(f"🧠 Lendo {arquivo} com encoding: {encoding_detectado}")

        df = pd.read_csv(
            caminho,
            sep=None,  # Deixe o pandas tentar inferir
            engine="python",
            encoding=encoding_detectado
        )

        df.columns = df.columns.str.strip()
        df["origem"] = arquivo.replace(".csv", "")
        bases.append(df)

    return pd.concat(bases, ignore_index=True)


def carregar_features_com_metadados(nome_arquivo, metadata_df):
    caminho = RAW_FEATURES_DIR + nome_arquivo
    df = pd.read_excel(caminho)

    print(metadata_df.columns.tolist())

    # Extrai o ID do áudio (ex: H0001) do nome do arquivo
    audio_id = nome_arquivo.split("_")[0]

    # Busca os metadados correspondentes
    meta = metadata_df[
        (metadata_df["Heart Sound ID"] == audio_id) |
        (metadata_df["Lung Sound ID"] == audio_id) |
        (metadata_df["Mixed Sound ID"] == audio_id)
    ]

    if meta.empty:
        print(f"[AVISO] Metadados não encontrados para {audio_id}")
        return None

    # Vamos só pegar a primeira linha de metadado (por segurança)
    meta = meta.iloc[0].to_dict()

    # Insere colunas de metadado em todas as linhas do df
    for chave, valor in meta.items():
        df[chave] = valor

    return df

# Codifica os rótulos em one-hot


def aplicar_one_hot_encoding(df, colunas_alvo):
    return pd.get_dummies(df, columns=colunas_alvo)

# Pipeline principal


def construir_dataset_final():
    metadata_df = load_metadata()

    todos_arquivos = [f for f in os.listdir(
        RAW_FEATURES_DIR) if f.endswith(".xlsx")]

    dfs = []
    for arq in todos_arquivos:
        print(f"⏳ Processando {arq}...")
        df = carregar_features_com_metadados(arq, metadata_df)
        if df is not None:
            dfs.append(df)

    df_geral = pd.concat(dfs, ignore_index=True)

    # Codificar rótulos de interesse (pode ajustar)
    df_geral = aplicar_one_hot_encoding(
        df_geral, ["Heart Sound Type", "Lung Sound Type", "origem"])

    # Salvar
    os.makedirs(os.path.dirname(FINAL_DATASET_PATH), exist_ok=True)
    df_geral.to_csv(FINAL_DATASET_PATH, index=False)
    print(f"✅ Dataset consolidado salvo em: {FINAL_DATASET_PATH}")

construir_dataset_final()

# if __name__ == "__main__":
#     construir_dataset_final()


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def remove_columns(df: pd.DataFrame, columns_to_remove: list):
         return df.drop(columns=columns_to_remove, errors='ignore')

def create_one_hot_encoded_df(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """
    Codifica apenas as colunas categóricas relevantes:
    - LS: 'Gender', 'Location' (ignora 'Heart Sound Type')
    - HS: 'Gender', 'Location' (ignora 'Lung Sound Type')
    - Mix: 'Gender', 'Heart Sound Type', 'Lung Sound Type', 'Location'
    """
    df = remove_columns(df, ['rqa_file_id', 'rqa_window_id'])

    colunas_para_codificar = [
        'Gender', 'Heart Sound Type', 'Lung Sound Type', 'Location']

    if tipo.upper() == "HS":
        colunas_para_codificar.remove("Lung Sound Type")
    elif tipo.upper() == "LS":
        colunas_para_codificar.remove("Heart Sound Type")
    else:
        if tipo.upper() != "MIX":
            raise ValueError(
                f"Tipo inválido: {tipo}. Use 'HS', 'LS' ou 'Mix'.")

    colunas_existentes = [
        col for col in colunas_para_codificar
        if col in df.columns.difference(["Heart Sound ID"])
    ]

    if not colunas_existentes:
        print("Nenhuma coluna categórica válida encontrada para codificação.")
        return df

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[colunas_existentes])

    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(
        colunas_existentes), index=df.index)

    df_final = pd.concat(
        [df.drop(columns=colunas_existentes), encoded_df], axis=1)

    return df_final

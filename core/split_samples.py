import pandas as pd

def get_splited_data(data: pd.DataFrame, target_col: str):
    y = data[target_col]
    X = data.drop(columns=[target_col])
    return X, y

import os
import pandas as pd

pasta = 'data/HS_normal_abnormal'
planilha = 'data/HS_normal_abnormal.csv'

df = pd.read_csv(planilha, sep=';')
ids_planilha = df['Heart Sound ID'].astype(str).str.strip().tolist()

arquivos_pasta = [os.path.splitext(f)[0] for f in os.listdir(pasta) if f.endswith('.wav')]

ids_set = set(ids_planilha)
arquivos_set = set(arquivos_pasta)

faltando_na_pasta = ids_set - arquivos_set
sobrando_na_pasta = arquivos_set - ids_set

print("IDs presentes na planilha mas ausentes na pasta:")
print(sorted(faltando_na_pasta))

print("\nArquivos .wav na pasta mas não listados na planilha:")
print(sorted(sobrando_na_pasta))

import subprocess

comando = [
    "python", "extract_dataset.py",
    "--audio_dir", "../../data/raw/HeartANDLung_Sounds_Dataset/HS/HS",
    "--rotulos_csv", "../../data/raw/HeartANDLung_Sounds_Dataset/HS.csv",
    "--id_col", "Heart Sound ID",
    "--output", "../../datasets/intermediate/dataset_HS.csv",
    "--processes", "6"
]

subprocess.run(comando)

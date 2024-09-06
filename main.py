import pandas as pd

def read_file_parquet(filepath):
    # Leggi il file Parquet
    df = pd.read_parquet(filepath)

    # Verifica la struttura del DataFrame per assicurarti che i dati siano stati letti correttamente
    print("Ecco le prime righe del DataFrame:")
    print(df.head())
    return df

if __name__ == "__main__":
    filepath="./challenge_campus_biomedico_2024.parquet"
    df= read_file_parquet(filepath)
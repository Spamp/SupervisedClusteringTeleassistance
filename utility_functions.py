# utility_functions.py

import pandas as pd

def read_file_parquet(filepath):
    """Reads a Parquet file into a pandas DataFrame."""
    df = pd.read_parquet(filepath)
    print("Ecco le prime righe del DataFrame:")
    print(df.head())
    print(f"Numero totale di record: {len(df)}")
    return df

def count_nulls(dataframe):
    """Counts null and NaN values in each column of the DataFrame."""
    null_counts = dataframe.isnull().sum()
    print("Conteggio dei valori nulli e NaN per colonna:")
    print(null_counts)
    return null_counts

def check_duplicates(dataframe):
    """Checks for duplicates in the 'id_professionista_sanitario' column."""
    unique_patients = dataframe['id_professionista_sanitario'].nunique()
    print(f"Numero di ID paziente univoci nella colonna 'id_paziente': {unique_patients}")
    return unique_patients

def salva_dataframe(df, nome_file):
    """Saves the DataFrame to a Parquet file."""
    try:
        df.to_parquet(nome_file, index=False)
        print(f"DataFrame salvato con successo in '{nome_file}'")
    except Exception as e:
        print(f"Errore durante il salvataggio del DataFrame: {e}")

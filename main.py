import pandas as pd

def read_file_parquet(filepath):
    # Leggi il file Parquet
    df = pd.read_parquet(filepath)

    # Verifica la struttura del DataFrame per assicurarti che i dati siano stati letti correttamente
    print("Ecco le prime righe del DataFrame:")
    print(df.head())

    # Conta il numero totale di record
    print(f"Numero totale di record: {len(df)}")
    return df

# Funzione per contare i valori null e NaN
def count_nulls(dataframe):
    null_counts = dataframe.isnull().sum()
    print("Conteggio dei valori nulli e NaN per colonna:")
    print(null_counts)
    return null_counts

# Funzione per controllare i duplicati nella colonna 'id_prenotazione'
def check_duplicates(dataframe):
    duplicate_count = dataframe['id_prenotazione'].duplicated().sum()
    print(f"Numero di duplicati nella colonna 'id_prenotazione': {duplicate_count}")
    return duplicate_count

# Funzione per ordinare cronologicamente in base a 'ora_inizio_erogazione'
def sort_chronologically_by_timestamp(dataframe):
    # Converti 'ora_inizio_erogazione' nel formato datetime
    dataframe['ora_inizio_erogazione'] = pd.to_datetime(dataframe['ora_inizio_erogazione'], errors='coerce')
    
    # Ordina il DataFrame
    sorted_dataframe = dataframe.sort_values(by='ora_inizio_erogazione')
    print("Ecco le prime righe del DataFrame ordinato cronologicamente:")
    print(sorted_dataframe.head())
    return sorted_dataframe

if __name__ == "__main__":
    filepath = "./challenge_campus_biomedico_2024.parquet"
    
    # Leggi il file Parquet
    df = read_file_parquet(filepath)
    
    # Conta i valori nulli e NaN
    count_nulls(df)
    
    # Controlla i duplicati nella colonna 'id_prenotazione'
    check_duplicates(df)
    
    # Ordina il DataFrame cronologicamente
    sorted_df = sort_chronologically_by_timestamp(df)

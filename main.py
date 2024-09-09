import pandas as pd
from datetime import datetime

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


# Funzione per calcolare l'età, rimuovere la colonna data di nascita e posizionare la colonna età
def calcola_eta_e_posiziona(df):
    # Funzione per convertire data di nascita in età
    def calcola_eta(data_nascita):
        today = datetime.today()
        # Conversione della data di nascita in formato datetime
        data_nascita = pd.to_datetime(data_nascita, format='%Y-%m-%d', errors='coerce')
        # Calcolo dell'età solo se la data di nascita è valida
        if pd.isnull(data_nascita):
            return None
        return today.year - data_nascita.year - ((today.month, today.day) < (data_nascita.month, data_nascita.day))

    # Trovare l'indice della colonna 'data_nascita' per reinserire 'età' nella stessa posizione
    index_col = df.columns.get_loc('data_nascita')

    # Applicare la funzione per calcolare l'età a ogni paziente
    df['età'] = df['data_nascita'].apply(calcola_eta)

    # Rimuovere la colonna 'data_nascita'
    df.drop(columns=['data_nascita'], inplace=True)

    # Spostare la colonna 'età' nella posizione originale di 'data_nascita'
    cols = df.columns.tolist()
    cols.insert(index_col, cols.pop(cols.index('età')))
    df = df[cols]

    # Visualizzare i risultati (prime 5 righe)
    print(df[['id_paziente', 'età']].head())
    
    return df




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

    # Calcola l'età, rimuovi la colonna 'data_nascita' e posiziona la colonna 'età'
    df = calcola_eta_e_posiziona(sorted_df)


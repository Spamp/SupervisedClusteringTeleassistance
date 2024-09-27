import pandas as pd
import json
import numpy as np

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
    unique_patients = dataframe['id_professionista_sanitario'].nunique()  # Conta gli ID univoci
    print(f"Numero di ID paziente univoci nella colonna 'id_paziente': {unique_patients}")
    return unique_patients

# Funzione per ordinare cronologicamente in base a 'ora_inizio_erogazione'
def sort_chronologically_by_timestamp(dataframe):
    # Converti 'ora_inizio_erogazione' nel formato datetime
    dataframe['ora_inizio_erogazione'] = pd.to_datetime(dataframe['data_erogazione'], errors='coerce')
    
    # Ordina il DataFrame
    sorted_dataframe = dataframe.sort_values(by='ora_inizio_erogazione')
    print("Ecco le prime righe del DataFrame ordinato cronologicamente:")
    return sorted_dataframe


# Funzione per visualizzare chiavi e numero di elementi per ogni chiave
def visualizza_dizionario_chiavi(dizionario, nome_dizionario, limite=10):
    print(f"Visualizzazione del dizionario {nome_dizionario} (limite {limite} chiavi):")
    
    for i, (chiave, valore) in enumerate(dizionario.items()):
        if i < limite:
            if isinstance(valore, list):
                print(f"Chiave: {chiave}, Numero di elementi: {len(valore)}")
            else:
                print(f"Chiave: {chiave}, Valore: {valore}")
        else:
            break
    print(f"Numero totale di chiavi in {nome_dizionario}: {len(dizionario)}\n")

# Modifica della funzione per mappare le colonne senza l'utilizzo di defaultdict
def mappa_colonne(df, coppie_colonne):
    mappatura_codice_to_nome = {}
    mappatura_nome_to_codice = {}
    valori_nulli = set()
    nomi_multipli = {}

    for nome_colonna, codice_colonna in coppie_colonne:
        mappatura_c_to_n = {}
        mappatura_n_to_c = {}

        for nome, codice in zip(df[nome_colonna], df[codice_colonna]):
            if pd.isnull(nome) or pd.isnull(codice):
                if pd.notnull(nome):
                    valori_nulli.add(f"Nella colonna '{codice_colonna}', il valore '{nome}' non ha codice.")
                if pd.notnull(codice):
                    valori_nulli.add(f"Nella colonna '{nome_colonna}', il codice '{codice}' non ha nome.")
            else:
                if codice not in mappatura_c_to_n:
                    mappatura_c_to_n[codice] = nome
                elif mappatura_c_to_n[codice] != nome:
                    raise ValueError(f"Conflitto di mappatura: il codice {codice} "
                                     f"è associato a '{mappatura_c_to_n[codice]}' e a '{nome}'")

                if nome not in mappatura_n_to_c:
                    mappatura_n_to_c[nome] = [codice]
                else:
                    if codice not in mappatura_n_to_c[nome]:
                        mappatura_n_to_c[nome].append(codice)
                        if len(mappatura_n_to_c[nome]) > 1:
                            nomi_multipli[nome] = mappatura_n_to_c[nome]

        mappatura_codice_to_nome[nome_colonna] = mappatura_c_to_n
        mappatura_nome_to_codice[nome_colonna] = mappatura_n_to_c
    
    return mappatura_codice_to_nome, mappatura_nome_to_codice, list(valori_nulli), nomi_multipli


# Funzione per visualizzare il numero di sub-chiavi per tutte le macro chiavi
def visualizza_sub_chiavi(dizionario_mappature):
    for macro_chiave, sub_chiavi in dizionario_mappature.items():
        num_sub_chiavi = len(sub_chiavi)
        print(f"Macro chiave: {macro_chiave} ha {num_sub_chiavi} sub-chiavi.")

# Funzione per controllare quanti record hanno ASL di residenza diversa da quella di erogazione
def conta_asl_differenti(df):
    # Confrontiamo le colonne 'asl_residenza' e 'asl_erogazione'
    record_differenti = df[df['asl_residenza'] != df['asl_erogazione']].shape[0]
    print(f"Numero di record con ASL di residenza diversa da quella di erogazione: {record_differenti}")
    return record_differenti


# Funzione per identificare i valori problematici nelle colonne 'data_nascita' e 'data_erogazione'
def esplora_formati_data(df):
    # Identifica i valori unici presenti in 'data_nascita' e 'data_erogazione'
    formati_data_nascita = df['data_nascita'].unique()
    formati_data_erogazione = df['data_erogazione'].unique()

    return df


# Funzione per verificare quali valori non possono essere convertiti in datetime
def verifica_date_non_convertibili(df):
    # Prova a convertire le date in datetime
    df['data_nascita'] = pd.to_datetime(df['data_nascita'], errors='coerce')
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], errors='coerce')

    # Trova i valori che non possono essere convertiti (NaT)
    date_nascita_non_convertibili = df[df['data_nascita'].isna()]
    date_erogazione_non_convertibili = df[df['data_erogazione'].isna()]

    return df


# Funzione principale per esplorare e diagnosticare le date
def diagnostica_date(df):
    # Esplora i formati unici nelle colonne 'data_nascita' e 'data_erogazione'
    df = esplora_formati_data(df)

    # Verifica quali valori non possono essere convertiti in datetime
    df = verifica_date_non_convertibili(df)

    return df


# Funzione per gestire il formato della colonna 'data_erogazione' mantenendo l'orario
def gestisci_data_erogazione(df):
    # Converti la colonna 'data_erogazione' in datetime, mantenendo l'ora e il fuso orario
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], errors='coerce')

    return df


# Funzione principale per calcolare l'età mantenendo l'orario in 'data_erogazione'
def calcola_eta_e_posiziona_vettorizzato(df):
    # Gestisci il formato della colonna 'data_erogazione' mantenendo l'orario
    df = gestisci_data_erogazione(df)

    # Converti 'data_nascita' in datetime
    df['data_nascita'] = pd.to_datetime(df['data_nascita'], errors='coerce')

    # Verifica se ci sono ancora valori NaT dopo la conversione
    date_nascita_non_convertibili = df[df['data_nascita'].isna()]
    if not date_nascita_non_convertibili.empty:
        print("Valori non convertibili in 'data_nascita':")
        print(date_nascita_non_convertibili[['id_paziente', 'data_nascita']])

    # Rimuovere le righe con valori NaT in 'data_nascita' o 'data_erogazione'
    df = df.dropna(subset=['data_nascita', 'data_erogazione'])

    # Calcola l'età basandosi solo sulla parte di data (ignorando l'ora e il fuso orario)
    df['età'] = df['data_erogazione'].dt.year - df['data_nascita'].dt.year

    # Aggiusta l'età se il compleanno non è ancora passato
    compleanno_non_passato = (df['data_erogazione'].dt.month < df['data_nascita'].dt.month) | \
                             ((df['data_erogazione'].dt.month == df['data_nascita'].dt.month) &
                              (df['data_erogazione'].dt.day < df['data_nascita'].dt.day))

    df.loc[compleanno_non_passato, 'età'] -= 1

    # Rimuovere la colonna 'data_nascita'
    df.drop(columns=['data_nascita'], inplace=True)

    # Spostare la colonna 'età' nella posizione originale di 'data_nascita'
    index_col = df.columns.get_loc('età')
    cols = df.columns.tolist()
    cols.insert(index_col, cols.pop(cols.index('età')))
    df = df[cols]

    # Visualizzare i risultati (prime 5 righe)
    print(df[['id_paziente', 'data_erogazione', 'età']].head())

    return df


# Funzione per calcolare la durata dell'assistenza, effettuare le correzioni e pulire il dataset
def calcola_e_correggi_durata_assistenza(df):
    # Converti 'ora_inizio_erogazione' e 'ora_fine_erogazione' in datetime (se non lo sono già)
    df['ora_inizio_erogazione'] = pd.to_datetime(df['ora_inizio_erogazione'], errors='coerce')
    df['ora_fine_erogazione'] = pd.to_datetime(df['ora_fine_erogazione'], errors='coerce')

    # Calcola la durata in ore solo per i record validi
    df['durata_assistenza'] = (df['ora_fine_erogazione'] - df['ora_inizio_erogazione']).dt.total_seconds() / 3600

    # Imposta 'durata_assistenza' a 0 per i record con disdetta (quando 'data_disdetta' non è NaN)
    df.loc[~df['data_disdetta'].isna() & (df['ora_inizio_erogazione'].isna() | df['ora_fine_erogazione'].isna()), 'durata_assistenza'] = 0

    # Rimuovi i record in cui 'durata_assistenza' è NaN (questo include i record con tutte le colonne NaN)
    df = df.dropna(subset=['durata_assistenza'])

    # Elimina le colonne 'ora_inizio_erogazione' e 'ora_fine_erogazione'
    df = df.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'])

    # Conta quanti valori NaN sono presenti in 'durata_assistenza' (dovrebbe essere 0 dopo la pulizia)
    nan_in_durata = df['durata_assistenza'].isna().sum()

    # Stampa il numero di valori NaN nella colonna 'durata_assistenza'
    print(f"Numero di valori NaN nella colonna 'durata_assistenza' dopo la pulizia: {nan_in_durata}")
    
    return df

# funzione per creare un dizionario con le regioni come chiavi, e i comuni di residenza come valori
def crea_dizionario_comuni_per_regione(df):
    dizionario_comuni_per_regione = {}
    for regione in df['regione_residenza'].unique():
        comuni = df.loc[df['regione_residenza'] == regione, 'comune_residenza'].unique()
        dizionario_comuni_per_regione[regione] = comuni
    return dizionario_comuni_per_regione

def reinsert_missing_codes(df):
    # Reinsert 'NA' for Napoli in 'codice_provincia_residenza' and 'codice_provincia_erogazione'
    df.loc[df['provincia_residenza'] == 'Napoli', 'codice_provincia_residenza'] = 'NA'
    df.loc[df['provincia_erogazione'] == 'Napoli', 'codice_provincia_erogazione'] = 'NA'

    # Reinsert 'None' for codice 1168 in 'comune_residenza'
    df.loc[df['comune_residenza'].isnull(), 'comune_residenza'] = 'Comune di None'

    return df

def unify_codes(df):
    #function which unify different structure codes reffering to the same struture, in one single code
    df['codice_struttura_erogazione'] = df['codice_struttura_erogazione'].astype(str).str.split('.').str[0]
    return df

def get_unique_province_for_presidio_ospedaliero(df):
    # Filter the dataframe where 'struttura_erogazione' is 'PRESIDIO OSPEDALIERO UNIFICATO'
    filtered_df = df[df['struttura_erogazione'] == 'PRESIDIO OSPEDALIERO UNIFICATO']
    
    # Group by 'codice_struttura_erogazione' and get unique 'provincia_erogazione' for each code
    province_per_codice = filtered_df.groupby('codice_struttura_erogazione')['provincia_erogazione'].unique()
    
    # Convert the result to a dictionary for easier reading
    result = province_per_codice.to_dict()
    
    return result

def modify_structure_name(df):
    #function which modify the name of the structure with same but different codes, add the provice they belong to
    df.loc[df['codice_struttura_erogazione'] == '70001', 'struttura_erogazione'] = 'PRESIDIO OSPEDALIERO UNIFICATO (Imperia)'
    df.loc[df['codice_struttura_erogazione'] == '100803', 'struttura_erogazione'] = 'PRESIDIO OSPEDALIERO UNIFICATO (Perugia)'
    return df

# Funzione per calcolare il tempo di attesa e convertire le date
def calcola_attesa_assistenza(df):
    # Converte le colonne 'data_contatto' e 'data_erogazione' in formato datetime mantenendo il fuso orario
    df['data_contatto'] = pd.to_datetime(df['data_contatto'], utc=True, errors='coerce')
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], utc=True, errors='coerce')

    # Calcola il tempo di attesa in giorni tra 'data_erogazione' e 'data_contatto' e arrotonda i risultati
    df['attesa_assistenza'] = (df['data_erogazione'] - df['data_contatto']).dt.total_seconds() / (60 * 60 * 24)

    # Arrotonda i valori di attesa_assistenza a giorni interi
    df['attesa_assistenza'] = df['attesa_assistenza'].round(0)

    # Visualizza i risultati arrotondati
    print(df[['data_contatto', 'data_erogazione', 'attesa_assistenza']].head())

    return df

# Funzione per convertire la colonna 'sesso' in valori booleani e posizionare 'sesso_bool' accanto a 'sesso'
def converti_sesso_in_booleano(df):
    # Converte 'sesso' in minuscolo e sostituisce direttamente 'male' con 1 e 'female' con 0
    df['sesso_bool'] = df['sesso'].str.lower().replace({'male': 1, 'female': 0})

    # Trova l'indice della colonna 'sesso'
    index_sesso = df.columns.get_loc('sesso')

    # Sposta la colonna 'sesso_bool' accanto a 'sesso'
    cols = df.columns.tolist()
    cols.insert(index_sesso + 1, cols.pop(cols.index('sesso_bool')))
    df = df[cols]

    # Mostra i primi record per verifica
    print(df[['sesso', 'sesso_bool']].head())

    return df

# Carica il dizionario delle coordinate dal file JSON
def load_coordinates(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")

# Funzione vettorizzata per sostituire i nomi delle città con le coordinate
def replace_city_columns_with_coordinates(df, city_columns, coordinates_dict):
    # Verifica se tutte le colonne esistono nel DataFrame
    for city_column in city_columns:
        if city_column not in df.columns:
            raise KeyError(f"The column '{city_column}' was not found in the DataFrame.")
    
    # Prepara la mappatura per latitudine e longitudine
    lat_dict = {city: coord[0] for city, coord in coordinates_dict.items()}
    lng_dict = {city: coord[1] for city, coord in coordinates_dict.items()}
    
    # Sostituisci i nomi delle città con le coordinate usando il metodo map vettorizzato
    for city_column in city_columns:
        df[city_column + '_lat'] = df[city_column].map(lat_dict)
        df[city_column + '_lng'] = df[city_column].map(lng_dict)
    
    # Opzionalmente, puoi rimuovere le colonne originali delle città
    #df.drop(columns=city_columns, inplace=True)
    
    return df



def crea_colonna_semestre(df):
    # Converti 'data_erogazione' in datetime se non lo è già
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'])
    
    # Estrarre l'anno
    df['anno'] = df['data_erogazione'].dt.year
    
    # Creare il semestre usando np.where per il controllo delle condizioni
    df['semestre'] = np.where(df['data_erogazione'].dt.month <= 6, 'S1', 'S2')
    
    # Combinare anno e semestre in una nuova colonna 'semestre'
    df['semestre'] = df['anno'].astype(str) + '-' + df['semestre']
    
    # Visualizzare i primi risultati
    print(df[['data_erogazione', 'anno', 'semestre']].head())
    print(df[['data_erogazione', 'anno', 'semestre']].tail())
    
    return df

# funzione che conta i valori NaN in una colonna specificata
def conta_nan_colonna(df, colonna):
    # Conta i valori NaN nella colonna specificata
    nan_count = df[colonna].isna().sum()
    
    # Stampa il risultato
    print(f"Numero di valori NaN nella colonna '{colonna}': {nan_count}")
    
    return nan_count

# fuzione che conta i valori NaN in una colonna specificata e rimuove le righe con valori NaN
def conta_e_rimuovi_nan_colonna(df, colonna, rimuovi=False):
    # Conta i valori NaN nella colonna specificata
    nan_count = df[colonna].isna().sum()
    
    # Stampa il risultato
    print(f"Numero di valori NaN nella colonna '{colonna}': {nan_count}")
    
    # Se rimuovi=True, elimina le righe con valori NaN nella colonna specificata dall'intero DataFrame
    if rimuovi:
        df = df.dropna(subset=[colonna])
        print(f"Le righe con valori NaN nella colonna '{colonna}' sono state rimosse.")
        print(f"Numero di righe rimanenti nel DataFrame: {len(df)}")
    
    return df

# Funzione per calcolare l'incremento percentuale delle teleassistenze per semestre
def calcola_incremento_teleassistenze(df):
    # Raggruppare per semestre
    grouped = df.groupby('semestre').size().reset_index(name='num_teleassistenze')

    # Calcolare l'incremento percentuale rispetto al semestre precedente
    grouped['incremento'] = grouped['num_teleassistenze'].pct_change() * 100

    # Riempiamo i valori NaN (per il primo semestre, non c'è incremento precedente) con 0
    grouped['incremento'] = grouped['incremento'].fillna(0)

    # Visualizza i risultati (prime righe)
    print(grouped.head())
    print(grouped.tail())

    return grouped

    

# Funzione aggiornata per classificare l'incremento per ogni semestre
def classifica_incremento_per_semestre(df_grouped):
    # Inizializziamo la colonna 'classificazione' come 'Stabile' di default
    df_grouped['classificazione'] = 'Stabile'
    
    # Classificazione per 'Decremento significativo'
    df_grouped.loc[df_grouped['incremento'] < 0, 'classificazione'] = 'Decremento'
    
    # Classificazione per 'Incremento moderato'
    df_grouped.loc[(df_grouped['incremento'] > 0) & (df_grouped['incremento'] <= 40), 'classificazione'] = 'Incremento moderato'
    
    # Classificazione per 'Incremento alto'
    df_grouped.loc[df_grouped['incremento'] > 40, 'classificazione'] = 'Incremento alto'
    
    # Visualizza i risultati per conferma
    print(df_grouped[['semestre', 'num_teleassistenze', 'incremento', 'classificazione']])
    
    return df_grouped





# Funzione per etichettare l'incremento per record nel DataFrame completo
def etichetta_incremento_per_record(df, df_incremento):
    # Uniamo df e df_incremento sulla colonna 'semestre'
    df = df.merge(df_incremento[['semestre', 'classificazione']], on='semestre', how='left')
    
    # Rinominiamo la colonna 'classificazione' in 'incremento_teleassistenza'
    df.rename(columns={'classificazione': 'incremento_teleassistenza'}, inplace=True)
    
    # Visualizza un campione del risultato
    print(df[['semestre', 'incremento_teleassistenza']].head())
    
    return df

# ... (tutte le altre funzioni e codice)

# Funzione per salvare il DataFrame in un file Parquet
def salva_dataframe(df, nome_file):
    """
    Salva il DataFrame in un file Parquet.

    Parametri:
    - df: il DataFrame da salvare
    - nome_file: nome del file Parquet (includi l'estensione .parquet)

    Restituisce:
    - None
    """
    try:
        df.to_parquet(nome_file, index=False)
        print(f"DataFrame salvato con successo in '{nome_file}'")
    except Exception as e:
        print(f"Errore durante il salvataggio del DataFrame: {e}")





if __name__ == "__main__":
    filepath = "./challenge_campus_biomedico_2024.parquet"
    
    # Leggi il file Parquet
    df = read_file_parquet(filepath)

    # Conta i valori nulli e NaN
    count_nulls(df)
    
    # Controlla i duplicati nella colonna 'id_pazienti'
    check_duplicates(df)

    #funcrtions which cleanse features with codes
    reinsert_missing_codes(df)

    df = unify_codes(df)

    unique_provinces = get_unique_province_for_presidio_ospedaliero(df)
    print(unique_provinces)

    df= modify_structure_name(df)

    coppie_colonne = [
    ('regione_residenza', 'codice_regione_residenza'),
    ('asl_residenza', 'codice_asl_residenza'),
    ('provincia_residenza', 'codice_provincia_residenza'),
    ('comune_residenza', 'codice_comune_residenza'),
    ('descrizione_attivita', 'codice_descrizione_attivita'),
    ('regione_erogazione', 'codice_regione_erogazione'),
    ('asl_erogazione', 'codice_asl_erogazione'),
    ('provincia_erogazione', 'codice_provincia_erogazione'),
    ('struttura_erogazione', 'codice_struttura_erogazione'),
    ('tipologia_struttura_erogazione', 'codice_tipologia_struttura_erogazione'),
    ('tipologia_professionista_sanitario', 'codice_tipologia_professionista_sanitario')
    ]


    
    mappatura_c_to_n, mappatura_n_to_c, valori_nulli, nomi_multipli = mappa_colonne(df, coppie_colonne)
    #create dict to map the cities for their regions
    dict_regioni=crea_dizionario_comuni_per_regione(df)

    # Output delle mappature e dei valori nulli
    print("Valori nulli:", valori_nulli)

    # Messaggio sui nomi associati a più codici
    if nomi_multipli:
        print(f"{len(nomi_multipli)} nomi sono associati a più di un codice:")
        for nome, codici in nomi_multipli.items():
            print(f"Il nome '{nome}' è associato ai codici: {codici}")
    else:
        print("Nessun nome è associato a più di un codice.")
    
     # Visualizza un campione delle ASL residenza e erogazione
    visualizza_dizionario_chiavi(mappatura_c_to_n['asl_residenza'], 'asl_residenza')
    visualizza_dizionario_chiavi(mappatura_c_to_n['asl_erogazione'], 'asl_erogazione')
    

    visualizza_sub_chiavi(mappatura_c_to_n)
    
    # Controllo dei record con ASL di residenza diversa da quella di erogazione
    conta_asl_differenti(df)
    
    
    # Ordina il DataFrame cronologicamente
    df = sort_chronologically_by_timestamp(df)

    # Calcola l'età, rimuovi la colonna 'data_nascita' e posiziona la colonna 'età'
    df = diagnostica_date(df)

    # Calcola l'età mantenendo l'orario in 'data_erogazione'
    df = calcola_eta_e_posiziona_vettorizzato(df)



    # Calcola la durata dell'assistenza e gestisci i casi particolari
    df = calcola_e_correggi_durata_assistenza(df)

    df= converti_sesso_in_booleano(df)

    df= calcola_attesa_assistenza(df)

    coordinates_file = './coordinate_dataset.json'

    # Carica il dizionario delle coordinate
    coordinates_dict = load_coordinates(coordinates_file)

    # Lista delle colonne che contengono i nomi delle città
    city_columns = ['comune_residenza', 'comune_destinazione']  # Aggiungi tutte le colonne desiderate

    # Sostituisci i nomi delle città con le coordinate nelle colonne specificate
    df = replace_city_columns_with_coordinates(df, ['provincia_residenza','comune_residenza', 'provincia_erogazione'], coordinates_dict)

    print(df.columns)
    # Crea la colonna 'semestre' basata su 'data_erogazione'
    df = crea_colonna_semestre(df)

    # Controlla i valori NaN nella colonna 'provincia_erogazione_lng' e 'provincia_erogazione_lat'
    conta_nan_colonna(df, 'provincia_erogazione_lng')
    conta_nan_colonna(df, 'provincia_erogazione_lat')

    # Calcola l'incremento percentuale delle teleassistenze per semestre
    df_incremento = calcola_incremento_teleassistenze(df)

    # Visualizza i primi 5 risultati
    print(df_incremento.head())
    # Visualizza gli ultimi 5 risultati
    print(df_incremento.tail())

    # Classifica l'incremento per ogni semestre
    df_incremento = classifica_incremento_per_semestre(df_incremento)


    selected_columns = [
        'id_prenotazione', 'asl_residenza', 'descrizione_attivita',
        'asl_erogazione', 'struttura_erogazione',
        'tipologia_struttura_erogazione', 'regione','tipologia_professionista_sanitario',
        'data_erogazione', 'età', 'durata_assistenza', 'sesso_bool', 'attesa_assistenza',
        'provincia_residenza_lat', 'provincia_residenza_lng', 'comune_residenza_lat',
        'comune_residenza_lng', 'provincia_erogazione_lat', 'provincia_erogazione_lng',
        'semestre'  
    ]

    df_feature_selezionate = df[selected_columns]

    # Etichetta l'incremento per record nel DataFrame completo
    df_feature_selezionate = etichetta_incremento_per_record(df_feature_selezionate, df_incremento)

    path_dataset_pulito='./dataset_pulito.parquet'

    salva_dataframe(df_feature_selezionate,path_dataset_pulito)
    # Visualizza i primi 5 risultati
    print(df_feature_selezionate.head())
    # Visualizza gli ultimi 5 risultati
    print(df_feature_selezionate.tail())














    

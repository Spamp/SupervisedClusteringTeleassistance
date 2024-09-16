import pandas as pd
from datetime import datetime
import hashlib

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
    print(sorted_dataframe.head())
    print(sorted_dataframe.tail())
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


# Funzione per controllare se c'è un comune corrispondente al codice 1168
def controllo_comune_residenza(df):
    comune = df.loc[df['codice_comune_residenza'] == 1168, 'comune_residenza'].unique()
    if len(comune) > 0:
        print(f"Il codice 1168 corrisponde al comune: {comune[0]}")
    else:
        print("Nessun comune trovato per il codice 1168.")

# Funzione per verificare i codici utilizzati per Napoli e aggiungere "NA" se necessario
def controllo_napoli(df):
    codici_napoli = df.loc[df['provincia_residenza'] == 'Napoli', 'codice_provincia_residenza'].unique()
    print(f"Codici utilizzati per Napoli: {codici_napoli}")

    # Aggiungiamo 'NA' se non è presente
    if 'NA' not in codici_napoli:
        print("Il codice 'NA' non è presente per Napoli, lo aggiungiamo.")
        # Aggiungere NA al DataFrame (simulazione)
        df.loc[df['comune_residenza'] == 'Napoli', 'codice_comune_residenza'] = 'NA'
    else:
        print("Il codice 'NA' è già presente per Napoli.")
    return df

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

if __name__ == "__main__":
    filepath = "./challenge_campus_biomedico_2024.parquet"
    
    # Leggi il file Parquet
    df = read_file_parquet(filepath)

    reinsert_missing_codes(df)
    
    # Conta i valori nulli e NaN
    count_nulls(df)
    
    # Controlla i duplicati nella colonna 'id_pazienti'
    check_duplicates(df)

    # Controllo se il codice 1168 ha un comune associato
    controllo_comune_residenza(df)
    
    # Controllo per Napoli e aggiunta del codice NA se necessario
    df = controllo_napoli(df)

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


    # Chiamata alla funzione
    mappatura_c_to_n, mappatura_n_to_c, valori_nulli, nomi_multipli = mappa_colonne(df, coppie_colonne)
    #create dict to map the cities for their regions
    dict_regioni=crea_dizionario_comuni_per_regione(df)

    # Output delle mappature e dei valori nulli
    #print("Mappatura Codice -> Nome:", mappatura_c_to_n)
    #print("Mappatura Nome -> Codice:", mappatura_n_to_c)
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
    sorted_df = sort_chronologically_by_timestamp(df)

    # Calcola l'età, rimuovi la colonna 'data_nascita' e posiziona la colonna 'età'
    df_diagnosticato = diagnostica_date(df)

    # Calcola l'età mantenendo l'orario in 'data_erogazione'
    df_eta = calcola_eta_e_posiziona_vettorizzato(df_diagnosticato)


    # Calcola la durata dell'assistenza e gestisci i casi particolari
    df = calcola_e_correggi_durata_assistenza(df_eta)

    # Mostra i primi risultati del DataFrame
    print(df.head())
    

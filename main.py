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

'''def mappa_colonne(df, coppie_colonne):
    mappatura_codice_to_nome = {}
    mappatura_nome_to_codice = defaultdict(list)
    valori_nulli = set()  # Set per mantenere i valori univoci
    nomi_multipli = {}  # Per tracciare nomi associati a più di un codice
    
    # Cicla su ogni coppia (descrittiva, codice)
    for nome_colonna, codice_colonna in coppie_colonne:
        # Dobbiamo mappare codice -> nome e nome -> codice
        mappatura_c_to_n = {}
        
        # Iteriamo su ogni riga delle due colonne
        for nome, codice in zip(df[nome_colonna], df[codice_colonna]):
            if pd.isnull(nome) or pd.isnull(codice):
                # Aggiungi il valore nella lista dei valori nulli
                if pd.notnull(nome):
                    valori_nulli.add(f"Nella colonna '{codice_colonna}', il valore '{nome}' non ha codice.")
                if pd.notnull(codice):
                    valori_nulli.add(f"Nella colonna '{nome_colonna}', il codice '{codice}' non ha nome.")
            else:
                # Gestione mappatura codice -> nome
                if codice not in mappatura_c_to_n:
                    mappatura_c_to_n[codice] = nome
                elif mappatura_c_to_n[codice] != nome:
                    raise ValueError(f"Conflitto di mappatura: il codice {codice} "
                                     f"è associato a '{mappatura_c_to_n[codice]}' e a '{nome}'")
                
                # Gestione mappatura nome -> codice (possiamo avere più codici per un nome)
                if codice not in mappatura_nome_to_codice[nome]:
                    mappatura_nome_to_codice[nome].append(codice)
                    # Traccia nomi associati a più codici
                    if len(mappatura_nome_to_codice[nome]) > 1:
                        nomi_multipli[nome] = mappatura_nome_to_codice[nome]
        
        # Aggiungiamo le mappe della colonna corrente ai risultati completi
        mappatura_codice_to_nome[nome_colonna] = mappatura_c_to_n
    
    # Convertiamo la lista dei valori nulli in una lista univoca
    return mappatura_codice_to_nome, mappatura_nome_to_codice, list(valori_nulli), nomi_multipli'''

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
    codici_napoli = df.loc[df['comune_residenza'] == 'Napoli', 'codice_comune_residenza'].unique()
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

if __name__ == "__main__":
    filepath = "./challenge_campus_biomedico_2024.parquet"
    
    # Leggi il file Parquet
    df = read_file_parquet(filepath)
    
    # Conta i valori nulli e NaN
    count_nulls(df)
    
    # Controlla i duplicati nella colonna 'id_pazienti'
    check_duplicates(df)

    # Controllo se il codice 1168 ha un comune associato
    controllo_comune_residenza(df)
    
    # Controllo per Napoli e aggiunta del codice NA se necessario
    df = controllo_napoli(df)

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
    #df = calcola_eta_e_posiziona(sorted_df)


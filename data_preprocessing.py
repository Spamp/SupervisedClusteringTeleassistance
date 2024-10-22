

import pandas as pd

def mappa_colonne(df, coppie_colonne):
    """Maps code columns to name columns without using defaultdict."""
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

def diagnostica_date(df):
    """Diagnoses date formats and identifies problematic values in 'data_nascita' and 'data_erogazione'."""
    df = esplora_formati_data(df)
    df = verifica_date_non_convertibili(df)
    return df

def esplora_formati_data(df):
    """Explores unique date formats in 'data_nascita' and 'data_erogazione'."""
    formati_data_nascita = df['data_nascita'].unique()
    formati_data_erogazione = df['data_erogazione'].unique()
    return df

def verifica_date_non_convertibili(df):
    """Checks which values cannot be converted to datetime."""
    df['data_nascita'] = pd.to_datetime(df['data_nascita'], errors='coerce')
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], errors='coerce')
    return df

def gestisci_data_erogazione(df):
    """Handles the format of the 'data_erogazione' column while keeping the time."""
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], errors='coerce')
    return df

def calcola_eta_e_posiziona_vettorizzato(df):
    """Calculates age while maintaining the time in 'data_erogazione'."""
    df = gestisci_data_erogazione(df)
    df['data_nascita'] = pd.to_datetime(df['data_nascita'], errors='coerce')

    df = df.dropna(subset=['data_nascita', 'data_erogazione'])
    df['età'] = df['data_erogazione'].dt.year - df['data_nascita'].dt.year

    compleanno_non_passato = (
        (df['data_erogazione'].dt.month < df['data_nascita'].dt.month) |
        ((df['data_erogazione'].dt.month == df['data_nascita'].dt.month) &
         (df['data_erogazione'].dt.day < df['data_nascita'].dt.day))
    )

    df.loc[compleanno_non_passato, 'età'] -= 1
    df.drop(columns=['data_nascita'], inplace=True)

    index_col = df.columns.get_loc('età')
    cols = df.columns.tolist()
    cols.insert(index_col, cols.pop(cols.index('età')))
    df = df[cols]

    print(df[['id_paziente', 'data_erogazione', 'età']].head())
    return df

def calcola_e_correggi_durata_assistenza(df):
    """Calculates the duration of assistance, makes corrections, and cleans the dataset."""
    df['ora_inizio_erogazione'] = pd.to_datetime(df['ora_inizio_erogazione'], errors='coerce')
    df['ora_fine_erogazione'] = pd.to_datetime(df['ora_fine_erogazione'], errors='coerce')

    df['durata_assistenza'] = (df['ora_fine_erogazione'] - df['ora_inizio_erogazione']).dt.total_seconds() / 3600

    df.loc[
        (~df['data_disdetta'].isna()) &
        (df['ora_inizio_erogazione'].isna() | df['ora_fine_erogazione'].isna()),
        'durata_assistenza'
    ] = 0

    df = df.dropna(subset=['durata_assistenza'])
    df = df.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'])

    nan_in_durata = df['durata_assistenza'].isna().sum()
    print(f"Numero di valori NaN nella colonna 'durata_assistenza' dopo la pulizia: {nan_in_durata}")
    return df

def calcola_attesa_assistenza(df):
    """Calculates the waiting time for assistance in days."""
    df['data_contatto'] = pd.to_datetime(df['data_contatto'], utc=True, errors='coerce')
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], utc=True, errors='coerce')

    df['attesa_assistenza'] = (df['data_erogazione'] - df['data_contatto']).dt.total_seconds() / (60 * 60 * 24)
    df['attesa_assistenza'] = df['attesa_assistenza'].round(0)

    print(df[['data_contatto', 'data_erogazione', 'attesa_assistenza']].head())
    return df

def converti_sesso_in_booleano(df):
    """Converts the 'sesso' column to boolean values and positions 'sesso_bool' next to 'sesso'."""
    df['sesso_bool'] = df['sesso'].str.lower().replace({'male': 1, 'female': 0})

    index_sesso = df.columns.get_loc('sesso')
    cols = df.columns.tolist()
    cols.insert(index_sesso + 1, cols.pop(cols.index('sesso_bool')))
    df = df[cols]

    print(df[['sesso', 'sesso_bool']].head())
    return df

def reinsert_missing_codes(df):
    """Re-inserts missing codes for specific provinces and municipalities."""
    df.loc[df['provincia_residenza'] == 'Napoli', 'codice_provincia_residenza'] = 'NA'
    df.loc[df['provincia_erogazione'] == 'Napoli', 'codice_provincia_erogazione'] = 'NA'
    df.loc[df['comune_residenza'].isnull(), 'comune_residenza'] = 'Comune di None'
    return df

def unify_codes(df):
    """Unifies different structure codes referring to the same structure into one single code."""
    df['codice_struttura_erogazione'] = df['codice_struttura_erogazione'].astype(str).str.split('.').str[0]
    return df

def get_unique_province_for_presidio_ospedaliero(df):
    """Gets unique provinces for 'PRESIDIO OSPEDALIERO UNIFICATO' structures."""
    filtered_df = df[df['struttura_erogazione'] == 'PRESIDIO OSPEDALIERO UNIFICATO']
    province_per_codice = filtered_df.groupby('codice_struttura_erogazione')['provincia_erogazione'].unique()
    result = province_per_codice.to_dict()
    return result

def modify_structure_name(df):
    """Modifies the name of structures with the same name but different codes by adding their province."""
    df.loc[df['codice_struttura_erogazione'] == '70001', 'struttura_erogazione'] = 'PRESIDIO OSPEDALIERO UNIFICATO (Imperia)'
    df.loc[df['codice_struttura_erogazione'] == '100803', 'struttura_erogazione'] = 'PRESIDIO OSPEDALIERO UNIFICATO (Perugia)'
    return df

def crea_dizionario_comuni_per_regione(df):
    """Creates a dictionary with regions as keys and resident municipalities as values."""
    dizionario_comuni_per_regione = {}
    for regione in df['regione_residenza'].unique():
        comuni = df.loc[df['regione_residenza'] == regione, 'comune_residenza'].unique()
        dizionario_comuni_per_regione[regione] = comuni
    return dizionario_comuni_per_regione

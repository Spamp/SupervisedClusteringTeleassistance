import pandas as pd
import json

# --- Data Loading and Initial Exploration ---

def read_file_parquet(filepath):
    """
    Reads a Parquet file into a pandas DataFrame and displays initial information.
    """
    df = pd.read_parquet(filepath)
    print("First few rows of the DataFrame:")
    print(df.head())
    print(f"Total number of records: {len(df)}")
    return df

def count_nulls(dataframe):
    """
    Counts null and NaN values in each column of the DataFrame.
    """
    null_counts = dataframe.isnull().sum()
    print("Count of null and NaN values per column:")
    print(null_counts)
    return null_counts

def check_duplicates(dataframe, column_name):
    """
    Checks for duplicates in the specified column and returns the count of unique values.
    """
    unique_values = dataframe[column_name].nunique()
    print(f"Number of unique IDs in column '{column_name}': {unique_values}")
    return unique_values

def sort_chronologically_by_timestamp(dataframe):
    """
    Sorts the DataFrame chronologically based on the 'data_erogazione' column.
    """
    dataframe['data_erogazione'] = pd.to_datetime(dataframe['data_erogazione'], errors='coerce')
    sorted_dataframe = dataframe.sort_values(by='data_erogazione')
    print("First few rows of the DataFrame sorted chronologically:")
    print(sorted_dataframe.head())
    return sorted_dataframe

# --- Data Cleaning and Preprocessing ---

def reinsert_missing_codes(df):
    """
    Re-inserts missing codes for specific provinces and communes.
    """
    df.loc[df['provincia_residenza'] == 'Napoli', 'codice_provincia_residenza'] = 'NA'
    df.loc[df['provincia_erogazione'] == 'Napoli', 'codice_provincia_erogazione'] = 'NA'
    df.loc[df['codice_comune_residenza'] == 1168, 'comune_residenza'] = 'Comune di None'
    return df

def unify_codes(df):
    """
    Unifies structure codes referring to the same structure by extracting the main code.
    """
    df['codice_struttura_erogazione'] = df['codice_struttura_erogazione'].astype(str).str.split('.').str[0]
    return df

def get_unique_province_for_presidio_ospedaliero(df):
    """
    Retrieves unique provinces for 'PRESIDIO OSPEDALIERO UNIFICATO' structures.
    """
    filtered_df = df[df['struttura_erogazione'] == 'PRESIDIO OSPEDALIERO UNIFICATO']
    province_per_codice = filtered_df.groupby('codice_struttura_erogazione')['provincia_erogazione'].unique()
    return province_per_codice.to_dict()

def modify_structure_name(df):
    """
    Modifies the name of the structure by adding the province they belong to.
    """
    mapping = {
        '70001': 'PRESIDIO OSPEDALIERO UNIFICATO (Imperia)',
        '100803': 'PRESIDIO OSPEDALIERO UNIFICATO (Perugia)'
    }
    df['struttura_erogazione'] = df['codice_struttura_erogazione'].map(mapping).fillna(df['struttura_erogazione'])
    return df

def mappa_colonne(df, coppie_colonne):
    """
    Efficiently maps codes to names and vice versa for specified column pairs without redundancy.
    """
    mappatura_codice_to_nome = {}
    mappatura_nome_to_codice = {}
    valori_nulli = set()
    nomi_multipli = {}

    for nome_colonna, codice_colonna in coppie_colonne:
        df_subset = df[[nome_colonna, codice_colonna]]
        # Identify null values in names or codes
        null_names = df_subset[df_subset[nome_colonna].isnull() & df_subset[codice_colonna].notnull()][codice_colonna].unique()
        null_codes = df_subset[df_subset[codice_colonna].isnull() & df_subset[nome_colonna].notnull()][nome_colonna].unique()
        for code in null_names:
            valori_nulli.add(f"In column '{nome_colonna}', code '{code}' has no name.")
        for name in null_codes:
            valori_nulli.add(f"In column '{codice_colonna}', name '{name}' has no code.")

        # Drop rows with null values in either column
        df_subset = df_subset.dropna(subset=[nome_colonna, codice_colonna])

        # Create mappings
        code_to_name = df_subset.drop_duplicates(subset=[codice_colonna]).set_index(codice_colonna)[nome_colonna].to_dict()
        name_to_codes_series = df_subset.groupby(nome_colonna)[codice_colonna].nunique()

        # Identify names associated with multiple codes
        multiple_codes = name_to_codes_series[name_to_codes_series > 1].index.tolist()
        if multiple_codes:
            nomi_multipli[nome_colonna] = df_subset[df_subset[nome_colonna].isin(multiple_codes)].groupby(nome_colonna)[codice_colonna].unique().to_dict()

        # Store mappings
        mappatura_codice_to_nome[nome_colonna] = code_to_name
        name_to_codes = df_subset.groupby(nome_colonna)[codice_colonna].unique().to_dict()
        mappatura_nome_to_codice[nome_colonna] = name_to_codes

    return mappatura_codice_to_nome, mappatura_nome_to_codice, list(valori_nulli), nomi_multipli

def print_nulls_and_multiple_codes(valori_nulli, nomi_multipli):
    """
    Prints information about null values and names associated with multiple codes.
    """
    print("Null values:", valori_nulli)
    if nomi_multipli:
        print(f"{len(nomi_multipli)} columns have names associated with more than one code:")
        for nome_colonna, nomi in nomi_multipli.items():
            for nome, codici in nomi.items():
                print(f"In column '{nome_colonna}', the name '{nome}' is associated with codes: {list(codici)}")
    else:
        print("No names are associated with more than one code.")

def visualizza_dizionario_chiavi(dizionario, nome_dizionario, limite=10):
    """
    Displays keys and their corresponding values from a dictionary.
    """
    print(f"Displaying dictionary {nome_dizionario} (limit {limite} keys):")
    for i, (chiave, valore) in enumerate(dizionario.items()):
        if i < limite:
            if isinstance(valore, list):
                print(f"Key: {chiave}, Number of elements: {len(valore)}")
            else:
                print(f"Key: {chiave}, Value: {valore}")
        else:
            break
    print(f"Total number of keys in {nome_dizionario}: {len(dizionario)}\n")

def visualizza_sub_chiavi(dizionario_mappature):
    """
    Displays the number of sub-keys for each macro key in the mapping dictionary.
    """
    for macro_chiave, sub_chiavi in dizionario_mappature.items():
        num_sub_chiavi = len(sub_chiavi)
        print(f"Macro key: {macro_chiave} has {num_sub_chiavi} sub-keys.")

def conta_asl_differenti(df):
    """
    Counts records where 'asl_residenza' and 'asl_erogazione' are different.
    """
    record_differenti = (df['asl_residenza'] != df['asl_erogazione']).sum()
    print(f"Number of records with different 'asl_residenza' and 'asl_erogazione': {record_differenti}")
    return record_differenti

def diagnostica_date(df):
    """
    Diagnoses issues with date conversions in 'data_nascita' and 'data_erogazione'.
    """
    df['data_nascita'] = pd.to_datetime(df['data_nascita'], errors='coerce')
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], errors='coerce')
    # Identify unconvertible dates
    date_nascita_non_convertibili = df[df['data_nascita'].isna()]
    date_erogazione_non_convertibili = df[df['data_erogazione'].isna()]
    if not date_nascita_non_convertibili.empty:
        print("Unconvertible values in 'data_nascita':")
        print(date_nascita_non_convertibili[['id_paziente', 'data_nascita']])
    if not date_erogazione_non_convertibili.empty:
        print("Unconvertible values in 'data_erogazione':")
        print(date_erogazione_non_convertibili[['id_paziente', 'data_erogazione']])
    return df

def calcola_eta_e_posiziona_vettorizzato(df):
    """
    Calculates age based on 'data_nascita' and 'data_erogazione', and positions the 'età' column appropriately.
    """
    # Remove rows with NaT in dates
    df = df.dropna(subset=['data_nascita', 'data_erogazione'])
    # Calculate age
    df['età'] = df['data_erogazione'].dt.year - df['data_nascita'].dt.year
    # Adjust age if birthday hasn't occurred yet
    birthday_passed = (df['data_erogazione'].dt.month > df['data_nascita'].dt.month) | \
                      ((df['data_erogazione'].dt.month == df['data_nascita'].dt.month) & \
                       (df['data_erogazione'].dt.day >= df['data_nascita'].dt.day))
    df.loc[~birthday_passed, 'età'] -= 1
    # Drop 'data_nascita' column
    df.drop(columns=['data_nascita'], inplace=True)
    print(df[['id_paziente', 'data_erogazione', 'età']].head())
    return df

def calcola_e_correggi_durata_assistenza(df):
    """
    Calculates the duration of assistance and handles special cases like cancellations.
    """
    df['ora_inizio_erogazione'] = pd.to_datetime(df['ora_inizio_erogazione'], errors='coerce')
    df['ora_fine_erogazione'] = pd.to_datetime(df['ora_fine_erogazione'], errors='coerce')

    # Calculate duration where both times are available
    mask_valid_times = df['ora_inizio_erogazione'].notnull() & df['ora_fine_erogazione'].notnull()
    df.loc[mask_valid_times, 'durata_assistenza'] = (df['ora_fine_erogazione'] - df['ora_inizio_erogazione']).dt.total_seconds() / 3600

    # Set duration to 0 for cancellations without start/end times
    mask_cancellations = df['data_disdetta'].notnull()
    df.loc[mask_cancellations, 'durata_assistenza'] = 0

    # Drop rows where 'durata_assistenza' is still NaN
    df = df.dropna(subset=['durata_assistenza'])

    df.drop(columns=['ora_inizio_erogazione', 'ora_fine_erogazione'], inplace=True)
    nan_in_durata = df['durata_assistenza'].isna().sum()
    print(f"Number of NaN values in 'durata_assistenza' after cleaning: {nan_in_durata}")
    return df

def converti_sesso_in_booleano(df):
    """
    Converts the 'sesso' column into a boolean column 'sesso_bool'.
    """
    df['sesso_bool'] = df['sesso'].str.lower().map({'male': 1, 'female': 0})
    print(df[['sesso', 'sesso_bool']].head())
    return df

# --- Data Transformation and Feature Engineering ---

def crea_dizionario_comuni_per_regione(df):
    """
    Creates a dictionary with regions as keys and arrays of resident communes as values.
    """
    dict_comuni_per_regione = df.groupby('regione_residenza')['comune_residenza'].unique().to_dict()
    return dict_comuni_per_regione

def calcola_attesa_assistenza(df):
    """
    Calculates the waiting time for assistance in days.
    """
    df['data_contatto'] = pd.to_datetime(df['data_contatto'], utc=True, errors='coerce')
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'], utc=True, errors='coerce')
    df['attesa_assistenza'] = (df['data_erogazione'] - df['data_contatto']).dt.days
    print(df[['data_contatto', 'data_erogazione', 'attesa_assistenza']].head())
    return df

def load_coordinates(file_path):
    """
    Loads a JSON file containing coordinates and returns it as a dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")

def replace_city_columns_with_coordinates(df, city_columns, coordinates_dict):
    """
    Replaces city names in specified columns with their corresponding coordinates.
    """
    for city_column in city_columns:
        if city_column not in df.columns:
            raise KeyError(f"The column '{city_column}' was not found in the DataFrame.")
    # Prepare mappings for latitude and longitude
    lat_dict = {city.lower(): coord[0] for city, coord in coordinates_dict.items()}
    lng_dict = {city.lower(): coord[1] for city, coord in coordinates_dict.items()}
    # Standardize city names to lowercase
    for city_column in city_columns:
        df[city_column] = df[city_column].str.lower()
        df[city_column + '_lat'] = df[city_column].map(lat_dict)
        df[city_column + '_lng'] = df[city_column].map(lng_dict)
    return df

# --- Main Execution ---

if __name__ == "__main__":
    filepath = "./challenge_campus_biomedico_2024.parquet"

    # Load the DataFrame
    df = read_file_parquet(filepath)

    # Initial Data Exploration
    count_nulls(df)
    check_duplicates(df, 'id_paziente')

    # Data Cleaning
    df = reinsert_missing_codes(df)
    df = unify_codes(df)
    unique_provinces = get_unique_province_for_presidio_ospedaliero(df)
    print("Unique provinces for 'PRESIDIO OSPEDALIERO UNIFICATO':", unique_provinces)
    df = modify_structure_name(df)

    # Mapping Codes to Names
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

    # Print Nulls and Multiple Codes
    print_nulls_and_multiple_codes(valori_nulli, nomi_multipli)

    # Create Dictionary of Communes per Region
    dict_regioni = crea_dizionario_comuni_per_regione(df)

    # Visualization
    visualizza_dizionario_chiavi(mappatura_c_to_n['asl_residenza'], 'asl_residenza')
    visualizza_dizionario_chiavi(mappatura_c_to_n['asl_erogazione'], 'asl_erogazione')
    visualizza_sub_chiavi(mappatura_c_to_n)

    # Data Analysis
    conta_asl_differenti(df)
    df = sort_chronologically_by_timestamp(df)
    df = diagnostica_date(df)
    df = calcola_eta_e_posiziona_vettorizzato(df)
    df = calcola_e_correggi_durata_assistenza(df)
    df = converti_sesso_in_booleano(df)
    df = calcola_attesa_assistenza(df)

    # Add Coordinates
    coordinates_file = './coordinate_dataset.json'
    coordinates_dict = load_coordinates(coordinates_file)
    city_columns = ['provincia_residenza', 'comune_residenza', 'provincia_erogazione']
    df = replace_city_columns_with_coordinates(df, city_columns, coordinates_dict)

    # Final DataFrame Preview
    print(df.head())
    print(df.columns)

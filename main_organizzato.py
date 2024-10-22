# main.py

import pandas as pd
from utility_functions import read_file_parquet, count_nulls, check_duplicates, salva_dataframe
from data_exploration import sort_chronologically_by_timestamp, visualizza_dizionario_chiavi, visualizza_sub_chiavi, conta_asl_differenti
from data_preprocessing import (mappa_colonne, diagnostica_date, calcola_eta_e_posiziona_vettorizzato,
                                calcola_e_correggi_durata_assistenza, calcola_attesa_assistenza,
                                converti_sesso_in_booleano, reinsert_missing_codes, unify_codes,
                                get_unique_province_for_presidio_ospedaliero, modify_structure_name,
                                crea_dizionario_comuni_per_regione)
from feature_engineering import (load_coordinates, replace_city_columns_with_coordinates,
                                 conta_nan_colonna, crea_colonna_semestre, crea_colonna_quadrimestre,
                                 calcola_incremento_teleassistenze, classifica_incremento_per_periodo,
                                 etichetta_incremento_per_record)


filepath = "./pija_la_posizione/challenge_campus_biomedico_2024.parquet"

# Read the Parquet file
df = read_file_parquet(filepath)

# Count null and NaN values
count_nulls(df)

# Check duplicates in 'id_paziente'
check_duplicates(df)

# Reinsert missing codes
df = reinsert_missing_codes(df)

# Unify structure codes
df = unify_codes(df)

# Get unique provinces for 'PRESIDIO OSPEDALIERO UNIFICATO'
unique_provinces = get_unique_province_for_presidio_ospedaliero(df)
print(unique_provinces)

# Modify structure names
df = modify_structure_name(df)

# Column mappings
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

# Create dictionary mapping cities to regions
dict_regioni = crea_dizionario_comuni_per_regione(df)

# Output mappings and null values
print("Valori nulli:", valori_nulli)

# Message about names associated with multiple codes
if nomi_multipli:
    print(f"{len(nomi_multipli)} nomi sono associati a più di un codice:")
    for nome, codici in nomi_multipli.items():
        print(f"Il nome '{nome}' è associato ai codici: {codici}")
else:
    print("Nessun nome è associato a più di un codice.")

# Display samples of ASL residence and delivery
visualizza_dizionario_chiavi(mappatura_c_to_n['asl_residenza'], 'asl_residenza')
visualizza_dizionario_chiavi(mappatura_c_to_n['asl_erogazione'], 'asl_erogazione')

visualizza_sub_chiavi(mappatura_c_to_n)

# Check records with different ASL
conta_asl_differenti(df)

# Sort DataFrame chronologically
df = sort_chronologically_by_timestamp(df)

# Diagnose dates
df = diagnostica_date(df)

# Calculate age
df = calcola_eta_e_posiziona_vettorizzato(df)

# Calculate duration of assistance
df = calcola_e_correggi_durata_assistenza(df)

# Convert 'sesso' to boolean
df = converti_sesso_in_booleano(df)

# Calculate waiting time
df = calcola_attesa_assistenza(df)

# Load coordinates
coordinates_file = './pija_la_posizione/coordinate_dataset.json'
coordinates_dict = load_coordinates(coordinates_file)

# Replace city names with coordinates
city_columns = ['provincia_residenza', 'comune_residenza', 'provincia_erogazione']
df = replace_city_columns_with_coordinates(df, city_columns, coordinates_dict)

print(df.columns)

# Check NaN values in coordinate columns
conta_nan_colonna(df, 'provincia_erogazione_lng')
conta_nan_colonna(df, 'provincia_erogazione_lat')

# Create 'semestre' and 'quadrimestre' columns
df = crea_colonna_semestre(df)
df = crea_colonna_quadrimestre(df)

# Check NaN values in period columns
conta_nan_colonna(df, 'semestre')
conta_nan_colonna(df, 'quadrimestre')

# Calculate percentage increase for 'semestre'
df_incremento_semestre = calcola_incremento_teleassistenze(df, 'semestre', 'incremento_semestre')

# Classify increase for each 'semestre'
df_incremento_semestre = classifica_incremento_per_periodo(df_incremento_semestre, 'semestre', 'incremento_semestre', 'classificazione_semestre')

# Calculate percentage increase for 'quadrimestre'
df_incremento_quadrimestre = calcola_incremento_teleassistenze(df, 'quadrimestre', 'incremento_quadrimestre')

# Classify increase for each 'quadrimestre'
df_incremento_quadrimestre = classifica_incremento_per_periodo(df_incremento_quadrimestre, 'quadrimestre', 'incremento_quadrimestre', 'classificazione_quadrimestre')

# Label increment for records in the main DataFrame
df = etichetta_incremento_per_record(df, df_incremento_semestre, 'semestre', 'classificazione_semestre', 'incremento_teleassistenza_semestre')
df = etichetta_incremento_per_record(df, df_incremento_quadrimestre, 'quadrimestre', 'classificazione_quadrimestre', 'incremento_teleassistenza_quadrimestre')

# Select features
selected_columns = [
    'id_prenotazione', 'asl_residenza', 'descrizione_attivita',
    'asl_erogazione', 'struttura_erogazione',
    'tipologia_struttura_erogazione', 'regione_erogazione', 'tipologia_professionista_sanitario',
    'data_erogazione', 'età', 'durata_assistenza', 'sesso_bool', 'attesa_assistenza',
    'provincia_residenza_lat', 'provincia_residenza_lng', 'comune_residenza_lat',
    'comune_residenza_lng', 'provincia_erogazione_lat', 'provincia_erogazione_lng',
    'semestre', 'incremento_teleassistenza_semestre',
    'quadrimestre', 'incremento_teleassistenza_quadrimestre'
]

df_feature_selezionate = df[selected_columns]

# Save the cleaned DataFrame
path_dataset_pulito = './dataset_pulito.parquet'
salva_dataframe(df_feature_selezionate, path_dataset_pulito)

# Display the first 5 results
print(df_feature_selezionate.head())
# Display the last 5 results
print(df_feature_selezionate.tail())

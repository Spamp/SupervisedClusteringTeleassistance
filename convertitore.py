import pandas as pd

# Leggi il file Parquet
df = pd.read_parquet("C:/Users/antos/Desktop/convertitore_csv/challenge_campus_biomedico_2024.parquet")

# Verifica la struttura del DataFrame per assicurarti che i dati siano stati letti correttamente
print("Ecco le prime righe del DataFrame:")
print(df.head())

print("\nEcco i nomi delle colonne del DataFrame:")
print(df.columns)

# Assicurati che il DataFrame non contenga colonne nidificate o dati non standard
if any(df.columns.str.contains(',')):
    print("\nAttenzione: Alcune colonne contengono virgole nei loro nomi!")

# Scrivi il DataFrame in un file CSV con i dati separati da virgole

# Scrivi il DataFrame in un file CSV
df.to_csv('C:/Users/antos/Desktop/convertitore_csv/challenge_campus_biomedico_2024.csv', index=False, sep=',')


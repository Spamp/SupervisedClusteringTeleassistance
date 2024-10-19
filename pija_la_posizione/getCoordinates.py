import time
import json
import pandas as pd
import ssl
import os
import certifi
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Configura SSL per usare i certificati di certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

# Funzione per ottenere le coordinate geografiche di un comune
def ottieni_coordinate_geografiche(comune, tentativi=3):
    geolocator = Nominatim(user_agent="geoapiExercises")
    
    for _ in range(tentativi):  # Tentativi per gestire i timeout
        try:
            location = geolocator.geocode(comune, timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except GeocoderTimedOut:
            time.sleep(1)  # Aspetta 1 secondo in caso di timeout e riprova
        except GeocoderServiceError as e:
            print(f"Errore di servizio per il comune {comune}: {e}")
            return None
    return None  # Restituisce None se il comune non viene trovato o se ci sono troppi timeout

# Funzione per caricare il dizionario da file, se esiste
def carica_dizionario_da_file(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}  # Restituisce un dizionario vuoto se il file non esiste

# Funzione per salvare il dizionario su file
def salva_dizionario_su_file(dizionario, file_path):
    with open(file_path, 'w') as file:
        json.dump(dizionario, file)

# Funzione per creare un dizionario di comuni e le loro coordinate usando insiemi
def crea_dizionario_coordinate(df, file_path='dizionario_coordinate.json'):
    # Carica il dizionario esistente, se presente
    dizionario_coordinate = carica_dizionario_da_file(file_path)
    
    # Estrai i comuni unici dal dataset
    comuni_unici = set(df['comune_residenza'].unique())
    
    # Estrai i comuni già presenti nel dizionario
    comuni_gia_presenti = set(dizionario_coordinate.keys())
    
    # Comuni mancanti (quelli che non hanno ancora coordinate)
    comuni_mancanti = comuni_unici - comuni_gia_presenti
    
    # Itera solo sui comuni che non hanno già le coordinate
    for comune in comuni_mancanti:
        coordinate = ottieni_coordinate_geografiche(comune)
        if coordinate:
            dizionario_coordinate[comune] = coordinate
            print(f"Coordinate trovate per {comune}: {coordinate}")
        else:
            print(f"Coordinate non trovate per {comune}")
        
        # Salva il dizionario aggiornato su file dopo ogni comune processato
        salva_dizionario_su_file(dizionario_coordinate, file_path)
        
        # Pausa tra le richieste per evitare di sovraccaricare il servizio (puoi regolare questo valore)
        time.sleep(6)  # Aumentato il tempo a 5 secondi per sicurezza
    
    return dizionario_coordinate

# Funzione per caricare il dataset (CSV o Parquet)
def carica_dataset(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Formato di file non supportato. Usa .csv o .parquet")
    return df

# Esempio di utilizzo
file_path_dataset = '/Users/edoardocaliano/Desktop/SupervisedClusteringTeleassistance/challenge_campus_biomedico_2024.parquet'  # Sostituisci con il percorso corretto del tuo file
df = carica_dataset(file_path_dataset)  # Carica il dataset
dizionario_coordinate = crea_dizionario_coordinate(df)  # Crea il dizionario delle coordinate
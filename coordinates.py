import time
import json
import pandas as pd
import os
import certifi
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Funzione per caricare il file JSON con le coordinate
def carica_coordinate_json(file_json):
    with open(file_json, 'r') as f:
        dati_json = json.load(f)
    
    # Crea un dizionario dove la chiave è il nome del comune e i valori sono le coordinate
    dizionario_coordinate = {}
    for item in dati_json:
        comune = item.get('comune')
        lat = item.get('lat')
        lng = item.get('lng')
        
        # Verifica che le coordinate esistano per questo comune
        if comune and lat and lng:
            dizionario_coordinate[comune] = (float(lat), float(lng))
        else:
            print(f"Coordinate mancanti per il comune: {comune}")

    return dizionario_coordinate

# Funzione per ottenere le coordinate geografiche di un comune con geopy e certifi
def ottieni_coordinate_geografiche(comune, tentativi=3):
    geolocator = Nominatim(user_agent="geoapiExercises")

    # Imposta il certificato SSL usando certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()

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

# Funzione per salvare il dizionario su file
def salva_dizionario_su_file(dizionario, file_path):
    with open(file_path, 'w') as file:
        json.dump(dizionario, file, indent=4)

# Funzione per aggiornare il dizionario con i comuni mancanti usando geopy
def aggiorna_coordinate_comuni_mancanti(comuni_mancanti, dizionario_coordinate, file_json):
    for comune in comuni_mancanti:
        if comune not in dizionario_coordinate:
            coordinate = ottieni_coordinate_geografiche(comune)
            if coordinate:
                dizionario_coordinate[comune] = coordinate
                print(f"Coordinate trovate per {comune}: {coordinate}")
            else:
                print(f"Coordinate non trovate per {comune}")
            
            # Salva il dizionario aggiornato su file dopo ogni nuovo comune
            salva_dizionario_su_file(dizionario_coordinate, file_json)
            
            # Pausa tra le richieste per evitare di sovraccaricare il servizio
            time.sleep(2)  # Pausa di 2 secondi tra le richieste per evitare blocchi

# Funzione per creare il dizionario delle coordinate dal dataset e gestire i mancanti con geopy
def crea_dizionario_coordinate_comuni(df, file_json):
    # Carica il dizionario delle coordinate dal file JSON
    dizionario_coordinate_json = carica_coordinate_json(file_json)
    
    # Estrai i comuni unici dal dataset
    comuni_unici = df['comune_residenza'].unique()
    
    # Dizionario finale con i comuni e le coordinate trovate
    dizionario_coordinate = {}
    
    # Lista per i comuni senza coordinate
    comuni_senza_coordinate = []
    
    # Itera sui comuni del dataset
    for comune in comuni_unici:
        if comune in dizionario_coordinate_json:
            dizionario_coordinate[comune] = dizionario_coordinate_json[comune]
        else:
            comuni_senza_coordinate.append(comune)
    
    # Stampa il numero e i nomi dei comuni senza coordinate
    if comuni_senza_coordinate:
        print(f"Numero di comuni senza coordinate: {len(comuni_senza_coordinate)}")
        print("Comuni senza coordinate:", comuni_senza_coordinate)

        # Aggiorna il dizionario con i comuni mancanti usando geopy
        aggiorna_coordinate_comuni_mancanti(comuni_senza_coordinate, dizionario_coordinate, file_json)
    
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
file_json = './italy_geo.json'  # Sostituisci con il percorso corretto del file JSON
file_parquet = './challenge_campus_biomedico_2024.parquet'  # Sostituisci con il percorso corretto del file Parquet

# Carica il dataset (in questo caso, un file Parquet)
df = carica_dataset(file_parquet)

# Crea il dizionario dei comuni e delle loro coordinate
dizionario_coordinate_comuni = crea_dizionario_coordinate_comuni(df, file_json)
print(dizionario_coordinate_comuni)
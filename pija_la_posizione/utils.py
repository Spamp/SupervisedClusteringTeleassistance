import os
import json
import certifi
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Set SSL environment variable for geopy
os.environ['SSL_CERT_FILE'] = certifi.where()

# Function to load dataset (CSV or Parquet)
def load_dataset(file_path):
    """Loads the dataset as a DataFrame from CSV or Parquet file."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

# Function to load JSON data from file
def load_json(file_path):
    """Loads JSON data from a file and returns as dictionary."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Function to save dictionary as JSON
def save_json(data, file_path):
    """Saves a dictionary to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# Function to fetch coordinates using geopy
def fetch_coordinates(comune, attempts=5):
    """Fetches geographic coordinates for a given city using geopy."""
    geolocator = Nominatim(user_agent="geoapiExercises")
    for attempt in range(attempts):
        try:
            location = geolocator.geocode(comune, timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except GeocoderTimedOut:
            print(f"Timeout for {comune}, retrying... ({attempt + 1}/{attempts})")
            time.sleep(2)  # Wait and retry in case of timeout
        except GeocoderServiceError as e:
            print(f"Service error for {comune}: {e}")
    return None

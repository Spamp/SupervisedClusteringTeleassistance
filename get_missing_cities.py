import time
import json
import os
import certifi
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Load the existing coordinate dictionary
def load_existing_coordinates(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save the updated dictionary of coordinates
def save_coordinates_to_file(coordinates, file_path):
    with open(file_path, 'w') as f:
        json.dump(coordinates, f, indent=4)

# Get geographic coordinates using geopy
def get_geopy_coordinates(comune, attempts=3):
    geolocator = Nominatim(user_agent="geoapiExercises")
    os.environ['SSL_CERT_FILE'] = certifi.where()
    
    for _ in range(attempts):
        try:
            location = geolocator.geocode(comune, timeout=10)
            if location:
                return (location.latitude, location.longitude)
        except GeocoderTimedOut:
            time.sleep(1)
        except GeocoderServiceError as e:
            print(f"Service error for {comune}: {e}")
            return None
    return None

# List of missing cities
missing_cities = ['Nardò', 'Valdilana', "Città Sant'Angelo", 'Corigliano-Rossano', 
                  'Gattico-Veruno', 'Paternò', 'Alì', 'Almè', 'Canicattì', 
                  'San Donà di Piave', 'Zerbolò', 'Città di Castello', 
                  'Merì', 'Codognè', 'Fossò', 'Cefalù']

# File path for the existing coordinates
coordinate_file = './coordinate_vere.json'

# Load existing coordinates
existing_coordinates = load_existing_coordinates(coordinate_file)

# Search for coordinates of missing cities and update the dictionary
for city in missing_cities:
    if city not in existing_coordinates:
        coordinates = get_geopy_coordinates(city)
        if coordinates:
            existing_coordinates[city] = coordinates
            print(f"Found coordinates for {city}: {coordinates}")
        else:
            print(f"Coordinates not found for {city}")
        # Adding a delay to avoid overloading the geopy service
        time.sleep(1)

# Save the updated dictionary back to the JSON file
save_coordinates_to_file(existing_coordinates, coordinate_file)

print(f"Updated coordinates for missing cities saved to {coordinate_file}")

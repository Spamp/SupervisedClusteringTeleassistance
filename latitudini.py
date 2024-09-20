import time
import json
import pandas as pd
import os
import certifi
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Load dataset (CSV or Parquet)
def load_dataset(file_path):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Use .csv or .parquet")

# Load JSON coordinates (italy_geo.json format)
def load_json_coordinates(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    coordinates_dict = {}
    for item in data:
        comune = item.get('comune')
        lat = item.get('lat')
        lng = item.get('lng')
        
        # Only add valid entries where lat and lng are not empty and convertible to floats
        if comune and lat and lng:
            try:
                coordinates_dict[comune] = (float(lat), float(lng))
            except ValueError:
                print(f"Skipping invalid coordinates for {comune}: lat={lat}, lng={lng}")
        else:
            print(f"Skipping entry with missing data for {comune}: lat={lat}, lng={lng}")
    
    return coordinates_dict

# Load coordinates from dizionario_coordinate.json
def load_dizionario_coordinates(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save the final dictionary of coordinates
def save_coordinates_to_file(coordinates, file_path):
    with open(file_path, 'w') as f:
        json.dump(coordinates, f, indent=4)

# Get geographic coordinates using geopy (fallback)
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

# Main function to integrate all steps
def create_coordinate_dict(df, italy_geo_file, dizionario_file, output_file):
    # Load JSON coordinates from italy_geo.json
    italy_coordinates = load_json_coordinates(italy_geo_file)
    
    # Load additional coordinates from dizionario_coordinate.json
    dizionario_coordinates = load_dizionario_coordinates(dizionario_file)

    # Extract unique cities from the dataset
    unique_cities = df['comune_residenza'].unique()
    
    # Prepare final coordinates dictionary
    final_coordinates = {}
    cities_not_found = []

    # Check for each city in the dataset
    for city in unique_cities:
        if city in italy_coordinates:
            final_coordinates[city] = italy_coordinates[city]
        elif city in dizionario_coordinates:
            final_coordinates[city] = dizionario_coordinates[city]
        else:
            cities_not_found.append(city)
    
    # Process cities not found using geopy
    for city in cities_not_found[:]:  # Iterate over a copy of the list
        coordinates = get_geopy_coordinates(city)
        if coordinates:
            final_coordinates[city] = coordinates
            cities_not_found.remove(city)
        else:
            print(f"Coordinates not found for {city}")

    # Save final coordinates dictionary to file
    save_coordinates_to_file(final_coordinates, output_file)

    # Report cities not found
    print(f"Number of cities not found: {len(cities_not_found)}")
    if cities_not_found:
        print(f"Cities not found: {cities_not_found}")

    # Count and report how many cities were saved in coordinate_vere.json
    print(f"Number of cities saved in {output_file}: {len(final_coordinates)}")

# Example usage
dataset_file = './challenge_campus_biomedico_2024.parquet'  # Dataset path
italy_geo_file = './italy_geo.json'  # Italy geo JSON file
dizionario_file = './dizionario_coordinate.json'  # Coordinate dictionary
output_file = './coordinate_dataset.json'  # Output file for the final coordinates

# Load dataset
df = load_dataset(dataset_file)

# Create final dictionary with coordinates and process missing ones
create_coordinate_dict(df, italy_geo_file, dizionario_file, output_file)

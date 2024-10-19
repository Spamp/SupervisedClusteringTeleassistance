import pandas as pd
import json

# Load the dataset (Parquet file)
def load_dataset(file_path):
    if file_path.endswith('.parquet'):
        return pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Only .parquet is supported.")

# Load the coordinate dictionary from the JSON file
def load_coordinates(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Check for missing or invalid coordinates (None values) in the coordinate dictionary
def check_missing_or_none_coordinates(coordinates_dict):
    missing_coordinates = {}
    for city, coordinates in coordinates_dict.items():
        if not coordinates or coordinates == [None, None]:
            missing_coordinates[city] = coordinates
    return missing_coordinates

# Find which cities from the dataset are missing in the coordinate dictionary
def find_missing_coordinates(df, coordinates_dict, city_column='provincia_residenza'):
    if city_column not in df.columns:
        raise KeyError(f"The column '{city_column}' was not found in the dataset.")
    
    # Get the unique cities from the dataset
    unique_cities = df[city_column].unique()

    # Identify cities that are missing from the coordinates dictionary
    missing_cities = [city for city in unique_cities if city not in coordinates_dict]
    
    return missing_cities

# Count how many records correspond to the missing cities
def count_records_for_cities(df, cities_list, city_column='provincia_residenza'):
    # Filter the dataset for records where the city is in the list of cities missing coordinates
    city_matches = df[df[city_column].isin(cities_list)]
    
    # Count the number of records for each city
    city_counts = city_matches[city_column].value_counts()

    return city_counts

# File paths
dataset_file = './challenge_campus_biomedico_2024.parquet'
coordinates_file = './coordinate_dataset.json'

# Load the dataset
df = load_dataset(dataset_file)

# Load the existing coordinates
coordinates_dict = load_coordinates(coordinates_file)

# Check for cities with missing or None coordinates
missing_or_none_coordinates = check_missing_or_none_coordinates(coordinates_dict)

# Find cities that are missing coordinates in the dictionary
missing_cities = find_missing_coordinates(df, coordinates_dict)

# Count records for the cities that are missing coordinates
city_counts = count_records_for_cities(df, missing_cities)

# Print results for cities missing or having None coordinates
print(f"Cities with missing or None coordinates: {missing_or_none_coordinates}")
print(f"Total cities missing or having None coordinates: {len(missing_or_none_coordinates)}\n")

# Print results for cities not present in the coordinate dictionary
print(f"Cities not in the coordinate dictionary: {missing_cities}")
print(f"Total cities missing from dictionary: {len(missing_cities)}\n")

# Print number of records for each city missing from dictionary
print("Number of records for each city missing from the dictionary:")
print(city_counts)

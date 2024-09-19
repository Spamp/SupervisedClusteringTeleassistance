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

# Find which cities from the dataset are missing in the coordinate dictionary
def find_missing_coordinates(df, coordinates_dict, city_column='comune_residenza'):
    if city_column not in df.columns:
        raise KeyError(f"The column '{city_column}' was not found in the dataset.")
    
    # Get the unique cities from the dataset
    unique_cities = df[city_column].unique()

    # Identify cities that are missing from the coordinates dictionary
    missing_cities = [city for city in unique_cities if city not in coordinates_dict]
    
    return missing_cities

# Count how many records correspond to the missing cities
def count_records_for_cities(df, cities_list, city_column='comune_residenza'):
    # Filter the dataset for records where the city is in the list of cities missing coordinates
    city_matches = df[df[city_column].isin(cities_list)]
    
    # Count the number of records for each city
    city_counts = city_matches[city_column].value_counts()

    return city_counts

# File paths
dataset_file = './challenge_campus_biomedico_2024.parquet'
coordinates_file = './coordinate_vere.json'

# Load the dataset
df = load_dataset(dataset_file)

# Load the existing coordinates
coordinates_dict = load_coordinates(coordinates_file)

# Find cities that are missing coordinates
missing_cities = find_missing_coordinates(df, coordinates_dict)

# Count records for the cities that are missing coordinates
city_counts = count_records_for_cities(df, missing_cities)

# Print the results
print(f"Cities missing coordinates: {missing_cities}")
print(f"Total cities missing coordinates: {len(missing_cities)}\n")
print("Number of records for each city missing coordinates:")
print(city_counts)

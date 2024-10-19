import pandas as pd
from utils import load_dataset, load_json

def count_missing_cities(dataset_file, coordinates_file):
    """Counts the records for cities missing in the coordinates file."""
    
    df = load_dataset(dataset_file)
    coordinates_dict = load_json(coordinates_file)
    
    # Identify cities missing in the coordinates dictionary
    missing_cities = [city for city in df['comune_residenza'].unique() if city not in coordinates_dict]
    
    # Count occurrences of missing cities
    missing_city_counts = df[df['comune_residenza'].isin(missing_cities)]['comune_residenza'].value_counts()
    
    print(f"Cities missing coordinates: {missing_cities}")
    print(f"Record counts for missing cities:\n{missing_city_counts}")

# Example usage
dataset_file = './challenge_campus_biomedico_2024.parquet'
coordinates_file = './coordinate_dataset.json'
count_missing_cities(dataset_file, coordinates_file)

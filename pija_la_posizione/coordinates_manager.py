import time
from utils import load_dataset, load_json, save_json, fetch_coordinates

# Function to create and update the coordinates dictionary, allowing resume from previous runs
def create_coordinate_dict(df, italy_geo_file, output_file):
    """Creates or updates a dictionary of city coordinates, resuming if necessary."""
    
    # Load coordinates from Italy geo file and existing output file
    italy_geo_data = load_json(italy_geo_file)  # Load italy_geo.json which has the 'comune' field
    final_coordinates = load_json(output_file)  # Load existing progress if available
    
    # Convert italy_geo to a dictionary for faster lookup (using 'comune' as the key)
    italy_geo_dict = {}
    for item in italy_geo_data:
        comune = item.get('comune')
        lat = item.get('lat')
        lng = item.get('lng')
        
        # Only include entries with valid latitude and longitude
        if comune and lat and lng:
            try:
                italy_geo_dict[comune] = (float(lat), float(lng))
            except ValueError:
                print(f"Skipping invalid coordinates for {comune}: lat={lat}, lng={lng}")
    
    # Extract unique cities from 'provincia_residenza' and 'comune_residenza' columns
    unique_province_cities = df['provincia_residenza'].unique()  # Unique cities from provincia_residenza
    unique_comune_cities = df['comune_residenza'].unique()  # Unique cities from comune_residenza
    
    # Combine the unique cities from both columns without duplicates
    unique_cities = set(unique_province_cities).union(set(unique_comune_cities))
    
    # Identify missing cities (those not already in the final coordinates)
    missing_cities = [city for city in unique_cities if city not in final_coordinates]
    total_missing = len(missing_cities)
    
    # First, try to fill missing cities using italy_geo.json
    for city in missing_cities[:]:  # Iterate over a copy of the missing cities list
        if city in italy_geo_dict:
            final_coordinates[city] = italy_geo_dict[city]
            missing_cities.remove(city)
            print(f"Coordinates found in italy_geo for {city}: {italy_geo_dict[city]}")
    
    # Fetch remaining missing cities using geopy
    for city in missing_cities[:]:  # Iterate over a copy to modify during loop
        coordinates = fetch_coordinates(city)
        if coordinates:
            final_coordinates[city] = coordinates
            missing_cities.remove(city)
            print(f"Coordinates found for {city} via geopy: {coordinates}")
        else:
            print(f"Coordinates not found for {city}")
        
        # Save progress after each successful city fetch
        save_json(final_coordinates, output_file)
        time.sleep(1)  # Delay to avoid overloading the geopy service
    
    # Print the results
    cities_found = total_missing - len(missing_cities)
    print(f"Total missing cities to start: {total_missing}")
    print(f"Cities found (including italy_geo.json): {cities_found}")
    print(f"Cities still missing coordinates: {missing_cities}")

    return final_coordinates

# Example usage
dataset_file = './challenge_campus_biomedico_2024.parquet'
italy_geo_file = './italy_geo.json'
output_file = './coordinate_dataset.json'  # This is the final output file

# Load dataset and process coordinates
df = load_dataset(dataset_file)
create_coordinate_dict(df, italy_geo_file, output_file)

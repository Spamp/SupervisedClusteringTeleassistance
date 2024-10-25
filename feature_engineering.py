

import pandas as pd
import numpy as np
import json

def load_coordinates(file_path):
    """Loads a dictionary of coordinates from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")

def replace_city_columns_with_coordinates(df, city_columns, coordinates_dict):
    """Replaces city names with their coordinates in specified columns."""
    for city_column in city_columns:
        if city_column not in df.columns:
            raise KeyError(f"The column '{city_column}' was not found in the DataFrame.")

    lat_dict = {city: coord[0] for city, coord in coordinates_dict.items()}
    lng_dict = {city: coord[1] for city, coord in coordinates_dict.items()}

    for city_column in city_columns:
        df[city_column + '_lat'] = df[city_column].map(lat_dict)
        df[city_column + '_lng'] = df[city_column].map(lng_dict)

    return df

def conta_nan_colonna(df, colonna):
    """Counts NaN values in a specified column."""
    nan_count = df[colonna].isna().sum()
    print(f"Numero di valori NaN nella colonna '{colonna}': {nan_count}")
    return nan_count

def crea_colonna_semestre(df):
    """Creates a 'semestre' column based on 'data_erogazione'."""
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'])
    df['anno'] = df['data_erogazione'].dt.year
    df['semestre'] = np.where(df['data_erogazione'].dt.month <= 6, 'S1', 'S2')
    df['semestre'] = df['anno'].astype(str) + '-' + df['semestre']

    print(df[['data_erogazione', 'anno', 'semestre']].head())
    print(df[['data_erogazione', 'anno', 'semestre']].tail())
    return df

def crea_colonna_quadrimestre(df):
    """Creates a 'quadrimestre' column based on 'data_erogazione'."""
    df['data_erogazione'] = pd.to_datetime(df['data_erogazione'])
    df['anno'] = df['data_erogazione'].dt.year

    condizioni = [
        (df['data_erogazione'].dt.month <= 4),
        (df['data_erogazione'].dt.month > 4) & (df['data_erogazione'].dt.month <= 8),
        (df['data_erogazione'].dt.month > 8)
    ]
    scelte = ['Q1', 'Q2', 'Q3']
    df['quadrimestre'] = np.select(condizioni, scelte, default='N/A')
    df['quadrimestre'] = df['anno'].astype(str) + '-' + df['quadrimestre']

    print(df[['data_erogazione', 'anno', 'quadrimestre']].head())
    print(df[['data_erogazione', 'anno', 'quadrimestre']].tail())
    return df

def calcola_incremento_teleassistenze(df, periodo_colonna, incremento_colonna):
    """Calculates the percentage increase of teleassistenze over specified periods."""
    grouped = df.groupby(periodo_colonna).size().reset_index(name='num_teleassistenze')
    grouped[incremento_colonna] = grouped['num_teleassistenze'].pct_change() * 100
    grouped[incremento_colonna] = grouped[incremento_colonna].fillna(0)

    print(grouped.head())
    print(grouped.tail())
    return grouped

def classifica_incremento_per_periodo(df_grouped, periodo_colonna, incremento_colonna, classificazione_colonna):
    """Classifies the percentage increase over periods into categories."""
    df_grouped[classificazione_colonna] = 'Stabile'
    df_grouped.loc[df_grouped[incremento_colonna] < -5, classificazione_colonna] = 'Decremento'
    df_grouped.loc[(df_grouped[incremento_colonna] > 5) & (df_grouped[incremento_colonna] <= 30), classificazione_colonna] = 'Incremento moderato'
    df_grouped.loc[df_grouped[incremento_colonna] > 30, classificazione_colonna] = 'Incremento alto'

    print(df_grouped[[periodo_colonna, 'num_teleassistenze', incremento_colonna, classificazione_colonna]])
    return df_grouped

def etichetta_incremento_per_record(df, df_incremento, periodo_colonna, classificazione_colonna, incremento_label):
    """Labels each record in the main DataFrame with the increment classification."""
    df = df.merge(df_incremento[[periodo_colonna, classificazione_colonna]], on=periodo_colonna, how='left')
    df.rename(columns={classificazione_colonna: incremento_label}, inplace=True)

    print(df[[periodo_colonna, incremento_label]].head())
    return df

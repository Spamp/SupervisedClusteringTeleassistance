

import pandas as pd

def sort_chronologically_by_timestamp(dataframe):
    """Sorts the DataFrame chronologically based on 'ora_inizio_erogazione'."""
    dataframe['ora_inizio_erogazione'] = pd.to_datetime(dataframe['data_erogazione'], errors='coerce')
    sorted_dataframe = dataframe.sort_values(by='ora_inizio_erogazione')
    print("Ecco le prime righe del DataFrame ordinato cronologicamente:")
    return sorted_dataframe

def visualizza_dizionario_chiavi(dizionario, nome_dizionario, limite=10):
    """Displays keys and number of elements for each key in a dictionary."""
    print(f"Visualizzazione del dizionario {nome_dizionario} (limite {limite} chiavi):")
    for i, (chiave, valore) in enumerate(dizionario.items()):
        if i < limite:
            if isinstance(valore, list):
                print(f"Chiave: {chiave}, Numero di elementi: {len(valore)}")
            else:
                print(f"Chiave: {chiave}, Valore: {valore}")
        else:
            break
    print(f"Numero totale di chiavi in {nome_dizionario}: {len(dizionario)}\n")

def visualizza_sub_chiavi(dizionario_mappature):
    """Displays the number of sub-keys for all macro keys in a dictionary."""
    for macro_chiave, sub_chiavi in dizionario_mappature.items():
        num_sub_chiavi = len(sub_chiavi)
        print(f"Macro chiave: {macro_chiave} ha {num_sub_chiavi} sub-chiavi.")

def conta_asl_differenti(df):
    """Counts the number of records where 'asl_residenza' and 'asl_erogazione' differ."""
    record_differenti = df[df['asl_residenza'] != df['asl_erogazione']].shape[0]
    print(f"Numero di record con ASL di residenza diversa da quella di erogazione: {record_differenti}")
    return record_differenti

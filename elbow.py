import matplotlib.pyplot as plt
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Funzione per leggere il file Parquet
def read_file_parquet(filepath):
    df = pd.read_parquet(filepath)
    print(f"Dataset caricato con {len(df)} record.")
    return df

# Funzione per preparare i dati per il clustering
def prepara_dati(df, numerical_columns, categorical_columns, incremento_column):
    df_clustering = df.copy()

    # Gestione delle variabili categoriali
    df_clustering[categorical_columns] = df_clustering[categorical_columns].astype(str)

    # Standardizza le variabili numeriche
    scaler = StandardScaler()
    df_clustering[numerical_columns] = scaler.fit_transform(df_clustering[numerical_columns])

    # Rimuovi eventuali righe con valori mancanti
    columns_to_keep = numerical_columns + categorical_columns + [incremento_column]
    df_clustering = df_clustering[columns_to_keep].dropna()

    # Converte in array numpy
    X_matrix = df_clustering[numerical_columns + categorical_columns].values

    # Indici delle colonne categoriali
    categorical_indices = [df_clustering.columns.get_loc(col) for col in categorical_columns]

    return X_matrix, categorical_indices, df_clustering

# Funzione per eseguire il metodo del gomito per K-Prototypes
def metodo_elbow(df, numerical_columns, categorical_columns, incremento_column, k_min=2, k_max=5):
    X_matrix, categorical_indices, df_clustering = prepara_dati(
        df, numerical_columns, categorical_columns, incremento_column)

    gamma_value = len(numerical_columns) / len(categorical_columns) if len(categorical_columns) > 0 else 0.5

    costi = []
    K_range = range(k_min, k_max + 1)

    for k in K_range:
        print(f"Eseguendo K-Prototypes con k = {k}...")
        kproto = KPrototypes(n_clusters=k, init='Cao', verbose=1, random_state=42, gamma=gamma_value)
        kproto.fit_predict(X_matrix, categorical=categorical_indices)
        costi.append(kproto.cost_)
        print(f"Valore del costo per k = {k}: {kproto.cost_}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(K_range, costi, 'bo-', markersize=8)
    plt.xlabel('Numero di Cluster (k)')
    plt.ylabel('Costo (ncost)')
    plt.title('Metodo del Gomito per K-Prototypes')
    plt.grid(True)
    plt.show()

    return costi

# Funzione per eseguire il metodo del gomito per ogni esperimento
def esegui_elbow_per_esperimenti(df, incremento_column, esperimenti):
    costi_per_esperimento = {}

    for esperimento, (numerical_columns, categorical_columns) in esperimenti.items():
        print(f"\nEsecuzione del Metodo del Gomito per {esperimento}")
        costi = metodo_elbow(df, numerical_columns, categorical_columns, incremento_column)
        costi_per_esperimento[esperimento] = costi

    return costi_per_esperimento

# --- Inizio del flusso principale ---

# Lettura del file Parquet
filepath = './dataset_pulito.parquet'
df = read_file_parquet(filepath)

# Definizione degli esperimenti (colonne numeriche e categoriali)
esperimenti = {
    # 'esperimento_1': (['provincia_erogazione_lat', 'provincia_erogazione_lng'], ['tipologia_professionista_sanitario', 'descrizione_attivita']),
    'esperimento_2': (['attesa_assistenza'], ['struttura_erogazione', 'tipologia_professionista_sanitario', 'asl_erogazione']),
    'esperimento_3': (['età'], ['tipologia_professionista_sanitario', 'descrizione_attivita']),
    'esperimento_4': (['attesa_assistenza', 'durata_assistenza'], []),
    'esperimento_5': (['attesa_assistenza'], ['descrizione_attivita'])
}

# Esegui il metodo del gomito per ogni esperimento
incremento_column = 'incremento_teleassistenza_semestre'  # Colonna di incremento aggiornata
costi_per_esperimento = esegui_elbow_per_esperimenti(df, incremento_column, esperimenti)

# Mostra i costi per ogni esperimento
for esperimento, costi in costi_per_esperimento.items():
    print(f"Costi per {esperimento}: {costi}")
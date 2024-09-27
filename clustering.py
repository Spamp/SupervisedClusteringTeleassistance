import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import seaborn as sns
import time  # Importa il modulo time
from sklearn.metrics import adjusted_rand_score, silhouette_score
import gower
from sklearn.metrics import silhouette_samples

def controlla_valori_numerici(df, numerical_columns):
    """
    Controlla le colonne numeriche per individuare valori non numerici.
    """
    for col in numerical_columns:
        # Seleziona la colonna
        col_data = df[col]
        # Crea una maschera per i valori non numerici
        mask_non_numerici = pd.to_numeric(col_data, errors='coerce').isna() & col_data.notna()
        # Verifica se ci sono valori non numerici
        if mask_non_numerici.any():
            print(f"\nValori non numerici trovati nella colonna '{col}':")
            print(df.loc[mask_non_numerici, [col]])
        else:
            print(f"La colonna '{col}' non contiene valori non numerici.")

def cerca_valore(df, valore='DIE'):
    """
    Cerca il valore specificato in tutte le colonne del DataFrame.
    """
    colonne_con_valore = []
    for col in df.columns:
        if df[col].astype(str).eq(valore).any():
            print(f"\nValore '{valore}' trovato nella colonna '{col}':")
            print(df[df[col].astype(str) == valore][[col]])
            colonne_con_valore.append(col)
    if not colonne_con_valore:
        print(f"Valore '{valore}' non trovato in nessuna colonna.")
    return colonne_con_valore

def read_file_parquet(filepath):
    """
    Legge un file Parquet e ritorna un DataFrame.
    """
    df = pd.read_parquet(filepath)

    # Verifica la struttura del DataFrame per assicurarti che i dati siano stati letti correttamente
    print("Ecco le prime righe del DataFrame:")
    print(df.head())

    # Conta il numero totale di record
    print(f"Numero totale di record: {len(df)}")

    # Stampa i nomi delle colonne
    print("\nColonne presenti nel DataFrame:")
    print(df.columns.tolist())

    return df

def prepara_dati(df, numerical_columns, categorical_columns, additional_columns=[]):
    """
    Prepara i dati per il clustering.

    Parametri aggiuntivi:
    - additional_columns: lista di colonne aggiuntive da mantenere nel DataFrame dei cluster (non incluse in X_matrix)
    """
    # Copia il DataFrame per evitare modifiche indesiderate
    df_clustering = df.copy()

    # Gestione delle variabili categoriali
    df_clustering[categorical_columns] = df_clustering[categorical_columns].astype(str)

    # Correggi i valori non numerici nelle colonne numeriche
    for col in numerical_columns:
        df_clustering[col] = pd.to_numeric(df_clustering[col], errors='coerce')

    # Standardizza le variabili numeriche
    scaler = StandardScaler()
    df_clustering[numerical_columns] = scaler.fit_transform(df_clustering[numerical_columns])

    # Rimuovi eventuali righe con valori mancanti
    columns_to_keep = numerical_columns + categorical_columns + ['incremento_teleassistenza'] + additional_columns
    df_clustering = df_clustering[columns_to_keep].dropna()

    # Converti in array numpy
    X_matrix = df_clustering[numerical_columns + categorical_columns].values

    # Calcola gli indici delle colonne categoriali rispetto a X_matrix
    categorical_indices = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))

    return X_matrix, categorical_indices, df_clustering

def esegui_clustering(X_matrix, categorical_indices, k_clusters=4, gamma=None, n_init=10, max_iter=100):
    """
    Esegue il clustering utilizzando K-Prototypes.

    Parametri:
    - X_matrix: array numpy dei dati
    - categorical_indices: lista degli indici delle colonne categoriali
    - k_clusters: numero di cluster
    - gamma: peso delle variabili numeriche rispetto a quelle categoriali
    - n_init: numero di inizializzazioni
    - max_iter: numero massimo di iterazioni per run

    Restituisce:
    - clusters: array degli assegnamenti dei cluster
    - kproto: modello KPrototypes addestrato
    """
    # Esegui K-Prototypes con i parametri specificati
    kproto = KPrototypes(n_clusters=k_clusters, init='Cao', verbose=1, random_state=42, gamma=gamma, n_init=n_init, max_iter=max_iter)
    clusters = kproto.fit_predict(X_matrix, categorical=categorical_indices)
    return clusters, kproto

def analisi_purezza(df_clustering, clusters):
    """
    Analizza la purezza dei cluster.
    """
    # Aggiungi i cluster al DataFrame
    df_clustering = df_clustering.copy()
    df_clustering['cluster'] = clusters

    print("\nAnalisi della Purezza dei Cluster:")
    total_elements = len(df_clustering)
    purity_sum = 0

    for cluster_label in np.unique(clusters):
        cluster_data = df_clustering[df_clustering['cluster'] == cluster_label]
        cluster_size = len(cluster_data)

        class_counts = cluster_data['incremento_teleassistenza'].value_counts()

        if not class_counts.empty:
            dominant_class_count = class_counts.max()
            cluster_purity = dominant_class_count / cluster_size
            purity_sum += dominant_class_count
            dominant_class = class_counts.idxmax()
        else:
            cluster_purity = 0
            dominant_class_count = 0
            dominant_class = 'Nessuna'

        print(f"Cluster {cluster_label}:")
        print(f"- Dimensione: {cluster_size}")
        print(f"- Classe dominante: {dominant_class} ({dominant_class_count} elementi)")
        print(f"- Purezza del cluster: {cluster_purity:.4f}\n")

    overall_purity = purity_sum / total_elements
    print(f"Purezza complessiva dei cluster: {overall_purity:.4f}")

    return overall_purity

def analisi_fitness(df_clustering, clusters, categorical_indices, additional_columns=[], calcola_silhouette=True):
    """
    Calcola l'Adjusted Rand Index e il Silhouette Score per valutare la qualità del clustering.
    
    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati.
    - clusters: Array degli assegnamenti dei cluster.
    - categorical_indices: Lista degli indici delle colonne categoriali.
    - additional_columns: Lista di colonne aggiuntive da escludere (default: []).
    - calcola_silhouette: Booleano per calcolare o meno il Silhouette Score (default: True).
    
    Restituisce:
    - ari_score: Adjusted Rand Index.
    - sil_score: Silhouette Score (se calcola).
    """
   

    # Calcolo dell'Adjusted Rand Index
    print("\nCalcolo dell'Adjusted Rand Index:")
    labels_true = df_clustering['incremento_teleassistenza'].astype('category').cat.codes
    labels_pred = clusters

    ari_score = adjusted_rand_score(labels_true, labels_pred)
    print(f"Adjusted Rand Index: {ari_score:.4f}")

    sil_score = None
    # Calcolo del Silhouette Score (usando Gower Distance per dati misti)
    if calcola_silhouette:
        print("\nCalcolo del Silhouette Score (Distanza di Gower):")
        
        # Identifica i nomi delle colonne categoriali basate sugli indici
        categorical_columns = df_clustering.columns[categorical_indices].tolist()
        
        # Definisci le colonne da escludere: categoriali, 'cluster', 'incremento_teleassistenza', e additional_columns
        exclude_columns = categorical_columns + ['cluster', 'incremento_teleassistenza'] + additional_columns
        
        # Crea una copia del DataFrame escludendo le colonne categoriali e aggiuntive
        numerical_df = df_clustering.drop(columns=exclude_columns, errors='ignore')
        
        # Crea una copia del DataFrame escludendo le colonne numeriche e aggiuntive
        categorical_df = df_clustering[categorical_columns].copy()
        
        if not numerical_df.empty or not categorical_df.empty:
            # Calcola la matrice di distanza di Gower
            gower_dist = gower.gower_matrix(df_clustering.drop(columns=['cluster', 'incremento_teleassistenza'] + additional_columns, errors='ignore'))
            
            # Calcola il Silhouette Score utilizzando la matrice di distanza precomputata
            if len(np.unique(clusters)) > 1:
                sil_score = silhouette_score(gower_dist, clusters, metric='precomputed')
                print(f"Silhouette Score: {sil_score:.4f}")
            else:
                print("Non è possibile calcolare il Silhouette Score con un solo cluster.")
        else:
            print("Non ci sono variabili numeriche o categoriali per calcolare il Silhouette Score.")
    else:
        print("Calcolo del Silhouette Score saltato.")

    return ari_score, sil_score

# --- Funzioni di Visualizzazione ---

def plot_regione_erogazione_distribution(df_clustering, clusters, region_column='regione_erogazione', k_clusters=4):
    """
    Crea un grafico a barre per ogni cluster mostrando la distribuzione delle regioni di erogazione.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - clusters: array degli assegnamenti dei cluster
    - region_column: nome della colonna che contiene le regioni
    - k_clusters: numero di cluster
    """
    # Aggiungi i cluster al DataFrame
    df_clustering['cluster'] = clusters

    # Definisci il numero di righe e colonne per i subplots
    n_cols = 2
    n_rows = (k_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for cluster_label in range(k_clusters):
        ax = axes[cluster_label]
        subset = df_clustering[df_clustering['cluster'] == cluster_label]
        conteggio_regioni = subset[region_column].value_counts()
        conteggio_regioni.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title(f'Cluster {cluster_label} - Distribuzione Regioni')
        ax.set_xlabel('Regione Erogazione')
        ax.set_ylabel('Conteggio')
        ax.tick_params(axis='x', rotation=45)

    # Rimuovi eventuali subplots vuoti
    for i in range(k_clusters, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def plot_cluster_composition(df_clustering, clusters, numerical_columns, categorical_columns, k_clusters=4):
    """
    Crea istogrammi per le variabili numeriche e grafici a barre per le variabili categoriali per ogni cluster.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - clusters: array degli assegnamenti dei cluster
    - numerical_columns: lista delle colonne numeriche utilizzate per il clustering
    - categorical_columns: lista delle colonne categoriali utilizzate per il clustering
    - k_clusters: numero di cluster
    """
    # Aggiungi i cluster al DataFrame
    df_clustering['cluster'] = clusters

    # Definisci il numero di righe e colonne per i subplots
    n_cols = 2
    n_rows_num = (len(numerical_columns) + n_cols - 1) // n_cols
    n_rows_cat = (len(categorical_columns) + n_cols - 1) // n_cols

    # Plot delle variabili numeriche
    fig_num, axes_num = plt.subplots(n_rows_num, n_cols, figsize=(15, 5 * n_rows_num))
    axes_num = axes_num.flatten()

    for i, col in enumerate(numerical_columns):
        ax = axes_num[i]
        sns.histplot(data=df_clustering, x=col, hue='cluster', multiple='stack', palette='husl', kde=True, ax=ax)
        ax.set_title(f'Distribuzione di {col} per Cluster')
        ax.set_xlabel(col)
        ax.set_ylabel('Conteggio')

    # Rimuovi eventuali subplots vuoti
    for i in range(len(numerical_columns), len(axes_num)):
        fig_num.delaxes(axes_num[i])

    plt.tight_layout()
    plt.show()

    # Plot delle variabili categoriali
    fig_cat, axes_cat = plt.subplots(n_rows_cat, n_cols, figsize=(15, 5 * n_rows_cat))
    axes_cat = axes_cat.flatten()

    for i, col in enumerate(categorical_columns):
        ax = axes_cat[i]
        sns.countplot(data=df_clustering, x=col, hue='cluster', palette='husl', ax=ax)
        ax.set_title(f'Conteggio di {col} per Cluster')
        ax.set_xlabel(col)
        ax.set_ylabel('Conteggio')
        ax.tick_params(axis='x', rotation=45)

    # Rimuovi eventuali subplots vuoti
    for i in range(len(categorical_columns), len(axes_cat)):
        fig_cat.delaxes(axes_cat[i])

    plt.tight_layout()
    plt.show()

def plot_silhouette_scores(X_numeric, clusters, silhouette_avg):
    """
    Crea un grafico del Silhouette Score per ogni campione e mostra il punteggio medio.

    Parametri:
    - X_numeric: array delle variabili numeriche
    - clusters: array degli assegnamenti dei cluster
    - silhouette_avg: punteggio medio del Silhouette Score
    """
    

    silhouette_vals = silhouette_samples(X_numeric, clusters, metric='euclidean')

    y_lower = 10
    fig, ax = plt.subplots(figsize=(10, 6))
    k = len(np.unique(clusters))
    palette = sns.color_palette("husl", k)

    for i in range(k):
        ith_cluster_silhouette_values = silhouette_vals[clusters == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = palette[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7)

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for spacing

    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_title("Silhouette plot per cluster")
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    plt.show()

def plot_purity_incremento(df_clustering, clusters, incremento_column='incremento_teleassistenza', k_clusters=4):
    """
    Crea un grafico a barre per ogni cluster mostrando la distribuzione di incremento_teleassistenza.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - clusters: array degli assegnamenti dei cluster
    - incremento_column: nome della colonna che contiene l'incremento di teleassistenza
    - k_clusters: numero di cluster
    """
    # Aggiungi i cluster al DataFrame
    df_clustering['cluster'] = clusters

    # Definisci il numero di righe e colonne per i subplots
    n_cols = 2
    n_rows = (k_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for cluster_label in range(k_clusters):
        ax = axes[cluster_label]
        subset = df_clustering[df_clustering['cluster'] == cluster_label]
        conteggio_incremento = subset[incremento_column].value_counts()
        conteggio_incremento.plot(kind='bar', ax=ax, color='salmon')
        ax.set_title(f'Cluster {cluster_label} - Distribuzione Incremento')
        ax.set_xlabel('Incremento Teleassistenza')
        ax.set_ylabel('Conteggio')
        ax.tick_params(axis='x', rotation=45)

    # Rimuovi eventuali subplots vuoti
    for i in range(k_clusters, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

# --- Integrazione nel Flusso Principale ---

# Lettura del file Parquet
filepath = './dataset_pulito.parquet'
df = read_file_parquet(filepath)

# Esperimento 1: Epidemia

# Definisci le feature per il clustering
numerical_columns_exp1 = ['provincia_erogazione_lat', 'provincia_erogazione_lng']
categorical_columns_exp1 = [
    'tipologia_professionista_sanitario',
    'descrizione_attivita',
    'regione_erogazione'  # Assicurati di includere 'regione_erogazione' qui
]

# Rimuovi 'regione_erogazione' dalla lista delle colonne aggiuntive poiché ora è categoriale
additional_columns = []  # ['altra_colonna'] se necessario

# Esegui il controllo per valori non numerici nelle colonne numeriche
controlla_valori_numerici(df, numerical_columns_exp1)

# Cerca il valore 'DIE' nel DataFrame
colonne_con_die = cerca_valore(df, valore='DIE')

# Se il valore 'DIE' è presente, identifica le colonne e aggiorna le colonne categoriali
if colonne_con_die:
    print(f"\nAggiornamento delle colonne categoriali con le colonne contenenti 'DIE': {colonne_con_die}")
    for col in colonne_con_die:
        if col not in categorical_columns_exp1:
            categorical_columns_exp1.append(col)
else:
    print("Il valore 'DIE' non è presente nel DataFrame.")

# Assicurati che 'incremento_teleassistenza' sia presente e correttamente codificato
df['incremento_teleassistenza'] = df['incremento_teleassistenza'].astype(str)

# Assicurati che le colonne categoriali siano di tipo stringa
df[categorical_columns_exp1] = df[categorical_columns_exp1].astype(str)

# Assicurati che le colonne numeriche siano di tipo float
df[numerical_columns_exp1] = df[numerical_columns_exp1].astype(float)

# Prepara i dati per il clustering includendo le colonne aggiuntive (ora vuote)
X_matrix_exp1, categorical_indices_exp1, df_clustering_exp1 = prepara_dati(
    df,
    numerical_columns=numerical_columns_exp1,
    categorical_columns=categorical_columns_exp1,
    additional_columns=additional_columns  # Ora vuoto
)

# Definizione delle Coordinate Geografiche per la Verifica
coordinate_columns = ['provincia_erogazione_lat', 'provincia_erogazione_lng']

# Verifica se le coordinate sono presenti nelle feature numeriche
if any(col in numerical_columns_exp1 for col in coordinate_columns):
    # Calcola gamma come rapporto tra numeriche e categoriali
    if len(categorical_columns_exp1) > 0:
        gamma_value_exp1 = len(numerical_columns_exp1) / len(categorical_columns_exp1)
    else:
        gamma_value_exp1 = 0.5  # Valore predefinito se non ci sono colonne categoriali
    print(f"\nValore di gamma calcolato: {gamma_value_exp1:.4f}")
else:
    # Imposta gamma a None per lasciare che K-Prototypes lo calcoli automaticamente
    gamma_value_exp1 = None
    print("\nGamma non impostato. K-Prototypes calcolerà automaticamente il valore di gamma.")

# Verifica gli indici delle colonne categoriali
print("\nIndici delle colonne categoriali rispetto a X_matrix:")
for idx in categorical_indices_exp1:
    print(f"Indice: {idx}, Colonna: {df_clustering_exp1.columns[idx]}")

# Verifica i tipi di dati in X_matrix
print("\nVerifica dei tipi di dati nell'array X_matrix:")
for i, col in enumerate(numerical_columns_exp1 + categorical_columns_exp1):
    if i in categorical_indices_exp1:
        print(f"Colonna {i} ('{col}') è categoriale e dovrebbe essere stringa.")
    else:
        print(f"Colonna {i} ('{col}') è numerica e dovrebbe essere float.")

# Controllo degli unici valori nelle colonne categoriali
for col in categorical_columns_exp1:
    unique_values = df_clustering_exp1[col].unique()
    print(f"\nValori unici nella colonna '{col}':")
    print(unique_values[:10])  # Stampa i primi 10 valori unici

# Misura il tempo di clustering
start_time = time.time()

# Esegui il clustering con n_init e max_iter configurati
clusters_exp1, kproto_exp1 = esegui_clustering(
    X_matrix_exp1,
    categorical_indices_exp1,
    k_clusters=4,
    gamma=gamma_value_exp1,
    n_init=1,        # Numero di run
    max_iter=50      # Numero massimo di iterazioni per run
)

# Fine del timer
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTempo totale di clustering: {elapsed_time:.2f} secondi")

# Analisi della purezza
overall_purity_exp1 = analisi_purezza(df_clustering_exp1, clusters_exp1)

# Analisi della fitness
'''ari_score_exp1, sil_score_exp1 = analisi_fitness(
    df_clustering_exp1, 
    clusters_exp1, 
    categorical_indices_exp1, 
    additional_columns=additional_columns,  # Passa le additional_columns
    calcola_silhouette=True
 )'''

# --- Grafici ---
# 1. Distribuzione delle regioni_erogazione per ogni cluster
plot_regione_erogazione_distribution(
    df_clustering_exp1, 
    clusters_exp1, 
    region_column='regione_erogazione', 
    k_clusters=4
)

# 2. Composizione di ogni cluster
plot_cluster_composition(
    df_clustering_exp1,
    clusters_exp1,
    numerical_columns=numerical_columns_exp1,
    categorical_columns=categorical_columns_exp1,
    k_clusters=4
)

# 3. Grafico dei Silhouette Scores
'''if sil_score_exp1 is not None:
    # Silhouette Score già calcolato con Gower Distance
    plot_silhouette_scores(
        X_numeric=None,  # Non necessario perché già calcolato
        clusters=clusters_exp1, 
        silhouette_avg=sil_score_exp1
    )'''

# 4. Distribuzione di 'incremento_teleassistenza' per ogni cluster
plot_purity_incremento(
    df_clustering_exp1,
    clusters_exp1,
    incremento_column='incremento_teleassistenza',
    k_clusters=4
)

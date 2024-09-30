import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import adjusted_rand_score

# --- Funzioni di Preprocessing ---

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

    # Salva le coordinate originali prima della standardizzazione
    coordinate_cols = ['provincia_erogazione_lat', 'provincia_erogazione_lng']
    for col in coordinate_cols:
        if col in df_clustering.columns:
            df_clustering[col + '_orig'] = df_clustering[col]

    # Standardizza le variabili numeriche
    scaler = StandardScaler()
    df_clustering[numerical_columns] = scaler.fit_transform(df_clustering[numerical_columns])

    # Rimuovi eventuali righe con valori mancanti
    columns_to_keep = numerical_columns + categorical_columns + ['incremento_teleassistenza'] + additional_columns + [col + '_orig' for col in coordinate_cols if col in df_clustering.columns]
    df_clustering = df_clustering[columns_to_keep].dropna()

    # Converti in array numpy
    X_matrix = df_clustering[numerical_columns + categorical_columns].values

    # Calcola gli indici delle colonne categoriali rispetto a X_matrix
    categorical_indices = list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns)))

    return X_matrix, categorical_indices, df_clustering

# --- Funzioni di Clustering e Valutazione ---

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

def analisi_fitness(df_clustering, clusters):
    """
    Calcola l'Adjusted Rand Index per valutare la qualità del clustering.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati.
    - clusters: Array degli assegnamenti dei cluster.

    Restituisce:
    - ari_score: Adjusted Rand Index.
    """
    # Calcolo dell'Adjusted Rand Index
    print("\nCalcolo dell'Adjusted Rand Index:")
    labels_true = df_clustering['incremento_teleassistenza'].astype('category').cat.codes
    labels_pred = clusters

    ari_score = adjusted_rand_score(labels_true, labels_pred)
    print(f"Adjusted Rand Index: {ari_score:.4f}")

    return ari_score

# --- Funzioni per l'Analisi delle Caratteristiche dei Cluster ---

def calcola_statistiche_cluster(df_clustering, clusters, numerical_columns, categorical_columns):
    """
    Calcola statistiche descrittive per ogni cluster.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - clusters: array degli assegnamenti dei cluster
    - numerical_columns: lista delle colonne numeriche
    - categorical_columns: lista delle colonne categoriali

    Restituisce:
    - stats_clusters: dizionario con statistiche per ogni cluster
    """
    df_clustering = df_clustering.copy()
    df_clustering['cluster'] = clusters
    stats_clusters = {}

    for cluster_label in np.unique(clusters):
        cluster_data = df_clustering[df_clustering['cluster'] == cluster_label]
        stats = {}

        # Statistiche sulle variabili numeriche
        stats['numerical'] = cluster_data[numerical_columns].describe().to_dict()

        # Statistiche sulle variabili categoriali
        stats['categorical'] = {}
        for col in categorical_columns:
            value_counts = cluster_data[col].value_counts(normalize=True).head(5)
            stats['categorical'][col] = value_counts.to_dict()

        stats_clusters[cluster_label] = stats

    return stats_clusters

def crea_tabella_riassuntiva(stats_clusters):
    """
    Crea una tabella riassuntiva delle caratteristiche dei cluster.

    Parametri:
    - stats_clusters: dizionario con statistiche per ogni cluster
    """
    for cluster_label, stats in stats_clusters.items():
        print(f"\nCluster {cluster_label}:\n")
        print("Variabili Numeriche:")
        for stat, values in stats['numerical'].items():
            print(f"  {stat}:")
            for var, value in values.items():
                print(f"    {var}: {value:.2f}")
        print("\nVariabili Categoriali:")
        for col, freqs in stats['categorical'].items():
            print(f"  {col}:")
            for category, freq in freqs.items():
                print(f"    {category}: {freq*100:.2f}%")

def assegnare_etichette_cluster(stats_clusters):
    """
    Assegna etichette ai cluster basate sulle caratteristiche distintive.

    Parametri:
    - stats_clusters: dizionario con statistiche per ogni cluster

    Restituisce:
    - etichette_cluster: dizionario che mappa i cluster alle etichette
    """
    etichette_cluster = {}
    for cluster_label, stats in stats_clusters.items():
        # Crea un'etichetta basata sulle categorie più frequenti
        principali_categorie = []
        for col in stats['categorical']:
            categorie = list(stats['categorical'][col].keys())
            if categorie:
                categoria_principale = categorie[0]
                principali_categorie.append(f"{col}: {categoria_principale}")
        etichetta = f"Cluster {cluster_label}"
        if principali_categorie:
            etichetta += " (" + ", ".join(principali_categorie) + ")"
        etichette_cluster[cluster_label] = etichetta
    return etichette_cluster

# --- Funzioni di Visualizzazione ---

def plot_variabili_numeriche(df_clustering, clusters, numerical_columns):
    """
    Crea boxplot per le variabili numeriche per ogni cluster.
    Per le coordinate (latitudine e longitudine), crea uno scatter plot.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - clusters: array degli assegnamenti dei cluster
    - numerical_columns: lista delle colonne numeriche
    """
    df_clustering = df_clustering.copy()
    df_clustering['cluster'] = clusters

    # Identifica le colonne delle coordinate
    coordinate_cols = ['provincia_erogazione_lat', 'provincia_erogazione_lng']
    coordinate_cols_orig = ['provincia_erogazione_lat_orig', 'provincia_erogazione_lng_orig']

    # Verifica se le coordinate sono presenti nelle variabili numeriche
    if all(col in numerical_columns for col in coordinate_cols):
        # Scatter plot per le coordinate
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            x='provincia_erogazione_lng_orig',
            y='provincia_erogazione_lat_orig',
            hue='cluster',
            data=df_clustering.sample(n=10000, random_state=42),  # Campione per velocizzare il plot
            palette='viridis',
            alpha=0.5
        )
        plt.title('Distribuzione Geografica dei Cluster')
        plt.xlabel('Longitudine')
        plt.ylabel('Latitudine')
        plt.legend(title='Cluster')
        plt.show()
        # Rimuovi le coordinate dalle variabili numeriche da plottare come boxplot
        numerical_columns = [col for col in numerical_columns if col not in coordinate_cols]

    # Boxplot per le altre variabili numeriche
    for col in numerical_columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='cluster', y=col, data=df_clustering, order=sorted(df_clustering['cluster'].unique()))
        plt.title(f'Distribuzione di {col} per Cluster')
        plt.show()

def plot_variabili_categoriali(df_clustering, clusters, categorical_columns, additional_categorical=[], top_n=10, frequency_threshold=0.01):
    """
    Crea grafici a barre per le variabili categoriali per ogni cluster, mostrando le categorie che superano una certa soglia di frequenza
    e raggruppando le altre in 'Altre'.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - clusters: array degli assegnamenti dei cluster
    - categorical_columns: lista delle colonne categoriali utilizzate nel clustering
    - additional_categorical: lista di colonne categoriali aggiuntive da plottare
    - top_n: numero massimo di categorie da visualizzare
    - frequency_threshold: soglia minima di frequenza relativa per mantenere una categoria (valore tra 0 e 1)
    """
    df_clustering = df_clustering.copy()
    df_clustering['cluster'] = clusters

    # Combina le colonne categoriali utilizzate nel clustering e quelle aggiuntive
    all_categorical_columns = categorical_columns + additional_categorical

    # Campiona i dati per velocizzare il plot
    df_sample = df_clustering.sample(n=10000, random_state=42)

    for col in all_categorical_columns:
        # Calcola la frequenza relativa delle categorie
        category_frequencies = df_sample[col].value_counts(normalize=True)
        # Identifica le categorie da mantenere
        categories_to_keep = category_frequencies[category_frequencies >= frequency_threshold].index
        # Se il numero di categorie da mantenere supera top_n, seleziona le top_n categorie
        if len(categories_to_keep) > top_n:
            categories_to_keep = category_frequencies.head(top_n).index
        # Raggruppa le categorie
        df_sample[col + '_plot'] = np.where(df_sample[col].isin(categories_to_keep), df_sample[col], 'Altre')
        # Ordina le categorie per la visualizzazione
        category_order = list(categories_to_keep) + ['Altre']

        plt.figure(figsize=(10, 6))
        sns.countplot(x=col + '_plot', hue='cluster', data=df_sample, order=category_order)
        plt.title(f'Distribuzione di {col} per Cluster (Categorie con frequenza >= {frequency_threshold*100:.1f}% o Top {top_n})')
        plt.xticks(rotation=45)
        plt.show()

def plot_variabili_categoriali_con_etichette(df_clustering, cluster_label_col, categorical_columns, additional_categorical=[], top_n=10, frequency_threshold=0.01):
    """
    Crea grafici a barre per le variabili categoriali utilizzando le etichette dei cluster, mostrando le categorie che superano una certa soglia di frequenza
    e raggruppando le altre in 'Altre'.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - cluster_label_col: nome della colonna con le etichette dei cluster
    - categorical_columns: lista delle colonne categoriali utilizzate nel clustering
    - additional_categorical: lista di colonne categoriali aggiuntive da plottare
    - top_n: numero massimo di categorie da visualizzare
    - frequency_threshold: soglia minima di frequenza relativa per mantenere una categoria (valore tra 0 e 1)
    """
    df_clustering = df_clustering.copy()

    # Combina le colonne categoriali utilizzate nel clustering e quelle aggiuntive
    all_categorical_columns = categorical_columns + additional_categorical

    # Campiona i dati per velocizzare il plot
    df_sample = df_clustering.sample(n=10000, random_state=42)

    for col in all_categorical_columns:
        # Calcola la frequenza relativa delle categorie
        category_frequencies = df_sample[col].value_counts(normalize=True)
        # Identifica le categorie da mantenere
        categories_to_keep = category_frequencies[category_frequencies >= frequency_threshold].index
        # Se il numero di categorie da mantenere supera top_n, seleziona le top_n categorie
        if len(categories_to_keep) > top_n:
            categories_to_keep = category_frequencies.head(top_n).index
        # Raggruppa le categorie
        df_sample[col + '_plot'] = np.where(df_sample[col].isin(categories_to_keep), df_sample[col], 'Altre')
        # Ordina le categorie per la visualizzazione
        category_order = list(categories_to_keep) + ['Altre']

        plt.figure(figsize=(10, 6))
        sns.countplot(x=col + '_plot', hue=cluster_label_col, data=df_sample, order=category_order)
        plt.title(f'Distribuzione di {col} per {cluster_label_col} (Categorie con frequenza >= {frequency_threshold*100:.1f}% o Top {top_n})')
        plt.xticks(rotation=45)
        plt.show()

def raggruppa_categorie_rare(df, colonna, soglia=0.01):
    """
    Raggruppa le categorie meno frequenti in una categoria "Altre".

    Parametri:
    - df: DataFrame
    - colonna: nome della colonna categoriale
    - soglia: frequenza minima per mantenere una categoria (valore tra 0 e 1)

    Restituisce:
    - df: DataFrame con le categorie raggruppate
    """
    frequenze = df[colonna].value_counts(normalize=True)
    categorie_da_mantenere = frequenze[frequenze >= soglia].index
    df[colonna] = np.where(df[colonna].isin(categorie_da_mantenere), df[colonna], 'Altre')
    return df

def plot_purity_incremento(df_clustering, clusters, incremento_column='incremento_teleassistenza'):
    """
    Crea un grafico a barre per ogni cluster mostrando la distribuzione di incremento_teleassistenza.

    Parametri:
    - df_clustering: DataFrame dei dati clusterizzati
    - clusters: array degli assegnamenti dei cluster
    - incremento_column: nome della colonna che contiene l'incremento di teleassistenza
    """
    # Aggiungi i cluster al DataFrame
    df_clustering['cluster'] = clusters
    k_clusters = len(np.unique(clusters))

    # Definisci il numero di righe e colonne per i subplots
    n_cols = 2
    n_rows = (k_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, cluster_label in enumerate(sorted(np.unique(clusters))):
        ax = axes[idx]
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

# Definisci il numero di cluster
k_clusters = 4  # Modifica questo valore per i tuoi esperimenti

# Definisci le feature per il clustering
numerical_columns_exp1 = ['provincia_erogazione_lat', 'provincia_erogazione_lng']
categorical_columns_exp1 = [
    'tipologia_professionista_sanitario',
    'descrizione_attivita',
    ]

additional_columns = ['regione_erogazione']

# Esegui il controllo per valori non numerici nelle colonne numeriche
controlla_valori_numerici(df, numerical_columns_exp1)

# Cerca il valore 'DIE' nel DataFrame
colonne_con_die = cerca_valore(df, valore='DIE')

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
df[categorical_columns_exp1 + additional_columns] = df[categorical_columns_exp1 + additional_columns].astype(str)

# Assicurati che le colonne numeriche siano di tipo float
df[numerical_columns_exp1] = df[numerical_columns_exp1].astype(float)

# Raggruppa le categorie rare nelle variabili categoriali
for col in categorical_columns_exp1 + additional_columns:
    df = raggruppa_categorie_rare(df, colonna=col, soglia=0.01)

# Prepara i dati per il clustering includendo le colonne aggiuntive
X_matrix_exp1, categorical_indices_exp1, df_clustering_exp1 = prepara_dati(
    df,
    numerical_columns=numerical_columns_exp1,
    categorical_columns=categorical_columns_exp1,
    additional_columns=additional_columns
)

# --- Calcolo di gamma con controllo ---

geographical_vars = {'provincia_erogazione_lat', 'provincia_erogazione_lng'}
numerical_vars_set = set(numerical_columns_exp1)

if geographical_vars.issubset(numerical_vars_set):
    # Imposta gamma a un valore fisso per ridurre l'influenza delle coordinate
    gamma_value_exp1 = 0.5  # Puoi regolare questo valore in base alle tue esigenze
    print(f"\nVariabili geografiche presenti. Gamma impostato a {gamma_value_exp1}.")
else:
    gamma_value_exp1 = None  # Non impostiamo gamma
    print("\nVariabili geografiche non presenti. Gamma non impostato (utilizzo del valore predefinito).")

# Esegui il clustering con n_init e max_iter configurati
start_time = time.time()
clusters_exp1, kproto_exp1 = esegui_clustering(
    X_matrix_exp1,
    categorical_indices_exp1,
    k_clusters=k_clusters,
    gamma=gamma_value_exp1,
    n_init=1,        # Numero di run
    max_iter=50      # Numero massimo di iterazioni per run
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"\nTempo totale di clustering: {elapsed_time:.2f} secondi")

# Aggiungi i cluster al DataFrame principale
df_clustering_exp1['cluster'] = clusters_exp1

# Analisi della purezza
overall_purity_exp1 = analisi_purezza(df_clustering_exp1, clusters_exp1)

# Analisi della fitness (Adjusted Rand Index)
ari_score_exp1 = analisi_fitness(df_clustering_exp1, clusters_exp1)

# Calcola le statistiche dei cluster
stats_clusters = calcola_statistiche_cluster(
    df_clustering_exp1,
    clusters_exp1,
    numerical_columns=numerical_columns_exp1,
    categorical_columns=categorical_columns_exp1
)

# Crea una tabella riassuntiva delle caratteristiche dei cluster
crea_tabella_riassuntiva(stats_clusters)

# Assegna etichette ai cluster (basate sulle caratteristiche distintive)
etichette_cluster = assegnare_etichette_cluster(stats_clusters)

# Aggiungi le etichette al DataFrame
df_clustering_exp1['cluster_label'] = df_clustering_exp1['cluster'].map(etichette_cluster)

# --- Grafici ---

# 1. Visualizzazione delle variabili numeriche per cluster
plot_variabili_numeriche(df_clustering_exp1, clusters_exp1, numerical_columns_exp1)

# 2. Visualizzazione delle variabili categoriali per cluster (includendo 'regione_erogazione')
plot_variabili_categoriali(
    df_clustering_exp1,
    clusters_exp1,
    categorical_columns=categorical_columns_exp1,
    additional_categorical=additional_columns,
    top_n=10,
    frequency_threshold=0.01
)

# 3. Distribuzione di 'incremento_teleassistenza' per ogni cluster
plot_purity_incremento(
    df_clustering_exp1,
    clusters_exp1,
    incremento_column='incremento_teleassistenza'
)

# 4. Visualizzazione delle variabili categoriali con le etichette dei cluster (includendo 'regione_erogazione')
plot_variabili_categoriali_con_etichette(
    df_clustering_exp1,
    cluster_label_col='cluster_label',
    categorical_columns=categorical_columns_exp1,
    additional_categorical=additional_columns,
    top_n=10,
    frequency_threshold=0.01
)

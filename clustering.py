import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt

# Supponiamo che 'df' sia il tuo DataFrame originale
def read_file_parquet(filepath):
    # Leggi il file Parquet
    df = pd.read_parquet(filepath)

    # Verifica la struttura del DataFrame per assicurarti che i dati siano stati letti correttamente
    print("Ecco le prime righe del DataFrame:")
    print(df.head())

    # Conta il numero totale di record
    print(f"Numero totale di record: {len(df)}")
    return df

# Funzione per preparare i dati per il clustering
def prepara_dati(df, numerical_columns, categorical_columns, include_coords=False):
    # Copia il DataFrame per evitare modifiche indesiderate
    df_clustering = df.copy()

    # Gestione delle variabili categoriali
    df_clustering[categorical_columns] = df_clustering[categorical_columns].astype(str)

    # Standardizza le variabili numeriche
    scaler = StandardScaler()
    df_clustering[numerical_columns] = scaler.fit_transform(df_clustering[numerical_columns])

    # Inclusione delle coordinate geografiche
    if include_coords:
        coord_columns = ['provincia_erogazione_lat', 'provincia_erogazione_lng']
        df_clustering[coord_columns] = df_clustering[coord_columns].astype(float)
        # Standardizza le coordinate
        df_clustering[coord_columns] = scaler.fit_transform(df_clustering[coord_columns])
        numerical_columns.extend(coord_columns)

    # Rimuovi eventuali righe con valori mancanti
    df_clustering = df_clustering[numerical_columns + categorical_columns + ['incremento_teleassistenza']].dropna()

    # Converti in array numpy
    X_matrix = df_clustering[numerical_columns + categorical_columns].values

    # Indici delle colonne categoriali
    categorical_indices = [df_clustering.columns.get_loc(col) for col in categorical_columns]

    return X_matrix, categorical_indices, df_clustering

# Funzione per eseguire il clustering
def esegui_clustering(X_matrix, categorical_indices, k_clusters=5, gamma=None):
    # Esegui K-Prototypes con il parametro gamma per bilanciare le variabili
    kproto = KPrototypes(n_clusters=k_clusters, init='Cao', verbose=1, random_state=42, gamma=gamma)
    clusters = kproto.fit_predict(X_matrix, categorical=categorical_indices)
    return clusters, kproto

# Funzione per analizzare i cluster e creare istogrammi delle regioni
def analisi_post_clustering(df_clustering, clusters, dizionario_regioni, coord_columns=['provincia_erogazione_lat', 'provincia_erogazione_lng']):
    # Aggiungi i cluster al DataFrame
    df_clustering['cluster'] = clusters

    # Mappa le coordinate alle regioni utilizzando il dizionario
    # Supponendo che 'dizionario_regioni' sia una funzione o un mapping che data una coppia di coordinate restituisce la regione
    def mappa_coordinate_a_regione(row):
        lat = row[coord_columns[0]]
        lng = row[coord_columns[1]]
        return ottieni_regione_da_coordinate(lat, lng, dizionario_regioni)

    df_clustering['regione'] = df_clustering.apply(mappa_coordinate_a_regione, axis=1)

    # Crea istogrammi delle concentrazioni delle regioni nei singoli cluster
    for cluster_label in df_clustering['cluster'].unique():
        dati_cluster = df_clustering[df_clustering['cluster'] == cluster_label]
        conteggio_regioni = dati_cluster['regione'].value_counts()
        conteggio_regioni.plot(kind='bar', title=f'Cluster {cluster_label} - Distribuzione delle Regioni')
        plt.xlabel('Regione')
        plt.ylabel('Conteggio')
        plt.show()

def ottieni_regione_da_coordinate(lat, lng, dizionario_regioni):
    # Percorri il dizionario per trovare la regione corrispondente
    for codice_regione, regione_data in dizionario_regioni.items():
        for codice_provincia, provincia_data in regione_data['province'].items():
            provincia_lat = provincia_data.get('lat')
            provincia_lng = provincia_data.get('lng')
            # Confronta le coordinate (puoi usare una soglia di tolleranza)
            if np.isclose(lat, provincia_lat, atol=0.1) and np.isclose(lng, provincia_lng, atol=0.1):
                return regione_data['nome_regione']
    return 'Regione Sconosciuta'

# Funzione per l'analisi della purezza
def analisi_purezza(df_clustering, clusters):
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

# Funzione per l'analisi della fitness
def analisi_fitness(df_clustering, clusters, categorical_indices, calcola_silhouette=True):
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    # Calcolo dell'Adjusted Rand Index
    print("\nCalcolo dell'Adjusted Rand Index:")
    labels_true = df_clustering['incremento_teleassistenza'].astype('category').cat.codes
    labels_pred = clusters

    ari_score = adjusted_rand_score(labels_true, labels_pred)
    print(f"Adjusted Rand Index: {ari_score:.4f}")

    sil_score = None
    # Calcolo del Silhouette Score (solo per variabili numeriche)
    if calcola_silhouette:
        print("\nCalcolo del Silhouette Score (variabili numeriche):")
        # Indici delle colonne numeriche
        numerical_indices = [i for i in range(df_clustering.shape[1]) if i not in categorical_indices and df_clustering.columns[i] != 'cluster']

        if numerical_indices:
            X_numeric = df_clustering.iloc[:, numerical_indices].values
            if len(np.unique(clusters)) > 1:
                sil_score = silhouette_score(X_numeric, clusters, metric='euclidean')
                print(f"Silhouette Score: {sil_score:.4f}")
            else:
                print("Non è possibile calcolare il Silhouette Score con un solo cluster.")
        else:
            print("Non ci sono variabili numeriche per calcolare il Silhouette Score.")
    else:
        print("Calcolo del Silhouette Score saltato.")

    return ari_score, sil_score

filepath=''
# Leggi il file Parquet
df = read_file_parquet(filepath)
# Esperimento 1: Epidemia

# Definisci le feature
numerical_columns_exp1 = []
categorical_columns_exp1 = [
    'codice_tipologia_professionista_sanitario',
    'codice_descrizione_attivita'
]

# Assicurati che 'incremento_teleassistenza' sia presente e correttamente codificato
df['incremento_teleassistenza'] = df['incremento_teleassistenza'].astype(str)

# Prepara i dati
X_matrix_exp1, categorical_indices_exp1, df_clustering_exp1 = prepara_dati(
    df,
    numerical_columns=numerical_columns_exp1,
    categorical_columns=categorical_columns_exp1,
    include_coords=True  # Includiamo le coordinate
)

# Calcola gamma
gamma_value_exp1 = len(numerical_columns_exp1) / len(categorical_columns_exp1) if len(categorical_columns_exp1) > 0 else 0.5

# Esegui il clustering
clusters_exp1, kproto_exp1 = esegui_clustering(
    X_matrix_exp1,
    categorical_indices_exp1,
    k_clusters=5,
    gamma=gamma_value_exp1
)

# Analisi della purezza
overall_purity_exp1 = analisi_purezza(df_clustering_exp1, clusters_exp1)

# Analisi della fitness
ari_score_exp1, sil_score_exp1 = analisi_fitness(df_clustering_exp1, clusters_exp1, categorical_indices_exp1, calcola_silhouette=True)

# Analisi post-clustering (includiamo ora la chiamata alla funzione)
#analisi_post_clustering(df_clustering_exp1, clusters_exp1, dizionario_regioni)

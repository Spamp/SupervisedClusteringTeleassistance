# SupervisedClusteringTeleassistance


Clustering Project

This repository performs clustering analysis on a dataset using K-Prototypes and K-Means algorithms. It includes data preprocessing, feature engineering, determining the optimal number of clusters, and visualizing the results.
Table of Contents

    Installation
    Repository Structure
    Usage
        1. Preprocessing and Feature Engineering
        2. Determine Optimal Number of Clusters (Elbow Method)
        3. Clustering and Analysis
    Adjusting Parameters
        Selecting Features
        Hyperparameters
    Visualization
    Additional Information

Installation
Prerequisites

    Python 3.6 or higher
    Git

Steps

    Clone the Repository

    bash

git clone https://github.com/your-username/clustering-project.git
cd clustering-project

Install Required Libraries

All necessary libraries are listed in the requirements.txt file. Install them using:

bash

    pip install -r requirements.txt


Usage
1. Preprocessing and Feature Engineering (main_organizzato.py)

Prepare the dataset by preprocessing and feature engineering.

bash

python main_organizzato.py

Output: dataset_pulito.parquet
2. Determine Optimal Number of Clusters (Elbow Method) (elbow.py)

Use the Elbow Method to find the best number of clusters for your experiments.

bash

python elbow.py

Note: Define your feature combinations in the script as needed.
3. Clustering and Analysis (clustering.py)

Perform clustering using K-Prototypes and analyze the results.

bash

python clustering.py

Output:

    Numerical results in the terminal
    Visual plots (scatter plots, boxplots, bar charts)

Adjusting Parameters
Selecting Features

To run experiments with different features:

    Open clustering.py.

    Locate Feature Selection:
        Numerical Features: Around line 470
        Categorical Features: Around line 471

    Insert Your Desired Features:

    python

    # Example:
    numerical_columns_exp1 = ['età']
    categorical_columns_exp1 = ['tipologia_professionista_sanitario']

Available Features: Refer to main_organizzato.py starting from line 139.
Hyperparameters

Modify the clustering parameters directly in clustering.py:

    Number of Clusters (k): Line 467

    python

k_clusters = 4

Number of Runs (n_init): Line 537

python

n_init = 4

Max Iterations (max_iter): Line 538

python

    max_iter = 50

Visualization

After running clustering.py, several plots will appear sequentially:

    Scatter Plot: Geographical distribution of clusters.
    Boxplots: Distribution of numerical variables per cluster.
    Bar Charts: Distribution of categorical variables per cluster.
    Incremento Teleassistenza Distribution: Target variable distribution within clusters.

Note: Save plots using the save button in the plot window. Close each plot to view the next one.
Additional Information
Coordinates Generation

To generate coordinates.json from dataset.parquet:

    Navigate to the coordinates Folder:

    bash

cd coordinates

Run the Script:

bash

    python generate_coordinates.py

Files:

    coordinates.json: Contains the coordinates of cities found in the dataset.
    dataset.parquet: Original dataset file used for generating coordinates.

Elbow Method Experiments

In elbow.py, define different feature combinations using a dictionary to generate elbow plots and identify the optimal k for each experiment.
Clustering Interactions

The clustering.py script allows you to:

    Numerical Variables: Modify which numerical features to include.
    Categorical Variables: Choose which categorical features to consider.
    Hyperparameters: Adjust k, n_init, and max_iter to fine-tune the clustering models.

Contact

For any questions or support, please contact your-email@example.com.


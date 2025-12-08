#!/usr/bin/env python
# coding: utf-8

import gdown
import pandas as pd


# Import des bibliothèques de viz
import matplotlib.pyplot as plt
import seaborn as sns

# Import split data
from sklearn.model_selection import train_test_split

# Import modèle de ML NON Supervisé
from sklearn.neighbors import NearestNeighbors

# Import outil standardisation de la donnée
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

# Import pipeline
from sklearn.pipeline import Pipeline

# Gestion des warnings
import warnings


# import du csv data_ml
file = "1h6zW1eKmDVZCJ2d2prtjVV0HqxlPMGP9"
gdown.download(f"https://drive.google.com/uc?id={file}", "data_ml.csv", quiet=False)
df = pd.read_csv("data_ml.csv")


# supprimer les films avec valeurs nulles dans Date de création et genres
# ne garder que les films avec minimum 500 votes et une note de plus de 1 (exclut les Nan en même temps) 
df = df.dropna(subset=['Date de création','genres'])
df_noted = df[(df['Note moyenne'] > 1.0) & (df['numVotes']>500)].copy()

df_noted['Date de création'] = df_noted['Date de création'].astype(int)

df_noted['Titre originale LC'] = df_noted['Titre originale'].str.lower()


df_noted.info()



# créer un nouvelle colonne note_ponderee qui permettra de trier les voisins
C = df_noted['Note moyenne'].mean()
m = df_noted['numVotes'].quantile(0.5) # pour les films en-dessous de m (environ 1800), la note sera tiré vers la moyenne

df_noted['Note_ponderee'] = (
    (df_noted['numVotes'] / (df_noted['numVotes'] + m)) * df_noted['Note moyenne']
    + (m / (df_noted['numVotes'] + m)) * C
)


# features
X = df_noted[['Action','Adventure','Animation','Biography','Comedy','Crime',
                'Documentary','Drama','Family','Fantasy','Film-Noir', 'History','Horror',
                'Music','Musical','Mystery','News', 'Romance','Sci-Fi','Short','Sport',
                'Thriller','War','Western','Date de création']]
num_features = ['Date de création']
bin_features = ['Action','Adventure','Animation','Biography','Comedy','Crime',
                'Documentary','Drama','Family','Fantasy','Film-Noir', 'History','Horror',
                'Music','Musical','Mystery','News', 'Romance','Sci-Fi','Short','Sport',
                'Thriller','War','Western']


# standardiser les colonnes 

preprocessor = ColumnTransformer(
    transformers=[
        ('num',MinMaxScaler(), num_features),
        ('bin','passthrough', bin_features)
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', NearestNeighbors(metric='euclidean'))
])



# créer le modèle 
nn_model = NearestNeighbors(metric='euclidean')

# entraîner sur les données standardisées X_scaled
pipeline.fit(X) 


def recommander_film(titre, n=10):

    # passer le titre cible en minuscule
    titre = titre.lower()
    
    # trouver le film dans le df_noted
    film_row = df_noted[df_noted['Titre originale LC'] == titre]

    if film_row.empty:
        return f"Le film '{titre}' n'a pas été trouvé dans le jeu de données d'entraînement."

    # utiliser l'index du premier résultat trouvé (peut-etre changer en dernier ?
    idx_label = film_row.index[0]
    
    # extraire le vecteur de features NON SCALÉ du film cible, en utilisant .loc (label index)
    film_df_unscaled = X.loc[[idx_label]]
    
    # Trouver 50 voisins sur le DataFrame de features non scalées (film_df_unscaled)
    distances, indices = pipeline.named_steps['model'].kneighbors(
        pipeline.named_steps['preprocessor'].transform(film_df_unscaled), # On transforme la donnée AVANT de l'envoyer à kneighbors
        n_neighbors=50
    )

    # Les indices renvoyés par nn_model sont des indices POSITIONNELS dans X_scaled/df_noted
    # utiliser .iloc sur df_noted pour récupérer les lignes correspondantes
    indices = indices[0]
    
    # récupérer les données complètes des films trouvés
    recos = df_noted.iloc[indices][['Titre originale', 'genres', 'Date de création', 'Note moyenne',"numVotes", 'Note_ponderee', 'directeur_name','Titre originale LC']]

    # exclure explicitement le film cible par titre ou index
    recos = recos[recos.index != idx_label]
    recos = recos[recos['Titre originale LC'] != titre]
    
    # trier les recommandations par Note pondérée décroissante
    recos_triées = recos.sort_values(by='Note_ponderee', ascending=False)

    # retourner le top 10
    return recos_triées[['Titre originale', 'genres', 'Date de création', 'Note moyenne','directeur_name']].head(n)


recommander_film("Pulp Fiction", n=10)



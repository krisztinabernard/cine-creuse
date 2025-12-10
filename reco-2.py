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

# Streamlit pour l'afichage
import streamlit as st

import base64

# Cache du loading et dernier procession de la data
@st.cache_data(show_spinner=True) # Affiche un message de chargement une seule fois
def load_data_and_preprocess():
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
    df_noted['titre_FR_LC'] = df_noted['titre_FR'].str.lower()

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
    return df_noted, X

# Cache du pipeline
@st.cache_resource(show_spinner=True)
def create_and_fit_pipeline(X):
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
    return pipeline

st.set_page_config(
    page_title="Recommendation de films",
    initial_sidebar_state="expanded"
)

# Configuration pour forcer un thème sombre avec des couleurs personnalisées
st.markdown(
    """
    <style>
    /* 1. Couleurs de base du thème */
    :root {
        --primary-color: #A0522D; /* Marron pour les accents */
        --background-color: transparent; 
        --secondary-background-color: rgba(0, 0, 0, 0.6); 
        --text-color: #F5F5DC; /* Beige clair */
        --font: sans-serif;
    }
    
    /* 2. Assurer que TOUT le texte est clair (Lisibilité générale) */
    h1, h2, h3, h4, p, li, label, .st-bh, .st-bq, .st-bb, .st-be, .st-bu, .st-bv {
        color: #F5F5DC !important; 
    }
    
    /* 3. Assurer que le fond des conteneurs est sombre/semi-transparent */
    .stForm, .stTextInput > div > div, .stDataFrame, .stExpander {
        background-color: rgba(0, 0, 0, 0.7) !important; 
        border: 1px solid #A0522D !important; 
        border-radius: 8px;
    }

    /* 4. Style spécifique pour le champ de saisie (input) */
    div.stTextInput input {
        color: #F5F5DC !important; 
        background-color: rgba(50, 50, 50, 0.8) !important; 
        border: 1px solid #A0522D; 
    }
    div.stTextInput input::placeholder {
        color: #D3D3D3 !important;
        opacity: 0.8;
    }

    /* Style spécifique pour le bouton - par son ID interne (le plus fort) */
    div[data-testid="stFormSubmitButton"] button {
        background-color: #DAA520 !important; /* FOND DU BOUTON : DORÉ (Gold) */
        color: #000000 !important; /* Texte sur le bouton : NOIR */
        border: none !important;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.5); 
        font-weight: bold;
    }

    /* La couleur du bouton en hover */
    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #B8860B !important; /* Nuance de doré plus foncée pour l'effet de survol */
        color: #000000 !important;
    }

    /* La couleur du bouton quand il est actif (cliqué) */
    div[data-testid="stFormSubmitButton"] button:active {
        background-color: #B8860B !important; /* Reste doré */
        color: #000000 !important;
    }
    
    /* 5. Changer la couleur des liens/icônes si nécessaire */
    .css-1d3w5ec, .css-1dp5r7f {
        color: #F5F5DC !important;
    }

    .st-emotion-lightgrey h4, .st-emotion-lightgrey h3 {
    margin-top: 0.5rem !important; /* Réduit la marge supérieure */
    margin-bottom: 0.5rem !important; /* Réduit la marge inférieure */
    line-height: 1.2; /* Assure que les lignes sont proches */
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

# --- FIN de la configuration du Thème Sombre ---

def add_bg_from_local(image_file):
    """
    Encode une image locale en Base64 et l'utilise comme fond d'écran pour Streamlit.
    Ajoute également un calque sombre semi-transparent (overlay) pour améliorer la lisibilité.
    """
    try:
        # 1. Lecture du fichier et encodage en Base64
        with open(image_file, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        
        bg_css = f"data:image/png;base64,{data}"
        
        # 2. Injection du CSS dans Streamlit
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url({bg_css});
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: center;
            }}
            /* Ajout d'un overlay (calque) pour le contraste */
            .stApp::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0, 0, 0, 0.4); /* 40% d'opacité noire */
                z-index: -1; 
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier d'image '{image_file}' n'a pas été trouvé.")
    except Exception as e:
        st.error(f"Erreur lors de l'application de l'image de fond : {e}")


add_bg_from_local('img/bg.png')

df_noted, X = load_data_and_preprocess()
pipeline = create_and_fit_pipeline(X)

def recommander_film(titre, n=10):

    # passer le titre cible en minuscule
    titre = titre.lower()
    
    # trouver le film dans le df_noted : chercher d'abord dans les titres FR LC, et si non trouve, dans titre originale
    film_row = df_noted[
        (df_noted['titre_FR_LC'] == titre) | 
        (df_noted['Titre originale LC'] == titre)]
    
    if film_row.empty:
        st.error(f"Le film **'{titre}'** n'est pas connu dans notre outil de recommandation")
        return

    # utiliser l'index du premier résultat trouvé (peut-etre changer en dernier ?
    idx_label = film_row.index[0]
    
    # extraire le vecteur de features NON SCALÉ du film cible, en utilisant .loc (label index)
    film_df_unscaled = X.loc[[idx_label]]
    
    # Trouver 50 voisins sur le DataFrame de features non scalées (film_df_unscaled)
    distances, indices = pipeline.named_steps['model'].kneighbors(
        pipeline.named_steps['preprocessor'].transform(film_df_unscaled), # On transforme la donnée AVANT de l'envoyer à kneighbors
        n_neighbors=50
    )

    # Les indices renvoyés par nn_model sont des indices positionnels dans X_scaled/df_noted
    # utiliser .iloc sur df_noted pour récupérer les lignes correspondantes
    indices = indices[0]
    
    # récupérer les données complètes des films trouvés
    recos = df_noted.iloc[indices][['Titre originale', 'genres', 'Date de création', 'Note moyenne',"numVotes", 'Note_ponderee', 'directeur_name','Titre originale LC','titre_FR','langue_originale','chemin_affiche','description', 'Durée (Min)']]


    # exclure explicitement le film cible par titre ou index
    recos = recos[recos.index != idx_label]
    recos = recos[recos['Titre originale LC'] != titre]
    
    # trier les recommandations par Note pondérée décroissante
    recos_triées = recos.sort_values(by='Note_ponderee', ascending=False)

    # retourner le top 10
    return recos_triées[['Titre originale', 'genres', 'Date de création', 'Note moyenne', 'numVotes','directeur_name', 'titre_FR','langue_originale','chemin_affiche','description', 'Durée (Min)']].head(n)


st.title("Recommendation de films")

st.header("Quel est votre film favori ?")

with st.form(key='mon_formulaire'):
    titre = st.text_input(label='Mon film favori')
    submit_form = st.form_submit_button(label="Recommande-moi des films")

# --- NOUVEL AFFICHAGE DES RÉSULTATS (Carte/Grille) ---

if submit_form:
    resultat = recommander_film(titre, n=10)
    
    if resultat is not None:
        st.subheader(f"Top {len(resultat)} des films similaires à **{titre.title()}** :")
        
        # Le nombre de colonnes
        COLUMNS_PER_ROW = 3 
        
        # Pour itérer sur les résultats et créer les lignes de colonnes
        for i in range(0, len(resultat), COLUMNS_PER_ROW):
            # Créer une ligne de 3 colonnes pour les films i, i+1, i+2
            cols = st.columns(COLUMNS_PER_ROW)
            
            # Traiter les films de cette ligne
            for j in range(COLUMNS_PER_ROW):
                if i + j < len(resultat):
                    row = resultat.iloc[i + j]
                    
                    # Utiliser l'index de la colonne pour placer la carte
                    with cols[j]:
                        
                        # --- DÉBUT DE LA CARTE DE FILM ---
                        # Utiliser un conteneur pour simuler la "carte" et appliquer un fond
                        with st.container(border=True): 
                            
                            # 1. AFFICHE
                            chemin_affiche = row['chemin_affiche']
                            if pd.notna(chemin_affiche):
                                url_affiche = f"https://image.tmdb.org/t/p/w500{chemin_affiche}"
                            else:
                                url_affiche = "https://via.placeholder.com/200x300.png?text=Aucune+affiche"
                            
            
                            st.image(url_affiche, width=180) # Réduire la taille de l'image pour les cartes
                            
                            
                            # 2. INFORMATIONS CLÉS
                            # H4 pour le titre pour qu'il ne prenne pas trop de place
                            st.markdown(f"**{row['Titre originale']}**") 

                            # Note et Année
                            note_format = f"{row['Note moyenne']:.1f}/10"
                            annee_format = f"({row['Date de création']})"
                            
                            st.markdown(f"⭐ **Note :** {note_format} {annee_format}")
                            
                            # Genres
                            st.markdown(f"**Genres :** {row['genres'].split(',')[0]}...") # Afficher juste le premier genre pour la concision

                            # 3. BOUTON DÉTAILS (Simuler un Expand/Popup)
                            # On utilise un expander DANS la carte pour afficher le reste des infos
                            with st.expander("Voir les détails"):
                                if pd.notna(row['titre_FR']):
                                    st.markdown(f"**Titre Français :** {row['titre_FR']}")
                                
                                st.markdown(f"**Directeur :** {row['directeur_name']}")
                                
                                if pd.notna(row['Durée (Min)']) and row['Durée (Min)'] > 0:
                                    st.markdown(f"**Durée :** {int(row['Durée (Min)'])} min")
                                
                                if pd.notna(row['description']):
                                    st.markdown(f"**Synopsis : **")
                                    st.caption(row['description'][:150] + "...") # Aperçu de la description
                                
                        # --- FIN DE LA CARTE DE FILM ---
from datetime import datetime

import streamlit as st

## ICI il y aura la Web app qui appelle les différentes fonctions et trucs à afficher de backend.py
# Supposons que votre script converti s'appelle notebook_script.py
from backend import plotAssetPrice

# Titre de l'application
st.title('Application Web utilisant Streamlit et un Notebook Jupyter')

asset = st.selectbox('Sélectionnez un asset', options=['S&P 500 PRICE IN USD', 'Autre Asset 1', 'Autre Asset 2'])

# Sélection des dates de début et de fin
startDate = st.date_input('Date de début', value=datetime(2022, 1, 1))
endDate = st.date_input('Date de fin', value=datetime(2023, 1, 1))

# Bouton pour générer le graphique
if st.button('Afficher le graphique'):
    try:
        # Appel de la fonction backend pour générer le graphique
        buf = plotAssetPrice(asset, startDate, endDate)

        # Affichage du graphique
        st.image(buf, use_column_width=True)

    except ValueError as e:
        st.error(e)